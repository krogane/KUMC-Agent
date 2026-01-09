from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import AppConfig, LLM_CHUNK_SYSTEM_PROMPT, build_llm_chunk_prompt
from indexing.constants import FILE_ID_SEPARATOR
from indexing.utils import ensure_dir, sanitize_filename

logger = logging.getLogger(__name__)



def _extract_drive_file_id(filename: str) -> str | None:
    if FILE_ID_SEPARATOR not in filename:
        return None
    prefix, _ = filename.split(FILE_ID_SEPARATOR, 1)
    return prefix or None


def _load_drive_metadata(source_path: Path) -> dict[str, str]:
    meta_path = source_path.with_suffix(source_path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read metadata sidecar %s: %s", meta_path.name, exc)
        return {}

    if not isinstance(data, dict):
        logger.warning("Invalid metadata sidecar %s: expected object", meta_path.name)
        return {}

    metadata: dict[str, str] = {}
    for key in ("drive_file_id", "drive_file_name", "drive_path", "drive_mime_type"):
        value = data.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value
    return metadata


def _build_splitter(
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
    )


def chunk_markdown_to_jsonl(
    *,
    raw_data_dir: Path,
    chunk_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
    source_type: str,
    file_extensions: Sequence[str] = (".md",),
) -> None:
    ensure_dir(chunk_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    input_files: list[Path] = []
    for ext in file_extensions:
        input_files.extend(raw_data_dir.rglob(f"*{ext}"))
    input_files = sorted(set(input_files), key=lambda path: str(path))
    if not input_files:
        logger.warning(
            "No files found under %s for extensions: %s",
            raw_data_dir,
            ", ".join(file_extensions),
        )
        return

    splitter = _build_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    for f in input_files:
        text = f.read_text(encoding="utf-8")
        rel_path = f.relative_to(raw_data_dir)
        metadata = {
            "source": f.name,
            "source_path": str(rel_path),
            "source_type": source_type,
        }
        metadata.update(_load_drive_metadata(f))
        if "drive_file_path" not in metadata and "drive_path" in metadata:
            metadata["drive_file_path"] = metadata["drive_path"]
        if "drive_file_id" not in metadata:
            drive_file_id = _extract_drive_file_id(f.name)
            if drive_file_id:
                metadata["drive_file_id"] = drive_file_id

        docs = splitter.split_documents(
            [Document(page_content=text, metadata=metadata)]
        )

        safe_rel = sanitize_filename(str(rel_path).replace(os.sep, "__"))
        out_path = chunk_dir / f"{safe_rel}.jsonl"
        with out_path.open("w", encoding="utf-8") as fw:
            for i, doc in enumerate(docs):
                record = {
                    "text": doc.page_content,
                    "metadata": {
                        **(doc.metadata or {}),
                        "chunk_id": i,
                    },
                }
                fw.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Chunked %s -> %s (%d chunks)", f.name, out_path.name, len(docs))


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fr:
        for line_no, line in enumerate(fr, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name} at line {line_no}: {exc}"
                ) from exc

            text = obj.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"Missing/invalid 'text' in {path.name} line {line_no}"
                )

            metadata = obj.get("metadata")
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError(
                    "Missing/invalid 'metadata' (must be dict) in "
                    f"{path.name} line {line_no}"
                )

            records.append({"text": text, "metadata": metadata})

    return records




def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _strip_trailing_output_comma(text: str) -> str:
    stripped = text.rstrip()
    cleaned = re.sub(r",(?=\s*\"\s*]\s*$)", "", stripped)
    cleaned = re.sub(r",(?=\s*]\s*$)", "", cleaned)
    if cleaned == stripped:
        return text
    trailing = text[len(stripped) :]
    return cleaned + trailing


def _strip_trailing_broken_quote(text: str) -> str:
    stripped = text.rstrip()
    cleaned = re.sub(r"(?m)^\s*\"\s*]\s*$", "]", stripped)
    if cleaned == stripped:
        return text
    trailing = text[len(stripped) :]
    return cleaned + trailing


def _parse_llm_chunks(response: str, *, source_name: str) -> list[str]:
    if not response:
        raise ValueError(f"Empty LLM response for {source_name}")

    payload = _strip_code_fences(response)
    payload = re.sub(r"\\(?![\"\\/bfnrtu])", "", payload)
    payload = _strip_trailing_output_comma(payload)
    payload = _strip_trailing_broken_quote(payload)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from LLM for {source_name}: {exc}") from exc

    if isinstance(data, dict):
        data = data.get("chunks")

    if not isinstance(data, list):
        raise ValueError(f"LLM output must be a list for {source_name}")

    chunks: list[str] = []
    for item in data:
        if not isinstance(item, str):
            raise ValueError(f"LLM chunk must be string for {source_name}")
        trimmed = item.strip()
        if trimmed:
            chunks.append(trimmed)

    if not chunks:
        raise ValueError(f"LLM produced no chunks for {source_name}")

    return chunks


def _generate_with_gemini_chunk(
    *,
    api_key: str,
    prompt: str,
    config: AppConfig,
) -> str:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for LLM chunking with Gemini."
        ) from exc

    client = _genai_client(api_key)
    response = client.models.generate_content(
        model=config.llm_chunk_model,
        contents=[
            {"role": "system", "parts": [{"text": LLM_CHUNK_SYSTEM_PROMPT}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ],
        config=genai.types.GenerateContentConfig(
            temperature=config.llm_chunk_temperature,
            max_output_tokens=config.llm_chunk_max_output_tokens,
            thinking_config=genai.types.ThinkingConfig(
                thinking_level=config.thinking_level
            ),
        ),
    )
    return (response.text or "").strip()


def _generate_with_llama_chunk(*, prompt: str, config: AppConfig) -> str:
    llama = _llama_client(
        config.llm_chunk_llama_model_path,
        config.llama_ctx_size,
        config.llama_threads,
        config.llama_gpu_layers,
    )
    result = llama.create_chat_completion(
        messages=[
            {"role": "system", "content": LLM_CHUNK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=config.llm_chunk_max_output_tokens,
        temperature=config.llm_chunk_temperature,
    )
    return (
        (result.get("choices", [{}])[0].get("message", {}) or {}).get("content")
        or ""
    ).strip()


@lru_cache(maxsize=1)
def _genai_client(api_key: str):
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Please set it in .env")
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for LLM chunking with Gemini."
        ) from exc
    return genai.Client(api_key=api_key)


@lru_cache(maxsize=1)
def _llama_client(
    model_path: str,
    ctx_size: int,
    threads: int,
    gpu_layers: int,
):
    if not model_path:
        raise RuntimeError(
            "LLM_CHUNK_LLAMA_MODEL_PATH is not set. Please set it in .env"
        )

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Please install it to use llama.cpp."
        ) from exc

    return Llama(
        model_path=model_path,
        n_ctx=ctx_size,
        n_threads=threads,
        n_gpu_layers=gpu_layers,
    )


def llm_chunk_jsonl_dir(
    *,
    input_chunk_dir: Path,
    output_chunk_dir: Path,
    config: AppConfig,
) -> None:
    ensure_dir(output_chunk_dir)

    if not input_chunk_dir.exists():
        raise FileNotFoundError(
            f"Input chunk directory does not exist: {input_chunk_dir}"
        )

    jsonl_files = sorted(input_chunk_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No .jsonl chunk files found under %s", input_chunk_dir)
        return

    provider = (config.llm_chunk_provider or "").lower()
    logger.info(
        "LLM rechunking enabled (%s) for %d files in %s",
        provider,
        len(jsonl_files),
        input_chunk_dir,
    )

    api_key = os.getenv("GEMINI_API_KEY", "")
    max_retries = max(1, config.llm_chunk_max_retries)
    skipped_chunks = 0
    total_chunks = 0

    for path in jsonl_files:
        records = _load_jsonl_records(path)
        if not records:
            logger.warning("Empty chunk file: %s", path.name)
            continue

        out_path = output_chunk_dir / path.name
        output_index = 0

        with out_path.open("w", encoding="utf-8") as fw:
            for record in records:
                source_text = record.get("text", "")
                if not source_text:
                    continue

                total_chunks += 1
                base_metadata = dict(record.get("metadata") or {})
                source_chunk_id = base_metadata.pop("chunk_id", None)

                prompt = build_llm_chunk_prompt(
                    text=source_text,
                    chunk_size=config.llm_chunk_size,
                )

                logger.info(
                    "LLM rechunking %s (source chunk: %s, chars: %d)",
                    path.name,
                    source_chunk_id if source_chunk_id is not None else "-",
                    len(source_text),
                )

                chunks: list[str] | None = None
                last_error: Exception | None = None
                last_response: str | None = None
                source_name = f"{path.name}#chunk{source_chunk_id}"
                for attempt in range(1, max_retries + 1):
                    try:
                        if provider == "gemini":
                            response = _generate_with_gemini_chunk(
                                api_key=api_key, prompt=prompt, config=config
                            )
                        elif provider == "llama":
                            response = _generate_with_llama_chunk(
                                prompt=prompt, config=config
                            )
                        else:
                            raise ValueError(
                                "Unsupported llm_chunk_provider: "
                                f"{config.llm_chunk_provider}. Use 'gemini' or 'llama'."
                            )

                        last_response = response
                        chunks = _parse_llm_chunks(response, source_name=source_name)
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if last_response:
                            logger.warning(
                                "LLM failed output for %s (attempt %d/%d): %s",
                                source_name,
                                attempt,
                                max_retries,
                                last_response,
                            )
                        if attempt < max_retries:
                            logger.warning(
                                "LLM rechunking failed for %s (attempt %d/%d): %s",
                                source_name,
                                attempt,
                                max_retries,
                                exc,
                            )
                            continue
                        logger.error(
                            "LLM rechunking failed for %s after %d attempts",
                            source_name,
                            max_retries,
                        )
                        chunks = []
                        break

                if chunks is None:
                    if last_error:
                        logger.error(
                            "Skipping %s due to repeated failures: %s",
                            source_name,
                            last_error,
                        )
                    else:
                        logger.error(
                            "Skipping %s due to repeated failures with no response",
                            source_name,
                        )
                    skipped_chunks += 1
                    continue

                for chunk in chunks:
                    metadata = dict(base_metadata)
                    if source_chunk_id is not None:
                        metadata["source_chunk_id"] = source_chunk_id
                    out_record = {
                        "text": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_id": output_index,
                        },
                    }
                    fw.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    output_index += 1

        logger.info(
            "LLM rechunked %s -> %s (%d chunks)",
            path.name,
            out_path.name,
            output_index,
        )

    if total_chunks:
        logger.info(
            "LLM rechunking skipped %d/%d chunks",
            skipped_chunks,
            total_chunks,
        )


def load_documents_from_jsonl(chunk_dir: Path) -> list[Document]:
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")

    docs: list[Document] = []
    jsonl_files = sorted(chunk_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No .jsonl chunk files found under %s", chunk_dir)
        return docs

    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as fr:
            for line_no, line in enumerate(fr, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in {path.name} at line {line_no}: {e}"
                    ) from e

                text = obj.get("text")
                if not isinstance(text, str):
                    raise ValueError(
                        f"Missing/invalid 'text' in {path.name} line {line_no}"
                    )

                metadata = obj.get("metadata")
                if metadata is None:
                    metadata = {}
                if not isinstance(metadata, dict):
                    raise ValueError(
                        "Missing/invalid 'metadata' (must be dict) in "
                        f"{path.name} line {line_no}"
                    )

                metadata.setdefault("chunk_file", path.name)
                metadata.setdefault("line_no", line_no)

                docs.append(Document(page_content=text, metadata=metadata))

    logger.info("Loaded %d documents from %d chunk files", len(docs), len(jsonl_files))
    return docs


def load_documents_from_jsonl_dirs(chunk_dirs: Iterable[Path]) -> list[Document]:
    docs: list[Document] = []
    for chunk_dir in chunk_dirs:
        docs.extend(load_documents_from_jsonl(chunk_dir))
    return docs
