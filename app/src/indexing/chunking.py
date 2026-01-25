from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    AppConfig,
    LLM_CHUNK_SYSTEM_PROMPT,
    build_proposition_chunk_prompt,
)
from indexing.chunks import Chunk, load_chunks, write_chunks
from indexing.constants import FILE_ID_SEPARATOR
from indexing.llm_client import generate_text
from indexing.token_utils import estimate_tokens
from indexing.utils import ensure_dir, sanitize_filename

logger = logging.getLogger(__name__)


_METADATA_KEYS = (
    "source_file_name",
    "source_type",
    "drive_file_name",
    "drive_mime_type",
    "drive_file_path",
    "drive_file_id",
)


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


def _build_base_metadata(
    *,
    source_file_name: str,
    source_type: str,
    drive_metadata: dict[str, str],
    fallback_drive_file_id: str | None,
) -> dict[str, object]:
    drive_file_id = drive_metadata.get("drive_file_id") or fallback_drive_file_id or ""
    drive_file_name = drive_metadata.get("drive_file_name") or ""
    drive_mime_type = drive_metadata.get("drive_mime_type") or ""
    drive_file_path = drive_metadata.get("drive_path") or drive_metadata.get(
        "drive_file_path", ""
    )

    return {
        "source_file_name": source_file_name,
        "source_type": source_type,
        "drive_file_name": drive_file_name,
        "drive_mime_type": drive_mime_type,
        "drive_file_path": drive_file_path,
        "drive_file_id": drive_file_id,
    }


def _with_stage(metadata: dict[str, object], stage: str) -> dict[str, object]:
    updated = dict(metadata)
    updated["chunk_stage"] = stage
    return updated


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


def _is_output_up_to_date(output_path: Path, input_path: Path) -> bool:
    try:
        return output_path.stat().st_mtime >= input_path.stat().st_mtime
    except FileNotFoundError:
        return False


def recursive_chunk_dir(
    *,
    raw_data_dir: Path,
    chunk_dir: Path,
    rec_chunk_size: int,
    rec_chunk_overlap: int,
    rec_min_chunk_tokens: int,
    tokenizer_model: str,
    separators: Sequence[str],
    source_type: str,
    file_extensions: Sequence[str] = (".md",),
    skip_existing: bool = False,
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
        chunk_size=rec_chunk_size,
        chunk_overlap=rec_chunk_overlap,
        separators=separators,
    )

    for path in input_files:
        rel_path = path.relative_to(raw_data_dir)
        safe_rel = sanitize_filename(str(rel_path).replace(os.sep, "__"))
        out_path = chunk_dir / f"{safe_rel}.jsonl"
        if skip_existing and out_path.exists():
            if _is_output_up_to_date(out_path, path):
                logger.info("Skip recursive chunking (up-to-date): %s", out_path.name)
                continue
            logger.info(
                "Rebuilding recursive chunking (source updated): %s", out_path.name
            )

        text = path.read_text(encoding="utf-8")
        drive_metadata = _load_drive_metadata(path)
        base_metadata = _build_base_metadata(
            source_file_name=path.name,
            source_type=source_type,
            drive_metadata=drive_metadata,
            fallback_drive_file_id=_extract_drive_file_id(path.name),
        )

        docs = splitter.split_text(text)
        output_chunks: list[Chunk] = []
        output_index = 0

        for doc in docs:
            if rec_min_chunk_tokens > 0 and estimate_tokens(
                text=doc, model_name=tokenizer_model
            ) < rec_min_chunk_tokens:
                continue
            metadata = dict(base_metadata)
            metadata["chunk_id"] = output_index
            metadata = _with_stage(metadata, "recursive")
            output_chunks.append(Chunk(text=doc, metadata=metadata))
            output_index += 1

        write_chunks(out_path, output_chunks)
        logger.info(
            "Recursive chunked %s -> %s (%d chunks)",
            path.name,
            out_path.name,
            len(output_chunks),
        )


def proposition_chunk_jsonl_dir(
    *,
    input_chunk_dir: Path,
    output_chunk_dir: Path,
    config: AppConfig,
    skip_existing: bool = False,
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

    provider = (config.prop_chunk_provider or "").lower()
    max_retries = max(1, config.prop_chunk_max_retries)
    logger.info(
        "Proposition chunking enabled (%s) for %d files in %s",
        provider,
        len(jsonl_files),
        input_chunk_dir,
    )

    for path in jsonl_files:
        out_path = output_chunk_dir / path.name
        if skip_existing and out_path.exists():
            if _is_output_up_to_date(out_path, path):
                logger.info("Skip proposition chunking (up-to-date): %s", out_path.name)
                continue
            logger.info(
                "Rebuilding proposition chunking (input updated): %s", out_path.name
            )
        chunks = load_chunks(path)
        if not chunks:
            logger.warning("Empty chunk file: %s", path.name)
            continue

        output_chunks: list[Chunk] = []
        output_index = 0

        for chunk in chunks:
            source_text = chunk.text
            if not source_text:
                continue
            base_metadata = _strip_chunk_metadata(chunk.metadata)
            parent_chunk_id = chunk.metadata.get("chunk_id")
            if parent_chunk_id is not None:
                base_metadata["parent_chunk_id"] = parent_chunk_id
            prompt = build_proposition_chunk_prompt(
                text=source_text,
                chunk_size=config.prop_chunk_size,
            )
            source_name = path.name
            chunk_texts = _run_llm_chunking(
                prompt=prompt,
                source_name=source_name,
                provider=provider,
                api_key=config.gemini_api_key,
                model=config.prop_chunk_model,
                llama_model_path=config.prop_chunk_llama_model_path,
                llama_ctx_size=config.prop_chunk_llama_ctx_size,
                temperature=config.prop_chunk_temperature,
                max_output_tokens=config.prop_chunk_max_output_tokens,
                max_retries=max_retries,
                thinking_level=config.thinking_level,
                llama_threads=config.llama_threads,
                llama_gpu_layers=config.llama_gpu_layers,
                action_label="Proposition chunking",
            )
            if chunk_texts is None:
                continue

            for chunk_text in chunk_texts:
                metadata = dict(base_metadata)
                metadata["chunk_id"] = output_index
                metadata = _with_stage(metadata, "proposition")
                output_chunks.append(Chunk(text=chunk_text, metadata=metadata))
                output_index += 1

        write_chunks(out_path, output_chunks)
        logger.info(
            "Proposition chunked %s -> %s (%d chunks)",
            path.name,
            out_path.name,
            len(output_chunks),
        )


def _strip_chunk_metadata(metadata: dict[str, object]) -> dict[str, object]:
    cleaned = {k: metadata.get(k, "") for k in _METADATA_KEYS}
    return cleaned


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




def _run_llm_chunking(
    *,
    prompt: str,
    source_name: str,
    provider: str,
    api_key: str,
    model: str,
    llama_model_path: str,
    llama_ctx_size: int,
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    thinking_level: str,
    llama_threads: int,
    llama_gpu_layers: int,
    action_label: str,
) -> list[str] | None:
    last_error: Exception | None = None
    last_response: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = generate_text(
                provider=provider,
                api_key=api_key,
                prompt=prompt,
                model=model,
                system_prompt=LLM_CHUNK_SYSTEM_PROMPT,
                llama_model_path=llama_model_path,
                llama_ctx_size=llama_ctx_size,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                thinking_level=thinking_level,
                llama_threads=llama_threads,
                llama_gpu_layers=llama_gpu_layers,
                response_mime_type="application/json",
            )
            last_response = response
            chunks = _parse_llm_chunks(response, source_name=source_name)
            return chunks
        except Exception as exc:
            last_error = exc
            if last_response:
                logger.warning(
                    "%s invalid output for %s (attempt %d/%d): %s",
                    action_label,
                    source_name,
                    attempt,
                    max_retries,
                    last_response,
                )
            if attempt < max_retries:
                logger.warning(
                    "%s failed for %s (attempt %d/%d): %s",
                    action_label,
                    source_name,
                    attempt,
                    max_retries,
                    exc,
                )
                continue
            logger.error(
                "%s failed for %s after %d attempts",
                action_label,
                source_name,
                max_retries,
            )

    if last_error:
        logger.error("Skipping %s due to repeated failures: %s", source_name, last_error)
    else:
        logger.error(
            "Skipping %s due to repeated failures with no response", source_name
        )
    return None
