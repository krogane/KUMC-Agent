from __future__ import annotations

import bisect
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    AppConfig,
    LLM_CHUNK_SYSTEM_PROMPT,
    build_proposition_chunk_prompt,
    build_summery_chunk_prompt,
)
from indexing.chunks import Chunk, load_chunks, write_chunks
from indexing.constants import FILE_ID_SEPARATOR, MESSAGE_SEPARATORS
from indexing.llm_client import generate_text
from indexing.utils import ensure_dir, sanitize_filename
from sparse_normalizer import SparseNormalizer, SparseNormalizerConfig

logger = logging.getLogger(__name__)


_METADATA_KEYS = (
    "source_file_name",
    "source_type",
    "guild_id",
    "guild_name",
    "channel_id",
    "channel_name",
    "first_message_id",
    "first_message_date",
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


_JST = timezone(timedelta(hours=9))


def _parse_message_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(_JST).strftime("%Y/%m/%d")


def _load_message_lines(
    path: Path,
) -> tuple[list[str], list[str | None], list[str | None], dict[str, object]]:
    lines: list[str] = []
    line_message_ids: list[str | None] = []
    line_message_dates: list[str | None] = []
    base_metadata: dict[str, object] = {}
    last_date: str | None = None

    with path.open("r", encoding="utf-8") as fr:
        for line_no, line in enumerate(fr, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name} at line {line_no}: {exc}"
                ) from exc

            text = obj.get("text")
            metadata = obj.get("metadata") or {}
            if not isinstance(text, str) or not isinstance(metadata, dict):
                continue

            if not base_metadata:
                base_metadata = {
                    "guild_id": str(metadata.get("guild_id") or ""),
                    "guild_name": str(metadata.get("guild_name") or ""),
                    "channel_id": str(metadata.get("channel_id") or ""),
                    "channel_name": str(metadata.get("channel_name") or ""),
                    "source_file_name": str(metadata.get("source_file_name") or ""),
                    "source_type": "messages",
                }

            message_id: str | None = None
            raw_message_id = metadata.get("message_id")
            if raw_message_id is None:
                raw_message_id = metadata.get("chunk_id")
            if raw_message_id is not None:
                message_id = str(raw_message_id).strip() or None

            date_str = _parse_message_date(
                str(metadata.get("message_timestamp") or "")
            )
            if last_date and date_str and date_str != last_date:
                lines.append(date_str)
                line_message_ids.append(None)
                line_message_dates.append(date_str)
            if date_str:
                last_date = date_str
            message_date = date_str or last_date

            author_name = str(metadata.get("author_name") or "unknown").strip()
            for part in text.splitlines():
                part = part.strip()
                if not part:
                    continue
                lines.append(f"{author_name}: {part}")
                line_message_ids.append(message_id)
                line_message_dates.append(message_date)

    if base_metadata:
        if not base_metadata.get("source_file_name"):
            guild_id = base_metadata.get("guild_id") or ""
            channel_id = base_metadata.get("channel_id") or ""
            if guild_id and channel_id:
                base_metadata["source_file_name"] = f"discord/{guild_id}/{channel_id}"

    return lines, line_message_ids, line_message_dates, base_metadata


def _build_message_text(lines: list[str]) -> tuple[str, list[int]]:
    parts: list[str] = []
    line_starts: list[int] = []
    offset = 0
    for idx, line in enumerate(lines):
        line_starts.append(offset)
        parts.append(line)
        offset += len(line)
        if idx < len(lines) - 1:
            parts.append("\n")
            offset += 1
    return "".join(parts), line_starts


def _first_message_id_for_span(
    *,
    line_starts: list[int],
    line_message_ids: list[str | None],
    start: int,
    end: int,
) -> str | None:
    if start < 0 or end <= start or not line_starts:
        return None

    idx = bisect.bisect_right(line_starts, start) - 1
    if idx < 0:
        idx = 0
    if idx < len(line_message_ids):
        current = line_message_ids[idx]
        if current:
            return current

    for next_idx in range(idx + 1, len(line_message_ids)):
        if line_starts[next_idx] >= end:
            break
        candidate = line_message_ids[next_idx]
        if candidate:
            return candidate
    return None


def _first_message_date_for_span(
    *,
    line_starts: list[int],
    line_message_dates: list[str | None],
    start: int,
    end: int,
) -> str | None:
    if start < 0 or end <= start or not line_starts:
        return None

    idx = bisect.bisect_right(line_starts, start) - 1
    if idx < 0:
        idx = 0

    for current_idx in range(idx, len(line_message_dates)):
        if line_starts[current_idx] >= end:
            break
        candidate = line_message_dates[current_idx]
        if candidate:
            return candidate
    return None


def _extract_channel_id_from_filename(stem: str) -> str:
    match = re.match(r"^(\d+)", stem)
    if match:
        return match.group(1)
    return stem


def _is_output_up_to_date(*, output_path: Path, input_path: Path) -> bool:
    try:
        return output_path.stat().st_mtime >= input_path.stat().st_mtime
    except OSError:
        return False


def _should_skip_existing_output(
    *,
    output_path: Path,
    input_path: Path,
    skip_existing: bool,
    update_existing: bool,
    action_label: str,
) -> bool:
    if not skip_existing or not output_path.exists():
        return False
    if not update_existing:
        logger.info("Skip %s (exists): %s", action_label, output_path.name)
        return True
    if _is_output_up_to_date(output_path=output_path, input_path=input_path):
        logger.info("Skip %s (up-to-date): %s", action_label, output_path.name)
        return True
    return False


def _cleanup_stale_jsonl_outputs(*, output_dir: Path, expected_names: set[str]) -> None:
    for path in output_dir.glob("*.jsonl"):
        if path.name in expected_names:
            continue
        try:
            path.unlink()
            logger.info("Removed stale chunk output: %s", path.name)
        except Exception as exc:
            logger.warning("Failed to remove stale chunk output %s: %s", path.name, exc)


def recursive_chunk_dir(
    *,
    raw_data_dir: Path,
    chunk_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
    source_type: str,
    stage: str,
    file_extensions: Sequence[str] = (".md",),
    skip_existing: bool = False,
    update_existing: bool = True,
    sync_deleted: bool = False,
) -> None:
    ensure_dir(chunk_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    input_files: list[Path] = []
    for ext in file_extensions:
        input_files.extend(raw_data_dir.rglob(f"*{ext}"))
    input_files = sorted(set(input_files), key=lambda path: str(path))
    if not input_files:
        if sync_deleted:
            _cleanup_stale_jsonl_outputs(output_dir=chunk_dir, expected_names=set())
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

    expected_output_names: set[str] = set()
    for path in input_files:
        rel_path = path.relative_to(raw_data_dir)
        safe_rel = sanitize_filename(str(rel_path).replace(os.sep, "__"))
        out_path = chunk_dir / f"{safe_rel}.jsonl"
        expected_output_names.add(out_path.name)
        if _should_skip_existing_output(
            output_path=out_path,
            input_path=path,
            skip_existing=skip_existing,
            update_existing=update_existing,
            action_label="recursive chunking",
        ):
            continue

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
            metadata = dict(base_metadata)
            metadata["chunk_id"] = output_index
            metadata = _with_stage(metadata, stage)
            output_chunks.append(Chunk(text=doc, metadata=metadata))
            output_index += 1

        write_chunks(out_path, output_chunks)
        logger.info(
            "Recursive chunked (%s) %s -> %s (%d chunks)",
            stage,
            path.name,
            out_path.name,
            len(output_chunks),
        )

    if sync_deleted:
        _cleanup_stale_jsonl_outputs(
            output_dir=chunk_dir,
            expected_names=expected_output_names,
        )


def message_chunk_jsonl_dir(
    *,
    raw_messages_dir: Path,
    chunk_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    stage: str,
    skip_existing: bool = False,
    update_existing: bool = True,
    sync_deleted: bool = False,
) -> None:
    ensure_dir(chunk_dir)

    if not raw_messages_dir.exists():
        raise FileNotFoundError(
            f"Raw messages directory does not exist: {raw_messages_dir}"
        )

    input_files = sorted(
        raw_messages_dir.rglob("*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not input_files:
        if sync_deleted:
            _cleanup_stale_jsonl_outputs(output_dir=chunk_dir, expected_names=set())
        logger.warning("No message .jsonl files found under %s", raw_messages_dir)
        return

    splitter = _build_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=MESSAGE_SEPARATORS,
    )

    seen_outputs: set[str] = set()
    expected_output_names: set[str] = set()
    for path in input_files:
        if path.name.endswith(".state.json"):
            continue

        lines, line_message_ids, line_message_dates, base_metadata = (
            _load_message_lines(path)
        )
        if not lines:
            logger.warning("Empty message file: %s", path.name)
            continue
        if len(line_message_ids) != len(lines) or len(line_message_dates) != len(lines):
            logger.warning(
                "Message line metadata mismatch in %s (lines=%d ids=%d dates=%d)",
                path.name,
                len(lines),
                len(line_message_ids),
                len(line_message_dates),
            )
            line_message_ids = [None] * len(lines)
            line_message_dates = [None] * len(lines)

        guild_id = str(base_metadata.get("guild_id") or "").strip()
        if not guild_id and path.parent != raw_messages_dir:
            guild_id = path.parent.name
            base_metadata["guild_id"] = guild_id

        channel_id = str(base_metadata.get("channel_id") or "").strip()
        if not channel_id:
            channel_id = _extract_channel_id_from_filename(path.stem)
            base_metadata["channel_id"] = channel_id

        if not base_metadata.get("source_file_name") and guild_id and channel_id:
            base_metadata["source_file_name"] = f"discord/{guild_id}/{channel_id}"
        base_metadata.setdefault("source_type", "messages")

        out_name = sanitize_filename(f"{guild_id}__{channel_id}.jsonl")
        expected_output_names.add(out_name)
        if out_name in seen_outputs:
            logger.info("Skip duplicate message file: %s", path.name)
            continue
        seen_outputs.add(out_name)

        out_path = chunk_dir / out_name
        if _should_skip_existing_output(
            output_path=out_path,
            input_path=path,
            skip_existing=skip_existing,
            update_existing=update_existing,
            action_label="message chunking",
        ):
            continue

        text, line_starts = _build_message_text(lines)
        docs = splitter.split_text(text)
        output_chunks: list[Chunk] = []
        output_index = 0
        search_pos = 0

        for doc in docs:
            metadata = dict(base_metadata)
            start = text.find(doc, search_pos)
            if start == -1:
                start = text.find(doc)
            if start != -1:
                end = start + len(doc)
                first_message_id = _first_message_id_for_span(
                    line_starts=line_starts,
                    line_message_ids=line_message_ids,
                    start=start,
                    end=end,
                )
                if first_message_id:
                    metadata["first_message_id"] = first_message_id
                first_message_date = _first_message_date_for_span(
                    line_starts=line_starts,
                    line_message_dates=line_message_dates,
                    start=start,
                    end=end,
                )
                if first_message_date:
                    metadata["first_message_date"] = first_message_date
                search_pos = max(search_pos, end)
            metadata["chunk_id"] = output_index
            metadata = _with_stage(metadata, stage)
            output_chunks.append(Chunk(text=doc, metadata=metadata))
            output_index += 1

        write_chunks(out_path, output_chunks)
        logger.info(
            "Message chunked (%s) %s -> %s (%d chunks)",
            stage,
            path.name,
            out_path.name,
            len(output_chunks),
        )

    if sync_deleted:
        _cleanup_stale_jsonl_outputs(
            output_dir=chunk_dir,
            expected_names=expected_output_names,
        )


def recursive_chunk_jsonl_dir(
    *,
    input_chunk_dir: Path,
    output_chunk_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
    stage: str,
    skip_existing: bool = False,
    update_existing: bool = True,
    sync_deleted: bool = False,
) -> None:
    ensure_dir(output_chunk_dir)

    if not input_chunk_dir.exists():
        raise FileNotFoundError(
            f"Input chunk directory does not exist: {input_chunk_dir}"
        )

    jsonl_files = sorted(input_chunk_dir.glob("*.jsonl"))
    if not jsonl_files:
        if sync_deleted:
            _cleanup_stale_jsonl_outputs(output_dir=output_chunk_dir, expected_names=set())
        logger.warning("No .jsonl chunk files found under %s", input_chunk_dir)
        return

    splitter = _build_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    expected_output_names = {path.name for path in jsonl_files}
    for path in jsonl_files:
        out_path = output_chunk_dir / path.name
        if _should_skip_existing_output(
            output_path=out_path,
            input_path=path,
            skip_existing=skip_existing,
            update_existing=update_existing,
            action_label="recursive chunking",
        ):
            continue

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

            docs = splitter.split_text(source_text)
            for doc in docs:
                metadata = dict(base_metadata)
                metadata["chunk_id"] = output_index
                metadata = _with_stage(metadata, stage)
                output_chunks.append(Chunk(text=doc, metadata=metadata))
                output_index += 1

        write_chunks(out_path, output_chunks)
        logger.info(
            "Recursive chunked (%s) %s -> %s (%d chunks)",
            stage,
            path.name,
            out_path.name,
            len(output_chunks),
        )

    if sync_deleted:
        _cleanup_stale_jsonl_outputs(
            output_dir=output_chunk_dir,
            expected_names=expected_output_names,
        )


def sparse_chunk_jsonl_dir(
    *,
    input_chunk_dir: Path,
    output_chunk_dir: Path,
    config: AppConfig,
    stage: str = "second_recursive_sparse",
    skip_existing: bool = False,
    update_existing: bool = True,
    sync_deleted: bool = False,
) -> None:
    ensure_dir(output_chunk_dir)

    if not input_chunk_dir.exists():
        raise FileNotFoundError(
            f"Input chunk directory does not exist: {input_chunk_dir}"
        )

    jsonl_files = sorted(input_chunk_dir.glob("*.jsonl"))
    if not jsonl_files:
        if sync_deleted:
            _cleanup_stale_jsonl_outputs(output_dir=output_chunk_dir, expected_names=set())
        logger.warning("No .jsonl chunk files found under %s", input_chunk_dir)
        return

    normalizer = SparseNormalizer(
        config=SparseNormalizerConfig(
            sudachi_mode=config.sudachi_mode,
            use_normalized_form=config.sparse_use_normalized_form,
            remove_symbols=True,
            remove_stopwords=True,
        )
    )
    logger.info(
        "Sparse chunking enabled for %d files in %s",
        len(jsonl_files),
        input_chunk_dir,
    )

    expected_output_names = {path.name for path in jsonl_files}
    for path in jsonl_files:
        out_path = output_chunk_dir / path.name
        if _should_skip_existing_output(
            output_path=out_path,
            input_path=path,
            skip_existing=skip_existing,
            update_existing=update_existing,
            action_label="sparse chunking",
        ):
            continue

        chunks = load_chunks(path)
        if not chunks:
            logger.warning("Empty chunk file: %s", path.name)
            continue

        output_chunks: list[Chunk] = []
        for chunk in chunks:
            tokens = normalizer.normalize_tokens(chunk.text or "")
            if not tokens:
                continue

            metadata = dict(chunk.metadata)
            metadata["chunk_stage"] = stage
            output_chunks.append(
                Chunk(text=" ".join(tokens), metadata=metadata)
            )

        write_chunks(out_path, output_chunks)
        logger.info(
            "Sparse chunked %s -> %s (%d chunks)",
            path.name,
            out_path.name,
            len(output_chunks),
        )

    if sync_deleted:
        _cleanup_stale_jsonl_outputs(
            output_dir=output_chunk_dir,
            expected_names=expected_output_names,
        )


def summery_chunk_jsonl_dir(
    *,
    input_chunk_dir: Path,
    output_chunk_dir: Path,
    config: AppConfig,
    skip_existing: bool = False,
    update_existing: bool = True,
    sync_deleted: bool = False,
) -> None:
    ensure_dir(output_chunk_dir)

    if not input_chunk_dir.exists():
        raise FileNotFoundError(
            f"Input chunk directory does not exist: {input_chunk_dir}"
        )

    jsonl_files = sorted(input_chunk_dir.glob("*.jsonl"))
    if not jsonl_files:
        if sync_deleted:
            _cleanup_stale_jsonl_outputs(output_dir=output_chunk_dir, expected_names=set())
        logger.warning("No .jsonl chunk files found under %s", input_chunk_dir)
        return

    provider = (config.summery_provider or "").lower()
    max_retries = max(1, config.summery_max_retries)
    logger.info(
        "Summery chunking enabled (%s) for %d files in %s",
        provider,
        len(jsonl_files),
        input_chunk_dir,
    )

    expected_output_names = {path.name for path in jsonl_files}
    for path in jsonl_files:
        out_path = output_chunk_dir / path.name
        if _should_skip_existing_output(
            output_path=out_path,
            input_path=path,
            skip_existing=skip_existing,
            update_existing=update_existing,
            action_label="summery chunking",
        ):
            continue

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

            source_type = str(base_metadata.get("source_type") or "").strip()
            drive_file_path = str(base_metadata.get("drive_file_path") or "").strip()
            prompt = build_summery_chunk_prompt(
                text=source_text,
                target_characters=config.summery_characters,
                source_type=source_type,
                drive_file_path=drive_file_path,
            )
            chunk_texts = _run_llm_chunking(
                prompt=prompt,
                source_name=path.name,
                provider=provider,
                api_key=config.gemini_api_key,
                model=_select_model_for_provider(
                    provider=provider,
                    gemini_model=config.summery_gemini_model,
                    llama_model=config.summery_llama_model,
                ),
                llama_model_path=config.summery_llama_model_path,
                llama_ctx_size=config.summery_llama_ctx_size,
                temperature=config.summery_temperature,
                max_output_tokens=config.summery_max_output_tokens,
                max_retries=max_retries,
                thinking_level=config.thinking_level,
                llama_threads=config.llama_threads,
                llama_gpu_layers=config.llama_gpu_layers,
                action_label="Summery chunking",
                output_format="raw_text",
                response_mime_type="text/plain",
            )
            if chunk_texts is None:
                continue

            for chunk_text in chunk_texts:
                metadata = dict(base_metadata)
                metadata["chunk_id"] = output_index
                metadata = _with_stage(metadata, "summery")
                output_chunks.append(Chunk(text=chunk_text, metadata=metadata))
                output_index += 1

        write_chunks(out_path, output_chunks)
        logger.info(
            "Summery chunked %s -> %s (%d chunks)",
            path.name,
            out_path.name,
            len(output_chunks),
        )

    if sync_deleted:
        _cleanup_stale_jsonl_outputs(
            output_dir=output_chunk_dir,
            expected_names=expected_output_names,
        )


def proposition_chunk_jsonl_dir(
    *,
    input_chunk_dir: Path,
    output_chunk_dir: Path,
    config: AppConfig,
    skip_existing: bool = False,
    update_existing: bool = True,
    sync_deleted: bool = False,
) -> None:
    ensure_dir(output_chunk_dir)

    if not input_chunk_dir.exists():
        raise FileNotFoundError(
            f"Input chunk directory does not exist: {input_chunk_dir}"
        )

    jsonl_files = sorted(input_chunk_dir.glob("*.jsonl"))
    if not jsonl_files:
        if sync_deleted:
            _cleanup_stale_jsonl_outputs(output_dir=output_chunk_dir, expected_names=set())
        logger.warning("No .jsonl chunk files found under %s", input_chunk_dir)
        return

    provider = (config.prop_provider or "").lower()
    max_retries = max(1, config.prop_max_retries)
    logger.info(
        "Proposition chunking enabled (%s) for %d files in %s",
        provider,
        len(jsonl_files),
        input_chunk_dir,
    )

    expected_output_names = {path.name for path in jsonl_files}
    for path in jsonl_files:
        out_path = output_chunk_dir / path.name
        if _should_skip_existing_output(
            output_path=out_path,
            input_path=path,
            skip_existing=skip_existing,
            update_existing=update_existing,
            action_label="proposition chunking",
        ):
            continue
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
            prompt = build_proposition_chunk_prompt(text=source_text)
            source_name = path.name
            chunk_texts = _run_llm_chunking(
                prompt=prompt,
                source_name=source_name,
                provider=provider,
                api_key=config.gemini_api_key,
                model=_select_model_for_provider(
                    provider=provider,
                    gemini_model=config.prop_gemini_model,
                    llama_model=config.prop_llama_model,
                ),
                llama_model_path=config.prop_llama_model_path,
                llama_ctx_size=config.prop_llama_ctx_size,
                temperature=config.prop_temperature,
                max_output_tokens=config.prop_max_output_tokens,
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

    if sync_deleted:
        _cleanup_stale_jsonl_outputs(
            output_dir=output_chunk_dir,
            expected_names=expected_output_names,
        )


def _select_model_for_provider(
    *,
    provider: str,
    gemini_model: str,
    llama_model: str,
) -> str:
    if (provider or "").lower() == "llama":
        return llama_model
    return gemini_model


def _strip_chunk_metadata(metadata: dict[str, object]) -> dict[str, object]:
    cleaned = {k: metadata.get(k, "") for k in _METADATA_KEYS}
    return cleaned


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
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


def _parse_llm_summary(response: str, *, source_name: str) -> list[str]:
    if not response:
        raise ValueError(f"Empty LLM response for {source_name}")

    payload = _strip_code_fences(response)
    if not payload.strip():
        raise ValueError(f"Empty LLM response for {source_name}")

    leading = payload.lstrip()
    if leading.startswith("[") or leading.startswith("{"):
        try:
            return _parse_llm_chunks(payload, source_name=source_name)
        except Exception:
            pass

    if leading.startswith('"') and leading.rstrip().endswith('"'):
        try:
            decoded = json.loads(payload)
        except Exception:
            decoded = None
        if isinstance(decoded, str):
            text = decoded.strip()
            if text:
                return [text]

    text = payload.strip()
    if not text:
        raise ValueError(f"LLM produced no summary for {source_name}")
    return [text]


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
    output_format: str = "json_list",
    response_mime_type: str | None = "application/json",
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
                response_mime_type=response_mime_type,
            )
            last_response = response
            if output_format == "raw_text":
                chunks = _parse_llm_summary(response, source_name=source_name)
            else:
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
