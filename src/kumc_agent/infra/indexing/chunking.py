from __future__ import annotations

import bisect
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from kumc_agent.infra.llm.gemini_rate_limit import index_summary_rate_limiter_name
from kumc_agent.infra.indexing.config import (
    AppConfig,
    build_summery_chunk_prompt,
    get_llm_chunk_system_prompt,
)
from kumc_agent.infra.indexing.date_metadata import (
    SOURCE_DATE_UNKNOWN,
    infer_source_date,
    source_date_from_vc_path,
)
from kumc_agent.infra.indexing.chunks import Chunk, load_chunks, write_chunks
from kumc_agent.infra.indexing.constants import FILE_ID_SEPARATOR, MESSAGE_SEPARATORS
from kumc_agent.infra.indexing.llm_client import generate_text
from kumc_agent.infra.indexing.summary_searchability import (
    SummarySearchabilityDecision,
    build_summary_searchability_prompt,
    normalize_summary_parent_id,
    parse_summary_searchability_response,
    summary_decision_sidecar_path,
    write_summary_searchability_decisions,
)
from kumc_agent.infra.indexing.summary_quality import (
    sanitize_summary_text,
    summary_quality_metadata,
)
from kumc_agent.infra.indexing.utils import ensure_dir, sanitize_filename
from kumc_agent.infra.indexing.sparse_normalizer import SparseNormalizer, SparseNormalizerConfig

logger = logging.getLogger(__name__)


_METADATA_KEYS = (
    "source_file_name",
    "source_kind",
    "source_type",
    "source_date",
    "updated_at",
    "meeting_date",
    "meeting_label",
    "guild_id",
    "guild_name",
    "category_id",
    "category_name",
    "channel_id",
    "channel_name",
    "first_message_id",
    "first_message_date",
    "drive_file_name",
    "drive_mime_type",
    "drive_file_path",
    "drive_file_id",
    "drive_url",
    "drive_modified_time",
    "content_sha256",
    "extraction_method",
    "extraction_status",
    "text_bytes",
    "nonempty_characters",
    "page_count",
    "slide_count",
    "ocr_page_count",
    "ocr_candidate_count",
    "embedded_image_count",
    "quality_flags",
    "index_status",
    "redaction_policy",
    "page_number",
    "page_ref",
    "slide_number",
    "slide_ref",
    "block_type",
    "normalized_record_id",
    "embedded_image_refs",
    "canonical_drive_file_id",
    "canonical_source_file_name",
    "variant_group_id",
    "duplicate_group_size",
    "variant_drive_file_ids",
    "sheet_id",
    "sheet_name",
    "sheet_index",
    "row_range",
    "column_range",
    "table_kind",
    "table_profile",
    "normalization_status",
    "sensitivity",
    "sensitivity_findings",
    "hatenablog_title",
    "hatenablog_entry_id",
    "hatenablog_created_at",
    "hatenablog_updated_at",
    "hatenablog_url",
    "hatenablog_html_normalized",
    "hatenablog_image_count",
    "hatenablog_images",
    "hatenablog_related_link_count",
    "hatenablog_related_links",
    "crafters_colony_title",
    "crafters_colony_article_id",
    "crafters_colony_published_at",
    "crafters_colony_updated_at",
    "crafters_colony_article_url",
    "notion_database_id",
    "notion_page_id",
    "notion_title",
    "notion_url",
    "notion_created_time",
    "notion_last_edited_time",
    "x_author_handle",
    "minecraft_wiki_title",
    "minecraft_wiki_page_id",
    "minecraft_wiki_revision_id",
    "minecraft_wiki_requested_title",
    "minecraft_wiki_is_redirect",
    "minecraft_wiki_redirect_from",
    "minecraft_wiki_redirect_to",
    "minecraft_wiki_resolved_title",
    "minecraft_wiki_resolved_page_id",
    "minecraft_wiki_aliases",
    "minecraft_wiki_cache_title",
    "minecraft_wiki_cache_file",
    "canonical_url",
    "heading_path",
    "visibility",
    "access_scope",
    "checksum",
)


def _extract_drive_file_id(filename: str) -> str | None:
    if FILE_ID_SEPARATOR not in filename:
        return None
    prefix, _ = filename.split(FILE_ID_SEPARATOR, 1)
    return prefix or None


def _load_drive_metadata(source_path: Path) -> dict[str, object]:
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

    metadata: dict[str, object] = {}
    for key in (
        "drive_file_id",
        "drive_file_name",
        "drive_path",
        "drive_mime_type",
        "drive_modified_time",
        "drive_url",
        "content_sha256",
        "extraction_method",
        "extraction_status",
        "text_bytes",
        "nonempty_characters",
        "page_count",
        "slide_count",
        "ocr_page_count",
        "ocr_candidate_count",
        "embedded_image_count",
        "quality_flags",
        "index_status",
        "redaction_policy",
        "page_number",
        "page_ref",
        "slide_number",
        "slide_ref",
        "block_type",
        "normalized_record_id",
        "embedded_image_refs",
        "canonical_drive_file_id",
        "canonical_source_file_name",
        "variant_group_id",
        "duplicate_group_size",
        "variant_drive_file_ids",
        "sheet_id",
        "sheet_name",
        "sheet_index",
        "row_range",
        "column_range",
        "table_kind",
        "table_profile",
        "normalization_status",
        "sensitivity",
        "sensitivity_findings",
        "hatenablog_title",
        "hatenablog_created_at",
        "hatenablog_url",
        "crafters_colony_title",
        "crafters_colony_article_id",
        "crafters_colony_published_at",
        "crafters_colony_updated_at",
        "crafters_colony_article_url",
        "source_date",
        "updated_at",
        "notion_database_id",
        "notion_page_id",
        "notion_title",
        "notion_url",
        "notion_created_time",
        "notion_last_edited_time",
        "minecraft_wiki_title",
        "minecraft_wiki_page_id",
        "minecraft_wiki_revision_id",
        "minecraft_wiki_requested_title",
        "minecraft_wiki_is_redirect",
        "minecraft_wiki_redirect_from",
        "minecraft_wiki_redirect_to",
        "minecraft_wiki_resolved_title",
        "minecraft_wiki_resolved_page_id",
        "minecraft_wiki_aliases",
        "minecraft_wiki_cache_title",
        "minecraft_wiki_cache_file",
        "canonical_url",
        "heading_path",
        "visibility",
        "access_scope",
        "checksum",
    ):
        value = data.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value
        elif isinstance(value, bool):
            metadata[key] = value
        elif key in {
            "minecraft_wiki_aliases",
            "sensitivity_findings",
            "quality_flags",
            "embedded_image_refs",
            "heading_path",
            "variant_drive_file_ids",
        } and isinstance(value, list):
            metadata[key] = value
        elif key in {"access_scope", "table_profile", "sensitivity"} and isinstance(value, dict):
            metadata[key] = value
        elif key in {
            "sheet_index",
            "text_bytes",
            "nonempty_characters",
            "page_count",
            "slide_count",
            "ocr_page_count",
            "ocr_candidate_count",
            "embedded_image_count",
            "page_number",
            "slide_number",
            "normalized_record_id",
            "duplicate_group_size",
            "hatenablog_image_count",
            "hatenablog_related_link_count",
        } and isinstance(value, int):
            metadata[key] = value
        elif key == "hatenablog_html_normalized" and isinstance(value, bool):
            metadata[key] = value
    return metadata


def _build_base_metadata(
    *,
    source_file_name: str,
    source_type: str,
    drive_metadata: dict[str, object],
    fallback_drive_file_id: str | None,
) -> dict[str, object]:
    drive_file_id = drive_metadata.get("drive_file_id") or fallback_drive_file_id or ""
    drive_file_name = drive_metadata.get("drive_file_name") or ""
    drive_mime_type = drive_metadata.get("drive_mime_type") or ""
    drive_file_path = drive_metadata.get("drive_path") or drive_metadata.get(
        "drive_file_path", ""
    )

    metadata: dict[str, object] = {
        "source_file_name": source_file_name,
        "source_kind": source_type,
        "source_type": source_type,
        "source_date": drive_metadata.get("source_date", SOURCE_DATE_UNKNOWN),
        "updated_at": drive_metadata.get("updated_at")
        or drive_metadata.get("crafters_colony_updated_at", ""),
        "meeting_date": "",
        "meeting_label": "",
        "drive_file_name": drive_file_name,
        "drive_mime_type": drive_mime_type,
        "drive_file_path": drive_file_path,
        "drive_file_id": drive_file_id,
        "hatenablog_title": drive_metadata.get("hatenablog_title", ""),
        "hatenablog_entry_id": drive_metadata.get("hatenablog_entry_id", ""),
        "hatenablog_created_at": drive_metadata.get("hatenablog_created_at", ""),
        "hatenablog_updated_at": drive_metadata.get("hatenablog_updated_at", ""),
        "hatenablog_url": drive_metadata.get("hatenablog_url", ""),
        "hatenablog_html_normalized": drive_metadata.get("hatenablog_html_normalized", False),
        "hatenablog_image_count": drive_metadata.get("hatenablog_image_count", 0),
        "hatenablog_images": drive_metadata.get("hatenablog_images", []),
        "hatenablog_related_link_count": drive_metadata.get(
            "hatenablog_related_link_count",
            0,
        ),
        "hatenablog_related_links": drive_metadata.get("hatenablog_related_links", []),
        "crafters_colony_title": drive_metadata.get("crafters_colony_title", ""),
        "crafters_colony_article_id": drive_metadata.get(
            "crafters_colony_article_id",
            "",
        ),
        "crafters_colony_published_at": drive_metadata.get(
            "crafters_colony_published_at",
            "",
        ),
        "crafters_colony_updated_at": drive_metadata.get(
            "crafters_colony_updated_at",
            "",
        ),
        "crafters_colony_article_url": drive_metadata.get(
            "crafters_colony_article_url",
            "",
        ),
        "notion_database_id": drive_metadata.get("notion_database_id", ""),
        "notion_page_id": drive_metadata.get("notion_page_id", ""),
        "notion_title": drive_metadata.get("notion_title", ""),
        "notion_url": drive_metadata.get("notion_url", ""),
        "notion_created_time": drive_metadata.get("notion_created_time", ""),
        "notion_last_edited_time": drive_metadata.get("notion_last_edited_time", ""),
        "minecraft_wiki_title": drive_metadata.get("minecraft_wiki_title", ""),
        "minecraft_wiki_page_id": drive_metadata.get("minecraft_wiki_page_id", ""),
        "minecraft_wiki_revision_id": drive_metadata.get("minecraft_wiki_revision_id", ""),
        "minecraft_wiki_requested_title": drive_metadata.get("minecraft_wiki_requested_title", ""),
        "minecraft_wiki_is_redirect": drive_metadata.get("minecraft_wiki_is_redirect", False),
        "minecraft_wiki_redirect_from": drive_metadata.get("minecraft_wiki_redirect_from", ""),
        "minecraft_wiki_redirect_to": drive_metadata.get("minecraft_wiki_redirect_to", ""),
        "minecraft_wiki_resolved_title": drive_metadata.get("minecraft_wiki_resolved_title", ""),
        "minecraft_wiki_resolved_page_id": drive_metadata.get("minecraft_wiki_resolved_page_id", ""),
        "minecraft_wiki_aliases": drive_metadata.get("minecraft_wiki_aliases", []),
        "minecraft_wiki_cache_title": drive_metadata.get("minecraft_wiki_cache_title", ""),
        "minecraft_wiki_cache_file": drive_metadata.get("minecraft_wiki_cache_file", ""),
        "canonical_url": drive_metadata.get("canonical_url", ""),
        "visibility": drive_metadata.get("visibility", ""),
        "access_scope": drive_metadata.get("access_scope", ""),
        "checksum": drive_metadata.get("checksum", ""),
    }
    for key in _METADATA_KEYS:
        if key in metadata:
            continue
        if key in drive_metadata:
            metadata[key] = drive_metadata[key]
    metadata["source_date"] = infer_source_date(metadata=metadata)
    return metadata


def _build_vc_meeting_metadata(path: Path) -> dict[str, str]:
    parent_name = path.parent.name
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})_\d+$", parent_name)
    if not match:
        return {}
    meeting_date = f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
    return {
        "meeting_date": meeting_date,
        "meeting_label": f"{meeting_date} 例会",
        "source_date": meeting_date,
    }


def _with_stage(metadata: dict[str, object], stage: str) -> dict[str, object]:
    updated = dict(metadata)
    updated["chunk_stage"] = stage
    return updated


def _is_minecraft_wiki_redirect_only(text: str) -> bool:
    return bool(
        re.match(
            r"(?is)^\s*#(?:転送|redirect)\s*:?\s*\[\[[^\]]+\]\]\s*$",
            text or "",
        )
    )


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
                source_type = str(metadata.get("source_type") or "").strip()
                if not source_type:
                    source_type = "messages"
                source_date = str(metadata.get("source_date") or "").strip()
                if not source_date:
                    source_date = SOURCE_DATE_UNKNOWN
                base_metadata = {
                    "guild_id": str(metadata.get("guild_id") or ""),
                    "guild_name": str(metadata.get("guild_name") or ""),
                    "category_id": str(metadata.get("category_id") or ""),
                    "category_name": str(metadata.get("category_name") or ""),
                    "channel_id": str(metadata.get("channel_id") or ""),
                    "channel_name": str(metadata.get("channel_name") or ""),
                    "source_file_name": str(metadata.get("source_file_name") or ""),
                    "source_type": source_type,
                    "source_date": source_date,
                    "x_author_handle": str(metadata.get("x_author_handle") or ""),
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


def _chunk_mtime_sidecar_path(chunk_path: Path) -> Path:
    return chunk_path.with_suffix(chunk_path.suffix + ".mtime.json")


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _read_chunk_mtime_sidecar(chunk_path: Path) -> dict[str, object]:
    sidecar = _chunk_mtime_sidecar_path(chunk_path)
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _write_chunk_mtime_sidecar(*, chunk_path: Path, input_path: Path) -> None:
    input_mtime = _safe_mtime(input_path)
    output_mtime = _safe_mtime(chunk_path)
    payload = {
        "source_path": str(input_path),
        "source_mtime": input_mtime,
        "output_mtime": output_mtime,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    sidecar = _chunk_mtime_sidecar_path(chunk_path)
    sidecar.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _remove_chunk_mtime_sidecar(chunk_path: Path) -> None:
    sidecar = _chunk_mtime_sidecar_path(chunk_path)
    if not sidecar.exists():
        return
    try:
        sidecar.unlink()
        logger.info("Removed stale chunk mtime sidecar: %s", sidecar.name)
    except Exception as exc:
        logger.warning(
            "Failed to remove stale chunk mtime sidecar %s: %s",
            sidecar.name,
            exc,
        )


def _is_output_up_to_date(*, output_path: Path, input_path: Path) -> bool:
    input_mtime = _safe_mtime(input_path)
    if input_mtime is None:
        return False
    sidecar = _read_chunk_mtime_sidecar(output_path)
    if sidecar:
        source_path = sidecar.get("source_path")
        source_mtime = sidecar.get("source_mtime")
        if (
            isinstance(source_path, str)
            and source_path == str(input_path)
            and isinstance(source_mtime, (int, float))
            and float(source_mtime) >= input_mtime
        ):
            return True
    output_mtime = _safe_mtime(output_path)
    if output_mtime is None:
        return False
    return output_mtime >= input_mtime


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
            _remove_chunk_mtime_sidecar(path)
        except Exception as exc:
            logger.warning("Failed to remove stale chunk output %s: %s", path.name, exc)
    for sidecar in output_dir.glob("*.jsonl.mtime.json"):
        chunk_name = sidecar.name[: -len(".mtime.json")]
        if chunk_name in expected_names:
            continue
        try:
            sidecar.unlink()
            logger.info("Removed stale chunk mtime sidecar: %s", sidecar.name)
        except Exception as exc:
            logger.warning(
                "Failed to remove stale chunk mtime sidecar %s: %s",
                sidecar.name,
                exc,
            )
    for sidecar in output_dir.glob("*.summary_decisions.json"):
        chunk_name = sidecar.name[: -len(".summary_decisions.json")] + ".jsonl"
        if chunk_name in expected_names:
            continue
        try:
            sidecar.unlink()
            logger.info("Removed stale summary decision sidecar: %s", sidecar.name)
        except Exception as exc:
            logger.warning(
                "Failed to remove stale summary decision sidecar %s: %s",
                sidecar.name,
                exc,
            )


_DENIED_INDEX_STATUSES = {"deleted", "quarantined", "permission_lost"}


def docs_chunk_dir(
    *,
    ingestion_data_dir: Path,
    structured_data_dir: Path | None,
    chunk_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
    stage: str,
    skip_existing: bool = False,
    update_existing: bool = True,
    sync_deleted: bool = False,
) -> None:
    ensure_dir(chunk_dir)
    if not ingestion_data_dir.exists():
        raise FileNotFoundError(
            f"Ingestion source directory does not exist: {ingestion_data_dir}"
        )

    structured_files = (
        sorted(structured_data_dir.glob("*.jsonl"), key=lambda path: str(path))
        if structured_data_dir is not None and structured_data_dir.exists()
        else []
    )
    markdown_files = sorted(ingestion_data_dir.rglob("*.md"), key=lambda path: str(path))
    structured_drive_ids = {
        drive_file_id
        for path in structured_files
        if (drive_file_id := _extract_drive_file_id(path.name))
    }
    fallback_markdown_files = [
        path
        for path in markdown_files
        if (_extract_drive_file_id(path.name) or "") not in structured_drive_ids
    ]
    if not structured_files and not fallback_markdown_files:
        if sync_deleted:
            _cleanup_stale_jsonl_outputs(output_dir=chunk_dir, expected_names=set())
        logger.warning("No Docs files found under %s", ingestion_data_dir)
        return

    splitter = _build_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    expected_output_names: set[str] = set()

    for path in structured_files:
        out_path = chunk_dir / path.name
        expected_output_names.add(out_path.name)
        if _should_skip_existing_output(
            output_path=out_path,
            input_path=path,
            skip_existing=skip_existing,
            update_existing=update_existing,
            action_label="docs structured chunking",
        ):
            continue
        from kumc_agent.infra.indexing.docs_normalizer import (
            load_structured_doc_chunks,
        )

        output_chunks = _split_docs_chunks_for_stage(
            chunks=load_structured_doc_chunks(path),
            splitter=splitter,
            stage=stage,
        )
        write_chunks(out_path, output_chunks)
        _write_chunk_mtime_sidecar(chunk_path=out_path, input_path=path)
        logger.info(
            "Docs structured chunked (%s) %s -> %s (%d chunks)",
            stage,
            path.name,
            out_path.name,
            len(output_chunks),
        )

    for path in fallback_markdown_files:
        rel_path = path.relative_to(ingestion_data_dir)
        safe_rel = sanitize_filename(str(rel_path).replace(os.sep, "__"))
        out_path = chunk_dir / f"{safe_rel}.jsonl"
        expected_output_names.add(out_path.name)
        if _should_skip_existing_output(
            output_path=out_path,
            input_path=path,
            skip_existing=skip_existing,
            update_existing=update_existing,
            action_label="docs markdown chunking",
        ):
            continue

        text = path.read_text(encoding="utf-8")
        drive_metadata = _load_drive_metadata(path)
        base_metadata = _build_base_metadata(
            source_file_name=path.name,
            source_type="docs",
            drive_metadata=drive_metadata,
            fallback_drive_file_id=_extract_drive_file_id(path.name),
        )
        if _is_denied_index_status(base_metadata):
            write_chunks(out_path, [])
            _write_chunk_mtime_sidecar(chunk_path=out_path, input_path=path)
            logger.info("Skipped quarantined Docs raw file: %s", path.name)
            continue

        docs = splitter.split_text(text)
        output_chunks: list[Chunk] = []
        output_index = 0
        for doc in docs:
            doc_text = doc.strip()
            if not doc_text:
                continue
            metadata = dict(base_metadata)
            metadata["chunk_id"] = output_index
            metadata = _with_stage(metadata, stage)
            output_chunks.append(Chunk(text=doc_text, metadata=metadata))
            output_index += 1
        write_chunks(out_path, output_chunks)
        _write_chunk_mtime_sidecar(chunk_path=out_path, input_path=path)
        logger.info(
            "Docs markdown chunked (%s) %s -> %s (%d chunks)",
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


def _split_docs_chunks_for_stage(
    *,
    chunks: Sequence[Chunk],
    splitter: RecursiveCharacterTextSplitter,
    stage: str,
) -> list[Chunk]:
    output_chunks: list[Chunk] = []
    output_index = 0
    for chunk in chunks:
        source_text = str(chunk.text or "").strip()
        if not source_text:
            continue
        base_metadata = dict(chunk.metadata)
        base_metadata.setdefault("source_type", "docs")
        base_metadata.setdefault("source_date", SOURCE_DATE_UNKNOWN)
        if _is_denied_index_status(base_metadata):
            continue
        base_metadata.pop("chunk_id", None)
        if "normalized_record_id" in base_metadata:
            base_metadata.setdefault(
                "parent_normalized_record_id",
                base_metadata.get("normalized_record_id"),
            )
        docs = splitter.split_text(source_text)
        for doc in docs:
            text = doc.strip()
            if not text:
                continue
            metadata = dict(base_metadata)
            metadata["chunk_id"] = output_index
            metadata = _with_stage(metadata, stage)
            output_chunks.append(Chunk(text=text, metadata=metadata))
            output_index += 1
    return output_chunks


def _is_denied_index_status(metadata: dict[str, object]) -> bool:
    index_status = str(metadata.get("index_status") or "active").strip().lower()
    if index_status in _DENIED_INDEX_STATUSES:
        return True
    redaction_policy = str(metadata.get("redaction_policy") or "quote_allowed").strip().lower()
    return redaction_policy == "deny"


def sheets_chunk_dir(
    *,
    ingestion_data_dir: Path,
    structured_data_dir: Path | None,
    chunk_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
    stage: str,
    skip_existing: bool = False,
    update_existing: bool = True,
    sync_deleted: bool = False,
) -> None:
    ensure_dir(chunk_dir)
    if not ingestion_data_dir.exists():
        raise FileNotFoundError(
            f"Ingestion source directory does not exist: {ingestion_data_dir}"
        )

    structured_files = (
        sorted(structured_data_dir.glob("*.jsonl"), key=lambda path: str(path))
        if structured_data_dir is not None and structured_data_dir.exists()
        else []
    )
    csv_files = sorted(ingestion_data_dir.rglob("*.csv"), key=lambda path: str(path))
    structured_drive_ids = {
        drive_file_id
        for path in structured_files
        if (drive_file_id := _extract_drive_file_id(path.name))
    }
    fallback_csv_files = [
        path
        for path in csv_files
        if (_extract_drive_file_id(path.name) or "") not in structured_drive_ids
    ]
    if not structured_files and not fallback_csv_files:
        if sync_deleted:
            _cleanup_stale_jsonl_outputs(output_dir=chunk_dir, expected_names=set())
        logger.warning("No Sheets files found under %s", ingestion_data_dir)
        return

    splitter = _build_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    expected_output_names: set[str] = set()

    for path in structured_files:
        out_path = chunk_dir / path.name
        expected_output_names.add(out_path.name)
        if _should_skip_existing_output(
            output_path=out_path,
            input_path=path,
            skip_existing=skip_existing,
            update_existing=update_existing,
            action_label="sheets structured chunking",
        ):
            continue
        from kumc_agent.infra.indexing.sheets_normalizer import (
            load_structured_sheet_chunks,
        )

        output_chunks = _split_sheet_chunks_for_stage(
            chunks=load_structured_sheet_chunks(path),
            splitter=splitter,
            stage=stage,
        )
        write_chunks(out_path, output_chunks)
        _write_chunk_mtime_sidecar(chunk_path=out_path, input_path=path)
        logger.info(
            "Sheets structured chunked (%s) %s -> %s (%d chunks)",
            stage,
            path.name,
            out_path.name,
            len(output_chunks),
        )

    for path in fallback_csv_files:
        rel_path = path.relative_to(ingestion_data_dir)
        safe_rel = sanitize_filename(str(rel_path).replace(os.sep, "__"))
        out_path = chunk_dir / f"{safe_rel}.jsonl"
        expected_output_names.add(out_path.name)
        if _should_skip_existing_output(
            output_path=out_path,
            input_path=path,
            skip_existing=skip_existing,
            update_existing=update_existing,
            action_label="sheets csv normalization",
        ):
            continue
        from kumc_agent.infra.indexing.sheets_normalizer import normalize_csv_file

        drive_metadata = _load_drive_metadata(path)
        normalized_chunks = normalize_csv_file(path, base_metadata=drive_metadata)
        output_chunks = _split_sheet_chunks_for_stage(
            chunks=normalized_chunks,
            splitter=splitter,
            stage=stage,
        )
        write_chunks(out_path, output_chunks)
        _write_chunk_mtime_sidecar(chunk_path=out_path, input_path=path)
        logger.info(
            "Sheets CSV normalized chunked (%s) %s -> %s (%d chunks)",
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


def _split_sheet_chunks_for_stage(
    *,
    chunks: Sequence[Chunk],
    splitter: RecursiveCharacterTextSplitter,
    stage: str,
) -> list[Chunk]:
    output_chunks: list[Chunk] = []
    output_index = 0
    for chunk in chunks:
        source_text = chunk.text
        if not source_text:
            continue
        base_metadata = dict(chunk.metadata)
        base_metadata.setdefault("source_type", "sheets")
        base_metadata.setdefault("source_date", SOURCE_DATE_UNKNOWN)
        base_metadata.pop("chunk_id", None)
        docs = splitter.split_text(source_text)
        for doc in docs:
            metadata = dict(base_metadata)
            metadata["chunk_id"] = output_index
            metadata = _with_stage(metadata, stage)
            output_chunks.append(Chunk(text=doc, metadata=metadata))
            output_index += 1
    return output_chunks


def recursive_chunk_dir(
    *,
    ingestion_data_dir: Path,
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

    if not ingestion_data_dir.exists():
        raise FileNotFoundError(
            f"Ingestion source directory does not exist: {ingestion_data_dir}"
        )

    input_files: list[Path] = []
    for ext in file_extensions:
        input_files.extend(ingestion_data_dir.rglob(f"*{ext}"))
    input_files = sorted(set(input_files), key=lambda path: str(path))
    if not input_files:
        if sync_deleted:
            _cleanup_stale_jsonl_outputs(output_dir=chunk_dir, expected_names=set())
        logger.warning(
            "No files found under %s for extensions: %s",
            ingestion_data_dir,
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
        rel_path = path.relative_to(ingestion_data_dir)
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
        if source_type == "vc_transcript":
            vc_meta = _build_vc_meeting_metadata(path)
            if vc_meta:
                base_metadata.update(vc_meta)
            base_metadata["source_file_name"] = (
                f"vc/{str(rel_path).replace(os.sep, '/')}"
            )
            if not str(base_metadata.get("source_date") or "").strip():
                base_metadata["source_date"] = source_date_from_vc_path(path)
        if source_type == "minecraft_wiki" and _is_minecraft_wiki_redirect_only(text):
            write_chunks(out_path, [])
            _write_chunk_mtime_sidecar(chunk_path=out_path, input_path=path)
            logger.info(
                "Skipped redirect-only Minecraft Wiki page %s -> %s",
                path.name,
                out_path.name,
            )
            continue

        docs = splitter.split_text(text)
        output_chunks: list[Chunk] = []
        output_index = 0

        for doc in docs:
            metadata = dict(base_metadata)
            metadata["chunk_id"] = output_index
            if source_type == "minecraft_wiki":
                metadata["heading_path"] = _minecraft_heading_path(
                    title=str(metadata.get("minecraft_wiki_title") or path.stem),
                    text=doc,
                )
            metadata = _with_stage(metadata, stage)
            output_chunks.append(Chunk(text=doc, metadata=metadata))
            output_index += 1

        write_chunks(out_path, output_chunks)
        _write_chunk_mtime_sidecar(chunk_path=out_path, input_path=path)
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
            source_type = str(base_metadata.get("source_type") or "").strip().lower()
            if source_type in {"messages", "discord_message"}:
                base_metadata["source_file_name"] = f"discord/{guild_id}/{channel_id}"
            else:
                base_metadata["source_file_name"] = f"{guild_id}/{channel_id}"
        base_metadata.setdefault("source_type", "messages")
        base_metadata.setdefault("source_date", SOURCE_DATE_UNKNOWN)

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
                    metadata["source_date"] = first_message_date
                search_pos = max(search_pos, end)
            metadata.setdefault("source_date", SOURCE_DATE_UNKNOWN)
            metadata["chunk_id"] = output_index
            metadata = _with_stage(metadata, stage)
            output_chunks.append(Chunk(text=doc, metadata=metadata))
            output_index += 1

        write_chunks(out_path, output_chunks)
        _write_chunk_mtime_sidecar(chunk_path=out_path, input_path=path)
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
            skip_parent_context = (
                stage == "second_recursive"
                and _is_passthrough_second_recursive(
                    source_text=source_text,
                    second_chunk_texts=docs,
                )
            )
            for doc in docs:
                metadata = dict(base_metadata)
                metadata["chunk_id"] = output_index
                if skip_parent_context:
                    metadata["skip_parent_context"] = True
                metadata = _with_stage(metadata, stage)
                output_chunks.append(Chunk(text=doc, metadata=metadata))
                output_index += 1

        write_chunks(out_path, output_chunks)
        _write_chunk_mtime_sidecar(chunk_path=out_path, input_path=path)
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
            remove_symbols=config.sparse_remove_symbols,
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
        _write_chunk_mtime_sidecar(chunk_path=out_path, input_path=path)
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
    second_chunk_dir: Path | None = None,
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
    summary_batch_size = max(1, int(getattr(config, "summery_batch_size", 1)))
    logger.info(
        "Summery chunking enabled (%s) for %d files in %s (batch_size=%d)",
        provider,
        len(jsonl_files),
        input_chunk_dir,
        summary_batch_size,
    )

    expected_output_names = {path.name for path in jsonl_files}
    output_chunks_by_path: dict[Path, list[Chunk]] = {}
    decisions_by_path: dict[Path, list[tuple[object, SummarySearchabilityDecision]]] = {}
    summary_jobs: list[tuple[Path, str, dict[str, object], str, str, str]] = []
    processed_paths: list[Path] = []
    input_path_by_output_path: dict[Path, Path] = {}
    summary_requests_per_minute = getattr(
        config,
        "gemini_summary_requests_per_minute",
        getattr(config, "gemini_requests_per_minute", 60),
    )
    summary_model = config.summery_gemini_model

    def _run_summary_prompt(
        prompt: str,
        *,
        source_name: str,
        fallback_summary: str,
    ) -> SummarySearchabilityDecision:
        disabled = provider.strip().lower() in {"", "none", "off", "disabled", "false", "0"}
        if disabled:
            return SummarySearchabilityDecision.keep(
                summary=fallback_summary,
                reason="provider_disabled",
                fallback_used=True,
            )
        return _run_llm_summary_decision(
            prompt=prompt,
            source_name=source_name,
            fallback_summary=fallback_summary,
            provider=provider,
            api_key=config.gemini_api_key,
            gemini_requests_per_minute=summary_requests_per_minute,
            model=summary_model,
            temperature=config.summery_temperature,
            max_output_tokens=config.summery_max_output_tokens,
            max_retries=max_retries,
            action_label="Summery chunking",
            gemini_rate_limiter_name=index_summary_rate_limiter_name(),
            response_mime_type="application/json",
        )

    for path in jsonl_files:
        out_path = output_chunk_dir / path.name
        if _should_skip_existing_output(
            output_path=out_path,
            input_path=path,
            skip_existing=skip_existing,
            update_existing=update_existing,
            action_label="summery chunking",
        ) and summary_decision_sidecar_path(out_path).exists():
            continue
        processed_paths.append(out_path)
        input_path_by_output_path[out_path] = path
        output_chunks_by_path[out_path] = []
        decisions_by_path[out_path] = []

        chunks = load_chunks(path)
        if not chunks:
            logger.warning("Empty chunk file: %s", path.name)
            continue

        skip_parent_chunk_ids: set[int] = set()
        if second_chunk_dir is not None:
            second_path = second_chunk_dir / path.name
            if second_path.exists():
                second_chunks = load_chunks(second_path)
                skip_parent_chunk_ids = _skip_parent_chunk_ids_from_second_chunks(
                    second_chunks
                )

        for chunk in chunks:
            source_text = chunk.text
            if not source_text:
                continue
            first_chunk_id = _normalize_chunk_id(chunk.metadata.get("chunk_id"))
            if first_chunk_id is not None and first_chunk_id in skip_parent_chunk_ids:
                logger.info(
                    "Skip summery chunking for %s (first_chunk_id=%s, unchanged second recursive chunk)",
                    path.name,
                    first_chunk_id,
                )
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
            prompt = build_summary_searchability_prompt(prompt)
            fallback_summary = str(source_text or "").strip()[: config.summery_characters]
            summary_jobs.append(
                (
                    out_path,
                    path.name,
                    base_metadata,
                    prompt,
                    fallback_summary,
                    source_text,
                )
            )

    batch_results: list[SummarySearchabilityDecision | None] = [None] * len(summary_jobs)
    for batch_start in range(0, len(summary_jobs), summary_batch_size):
        batch = summary_jobs[batch_start : batch_start + summary_batch_size]
        if len(batch) == 1:
            _out_path, source_name, _metadata, prompt, fallback_summary, _source_text = batch[0]
            batch_results[batch_start] = _run_summary_prompt(
                prompt,
                source_name=source_name,
                fallback_summary=fallback_summary,
            )
            continue

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures: dict[Future[SummarySearchabilityDecision], int] = {}
            for offset, (
                _out_path,
                source_name,
                _metadata,
                prompt,
                fallback_summary,
                _source_text,
            ) in enumerate(batch):
                futures[
                    executor.submit(
                        _run_summary_prompt,
                        prompt,
                        source_name=source_name,
                        fallback_summary=fallback_summary,
                    )
                ] = batch_start + offset
            for future in as_completed(futures):
                batch_results[futures[future]] = future.result()

    output_indexes: dict[Path, int] = {path: 0 for path in processed_paths}
    for (
        out_path,
        _source_name,
        base_metadata,
        _prompt,
        _fallback_summary,
        source_text,
    ), decision in zip(
        summary_jobs,
        batch_results,
        strict=False,
    ):
        if decision is None:
            continue
        parent_id = normalize_summary_parent_id(base_metadata.get("parent_chunk_id"))
        decisions_by_path.setdefault(out_path, []).append((parent_id, decision))
        if not decision.searchable:
            continue
        chunk_text = sanitize_summary_text(decision.summary)
        if not chunk_text:
            continue
        metadata = dict(base_metadata)
        metadata["chunk_id"] = output_indexes[out_path]
        metadata.update(
            summary_quality_metadata(
                source_text=str(source_text or ""),
                summary_text=chunk_text,
                decision=decision,
            )
        )
        metadata = _with_stage(metadata, "summery")
        output_chunks_by_path[out_path].append(
            Chunk(text=chunk_text, metadata=metadata)
        )
        output_indexes[out_path] += 1

    for out_path in processed_paths:
        write_chunks(out_path, output_chunks_by_path.get(out_path, []))
        write_summary_searchability_decisions(
            path=summary_decision_sidecar_path(out_path),
            decisions=decisions_by_path.get(out_path, []),
        )
        input_path = input_path_by_output_path[out_path]
        _write_chunk_mtime_sidecar(chunk_path=out_path, input_path=input_path)
        logger.info(
            "Summery chunked %s -> %s (%d chunks)",
            input_path.name,
            out_path.name,
            len(output_chunks_by_path.get(out_path, [])),
        )

    if sync_deleted:
        _cleanup_stale_jsonl_outputs(
            output_dir=output_chunk_dir,
            expected_names=expected_output_names,
        )


def _strip_chunk_metadata(metadata: dict[str, object]) -> dict[str, object]:
    cleaned = {k: metadata.get(k, "") for k in _METADATA_KEYS}
    return cleaned


def _minecraft_heading_path(*, title: str, text: str) -> list[str]:
    headings = [
        match.group(2).strip()
        for match in re.finditer(r"(?m)^(#{1,6})\s+(.+)$", text or "")
        if match.group(2).strip()
    ]
    base = [title.strip()] if title.strip() else []
    if not headings:
        return base
    return base + headings[:3]


def _normalize_chunk_id(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _metadata_flag_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _normalize_text_for_comparison(text: str) -> str:
    return " ".join((text or "").split())


def _is_passthrough_second_recursive(
    *,
    source_text: str,
    second_chunk_texts: Sequence[str],
) -> bool:
    if len(second_chunk_texts) != 1:
        return False
    return _normalize_text_for_comparison(
        source_text
    ) == _normalize_text_for_comparison(second_chunk_texts[0])


def _skip_parent_chunk_ids_from_second_chunks(chunks: Sequence[Chunk]) -> set[int]:
    parent_ids: set[int] = set()
    for chunk in chunks:
        if not _metadata_flag_enabled(chunk.metadata.get("skip_parent_context")):
            continue
        parent_chunk_id = _normalize_chunk_id(chunk.metadata.get("parent_chunk_id"))
        if parent_chunk_id is None:
            continue
        parent_ids.add(parent_chunk_id)
    return parent_ids


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


def _run_llm_summary_decision(
    *,
    prompt: str,
    source_name: str,
    fallback_summary: str,
    provider: str,
    api_key: str,
    gemini_requests_per_minute: int,
    model: str,
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    action_label: str,
    gemini_rate_limiter_name: str = "",
    response_mime_type: str | None = "application/json",
) -> SummarySearchabilityDecision:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = generate_text(
                provider=provider,
                api_key=api_key,
                prompt=prompt,
                model=model,
                system_prompt=get_llm_chunk_system_prompt(),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type=response_mime_type,
                gemini_requests_per_minute=gemini_requests_per_minute,
                gemini_rate_limiter_name=gemini_rate_limiter_name,
            )
            return parse_summary_searchability_response(
                response,
                fallback_summary=fallback_summary,
            )
        except Exception as exc:
            last_error = exc
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
    return SummarySearchabilityDecision.keep(
        summary=fallback_summary,
        reason=str(last_error or "generation_failed"),
        fallback_used=True,
        parse_failed=True,
    )


def _run_llm_chunking(
    *,
    prompt: str,
    source_name: str,
    provider: str,
    api_key: str,
    gemini_requests_per_minute: int,
    model: str,
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    action_label: str,
    gemini_rate_limiter_name: str = "",
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
                system_prompt=get_llm_chunk_system_prompt(),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type=response_mime_type,
                gemini_requests_per_minute=gemini_requests_per_minute,
                gemini_rate_limiter_name=gemini_rate_limiter_name,
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
