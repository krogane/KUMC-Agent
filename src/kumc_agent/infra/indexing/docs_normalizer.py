from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kumc_agent.infra.indexing.chunks import Chunk, load_chunks, write_chunks
from kumc_agent.infra.indexing.date_metadata import (
    SOURCE_DATE_UNKNOWN,
    infer_source_date,
)


_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
_PAGE_HEADING_RE = re.compile(r"^#+\s*(page|slide)\s+\d+\s*$", re.IGNORECASE)
_NUMBER_ONLY_RE = re.compile(r"^[#\s\-_*|:;,.(){}\[\]0-9０-９ivxlcdmIVXLCDM]+$")
_WORDISH_RE = re.compile(r"[A-Za-zぁ-んァ-ン一-龥]")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")


def content_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def text_bytes(text: str) -> int:
    return len((text or "").encode("utf-8"))


def nonempty_character_count(text: str) -> int:
    return sum(1 for char in text or "" if not char.isspace())


def is_page_number_only_text(text: str) -> bool:
    lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or _PAGE_HEADING_RE.match(line):
            continue
        lines.append(line)
    if not lines:
        return False
    joined = " ".join(lines)
    if _WORDISH_RE.search(joined):
        return False
    return bool(_NUMBER_ONLY_RE.match(joined))


def quality_flags_for_text(
    text: str,
    *,
    min_nonempty_characters: int = 200,
    embedded_image_count: int = 0,
    ocr_candidate_count: int = 0,
    ocr_page_count: int = 0,
) -> list[str]:
    flags: list[str] = []
    if nonempty_character_count(text) < max(0, int(min_nonempty_characters)):
        flags.append("too_short")
    if is_page_number_only_text(text):
        flags.append("page_number_only")
    if int(embedded_image_count or 0) > 0 and (
        "too_short" in flags or nonempty_character_count(text) < max(1, min_nonempty_characters) * 2
    ):
        flags.append("image_heavy")
    if int(ocr_candidate_count or 0) > int(ocr_page_count or 0):
        flags.append("ocr_needed")
    return _dedupe(flags)


def build_docs_quality_metadata(
    *,
    base_metadata: dict[str, object],
    text: str,
    extraction_method: str,
    extraction_status: str = "ok",
    page_count: int = 0,
    slide_count: int = 0,
    ocr_page_count: int = 0,
    ocr_candidate_count: int = 0,
    embedded_image_count: int = 0,
    min_nonempty_characters: int = 200,
    quarantine_low_information: bool = True,
    extra_quality_flags: Sequence[str] = (),
) -> dict[str, object]:
    metadata = dict(base_metadata)
    modified_time = str(metadata.get("drive_modified_time") or "").strip()
    updated_at = str(metadata.get("updated_at") or "").strip() or modified_time
    if not updated_at:
        updated_at = datetime.now(timezone.utc).isoformat()
    metadata["updated_at"] = updated_at
    metadata["source_date"] = infer_source_date(
        metadata={
            **metadata,
            "source_type": "docs",
            "drive_file_path": metadata.get("drive_path")
            or metadata.get("drive_file_path")
            or "",
        }
    )
    digest = content_sha256(text)
    metadata["checksum"] = digest
    metadata["content_sha256"] = digest
    metadata["extraction_method"] = extraction_method
    metadata["extraction_status"] = extraction_status
    metadata["text_bytes"] = text_bytes(text)
    metadata["nonempty_characters"] = nonempty_character_count(text)
    metadata["page_count"] = max(0, int(page_count or 0))
    metadata["slide_count"] = max(0, int(slide_count or 0))
    metadata["ocr_page_count"] = max(0, int(ocr_page_count or 0))
    metadata["ocr_candidate_count"] = max(0, int(ocr_candidate_count or 0))
    metadata["embedded_image_count"] = max(0, int(embedded_image_count or 0))
    flags = quality_flags_for_text(
        text,
        min_nonempty_characters=min_nonempty_characters,
        embedded_image_count=embedded_image_count,
        ocr_candidate_count=ocr_candidate_count,
        ocr_page_count=ocr_page_count,
    )
    flags.extend(str(flag) for flag in extra_quality_flags if str(flag).strip())
    metadata["quality_flags"] = _dedupe(flags)
    if quarantine_low_information and _is_low_information_flags(metadata["quality_flags"]):
        metadata["index_status"] = "quarantined"
    else:
        metadata["index_status"] = str(metadata.get("index_status") or "active")
    metadata.setdefault("redaction_policy", "quote_allowed")
    metadata.setdefault("visibility", "guild")
    access_scope = metadata.get("access_scope")
    if not isinstance(access_scope, dict):
        metadata["access_scope"] = {"visibility": metadata["visibility"]}
    return metadata


def normalize_markdown_text(
    text: str,
    *,
    base_metadata: dict[str, object],
    source_file_name: str,
    min_nonempty_characters: int = 200,
    quarantine_low_information: bool = True,
) -> list[Chunk]:
    sections = _markdown_sections(text)
    if not sections and (text or "").strip():
        sections = [("markdown_body", [], (text or "").strip())]
    chunks: list[Chunk] = []
    for idx, (block_type, heading_path, body) in enumerate(sections):
        body = body.strip()
        if not body:
            continue
        metadata = _record_metadata(
            base_metadata=base_metadata,
            source_file_name=source_file_name,
            record_index=idx,
            block_type=block_type,
            min_nonempty_characters=min_nonempty_characters,
            quarantine_low_information=quarantine_low_information,
            text=body,
        )
        if heading_path:
            metadata["heading_path"] = list(heading_path)
        chunks.append(Chunk(text=body, metadata=metadata))
    return chunks


def normalize_pdf_pages(
    pages: Sequence[dict[str, object]],
    *,
    base_metadata: dict[str, object],
    source_file_name: str,
    min_nonempty_characters: int = 200,
    quarantine_low_information: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, page in enumerate(pages):
        text = str(page.get("text") or "").strip()
        if not text:
            continue
        page_number = _to_int(page.get("page_number"), idx + 1)
        metadata = _record_metadata(
            base_metadata=base_metadata,
            source_file_name=source_file_name,
            record_index=idx,
            block_type="pdf_page",
            min_nonempty_characters=min_nonempty_characters,
            quarantine_low_information=quarantine_low_information,
            text=text,
            record_quality_flags=_as_list(page.get("quality_flags")),
        )
        metadata["page_number"] = page_number
        metadata["page_ref"] = f"page:{page_number}"
        metadata["ocr_status"] = str(page.get("ocr_status") or "")
        if str(page.get("ocr_text") or "").strip():
            metadata["ocr_text_present"] = True
        chunks.append(Chunk(text=text, metadata=metadata))
    return chunks


def normalize_pptx_slides(
    slides: Sequence[dict[str, object]],
    *,
    base_metadata: dict[str, object],
    source_file_name: str,
    min_nonempty_characters: int = 200,
    quarantine_low_information: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, slide in enumerate(slides):
        text = str(slide.get("text") or "").strip()
        notes = str(slide.get("speaker_notes") or "").strip()
        body_parts = [part for part in (text, f"Speaker notes:\n{notes}" if notes else "") if part]
        body = "\n".join(body_parts).strip()
        if not body:
            continue
        slide_number = _to_int(slide.get("slide_number"), idx + 1)
        image_refs = _as_list(slide.get("embedded_image_refs"))
        metadata = _record_metadata(
            base_metadata=base_metadata,
            source_file_name=source_file_name,
            record_index=idx,
            block_type="slide",
            min_nonempty_characters=min_nonempty_characters,
            quarantine_low_information=quarantine_low_information,
            text=body,
            embedded_image_count=len(image_refs),
        )
        metadata["slide_number"] = slide_number
        metadata["slide_ref"] = f"slide:{slide_number}"
        metadata["embedded_image_refs"] = image_refs
        chunks.append(Chunk(text=body, metadata=metadata))
    return chunks


def normalize_docx_blocks(
    blocks: Sequence[dict[str, object]],
    *,
    base_metadata: dict[str, object],
    source_file_name: str,
    min_nonempty_characters: int = 200,
    quarantine_low_information: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, block in enumerate(blocks):
        body = str(block.get("text") or "").strip()
        if not body:
            continue
        block_type = str(block.get("block_type") or "paragraph")
        metadata = _record_metadata(
            base_metadata=base_metadata,
            source_file_name=source_file_name,
            record_index=idx,
            block_type=f"docx_{block_type}",
            min_nonempty_characters=min_nonempty_characters,
            quarantine_low_information=quarantine_low_information,
            text=body,
        )
        heading_path = _as_list(block.get("heading_path"))
        if heading_path:
            metadata["heading_path"] = heading_path
        chunks.append(Chunk(text=body, metadata=metadata))
    return chunks


def write_normalized_docs_jsonl(
    *,
    output_path: Path,
    chunks: Sequence[Chunk],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_chunks(output_path, chunks)


def load_structured_doc_chunks(path: Path) -> list[Chunk]:
    return load_chunks(path)


def _markdown_sections(text: str) -> list[tuple[str, list[str], str]]:
    sections: list[tuple[str, list[str], str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_heading_path: list[str] = []

    def flush(block_type: str = "markdown_section") -> None:
        nonlocal current_lines, current_heading_path
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((block_type, list(current_heading_path), body))
        current_lines = []

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        heading_match = _HEADING_RE.match(line.strip())
        if heading_match:
            flush()
            level = len(heading_match.group("marks"))
            title = heading_match.group("title").strip()
            heading_stack = [
                item for item in heading_stack if int(item[0]) < level
            ]
            heading_stack.append((level, title))
            current_heading_path = [title for _, title in heading_stack]
            current_lines = [line]
            continue
        current_lines.append(line)
    flush()

    normalized: list[tuple[str, list[str], str]] = []
    for block_type, heading_path, body in sections:
        if _looks_like_markdown_table(body):
            block_type = "markdown_table"
        normalized.append((block_type, heading_path, body))
    return normalized


def _looks_like_markdown_table(text: str) -> bool:
    rows = [line for line in (text or "").splitlines() if line.strip()]
    table_rows = sum(1 for line in rows if _TABLE_ROW_RE.match(line))
    return table_rows >= 2 and table_rows >= max(1, len(rows) - 1)


def _record_metadata(
    *,
    base_metadata: dict[str, object],
    source_file_name: str,
    record_index: int,
    block_type: str,
    min_nonempty_characters: int,
    quarantine_low_information: bool,
    text: str,
    embedded_image_count: int = 0,
    record_quality_flags: Sequence[str] = (),
) -> dict[str, object]:
    metadata = dict(base_metadata)
    metadata["source_type"] = "docs"
    metadata["source_file_name"] = source_file_name
    metadata["normalized_record_id"] = record_index
    metadata["block_type"] = block_type
    metadata.setdefault("source_date", SOURCE_DATE_UNKNOWN)
    flags = list(_as_list(metadata.get("quality_flags")))
    flags.extend(
        quality_flags_for_text(
            text,
            min_nonempty_characters=min_nonempty_characters,
            embedded_image_count=embedded_image_count,
        )
    )
    flags.extend(record_quality_flags)
    metadata["quality_flags"] = _dedupe(str(flag) for flag in flags)
    if quarantine_low_information and _is_low_information_flags(metadata["quality_flags"]):
        metadata["index_status"] = "quarantined"
    else:
        metadata["index_status"] = str(metadata.get("index_status") or "active")
    return metadata


def _is_low_information_flags(flags: object) -> bool:
    values = set(_as_list(flags))
    return bool(values & {"too_short", "page_number_only"})


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dedupe(values: Sequence[str] | Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _to_int(value: object, fallback: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return fallback
    return fallback
