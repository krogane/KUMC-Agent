from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from kumc_agent.infra.indexing.date_metadata import SOURCE_DATE_UNKNOWN, infer_source_date
from kumc_agent.infra.indexing.docs_normalizer import (
    content_sha256,
    is_page_number_only_text,
    nonempty_character_count,
    text_bytes,
)


_REQUIRED_METADATA_KEYS = (
    "drive_file_id",
    "drive_file_name",
    "drive_path",
    "drive_mime_type",
    "drive_modified_time",
    "drive_url",
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*\bsrc=[\"'](?P<url>[^\"']+)[\"'][^>]*>", re.IGNORECASE)


@dataclass(frozen=True)
class GoogleDriveDocsQualityThresholds:
    enabled: bool = True
    policy: str = "warn"
    min_text_bytes: int = 100
    min_nonempty_characters: int = 200
    max_short_document_ratio: float = 0.4
    max_source_date_unknown_ratio: float = 0.2
    required_metadata_keys: tuple[str, ...] = _REQUIRED_METADATA_KEYS


@dataclass(frozen=True)
class GoogleDriveDocsRawAuditReport:
    source: str
    raw_dir: Path
    exists: bool
    status: str
    can_continue: bool
    critical_failures: tuple[str, ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "status": self.status,
            "can_continue": self.can_continue,
            "metadata": {
                "raw_dir": str(self.raw_dir),
                "exists": self.exists,
                "critical_failures": list(self.critical_failures),
                "warnings": list(self.warnings),
                **self.metadata,
            },
        }

    def to_markdown(self) -> str:
        meta = self.metadata
        lines = [
            "# Google Drive Docs Raw品質監査",
            "",
            f"- source: `{self.source}`",
            f"- raw_dir: `{self.raw_dir}`",
            f"- status: `{self.status}`",
            f"- can_continue: `{str(self.can_continue).lower()}`",
            "",
            "## 集計",
            "",
            f"- Markdown本文: {meta.get('markdown_count', 0)}",
            f"- sidecar metadata: {meta.get('metadata_count', 0)}",
            f"- metadata欠落: {meta.get('missing_metadata_count', 0)}",
            f"- orphan metadata: {meta.get('orphan_metadata_count', 0)}",
            f"- JSON parse失敗: {meta.get('invalid_metadata_count', 0)}",
            f"- 短文資料: {meta.get('short_document_count', 0)}",
            f"- 短文率: {meta.get('short_document_ratio', 0.0)}",
            f"- source_date不明: {meta.get('source_date_unknown_count', 0)}",
            f"- 画像artifact: {meta.get('image_artifact_count', 0)}",
            f"- 本文内画像参照あり: {meta.get('markdown_image_reference_count', 0)}",
            f"- 重複本文グループ: {meta.get('duplicate_group_count', 0)}",
            "",
            "## MIME type",
            "",
        ]
        mime_counts = meta.get("mime_type_counts")
        if isinstance(mime_counts, dict):
            for key, value in mime_counts.items():
                lines.append(f"- `{key or '(empty)'}`: {value}")
        lines.extend(["", "## 品質フラグ", ""])
        flag_counts = meta.get("quality_flag_counts")
        if isinstance(flag_counts, dict):
            for key, value in flag_counts.items():
                lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## 短文資料", ""])
        short_docs = meta.get("top_short_documents")
        if isinstance(short_docs, list):
            for item in short_docs:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- `{item.get('file', '')}`: {item.get('bytes', 0)} bytes / "
                    f"{item.get('nonempty_characters', 0)} chars"
                )
        duplicates = meta.get("duplicate_groups")
        if isinstance(duplicates, list) and duplicates:
            lines.extend(["", "## 重複本文", ""])
            for item in duplicates:
                if not isinstance(item, dict):
                    continue
                files = ", ".join(str(file) for file in item.get("files", []))
                lines.append(f"- `{item.get('content_sha256', '')}`: {files}")
        if self.critical_failures:
            lines.extend(["", "## critical_failures", ""])
            lines.extend(f"- {item}" for item in self.critical_failures)
        if self.warnings:
            lines.extend(["", "## warnings", ""])
            lines.extend(f"- {item}" for item in self.warnings)
        return "\n".join(lines).rstrip() + "\n"


def audit_google_drive_docs_raw_dir(
    *,
    raw_dir: Path,
    thresholds: GoogleDriveDocsQualityThresholds | None = None,
    normalized_dir: Path | None = None,
    image_dir: Path | None = None,
    chunk_count: int | None = None,
    top_n: int = 20,
) -> GoogleDriveDocsRawAuditReport:
    resolved_thresholds = thresholds or GoogleDriveDocsQualityThresholds()
    if not raw_dir.exists():
        return _report(
            raw_dir=raw_dir,
            thresholds=resolved_thresholds,
            failures=("raw_dir_missing",),
            warnings=tuple(),
            metadata={
                "markdown_count": 0,
                "metadata_count": 0,
                "thresholds": resolved_thresholds.__dict__,
                "chunk_count": int(chunk_count or 0),
            },
        )

    markdown_files = sorted(path for path in raw_dir.glob("*.md") if path.is_file())
    metadata_files = sorted(path for path in raw_dir.glob("*.md.meta.json") if path.is_file())
    metadata_by_markdown: dict[str, dict[str, object]] = {}
    invalid_metadata: list[str] = []
    for path in metadata_files:
        base_name = path.name[: -len(".meta.json")]
        payload = _read_json_object(path)
        if payload is None:
            invalid_metadata.append(path.name)
            continue
        metadata_by_markdown[base_name] = payload

    markdown_names = {path.name for path in markdown_files}
    missing_metadata: list[str] = []
    incomplete_metadata: list[dict[str, object]] = []
    mime_type_counts: Counter[str] = Counter()
    index_status_counts: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    length_values: list[int] = []
    short_documents: list[dict[str, object]] = []
    source_date_unknown: list[str] = []
    markdown_image_reference_count = 0
    page_heading_count = 0
    slide_heading_count = 0
    page_count_sum = 0
    slide_count_sum = 0
    ocr_page_count_sum = 0
    embedded_image_count_sum = 0
    duplicate_candidates: dict[str, list[str]] = defaultdict(list)

    for path in markdown_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        size = text_bytes(text)
        chars = nonempty_character_count(text)
        length_values.append(size)
        metadata = metadata_by_markdown.get(path.name)
        if metadata is None:
            missing_metadata.append(path.name)
            metadata = {}
        missing_keys = [
            key
            for key in resolved_thresholds.required_metadata_keys
            if not str(metadata.get(key) or "").strip()
        ]
        if missing_keys:
            incomplete_metadata.append({"file": path.name, "missing_keys": missing_keys})
        mime_type_counts[str(metadata.get("drive_mime_type") or "")] += 1
        index_status_counts[str(metadata.get("index_status") or "active")] += 1
        for flag in _as_list(metadata.get("quality_flags")):
            quality_flag_counts[flag] += 1
        if is_page_number_only_text(text):
            quality_flag_counts["page_number_only_detected"] += 1
        if size < resolved_thresholds.min_text_bytes or chars < resolved_thresholds.min_nonempty_characters:
            short_documents.append(
                {
                    "file": path.name,
                    "bytes": size,
                    "nonempty_characters": chars,
                    "drive_mime_type": str(metadata.get("drive_mime_type") or ""),
                }
            )
        if _has_markdown_image_ref(text):
            markdown_image_reference_count += 1
        page_heading_count += len(re.findall(r"(?im)^##\s+Page\s+\d+\s*$", text))
        slide_heading_count += len(re.findall(r"(?im)^##\s+Slide\s+\d+\s*$", text))
        source_date = str(metadata.get("source_date") or "").strip()
        if not source_date:
            source_date = infer_source_date(
                metadata={
                    **metadata,
                    "source_type": "docs",
                    "drive_file_path": metadata.get("drive_path")
                    or metadata.get("drive_file_path")
                    or "",
                }
            )
        if not source_date or source_date == SOURCE_DATE_UNKNOWN:
            source_date_unknown.append(path.name)
        page_count_sum += _to_int(metadata.get("page_count"))
        slide_count_sum += _to_int(metadata.get("slide_count"))
        ocr_page_count_sum += _to_int(metadata.get("ocr_page_count"))
        embedded_image_count_sum += _to_int(metadata.get("embedded_image_count"))
        digest = str(metadata.get("content_sha256") or metadata.get("checksum") or "").strip()
        if not digest:
            digest = content_sha256(text)
        duplicate_candidates[digest].append(path.name)

    orphan_metadata = [
        path.name
        for path in metadata_files
        if path.name[: -len(".meta.json")] not in markdown_names
    ]
    duplicate_groups = [
        {"content_sha256": digest, "files": sorted(files)}
        for digest, files in sorted(duplicate_candidates.items())
        if digest and len(files) > 1
    ]
    short_ratio = (
        round(len(short_documents) / len(markdown_files), 4)
        if markdown_files
        else 0.0
    )
    source_date_unknown_ratio = (
        round(len(source_date_unknown) / len(markdown_files), 4)
        if markdown_files
        else 0.0
    )
    normalized_count = _count_normalized_records(normalized_dir)
    image_stats = _image_artifact_stats(
        image_dir or raw_dir.parent / "images" / "google_drive",
        markdown_drive_ids={
            str(meta.get("drive_file_id") or "").strip()
            for meta in metadata_by_markdown.values()
            if str(meta.get("drive_file_id") or "").strip()
        },
    )

    failures: list[str] = []
    warnings: list[str] = []
    if missing_metadata:
        failures.append("metadata_missing")
    if invalid_metadata:
        failures.append("metadata_invalid_json")
    if incomplete_metadata:
        failures.append("metadata_incomplete")
    if short_ratio > resolved_thresholds.max_short_document_ratio:
        failures.append("short_document_ratio_too_high")
    if source_date_unknown_ratio > resolved_thresholds.max_source_date_unknown_ratio:
        failures.append("source_date_unknown_ratio_too_high")
    if duplicate_groups:
        warnings.append("duplicate_content_present")
    if markdown_files and normalized_dir is not None and normalized_count == 0:
        warnings.append("normalized_docs_missing")
    if not markdown_files:
        warnings.append("raw_markdown_empty")

    metadata = {
        "markdown_count": len(markdown_files),
        "metadata_count": len(metadata_by_markdown),
        "missing_metadata_count": len(missing_metadata),
        "orphan_metadata_count": len(orphan_metadata),
        "invalid_metadata_count": len(invalid_metadata),
        "incomplete_metadata_count": len(incomplete_metadata),
        "mime_type_counts": dict(sorted(mime_type_counts.items())),
        "index_status_counts": dict(sorted(index_status_counts.items())),
        "quality_flag_counts": dict(sorted(quality_flag_counts.items())),
        "length_distribution": {
            "lt_100_bytes": sum(1 for value in length_values if value < 100),
            "lt_500_bytes": sum(1 for value in length_values if value < 500),
            "lt_1000_bytes": sum(1 for value in length_values if value < 1000),
            "gte_1000_bytes": sum(1 for value in length_values if value >= 1000),
            "gte_10000_bytes": sum(1 for value in length_values if value >= 10000),
        },
        "short_document_count": len(short_documents),
        "short_document_ratio": short_ratio,
        "source_date_unknown_count": len(source_date_unknown),
        "source_date_unknown_ratio": source_date_unknown_ratio,
        "page_heading_count": page_heading_count,
        "slide_heading_count": slide_heading_count,
        "page_count_sum": page_count_sum,
        "slide_count_sum": slide_count_sum,
        "ocr_page_count_sum": ocr_page_count_sum,
        "embedded_image_count_sum": embedded_image_count_sum,
        "markdown_image_reference_count": markdown_image_reference_count,
        "image_artifact_count": image_stats["image_artifact_count"],
        "linked_image_artifact_count": image_stats["linked_image_artifact_count"],
        "normalized_record_count": normalized_count,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups[: max(0, top_n)],
        "top_short_documents": sorted(
            short_documents,
            key=lambda item: (int(item["bytes"]), str(item["file"])),
        )[: max(0, top_n)],
        "missing_metadata_files": missing_metadata[: max(0, top_n)],
        "invalid_metadata_files": invalid_metadata[: max(0, top_n)],
        "incomplete_metadata_files": incomplete_metadata[: max(0, top_n)],
        "orphan_metadata_files": orphan_metadata[: max(0, top_n)],
        "source_date_unknown_files": source_date_unknown[: max(0, top_n)],
        "thresholds": resolved_thresholds.__dict__,
        "chunk_count": int(chunk_count or 0),
    }
    return _report(
        raw_dir=raw_dir,
        thresholds=resolved_thresholds,
        failures=tuple(failures),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def build_google_drive_docs_quality_payload(
    *,
    raw_dir: Path,
    normalized_dir: Path | None = None,
    image_dir: Path | None = None,
    thresholds: GoogleDriveDocsQualityThresholds | None = None,
    chunk_count: int | None = None,
) -> dict[str, object]:
    report = audit_google_drive_docs_raw_dir(
        raw_dir=raw_dir,
        normalized_dir=normalized_dir,
        image_dir=image_dir,
        thresholds=thresholds,
        chunk_count=chunk_count,
    )
    return report.to_payload()


def _report(
    *,
    raw_dir: Path,
    thresholds: GoogleDriveDocsQualityThresholds,
    failures: tuple[str, ...],
    warnings: tuple[str, ...],
    metadata: dict[str, object],
) -> GoogleDriveDocsRawAuditReport:
    policy = str(thresholds.policy or "warn").strip().lower()
    if not thresholds.enabled:
        status = "disabled"
        can_continue = True
    elif failures and policy == "fail":
        status = "failed"
        can_continue = False
    elif failures or warnings:
        status = "warning"
        can_continue = True
    else:
        status = "passed"
        can_continue = True
    return GoogleDriveDocsRawAuditReport(
        source="docs",
        raw_dir=raw_dir,
        exists=raw_dir.exists(),
        status=status,
        can_continue=can_continue,
        critical_failures=failures,
        warnings=warnings,
        metadata=metadata,
    )


def _read_json_object(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): value for key, value in payload.items()}


def _has_markdown_image_ref(text: str) -> bool:
    return bool(_MARKDOWN_IMAGE_RE.search(text or "") or _HTML_IMAGE_RE.search(text or ""))


def _count_normalized_records(normalized_dir: Path | None) -> int:
    if normalized_dir is None or not normalized_dir.exists():
        return 0
    count = 0
    for path in normalized_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                if line.strip():
                    count += 1
    return count


def _image_artifact_stats(
    image_dir: Path,
    *,
    markdown_drive_ids: set[str],
) -> dict[str, int]:
    if not image_dir.exists():
        return {"image_artifact_count": 0, "linked_image_artifact_count": 0}
    count = 0
    linked = 0
    for path in image_dir.glob("**/*"):
        if not path.is_file() or path.name.endswith(".meta.json"):
            continue
        count += 1
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        metadata = _read_json_object(sidecar) if sidecar.exists() else {}
        drive_file_id = str((metadata or {}).get("drive_file_id") or "").strip()
        if drive_file_id and drive_file_id in markdown_drive_ids:
            linked += 1
    return {"image_artifact_count": count, "linked_image_artifact_count": linked}


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _to_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
