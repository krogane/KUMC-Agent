from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import re
from pathlib import Path

from kumc_agent.domain.models.source import SourceRawItem
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class NotionQualityThresholds:
    enabled: bool = True
    policy: str = "warn"
    min_text_bytes: int = 200
    min_nonempty_characters: int = 50
    max_short_document_ratio: float = 0.4
    max_heading_only_ratio: float = 0.3
    max_duplicate_text_ratio: float = 0.05
    min_repository_coverage_ratio: float = 1.0
    min_index_coverage_ratio: float = 1.0


@dataclass(frozen=True)
class NotionRawAuditReport:
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
            "# Notion Raw/Indexing品質監査",
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
            f"- 短文ページ: {meta.get('short_document_count', 0)}",
            f"- 短文率: {meta.get('short_document_ratio', 0.0)}",
            f"- heading/url only: {meta.get('heading_only_count', 0)}",
            f"- heading/url only率: {meta.get('heading_only_ratio', 0.0)}",
            f"- 重複本文ページ: {meta.get('duplicate_document_count', 0)}",
            f"- repository coverage: {meta.get('repository_coverage_ratio', 0.0)}",
            f"- index coverage: {meta.get('index_coverage_ratio', 0.0)}",
            f"- Notion asset block: {meta.get('asset_block_count', 0)}",
            "",
            "## access_scope",
            "",
        ]
        visibility_counts = meta.get("visibility_counts")
        if isinstance(visibility_counts, dict):
            for key, value in visibility_counts.items():
                lines.append(f"- `{key or '(empty)'}`: {value}")
        lines.extend(["", "## unsupported block types", ""])
        unsupported = meta.get("unsupported_block_type_counts")
        if isinstance(unsupported, dict):
            for key, value in unsupported.items():
                lines.append(f"- `{key}`: {value}")
        if self.critical_failures:
            lines.extend(["", "## critical_failures", ""])
            lines.extend(f"- {item}" for item in self.critical_failures)
        if self.warnings:
            lines.extend(["", "## warnings", ""])
            lines.extend(f"- {item}" for item in self.warnings)
        return "\n".join(lines).rstrip() + "\n"


def audit_notion_raw_dir(
    *,
    raw_dir: Path,
    thresholds: NotionQualityThresholds | None = None,
    repository_dir: Path | None = None,
    index_page_ids: set[str] | None = None,
    stage_dirs: tuple[Path, ...] = tuple(),
    object_storage_dir: Path | None = None,
    top_n: int = 20,
) -> NotionRawAuditReport:
    resolved = thresholds or NotionQualityThresholds()
    if not raw_dir.exists():
        return _report(
            raw_dir=raw_dir,
            thresholds=resolved,
            failures=("raw_dir_missing",),
            warnings=tuple(),
            metadata={
                "markdown_count": 0,
                "metadata_count": 0,
                "thresholds": resolved.__dict__,
            },
        )

    markdown_files = sorted(path for path in raw_dir.rglob("*.md") if path.is_file())
    metadata_files = sorted(
        path for path in raw_dir.rglob("*.md.meta.json") if path.is_file()
    )
    metadata_by_markdown = _metadata_by_markdown(raw_dir=raw_dir, paths=metadata_files)
    markdown_keys = {
        str(path.relative_to(raw_dir)).replace("\\", "/") for path in markdown_files
    }

    page_ids: set[str] = set()
    missing_metadata: list[str] = []
    invalid_metadata: list[str] = []
    incomplete_metadata: list[dict[str, object]] = []
    short_documents: list[dict[str, object]] = []
    heading_only: list[dict[str, object]] = []
    duplicate_candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    visibility_counts: Counter[str] = Counter()
    asset_block_count = 0
    unsupported_block_type_counts: Counter[str] = Counter()
    path_present_count = 0
    length_values: list[int] = []

    for path in markdown_files:
        rel = str(path.relative_to(raw_dir)).replace("\\", "/")
        metadata = metadata_by_markdown.get(rel)
        if metadata is None:
            sidecar = path.with_suffix(path.suffix + ".meta.json")
            if sidecar.exists():
                invalid_metadata.append(rel)
            else:
                missing_metadata.append(rel)
            metadata = {}
        page_id = _normalize_page_id(str(metadata.get("notion_page_id") or ""))
        if page_id:
            page_ids.add(page_id)
        missing_keys = [
            key
            for key in (
                "notion_page_id",
                "notion_title",
                "notion_url",
                "notion_created_time",
                "notion_last_edited_time",
                "source_type",
            )
            if not str(metadata.get(key) or "").strip()
        ]
        if missing_keys:
            incomplete_metadata.append({"file": rel, "missing_keys": missing_keys})
        if str(metadata.get("notion_page_path") or "").strip():
            path_present_count += 1
        visibility_counts[_visibility(metadata)] += 1
        asset_block_count += _to_int(metadata.get("notion_asset_count"))
        for block_type in _as_list(metadata.get("notion_unsupported_block_types")):
            unsupported_block_type_counts[block_type] += 1

        text = path.read_text(encoding="utf-8", errors="replace")
        size = len(text.encode("utf-8"))
        chars = _nonempty_character_count(text)
        length_values.append(size)
        if size < resolved.min_text_bytes or chars < resolved.min_nonempty_characters:
            short_documents.append(
                {
                    "file": rel,
                    "notion_page_id": page_id,
                    "bytes": size,
                    "nonempty_characters": chars,
                }
            )
        if _is_heading_or_url_only(text):
            heading_only.append(
                {
                    "file": rel,
                    "notion_page_id": page_id,
                    "bytes": size,
                    "nonempty_characters": chars,
                }
            )
        digest = _content_digest(text)
        duplicate_candidates[digest].append(
            {"file": rel, "notion_page_id": page_id, "bytes": size}
        )

    orphan_metadata = [
        str(path.relative_to(raw_dir)).replace("\\", "/")
        for path in metadata_files
        if str(path.relative_to(raw_dir)).replace("\\", "/")[: -len(".meta.json")]
        not in markdown_keys
    ]
    duplicate_groups = [
        {
            "content_sha256": digest,
            "notion_page_ids": [
                str(item.get("notion_page_id") or "")
                for item in items
                if str(item.get("notion_page_id") or "")
            ],
            "files": [str(item.get("file") or "") for item in items],
        }
        for digest, items in sorted(duplicate_candidates.items())
        if digest and len(items) > 1
    ]
    duplicate_document_count = sum(len(group["files"]) for group in duplicate_groups)
    short_ratio = round(len(short_documents) / len(markdown_files), 4) if markdown_files else 0.0
    heading_only_ratio = round(len(heading_only) / len(markdown_files), 4) if markdown_files else 0.0
    duplicate_ratio = round(duplicate_document_count / len(markdown_files), 4) if markdown_files else 0.0

    repository_page_ids = (
        _active_repository_page_ids(repository_dir) if repository_dir is not None else set()
    )
    repository_missing_page_ids = sorted(page_ids - repository_page_ids)
    repository_extra_page_ids = sorted(repository_page_ids - page_ids)
    repository_coverage_ratio = _coverage_ratio(page_ids, repository_page_ids)

    index_ids = {_normalize_page_id(value) for value in (index_page_ids or set())}
    index_ids.discard("")
    index_missing_page_ids = sorted(page_ids - index_ids)
    index_extra_page_ids = sorted(index_ids - page_ids)
    index_coverage_ratio = _coverage_ratio(page_ids, index_ids)

    stage_layout = _stage_layout_report(stage_dirs=stage_dirs)
    failures: list[str] = []
    warnings: list[str] = []
    if missing_metadata:
        failures.append("metadata_missing")
    if invalid_metadata:
        failures.append("metadata_invalid_json")
    if incomplete_metadata:
        failures.append("metadata_incomplete")
    if short_ratio > resolved.max_short_document_ratio:
        failures.append("short_document_ratio_too_high")
    if heading_only_ratio > resolved.max_heading_only_ratio:
        failures.append("heading_only_ratio_too_high")
    if duplicate_ratio > resolved.max_duplicate_text_ratio:
        failures.append("duplicate_text_ratio_too_high")
    if repository_dir is not None and repository_coverage_ratio < resolved.min_repository_coverage_ratio:
        failures.append("repository_coverage_too_low")
    if index_page_ids is not None and index_coverage_ratio < resolved.min_index_coverage_ratio:
        failures.append("index_coverage_too_low")
    if duplicate_groups:
        warnings.append("duplicate_content_present")
    if not markdown_files:
        warnings.append("raw_markdown_empty")
    if any(bool(item.get("legacy_flat_file_present")) for item in stage_layout.values()):
        warnings.append("legacy_flat_notion_stage_file_present")
    if asset_block_count and not unsupported_block_type_counts:
        warnings.append("notion_assets_detected_without_block_type_detail")

    metadata = {
        "markdown_count": len(markdown_files),
        "metadata_count": len(metadata_by_markdown),
        "missing_metadata_count": len(missing_metadata),
        "orphan_metadata_count": len(orphan_metadata),
        "invalid_metadata_count": len(invalid_metadata),
        "incomplete_metadata_count": len(incomplete_metadata),
        "unique_page_ids": len(page_ids),
        "repository_unique_page_ids": len(repository_page_ids),
        "repository_missing_page_ids": repository_missing_page_ids[: max(0, top_n)],
        "repository_extra_page_ids": repository_extra_page_ids[: max(0, top_n)],
        "repository_coverage_ratio": repository_coverage_ratio,
        "index_unique_page_ids": len(index_ids),
        "index_missing_page_ids": index_missing_page_ids[: max(0, top_n)],
        "index_extra_page_ids": index_extra_page_ids[: max(0, top_n)],
        "index_coverage_ratio": index_coverage_ratio,
        "short_document_count": len(short_documents),
        "short_document_ratio": short_ratio,
        "heading_only_count": len(heading_only),
        "heading_only_ratio": heading_only_ratio,
        "duplicate_document_count": duplicate_document_count,
        "duplicate_text_ratio": duplicate_ratio,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups[: max(0, top_n)],
        "length_distribution": {
            "lt_50_bytes": sum(1 for value in length_values if value < 50),
            "lt_100_bytes": sum(1 for value in length_values if value < 100),
            "lt_200_bytes": sum(1 for value in length_values if value < 200),
            "lt_500_bytes": sum(1 for value in length_values if value < 500),
            "gte_1000_bytes": sum(1 for value in length_values if value >= 1000),
        },
        "top_short_documents": sorted(
            short_documents,
            key=lambda item: (int(item["bytes"]), str(item["file"])),
        )[: max(0, top_n)],
        "heading_only_documents": heading_only[: max(0, top_n)],
        "missing_metadata_files": missing_metadata[: max(0, top_n)],
        "invalid_metadata_files": invalid_metadata[: max(0, top_n)],
        "incomplete_metadata_files": incomplete_metadata[: max(0, top_n)],
        "orphan_metadata_files": orphan_metadata[: max(0, top_n)],
        "visibility_counts": dict(sorted(visibility_counts.items())),
        "page_path_present_count": path_present_count,
        "asset_block_count": asset_block_count,
        "unsupported_block_type_counts": dict(sorted(unsupported_block_type_counts.items())),
        "stage_layout": stage_layout,
        "object_storage_raw_count": _object_storage_raw_count(object_storage_dir),
        "thresholds": resolved.__dict__,
    }
    return _report(
        raw_dir=raw_dir,
        thresholds=resolved,
        failures=tuple(failures),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def build_notion_quality_payload(
    *,
    raw_dir: Path,
    thresholds: NotionQualityThresholds,
    repository_dir: Path | None = None,
    index_page_ids: set[str] | None = None,
    stage_dirs: tuple[Path, ...] = tuple(),
    object_storage_dir: Path | None = None,
) -> dict[str, object]:
    report = audit_notion_raw_dir(
        raw_dir=raw_dir,
        thresholds=thresholds,
        repository_dir=repository_dir,
        index_page_ids=index_page_ids,
        stage_dirs=stage_dirs,
        object_storage_dir=object_storage_dir,
    )
    return report.to_payload()


def annotate_notion_raw_items(
    items: list[SourceRawItem],
    *,
    min_text_bytes: int,
    min_nonempty_characters: int,
    quarantine_low_information: bool,
) -> list[SourceRawItem]:
    duplicate_groups: dict[str, list[SourceRawItem]] = defaultdict(list)
    for item in items:
        duplicate_groups[_content_digest(item.text)].append(item)

    duplicate_metadata: dict[str, dict[str, object]] = {}
    for digest, group in duplicate_groups.items():
        if len(group) <= 1:
            continue
        group_id = stable_hash(f"notion-duplicate:{digest}")[:16]
        for item in group:
            duplicate_metadata[item.external_id] = {
                "duplicate_group_id": group_id,
                "duplicate_group_size": len(group),
            }

    annotated: list[SourceRawItem] = []
    for item in items:
        metadata = dict(item.metadata or {})
        flags = set(_as_list(metadata.get("quality_flags")))
        size = len((item.text or "").encode("utf-8"))
        chars = _nonempty_character_count(item.text)
        short = size < min_text_bytes or chars < min_nonempty_characters
        heading_only = _is_heading_or_url_only(item.text)
        if short:
            flags.add("short_text")
        if heading_only:
            flags.add("heading_or_url_only")
        if short or heading_only:
            flags.add("low_information")
        duplicate = duplicate_metadata.get(item.external_id)
        if duplicate:
            flags.add("duplicate_text")
            metadata.update(duplicate)
        if flags:
            metadata["quality_flags"] = sorted(flags)
        if quarantine_low_information and "low_information" in flags:
            metadata["index_status"] = "quarantined"
        annotated.append(replace(item, metadata=metadata))
    return annotated


def page_ids_from_chunks(chunks: list[object]) -> set[str]:
    page_ids: set[str] = set()
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        source_type = str(
            metadata.get("source_type") or metadata.get("source_kind") or ""
        ).strip().lower()
        if source_type != "notion":
            continue
        page_id = _normalize_page_id(str(metadata.get("notion_page_id") or ""))
        if page_id:
            page_ids.add(page_id)
    return page_ids


def _report(
    *,
    raw_dir: Path,
    thresholds: NotionQualityThresholds,
    failures: tuple[str, ...],
    warnings: tuple[str, ...],
    metadata: dict[str, object],
) -> NotionRawAuditReport:
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
    return NotionRawAuditReport(
        source="notion",
        raw_dir=raw_dir,
        exists=raw_dir.exists(),
        status=status,
        can_continue=can_continue,
        critical_failures=failures,
        warnings=warnings,
        metadata=metadata,
    )


def _metadata_by_markdown(*, raw_dir: Path, paths: list[Path]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for path in paths:
        key = str(path.relative_to(raw_dir)).replace("\\", "/")
        if key.endswith(".meta.json"):
            key = key[: -len(".meta.json")]
        payload = _read_json_object(path)
        if payload is not None:
            out[key] = payload
    return out


def _read_json_object(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): value for key, value in payload.items()}


def _active_repository_page_ids(repository_dir: Path) -> set[str]:
    source_rows = _read_jsonl(repository_dir / "source_items.jsonl")
    delete_rows = _read_jsonl(repository_dir / "source_deletes.jsonl")
    latest: dict[str, dict[str, object]] = {}
    for payload in source_rows:
        if str(payload.get("source_kind") or "") != "notion":
            continue
        key = str(payload.get("external_id") or "")
        if key:
            latest[key] = payload
    for payload in delete_rows:
        if str(payload.get("source_kind") or "") != "notion":
            continue
        key = str(payload.get("external_id") or "")
        if key and key in latest:
            current = dict(latest[key])
            current["index_status"] = str(payload.get("index_status") or "deleted")
            latest[key] = current
    return {
        _normalize_page_id(str(payload.get("external_id") or ""))
        for payload in latest.values()
        if str(payload.get("index_status") or "active") == "active"
    } - {""}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _stage_layout_report(*, stage_dirs: tuple[Path, ...]) -> dict[str, object]:
    report: dict[str, object] = {}
    for stage_dir in stage_dirs:
        if not stage_dir:
            continue
        notion_dir = stage_dir / "notion"
        flat_file = stage_dir / "notion.jsonl"
        report[stage_dir.name] = {
            "stage_dir": str(stage_dir),
            "source_directory_present": notion_dir.exists(),
            "source_directory_files": (
                len(list(notion_dir.glob("*.jsonl"))) if notion_dir.exists() else 0
            ),
            "legacy_flat_file_present": flat_file.exists(),
        }
    return report


def _object_storage_raw_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _coverage_ratio(expected: set[str], actual: set[str]) -> float:
    if not expected:
        return 1.0
    return round(len(expected & actual) / len(expected), 4)


def _visibility(metadata: dict[str, object]) -> str:
    scope = metadata.get("access_scope")
    if isinstance(scope, dict):
        visibility = str(scope.get("visibility") or "").strip().lower()
        if visibility:
            return visibility
    return str(metadata.get("visibility") or "").strip().lower()


def _is_heading_or_url_only(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return True
    body_lines = [line for line in lines if not re.match(r"^#{1,6}\s*", line)]
    if not body_lines:
        return True
    return all(_is_url_or_separator(line) for line in body_lines)


def _is_url_or_separator(value: str) -> bool:
    raw = value.strip()
    if raw in {"-", "---", "*"}:
        return True
    return bool(re.match(r"^https?://\S+$", raw))


def _nonempty_character_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _content_digest(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in (text or "").strip().splitlines())
    return sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_page_id(value: str) -> str:
    compact = re.sub(r"[^0-9a-fA-F]", "", str(value or ""))
    return compact.lower() if len(compact) == 32 else ""


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
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
