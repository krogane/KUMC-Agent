from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from urllib.parse import urlparse


@dataclass(frozen=True)
class MinecraftWikiQualityThresholds:
    enabled: bool = True
    min_article_characters: int = 500
    max_redirect_ratio: float = 0.2
    min_indexable_pages: int = 1
    min_chunk_count: int = 1
    required_canonical_host: str = "ja.minecraft.wiki"
    policy: str = "warn"


@dataclass(frozen=True)
class MinecraftWikiRawAuditReport:
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
            "# Minecraft Wiki Raw品質監査",
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
            f"- 転送ページ: {meta.get('redirect_count', 0)}",
            f"- 転送率: {meta.get('redirect_ratio', 0.0)}",
            f"- index可能ページ: {meta.get('indexable_page_count', 0)}",
            f"- chunk数: {meta.get('chunk_count', 0)}",
            f"- metadata欠落: {meta.get('missing_metadata_count', 0)}",
            f"- revision id欠落: {meta.get('missing_revision_count', 0)}",
            "",
            "## 本文長分布",
            "",
        ]
        length_distribution = meta.get("length_distribution")
        if isinstance(length_distribution, dict):
            for key, value in length_distribution.items():
                lines.append(f"- {key}: {value}")
        lines.extend(["", "## canonical host", ""])
        hosts = meta.get("canonical_hosts")
        if isinstance(hosts, dict):
            for key, value in hosts.items():
                lines.append(f"- `{key or '(empty)'}`: {value}")
        lines.extend(["", "## 短文ページ", ""])
        short_pages = meta.get("top_short_pages")
        if isinstance(short_pages, list):
            for item in short_pages:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- `{item.get('file', '')}`: {item.get('bytes', 0)} bytes"
                )
        if self.critical_failures:
            lines.extend(["", "## critical_failures", ""])
            lines.extend(f"- {item}" for item in self.critical_failures)
        if self.warnings:
            lines.extend(["", "## warnings", ""])
            lines.extend(f"- {item}" for item in self.warnings)
        return "\n".join(lines).rstrip() + "\n"


def audit_minecraft_wiki_raw_dir(
    *,
    raw_dir: Path,
    thresholds: MinecraftWikiQualityThresholds | None = None,
    chunk_count: int | None = None,
    top_n: int = 20,
) -> MinecraftWikiRawAuditReport:
    resolved_thresholds = thresholds or MinecraftWikiQualityThresholds()
    if not raw_dir.exists():
        return _report(
            raw_dir=raw_dir,
            thresholds=resolved_thresholds,
            failures=("raw_dir_missing",),
            warnings=tuple(),
            metadata={
                "markdown_count": 0,
                "metadata_count": 0,
                "redirect_count": 0,
                "redirect_ratio": 0.0,
                "indexable_page_count": 0,
                "chunk_count": int(chunk_count or 0),
                "thresholds": resolved_thresholds.__dict__,
            },
        )

    markdown_files = sorted(path for path in raw_dir.glob("*.md") if path.is_file())
    metadata_count = 0
    missing_metadata: list[str] = []
    missing_revision: list[str] = []
    redirect_pages: list[str] = []
    canonical_hosts: dict[str, int] = {}
    updated_at_values: list[str] = []
    category_counts: dict[str, int] = {}
    length_values: list[int] = []
    short_pages: list[dict[str, object]] = []
    indexable_page_count = 0

    for path in markdown_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        size = len(text.encode("utf-8"))
        length_values.append(size)
        metadata = _read_sidecar(path)
        if metadata:
            metadata_count += 1
        else:
            missing_metadata.append(path.name)
        if not str(metadata.get("minecraft_wiki_revision_id") or "").strip():
            missing_revision.append(path.name)
        canonical_url = str(metadata.get("canonical_url") or "").strip()
        host = (urlparse(canonical_url).hostname or "").strip().lower()
        canonical_hosts[host] = canonical_hosts.get(host, 0) + 1
        updated_at = str(metadata.get("updated_at") or "").strip()
        if updated_at:
            updated_at_values.append(updated_at)
        for category in _categories(metadata):
            category_counts[category] = category_counts.get(category, 0) + 1
        if _is_redirect_only(text) or bool(metadata.get("minecraft_wiki_is_redirect") is True and size < resolved_thresholds.min_article_characters):
            redirect_pages.append(path.name)
        if size >= resolved_thresholds.min_article_characters and not _is_redirect_only(text):
            indexable_page_count += 1
        short_pages.append({"file": path.name, "bytes": size})

    redirect_ratio = (
        round(len(redirect_pages) / len(markdown_files), 4)
        if markdown_files
        else 0.0
    )
    length_distribution = {
        "lt_100_bytes": sum(1 for value in length_values if value < 100),
        "lt_500_bytes": sum(1 for value in length_values if value < 500),
        "gte_1kb": sum(1 for value in length_values if value >= 1024),
        "gte_3kb": sum(1 for value in length_values if value >= 3 * 1024),
        "gte_10kb": sum(1 for value in length_values if value >= 10 * 1024),
    }
    failures: list[str] = []
    warnings: list[str] = []
    if markdown_files and redirect_ratio > resolved_thresholds.max_redirect_ratio:
        failures.append("redirect_ratio_too_high")
    if indexable_page_count < resolved_thresholds.min_indexable_pages:
        failures.append("indexable_pages_too_few")
    if chunk_count is not None and int(chunk_count) < resolved_thresholds.min_chunk_count:
        failures.append("chunk_count_too_few")
    required_host = resolved_thresholds.required_canonical_host
    if required_host:
        invalid_hosts = sorted(
            host for host in canonical_hosts if host and host != required_host
        )
        if invalid_hosts:
            failures.append("invalid_canonical_host")
    if missing_metadata:
        failures.append("metadata_missing")
    if missing_revision:
        failures.append("revision_id_missing")
    if not markdown_files:
        warnings.append("raw_markdown_empty")

    metadata = {
        "markdown_count": len(markdown_files),
        "metadata_count": metadata_count,
        "missing_metadata_count": len(missing_metadata),
        "missing_revision_count": len(missing_revision),
        "redirect_count": len(redirect_pages),
        "redirect_ratio": redirect_ratio,
        "indexable_page_count": indexable_page_count,
        "chunk_count": int(chunk_count or 0),
        "length_distribution": length_distribution,
        "canonical_hosts": dict(sorted(canonical_hosts.items())),
        "updated_at_min": min(updated_at_values) if updated_at_values else "",
        "updated_at_max": max(updated_at_values) if updated_at_values else "",
        "category_counts": dict(sorted(category_counts.items())),
        "top_short_pages": sorted(short_pages, key=lambda item: int(item["bytes"]))[
            : max(0, top_n)
        ],
        "redirect_pages": redirect_pages[: max(0, top_n)],
        "thresholds": resolved_thresholds.__dict__,
    }
    return _report(
        raw_dir=raw_dir,
        thresholds=resolved_thresholds,
        failures=tuple(failures),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def _report(
    *,
    raw_dir: Path,
    thresholds: MinecraftWikiQualityThresholds,
    failures: tuple[str, ...],
    warnings: tuple[str, ...],
    metadata: dict[str, object],
) -> MinecraftWikiRawAuditReport:
    if not thresholds.enabled:
        status = "disabled"
        can_continue = True
    elif failures and thresholds.policy == "fail":
        status = "failed"
        can_continue = False
    elif failures or warnings:
        status = "warning"
        can_continue = True
    else:
        status = "passed"
        can_continue = True
    return MinecraftWikiRawAuditReport(
        source="minecraft_wiki",
        raw_dir=raw_dir,
        exists=raw_dir.exists(),
        status=status,
        can_continue=can_continue,
        critical_failures=failures,
        warnings=warnings,
        metadata=metadata,
    )


def _read_sidecar(path: Path) -> dict[str, object]:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not sidecar.exists():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}


def _categories(metadata: dict[str, object]) -> list[str]:
    raw = metadata.get("minecraft_wiki_categories") or metadata.get("categories")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    category = str(metadata.get("minecraft_wiki_category") or "").strip()
    return [category] if category else []


def _is_redirect_only(text: str) -> bool:
    return bool(
        re.match(
            r"(?is)^\s*#(?:転送|redirect)\s*:?\s*\[\[[^\]]+\]\]\s*$",
            text or "",
        )
    )
