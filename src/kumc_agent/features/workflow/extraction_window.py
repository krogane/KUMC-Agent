from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from kumc_agent.domain.models.chunk import Chunk

TIMESTAMP_KEYS = (
    "updated_at",
    "source_updated_at",
    "created_at",
    "source_created_at",
    "published_at",
    "message_timestamp",
    "hatenablog_updated_at",
    "hatenablog_created_at",
    "indexed_at",
    "ingested_at",
)


@dataclass(frozen=True)
class ExtractionWindow:
    lookback_days: int
    extraction_at: datetime
    since: datetime

    def as_metadata(self) -> dict[str, object]:
        return {
            "lookback_days": self.lookback_days,
            "extraction_since": self.since.isoformat(),
            "extraction_at": self.extraction_at.isoformat(),
        }


@dataclass(frozen=True)
class ChunkWindowSelection:
    chunks: list[Chunk]
    window: ExtractionWindow
    excluded_older_chunks: int = 0
    excluded_missing_timestamp_chunks: int = 0

    def as_metadata(self, *, selected_chunks: int | None = None) -> dict[str, object]:
        selected_count = len(self.chunks) if selected_chunks is None else selected_chunks
        return {
            **self.window.as_metadata(),
            "selected_chunks": selected_count,
            "excluded_older_chunks": self.excluded_older_chunks,
            "excluded_missing_timestamp_chunks": self.excluded_missing_timestamp_chunks,
        }


def normalize_lookback_days(value: object) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def build_extraction_window(
    *,
    lookback_days: object,
    extraction_at: datetime | None = None,
) -> ExtractionWindow:
    current = _aware(extraction_at or datetime.now(UTC))
    days = normalize_lookback_days(lookback_days)
    return ExtractionWindow(
        lookback_days=days,
        extraction_at=current,
        since=current - timedelta(days=days),
    )


def select_recent_chunks(
    chunks: list[Chunk],
    *,
    lookback_days: object,
    extraction_at: datetime | None = None,
) -> ChunkWindowSelection:
    window = build_extraction_window(
        lookback_days=lookback_days,
        extraction_at=extraction_at,
    )
    selected: list[Chunk] = []
    older = 0
    missing_timestamp = 0
    for chunk in chunks:
        changed_at = changed_at_from_metadata(dict(chunk.metadata or {}))
        if changed_at is None:
            missing_timestamp += 1
            continue
        if changed_at < window.since:
            older += 1
            continue
        selected.append(chunk)
    return ChunkWindowSelection(
        chunks=selected,
        window=window,
        excluded_older_chunks=older,
        excluded_missing_timestamp_chunks=missing_timestamp,
    )


def changed_at_from_metadata(metadata: dict[str, Any]) -> datetime | None:
    for key in TIMESTAMP_KEYS:
        value = metadata.get(key)
        if value:
            parsed = parse_datetime(value)
            if parsed is not None:
                return parsed
    return None


def parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _aware(value)
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return _aware(datetime.fromisoformat(raw.replace("Z", "+00:00")))
    except ValueError:
        return None


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
