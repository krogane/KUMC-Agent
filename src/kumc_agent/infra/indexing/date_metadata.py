from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo

SOURCE_DATE_UNKNOWN = "不明"
_JST = ZoneInfo("Asia/Tokyo")

_YMD_COMPACT_RE = re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})$")
_YMD_SLASH_RE = re.compile(r"^(?P<y>\d{4})/(?P<m>\d{2})/(?P<d>\d{2})$")
_YMD_DASH_RE = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})$")
_VC_PARENT_RE = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})_\d+$")


def source_date_to_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    normalized = normalize_source_date(value)
    if normalized is None:
        return None
    try:
        return datetime.strptime(normalized, "%Y/%m/%d").date()
    except ValueError:
        return None


def normalize_source_date(value: str | None) -> str | None:
    raw = (value or "").strip()
    if not raw or raw == SOURCE_DATE_UNKNOWN:
        return None

    for pattern in (_YMD_SLASH_RE, _YMD_DASH_RE, _YMD_COMPACT_RE):
        match = pattern.match(raw)
        if match is None:
            continue
        return _safe_ymd(
            int(match.group("y")),
            int(match.group("m")),
            int(match.group("d")),
        )

    iso_candidate = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(_JST).strftime("%Y/%m/%d")


def source_date_from_discord_timestamp(message_timestamp: str | None) -> str:
    normalized = normalize_source_date(message_timestamp)
    return normalized or SOURCE_DATE_UNKNOWN


def source_date_from_hatenablog_created_at(hatenablog_created_at: str | None) -> str:
    normalized = normalize_source_date(hatenablog_created_at)
    return normalized or SOURCE_DATE_UNKNOWN


def source_date_from_crafters_colony_published_at(value: str | None) -> str:
    normalized = normalize_source_date(value)
    return normalized or SOURCE_DATE_UNKNOWN


def source_date_from_crafters_colony_timestamp(value: str | None) -> str:
    normalized = normalize_source_date(value)
    return normalized or SOURCE_DATE_UNKNOWN


def source_date_from_vc_path(path: Path | str | None) -> str:
    if path is None:
        return SOURCE_DATE_UNKNOWN
    parent_name = Path(path).parent.name
    match = _VC_PARENT_RE.match(parent_name)
    if match is None:
        return SOURCE_DATE_UNKNOWN
    normalized = _safe_ymd(
        int(match.group("y")),
        int(match.group("m")),
        int(match.group("d")),
    )
    return normalized or SOURCE_DATE_UNKNOWN


def source_date_from_drive(
    *,
    drive_file_name: str | None,
    drive_file_path: str | None,
) -> str:
    file_name = (drive_file_name or "").strip()
    if file_name:
        match = re.match(r"^(?P<ymd>\d{8})", file_name)
        if match is not None:
            normalized = normalize_source_date(match.group("ymd"))
            if normalized is not None:
                return normalized

    path_value = (drive_file_path or "").strip()
    if path_value:
        parts = [part.strip() for part in path_value.split("/") if part.strip()]
        for idx in range(len(parts) - 1):
            year_part = parts[idx]
            month_part = parts[idx + 1]
            if not re.fullmatch(r"\d{4}", year_part):
                continue
            month_match = re.match(r"^(?P<m>\d{2}):\s*", month_part)
            if month_match is None:
                continue
            normalized = _safe_ymd(
                int(year_part),
                int(month_match.group("m")),
                1,
            )
            if normalized is not None:
                return normalized

    return SOURCE_DATE_UNKNOWN


def infer_source_date(
    *,
    metadata: Mapping[str, object] | None = None,
    source_path: Path | None = None,
) -> str:
    meta = metadata or {}

    existing = normalize_source_date(str(meta.get("source_date") or ""))
    if existing is not None:
        return existing

    source_type = str(meta.get("source_type") or "").strip().lower()
    if source_type in {"messages", "discord_message", "x_posts"}:
        first_message_date = normalize_source_date(str(meta.get("first_message_date") or ""))
        if first_message_date is not None:
            return first_message_date
        return source_date_from_discord_timestamp(str(meta.get("message_timestamp") or ""))

    if source_type == "hatenablog":
        return source_date_from_hatenablog_created_at(
            str(meta.get("hatenablog_created_at") or "")
        )

    if source_type == "crafters_colony":
        return source_date_from_crafters_colony_timestamp(
            str(
                meta.get("crafters_colony_updated_at")
                or meta.get("crafters_colony_published_at")
                or ""
            )
        )

    if source_type == "notion":
        notion_last_edited = normalize_source_date(
            str(meta.get("notion_last_edited_time") or "")
        )
        if notion_last_edited is not None:
            return notion_last_edited
        notion_created = normalize_source_date(
            str(meta.get("notion_created_time") or "")
        )
        if notion_created is not None:
            return notion_created
        updated_at = normalize_source_date(str(meta.get("updated_at") or ""))
        if updated_at is not None:
            return updated_at
        return SOURCE_DATE_UNKNOWN

    if source_type == "vc_transcript":
        meeting_date = normalize_source_date(str(meta.get("meeting_date") or ""))
        if meeting_date is not None:
            return meeting_date
        if source_path is not None:
            return source_date_from_vc_path(source_path)
        return SOURCE_DATE_UNKNOWN

    if source_type in {"docs", "sheets"}:
        return source_date_from_drive(
            drive_file_name=str(meta.get("drive_file_name") or ""),
            drive_file_path=str(meta.get("drive_file_path") or ""),
        )

    return SOURCE_DATE_UNKNOWN


def _safe_ymd(year: int, month: int, day: int) -> str | None:
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    return parsed.strftime("%Y/%m/%d")
