from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_SHEETS_METADATA_KEYS: tuple[str, ...] = (
    "drive_file_id",
    "drive_file_name",
    "drive_path",
    "drive_mime_type",
    "drive_modified_time",
    "drive_url",
)

_SHEET_MARKER_RE = re.compile(r"^#\s*sheet:\s*(?P<name>.+?)\s*$", re.IGNORECASE)
_HIGH_RISK_COLUMN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(mail|e-mail|email|メール)", re.IGNORECASE), "email"),
    (re.compile(r"(phone|tel|電話|携帯)", re.IGNORECASE), "phone"),
    (re.compile(r"(address|住所)", re.IGNORECASE), "address"),
    (re.compile(r"(氏名|名前|name)", re.IGNORECASE), "name"),
    (re.compile(r"(student|学籍|会員番号|member.?id)", re.IGNORECASE), "identifier"),
    (re.compile(r"(birthday|birth|生年月日|誕生日)", re.IGNORECASE), "birth_date"),
    (re.compile(r"(password|passwd|secret|token|api.?key)", re.IGNORECASE), "secret"),
    (re.compile(r"(bank|銀行|口座|振込)", re.IGNORECASE), "financial"),
)


def profile_sheets_raw(
    *,
    sheets_dir: Path,
    structured_sheets_dir: Path | None = None,
) -> dict[str, object]:
    csv_files = sorted(sheets_dir.glob("*.csv")) if sheets_dir.exists() else []
    file_profiles: list[dict[str, object]] = []
    totals = {
        "csv_files": 0,
        "metadata_files": 0,
        "csv_bytes": 0,
        "rows": 0,
        "non_empty_rows": 0,
        "empty_rows": 0,
        "non_empty_cells": 0,
        "replacement_character_files": 0,
        "metadata_missing": 0,
        "sheet_marker_files": 0,
        "sensitivity_findings": 0,
    }

    for csv_path in csv_files:
        profile = profile_sheets_csv(csv_path)
        file_profiles.append(profile)
        totals["csv_files"] += 1
        totals["csv_bytes"] += int(profile["size_bytes"])
        totals["rows"] += int(profile["row_count"])
        totals["non_empty_rows"] += int(profile["non_empty_row_count"])
        totals["empty_rows"] += int(profile["empty_row_count"])
        totals["non_empty_cells"] += int(profile["non_empty_cell_count"])
        if profile["has_replacement_character"]:
            totals["replacement_character_files"] += 1
        metadata = profile["metadata"]
        if isinstance(metadata, dict) and metadata.get("present"):
            totals["metadata_files"] += 1
        else:
            totals["metadata_missing"] += 1
        if profile["has_sheet_markers"]:
            totals["sheet_marker_files"] += 1
        sensitivity_findings = metadata.get("sensitivity_findings") if isinstance(metadata, dict) else []
        if isinstance(sensitivity_findings, list):
            totals["sensitivity_findings"] += len(sensitivity_findings)

    structured_summary = _profile_structured_sheets(structured_sheets_dir)
    return {
        "schema_version": 1,
        "sheets_dir": str(sheets_dir),
        "structured_sheets_dir": str(structured_sheets_dir) if structured_sheets_dir else "",
        "totals": totals,
        "files": file_profiles,
        "structured": structured_summary,
    }


def write_sheets_profile(
    *,
    sheets_dir: Path,
    output_path: Path,
    structured_sheets_dir: Path | None = None,
) -> dict[str, object]:
    profile = profile_sheets_raw(
        sheets_dir=sheets_dir,
        structured_sheets_dir=structured_sheets_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return profile


def profile_sheets_csv(csv_path: Path) -> dict[str, object]:
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    rows = _read_csv_rows(text)
    row_count = len(rows)
    non_empty_row_count = sum(1 for row in rows if _row_has_value(row))
    empty_row_count = row_count - non_empty_row_count
    max_column_count = max((len(row) for row in rows), default=0)
    cell_count = sum(len(row) for row in rows)
    non_empty_cell_count = sum(
        1 for row in rows for cell in row if str(cell or "").strip()
    )
    sheet_markers = [
        match.group("name").strip()
        for line in text.splitlines()
        if (match := _SHEET_MARKER_RE.match(line.strip()))
    ]
    metadata_present, metadata, metadata_error = _load_metadata(csv_path)
    missing_keys = [
        key
        for key in REQUIRED_SHEETS_METADATA_KEYS
        if not str(metadata.get(key) or "").strip()
    ]
    sensitivity_findings = _find_sensitivity_columns(rows)
    return {
        "file_name": csv_path.name,
        "size_bytes": csv_path.stat().st_size,
        "row_count": row_count,
        "non_empty_row_count": non_empty_row_count,
        "empty_row_count": empty_row_count,
        "empty_row_ratio": _ratio(empty_row_count, row_count),
        "max_column_count": max_column_count,
        "cell_count": cell_count,
        "non_empty_cell_count": non_empty_cell_count,
        "non_empty_cell_ratio": _ratio(non_empty_cell_count, cell_count),
        "has_replacement_character": "\ufffd" in text,
        "has_sheet_markers": bool(sheet_markers),
        "sheet_markers": sheet_markers[:50],
        "metadata": {
            "present": metadata_present,
            "error": metadata_error,
            "missing_keys": missing_keys,
            "drive_file_id": str(metadata.get("drive_file_id") or ""),
            "drive_file_name": str(metadata.get("drive_file_name") or ""),
            "drive_path": str(metadata.get("drive_path") or ""),
            "drive_mime_type": str(metadata.get("drive_mime_type") or ""),
            "sensitivity_findings": sensitivity_findings,
        },
    }


def _profile_structured_sheets(structured_sheets_dir: Path | None) -> dict[str, object]:
    if structured_sheets_dir is None or not structured_sheets_dir.exists():
        return {
            "present": False,
            "jsonl_files": 0,
            "metadata_files": 0,
            "records": 0,
            "drive_file_ids": [],
            "sheet_names": [],
        }
    jsonl_files = sorted(structured_sheets_dir.glob("*.jsonl"))
    drive_file_ids: set[str] = set()
    sheet_names: set[str] = set()
    records = 0
    for path in jsonl_files:
        records += _count_jsonl_records(path)
        metadata = _load_json(path.with_suffix(path.suffix + ".meta.json"))
        drive_file_id = str(metadata.get("drive_file_id") or "").strip()
        sheet_name = str(metadata.get("sheet_name") or "").strip()
        if drive_file_id:
            drive_file_ids.add(drive_file_id)
        if sheet_name:
            sheet_names.add(sheet_name)
    return {
        "present": True,
        "jsonl_files": len(jsonl_files),
        "metadata_files": len(list(structured_sheets_dir.glob("*.jsonl.meta.json"))),
        "records": records,
        "drive_file_ids": sorted(drive_file_ids),
        "sheet_names": sorted(sheet_names)[:200],
    }


def _read_csv_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    reader = csv.reader(text.splitlines())
    for row in reader:
        rows.append([str(cell or "") for cell in row])
    return rows


def _load_metadata(csv_path: Path) -> tuple[bool, dict[str, Any], str]:
    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    if not meta_path.exists():
        return False, {}, ""
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return True, {}, str(exc)
    if not isinstance(data, dict):
        return True, {}, "metadata_not_object"
    return True, data, ""


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _count_jsonl_records(path: Path) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                if line.strip():
                    count += 1
    except OSError:
        return 0
    return count


def _find_sensitivity_columns(rows: list[list[str]]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for row_index, row in enumerate(rows[:10], start=1):
        for column_index, value in enumerate(row, start=1):
            header = str(value or "").strip()
            if not header or len(header) > 80:
                continue
            for pattern, category in _HIGH_RISK_COLUMN_PATTERNS:
                if pattern.search(header):
                    findings.append(
                        {
                            "row": row_index,
                            "column": column_index,
                            "header": header,
                            "category": category,
                        }
                    )
                    break
    return findings[:50]


def _row_has_value(row: list[str]) -> bool:
    return any(str(cell or "").strip() for cell in row)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)
