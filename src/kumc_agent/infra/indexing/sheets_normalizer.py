from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

from kumc_agent.infra.indexing.chunks import Chunk, load_chunks, write_chunks


_SHEET_MARKER_RE = re.compile(r"^#\s*sheet:\s*(?P<name>.+?)\s*$", re.IGNORECASE)
_DATE_LIKE_RE = re.compile(r"(\d{1,2}[/-]\d{1,2}|\d{4}[/-]\d{1,2}|月|火|水|木|金|土|日)")
_TIME_LIKE_RE = re.compile(r"(\d{1,2}:\d{2}|\d{1,2}時)")
_FORM_HEADER_RE = re.compile(r"(タイムスタンプ|timestamp|メール|email|回答|response)", re.IGNORECASE)


def normalize_csv_file(
    path: Path,
    *,
    base_metadata: dict[str, object] | None = None,
) -> list[Chunk]:
    return normalize_csv_text(
        path.read_text(encoding="utf-8", errors="replace"),
        base_metadata=base_metadata or {},
        source_file_name=path.name,
    )


def normalize_csv_text(
    csv_text: str,
    *,
    base_metadata: dict[str, object] | None = None,
    source_file_name: str = "",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sheet_index, (sheet_name, rows) in enumerate(_parse_csv_sections(csv_text)):
        chunks.extend(
            normalize_worksheet_rows(
                rows,
                base_metadata=base_metadata or {},
                sheet_name=sheet_name,
                sheet_index=sheet_index,
                sheet_id=str(sheet_index),
                source_file_name=source_file_name,
            )
        )
    if chunks:
        return chunks
    return _fallback_chunks(
        csv_text=csv_text,
        base_metadata=base_metadata or {},
        source_file_name=source_file_name,
        sheet_name="",
        sheet_index=0,
    )


def normalize_worksheet_rows(
    rows: list[list[str]],
    *,
    base_metadata: dict[str, object] | None = None,
    sheet_name: str = "",
    sheet_index: int = 0,
    sheet_id: str | int | None = None,
    source_file_name: str = "",
    grid_row_count: int | None = None,
    grid_column_count: int | None = None,
) -> list[Chunk]:
    metadata_base = _base_metadata(
        base_metadata or {},
        source_file_name=source_file_name,
        sheet_name=sheet_name,
        sheet_index=sheet_index,
        sheet_id=sheet_id,
    )
    trimmed = _trim_table(rows)
    if trimmed is None:
        return _fallback_chunks(
            csv_text="",
            base_metadata=metadata_base,
            source_file_name=source_file_name,
            sheet_name=sheet_name,
            sheet_index=sheet_index,
        )

    table_rows, row_start, row_end, col_start, col_end = trimmed
    header_index = _detect_header_row(table_rows)
    if header_index is None:
        return _fallback_chunks(
            csv_text=_rows_to_csv_text(table_rows),
            base_metadata=metadata_base,
            source_file_name=source_file_name,
            sheet_name=sheet_name,
            sheet_index=sheet_index,
        )

    headers = _build_headers(table_rows[header_index], col_start=col_start)
    table_kind = _detect_table_kind(headers=headers, rows=table_rows[header_index + 1 :])
    table_profile = _table_profile(
        rows=table_rows,
        header_index=header_index,
        row_start=row_start,
        row_end=row_end,
        col_start=col_start,
        col_end=col_end,
        grid_row_count=grid_row_count,
        grid_column_count=grid_column_count,
        table_kind=table_kind,
    )

    chunks: list[Chunk] = []
    for local_row_index, row in enumerate(table_rows[header_index + 1 :], start=header_index + 1):
        if not _row_has_value(row):
            continue
        row_number = row_start + local_row_index
        text = _record_text(
            row=row,
            headers=headers,
            metadata=metadata_base,
            sheet_name=sheet_name,
            row_number=row_number,
            row_range=f"{row_number}-{row_number}",
            column_range=f"{_column_name(col_start)}-{_column_name(col_end)}",
            table_kind=table_kind,
        )
        if not text.strip():
            continue
        metadata = dict(metadata_base)
        metadata.update(
            {
                "row_range": f"{row_number}-{row_number}",
                "column_range": f"{_column_name(col_start)}-{_column_name(col_end)}",
                "table_kind": table_kind,
                "table_profile": table_profile,
                "normalization_status": "normalized",
            }
        )
        chunks.append(Chunk(text=text, metadata=metadata))

    if chunks:
        return chunks
    fallback_metadata = dict(metadata_base)
    fallback_metadata["table_profile"] = table_profile
    fallback_metadata["table_kind"] = table_kind
    return _fallback_chunks(
        csv_text=_rows_to_csv_text(table_rows),
        base_metadata=fallback_metadata,
        source_file_name=source_file_name,
        sheet_name=sheet_name,
        sheet_index=sheet_index,
    )


def write_normalized_worksheet_jsonl(
    *,
    output_path: Path,
    rows: list[list[str]],
    base_metadata: dict[str, object],
    sheet_name: str,
    sheet_index: int,
    sheet_id: str | int | None,
    source_file_name: str,
    grid_row_count: int | None = None,
    grid_column_count: int | None = None,
) -> list[Chunk]:
    chunks = normalize_worksheet_rows(
        rows,
        base_metadata=base_metadata,
        sheet_name=sheet_name,
        sheet_index=sheet_index,
        sheet_id=sheet_id,
        source_file_name=source_file_name,
        grid_row_count=grid_row_count,
        grid_column_count=grid_column_count,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_chunks(output_path, chunks)
    return chunks


def load_structured_sheet_chunks(path: Path) -> list[Chunk]:
    return load_chunks(path)


def _parse_csv_sections(csv_text: str) -> list[tuple[str, list[list[str]]]]:
    sections: list[tuple[str, list[str]]] = []
    current_name = ""
    current_lines: list[str] = []
    for line in csv_text.splitlines():
        match = _SHEET_MARKER_RE.match(line.strip())
        if match:
            if current_lines or current_name:
                sections.append((current_name, current_lines))
            current_name = match.group("name").strip()
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines or current_name:
        sections.append((current_name, current_lines))
    if not sections:
        sections.append(("", csv_text.splitlines()))

    parsed: list[tuple[str, list[list[str]]]] = []
    for sheet_name, lines in sections:
        reader = csv.reader(io.StringIO("\n".join(lines)))
        parsed.append((sheet_name, [[str(cell or "") for cell in row] for row in reader]))
    return parsed


def _trim_table(rows: list[list[str]]) -> tuple[list[list[str]], int, int, int, int] | None:
    positions: list[tuple[int, int]] = []
    for row_index, row in enumerate(rows):
        for col_index, cell in enumerate(row):
            if str(cell or "").strip():
                positions.append((row_index, col_index))
    if not positions:
        return None
    min_row = min(pos[0] for pos in positions)
    max_row = max(pos[0] for pos in positions)
    min_col = min(pos[1] for pos in positions)
    max_col = max(pos[1] for pos in positions)
    trimmed: list[list[str]] = []
    for row in rows[min_row : max_row + 1]:
        trimmed.append([_clean_cell(row[col] if col < len(row) else "") for col in range(min_col, max_col + 1)])
    return trimmed, min_row + 1, max_row + 1, min_col + 1, max_col + 1


def _detect_header_row(rows: list[list[str]]) -> int | None:
    best_index: int | None = None
    best_score = 0
    for index, row in enumerate(rows[:10]):
        values = [_clean_cell(cell) for cell in row]
        non_empty = [value for value in values if value]
        if not non_empty:
            continue
        unique = len(set(non_empty))
        alphaish = sum(1 for value in non_empty if re.search(r"[A-Za-zぁ-んァ-ン一-龥]", value))
        score = len(non_empty) * 3 + unique + alphaish - index
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _build_headers(header_row: list[str], *, col_start: int) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}
    inherited = ""
    for offset, raw_header in enumerate(header_row):
        header = _clean_cell(raw_header)
        if header:
            inherited = header
        elif inherited:
            header = f"{inherited} 同上"
        else:
            header = f"column_{_column_name(col_start + offset)}"
        seen[header] = seen.get(header, 0) + 1
        if seen[header] > 1:
            header = f"{header} {seen[header]}"
        headers.append(header)
    return headers


def _detect_table_kind(*, headers: list[str], rows: list[list[str]]) -> str:
    joined_headers = "\n".join(headers)
    if _FORM_HEADER_RE.search(joined_headers):
        return "form_response"
    date_like_headers = sum(1 for header in headers if _DATE_LIKE_RE.search(header))
    first_column_time_like = sum(
        1 for row in rows[:20]
        if row and _TIME_LIKE_RE.search(_clean_cell(row[0]))
    )
    if len(headers) >= 6 and (date_like_headers >= 3 or first_column_time_like >= 3):
        return "schedule_grid"
    return "table"


def _table_profile(
    *,
    rows: list[list[str]],
    header_index: int,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    grid_row_count: int | None,
    grid_column_count: int | None,
    table_kind: str,
) -> dict[str, object]:
    total_rows = len(rows)
    total_cells = sum(len(row) for row in rows)
    non_empty_rows = sum(1 for row in rows if _row_has_value(row))
    non_empty_cells = sum(1 for row in rows for cell in row if _clean_cell(cell))
    return {
        "table_kind": table_kind,
        "row_count": total_rows,
        "column_count": max((len(row) for row in rows), default=0),
        "non_empty_row_count": non_empty_rows,
        "empty_row_ratio": _ratio(total_rows - non_empty_rows, total_rows),
        "non_empty_cell_count": non_empty_cells,
        "non_empty_cell_ratio": _ratio(non_empty_cells, total_cells),
        "header_row": row_start + header_index,
        "row_start": row_start,
        "row_end": row_end,
        "column_start": _column_name(col_start),
        "column_end": _column_name(col_end),
        "grid_row_count": grid_row_count,
        "grid_column_count": grid_column_count,
    }


def _record_text(
    *,
    row: list[str],
    headers: list[str],
    metadata: dict[str, object],
    sheet_name: str,
    row_number: int,
    row_range: str,
    column_range: str,
    table_kind: str,
) -> str:
    drive_path = str(metadata.get("drive_file_path") or metadata.get("drive_path") or "").strip()
    drive_name = str(metadata.get("drive_file_name") or metadata.get("drive_name") or "").strip()
    lines = [
        f"Drive: {drive_path or drive_name or str(metadata.get('source_file_name') or '').strip()}",
        f"Sheet: {sheet_name or str(metadata.get('sheet_name') or '').strip() or 'unknown'}",
        f"Table: {table_kind}, rows {row_range}, columns {column_range}",
        f"Row {row_number}:",
    ]
    for index, header in enumerate(headers):
        value = _clean_cell(row[index] if index < len(row) else "")
        if not value:
            continue
        lines.append(f"- {header}: {value}")
    return "\n".join(lines)


def _fallback_chunks(
    *,
    csv_text: str,
    base_metadata: dict[str, object],
    source_file_name: str,
    sheet_name: str,
    sheet_index: int,
) -> list[Chunk]:
    text = csv_text.strip()
    if not text:
        return []
    metadata = _base_metadata(
        base_metadata,
        source_file_name=source_file_name,
        sheet_name=sheet_name,
        sheet_index=sheet_index,
        sheet_id=base_metadata.get("sheet_id"),
    )
    metadata.update(
        {
            "row_range": "",
            "column_range": "",
            "table_kind": str(base_metadata.get("table_kind") or "csv_fallback"),
            "normalization_status": "fallback",
        }
    )
    if isinstance(base_metadata.get("table_profile"), dict):
        metadata["table_profile"] = base_metadata["table_profile"]
    return [Chunk(text=text, metadata=metadata)]


def _base_metadata(
    metadata: dict[str, object],
    *,
    source_file_name: str,
    sheet_name: str,
    sheet_index: int,
    sheet_id: str | int | None,
) -> dict[str, object]:
    base = dict(metadata)
    if source_file_name:
        base["source_file_name"] = source_file_name
    base["source_type"] = "sheets"
    if not str(base.get("drive_file_path") or "").strip():
        base["drive_file_path"] = str(base.get("drive_path") or "")
    if not str(base.get("drive_file_name") or "").strip():
        base["drive_file_name"] = str(base.get("drive_name") or "")
    modified_time = str(base.get("drive_modified_time") or "").strip()
    if modified_time and not str(base.get("updated_at") or "").strip():
        base["updated_at"] = modified_time
    if modified_time and not str(base.get("source_date") or "").strip():
        base["source_date"] = modified_time[:10]
    base["sheet_name"] = sheet_name or str(base.get("sheet_name") or "")
    base["sheet_index"] = sheet_index
    if sheet_id is not None:
        base["sheet_id"] = str(sheet_id)
    return base


def _rows_to_csv_text(rows: list[list[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().strip()


def _clean_cell(value: Any) -> str:
    return str(value or "").replace("\r", " ").strip()


def _row_has_value(row: list[str]) -> bool:
    return any(_clean_cell(cell) for cell in row)


def _column_name(index: int) -> str:
    if index <= 0:
        return ""
    value = index
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)
