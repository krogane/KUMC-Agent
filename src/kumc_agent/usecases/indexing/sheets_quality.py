from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.infra.loaders.common import DRIVE_SHEET_MIME
from kumc_agent.infra.loaders.sheets_profile import profile_sheets_raw


@dataclass(frozen=True)
class SheetsQualityResult:
    passed: bool
    warnings: tuple[str, ...]
    metadata: dict[str, object]


def build_sheets_quality_payload(
    *,
    sheets_dir: Path,
    structured_sheets_dir: Path,
    fail_fast: bool = False,
    max_empty_row_ratio: float = 0.8,
    min_non_empty_cells: int = 1,
) -> dict[str, object]:
    result = check_sheets_quality(
        sheets_dir=sheets_dir,
        structured_sheets_dir=structured_sheets_dir,
        fail_fast=fail_fast,
        max_empty_row_ratio=max_empty_row_ratio,
        min_non_empty_cells=min_non_empty_cells,
    )
    return {
        "can_continue": result.passed or not fail_fast,
        "metadata": result.metadata
        | {
            "passed": result.passed,
            "warnings": list(result.warnings),
            "fail_fast": fail_fast,
        },
    }


def check_sheets_quality(
    *,
    sheets_dir: Path,
    structured_sheets_dir: Path,
    fail_fast: bool = False,
    max_empty_row_ratio: float = 0.8,
    min_non_empty_cells: int = 1,
) -> SheetsQualityResult:
    profile = profile_sheets_raw(
        sheets_dir=sheets_dir,
        structured_sheets_dir=structured_sheets_dir,
    )
    warnings: list[str] = []
    structured_ids = set(_structured_drive_file_ids(profile))
    files = profile.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            file_name = str(item.get("file_name") or "")
            metadata = item.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            drive_file_id = str(metadata.get("drive_file_id") or "").strip()
            mime_type = str(metadata.get("drive_mime_type") or "").strip()
            if not metadata.get("present"):
                warnings.append(f"metadata_missing:{file_name}")
            missing_keys = metadata.get("missing_keys")
            if isinstance(missing_keys, list) and missing_keys:
                warnings.append(f"metadata_incomplete:{file_name}")
            if float(item.get("empty_row_ratio") or 0.0) > max_empty_row_ratio:
                warnings.append(f"empty_row_ratio_high:{file_name}")
            if int(item.get("non_empty_cell_count") or 0) < min_non_empty_cells:
                warnings.append(f"non_empty_cells_too_low:{file_name}")
            if mime_type == DRIVE_SHEET_MIME and drive_file_id and drive_file_id not in structured_ids:
                warnings.append(f"google_sheets_tab_count_unknown:{file_name}")
            sensitivity_findings = metadata.get("sensitivity_findings")
            if isinstance(sensitivity_findings, list) and sensitivity_findings:
                warnings.append(f"sensitivity_findings_unmasked:{file_name}")

    metadata = {
        "status": "succeeded",
        "sheets_dir": str(sheets_dir),
        "structured_sheets_dir": str(structured_sheets_dir),
        "max_empty_row_ratio": max_empty_row_ratio,
        "min_non_empty_cells": min_non_empty_cells,
        "profile": profile,
    }
    return SheetsQualityResult(
        passed=not warnings,
        warnings=tuple(warnings),
        metadata=metadata,
    )


def _structured_drive_file_ids(profile: dict[str, object]) -> list[str]:
    structured = profile.get("structured")
    if not isinstance(structured, dict):
        return []
    drive_file_ids = structured.get("drive_file_ids")
    if not isinstance(drive_file_ids, list):
        return []
    return [str(item) for item in drive_file_ids if str(item).strip()]
