from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import kumc_agent.infra.loaders.google_drive_impl as drive_impl
from kumc_agent.infra.loaders.common import DRIVE_SHEET_MIME
from kumc_agent.infra.loaders.google_drive_impl import DriveFile


class _Request:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def execute(self) -> dict[str, object]:
        return self._payload


class _SheetsService:
    def __init__(self) -> None:
        self.requested_ranges: list[str] = []

    def spreadsheets(self) -> "_SheetsService":
        return self

    def values(self) -> "_SheetsService":
        return self

    def get(self, **_: object) -> _Request:
        return _Request(
            {
                "sheets": [
                    {
                        "properties": {
                            "sheetId": 11,
                            "title": "Members",
                            "index": 0,
                            "gridProperties": {"rowCount": 10, "columnCount": 3},
                        }
                    },
                    {
                        "properties": {
                            "sheetId": 12,
                            "title": "Schedule",
                            "index": 1,
                            "gridProperties": {"rowCount": 10, "columnCount": 3},
                        }
                    },
                ]
            }
        )

    def batchGet(self, **kwargs: object) -> _Request:
        self.requested_ranges = [str(item) for item in kwargs.get("ranges", [])]
        return _Request(
            {
                "valueRanges": [
                    {"values": [["名前", "メール"], ["Alice", "a@example.test"]]},
                    {"values": [["日付", "担当"], ["4/29", "Alice"]]},
                ]
            }
        )


class GoogleDriveSheetsLoadingTests(unittest.TestCase):
    def test_download_drive_markdown_writes_structured_sheet_jsonl_per_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs_dir = root / "docs"
            sheets_dir = root / "sheets"
            structured_dir = root / "sheets_structured"
            docs_dir.mkdir()
            sheets_dir.mkdir()
            structured_dir.mkdir()
            sheet_file = DriveFile(
                file_id="sheet-id",
                name="会員表",
                mime_type=DRIVE_SHEET_MIME,
                path="folder/会員表",
                modified_time="2026-04-29T00:00:00Z",
            )
            sheets_service = _SheetsService()

            with (
                patch.object(drive_impl, "_build_google_credentials", return_value=object()),
                patch.object(drive_impl, "_build_drive_service", return_value=object()),
                patch.object(drive_impl, "_build_sheets_service", return_value=sheets_service),
                patch.object(drive_impl, "_list_drive_files", return_value=[sheet_file]),
                patch.object(drive_impl, "_download_export_bytes") as export_mock,
            ):
                docs_count, sheets_count = drive_impl.download_drive_markdown(
                    drive_folder_id="folder-id",
                    docs_dir=docs_dir,
                    sheets_dir=sheets_dir,
                    sheets_structured_dir=structured_dir,
                    google_application_credentials="",
                    pdf_ocr_model_path="",
                )

            self.assertEqual(0, docs_count)
            self.assertEqual(1, sheets_count)
            export_mock.assert_not_called()
            self.assertEqual(["'Members'", "'Schedule'"], sheets_service.requested_ranges)
            csv_outputs = list(sheets_dir.glob("*.csv"))
            self.assertEqual(1, len(csv_outputs))
            self.assertIn("# sheet: Members", csv_outputs[0].read_text(encoding="utf-8"))
            structured_outputs = sorted(structured_dir.glob("*.jsonl"))
            self.assertEqual(2, len(structured_outputs))
            first_record = json.loads(structured_outputs[0].read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual("Members", first_record["metadata"]["sheet_name"])
            self.assertEqual("2-2", first_record["metadata"]["row_range"])


if __name__ == "__main__":
    unittest.main()
