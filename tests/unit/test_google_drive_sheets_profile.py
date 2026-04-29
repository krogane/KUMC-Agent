from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.infra.loaders.sheets_profile import profile_sheets_raw


class GoogleDriveSheetsProfileTests(unittest.TestCase):
    def test_profile_reports_empty_rows_markers_and_sensitive_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sheets_dir = Path(tmp) / "sheets"
            sheets_dir.mkdir()
            csv_path = sheets_dir / "sheet-id__sample.csv"
            csv_path.write_text(
                "# sheet: Main\n氏名,メール,参加\nAlice,a@example.test,yes\n,,\n",
                encoding="utf-8",
            )
            csv_path.with_suffix(".csv.meta.json").write_text(
                json.dumps(
                    {
                        "drive_file_id": "sheet-id",
                        "drive_file_name": "sample",
                        "drive_path": "folder/sample",
                        "drive_mime_type": "application/vnd.google-apps.spreadsheet",
                        "drive_modified_time": "2026-04-29T00:00:00Z",
                        "drive_url": "https://drive.google.com/file/d/sheet-id/view",
                    }
                ),
                encoding="utf-8",
            )

            profile = profile_sheets_raw(sheets_dir=sheets_dir)

        self.assertEqual(1, profile["totals"]["csv_files"])
        self.assertEqual(1, profile["totals"]["sheet_marker_files"])
        file_profile = profile["files"][0]
        self.assertTrue(file_profile["has_sheet_markers"])
        findings = file_profile["metadata"]["sensitivity_findings"]
        self.assertGreaterEqual(len(findings), 2)


if __name__ == "__main__":
    unittest.main()
