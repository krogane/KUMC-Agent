from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.infra.indexing.sheets_normalizer import normalize_worksheet_rows


class SheetsNormalizerTests(unittest.TestCase):
    def test_normalize_worksheet_rows_keeps_sheet_and_range_metadata(self) -> None:
        rows = [
            ["", "", ""],
            ["名前", "担当", "4/29"],
            ["Alice", "受付", "10:00"],
            ["", "", ""],
        ]
        chunks = normalize_worksheet_rows(
            rows,
            base_metadata={
                "drive_file_id": "file-id",
                "drive_file_name": "当番表",
                "drive_path": "folder/当番表",
                "drive_modified_time": "2026-04-29T00:00:00Z",
            },
            sheet_name="Main",
            sheet_index=0,
            sheet_id=123,
            source_file_name="file.jsonl",
        )

        self.assertEqual(1, len(chunks))
        self.assertIn("Sheet: Main", chunks[0].text)
        self.assertIn("- 名前: Alice", chunks[0].text)
        self.assertEqual("Main", chunks[0].metadata["sheet_name"])
        self.assertEqual("3-3", chunks[0].metadata["row_range"])
        self.assertEqual("A-C", chunks[0].metadata["column_range"])


if __name__ == "__main__":
    unittest.main()
