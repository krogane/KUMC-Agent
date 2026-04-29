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

from kumc_agent.usecases.ingestion.google_drive_docs_audit import (
    GoogleDriveDocsQualityThresholds,
    audit_google_drive_docs_raw_dir,
)


class GoogleDriveDocsAuditTests(unittest.TestCase):
    def test_audit_reports_metadata_short_docs_duplicates_and_normalized_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "docs"
            normalized_dir = root / "docs_normalized"
            raw_dir.mkdir()
            normalized_dir.mkdir()
            for file_id, name in (("file-1", "a.md"), ("file-2", "b.md")):
                path = raw_dir / name
                path.write_text("40", encoding="utf-8")
                path.with_suffix(".md.meta.json").write_text(
                    json.dumps(
                        {
                            "drive_file_id": file_id,
                            "drive_file_name": name,
                            "drive_path": f"folder/{name}",
                            "drive_mime_type": "application/pdf",
                            "drive_modified_time": "2026-04-29T00:00:00Z",
                            "drive_url": f"https://drive.google.com/file/d/{file_id}/view",
                            "content_sha256": "same-hash",
                            "source_date": "2026/04/29",
                            "quality_flags": ["too_short"],
                        }
                    ),
                    encoding="utf-8",
                )
            (normalized_dir / "file-1__a.jsonl").write_text(
                json.dumps({"text": "record", "metadata": {"source_type": "docs"}}) + "\n",
                encoding="utf-8",
            )

            report = audit_google_drive_docs_raw_dir(
                raw_dir=raw_dir,
                normalized_dir=normalized_dir,
                thresholds=GoogleDriveDocsQualityThresholds(
                    policy="warn",
                    min_text_bytes=100,
                    min_nonempty_characters=10,
                    max_short_document_ratio=0.2,
                ),
            )

        payload = report.to_payload()
        metadata = payload["metadata"]
        self.assertEqual("warning", payload["status"])
        self.assertTrue(payload["can_continue"])
        self.assertIn("short_document_ratio_too_high", metadata["critical_failures"])
        self.assertEqual(2, metadata["markdown_count"])
        self.assertEqual(1, metadata["duplicate_group_count"])
        self.assertEqual(1, metadata["normalized_record_count"])
        self.assertEqual(2, metadata["short_document_count"])


if __name__ == "__main__":
    unittest.main()
