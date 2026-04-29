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

from kumc_agent.infra.connectors.file_scanner import iter_structured_jsonl_records


class DocsConnectorRecordTests(unittest.TestCase):
    def test_iter_structured_jsonl_records_emits_record_level_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "file-id__doc.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "text": "ページ本文",
                        "metadata": {
                            "source_type": "docs",
                            "drive_file_id": "file-id",
                            "drive_file_name": "doc.pdf",
                            "page_number": 2,
                            "normalized_record_id": 5,
                            "index_status": "active",
                            "access_scope": {"visibility": "guild", "guild_id": "guild-1"},
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            path.with_suffix(".jsonl.meta.json").write_text(
                json.dumps(
                    {
                        "drive_file_id": "file-id",
                        "drive_file_name": "doc.pdf",
                        "drive_modified_time": "2026-04-29T00:00:00Z",
                    }
                ),
                encoding="utf-8",
            )

            items = iter_structured_jsonl_records(
                source_kind="google_drive",
                root_dir=root,
                default_visibility="admin",
            )

        self.assertEqual(1, len(items))
        self.assertEqual("file-id:5", items[0].external_id)
        self.assertEqual(2, items[0].metadata["page_number"])
        self.assertEqual("guild", items[0].access_scope.visibility)
        self.assertEqual("guild-1", items[0].access_scope.guild_id)


if __name__ == "__main__":
    unittest.main()
