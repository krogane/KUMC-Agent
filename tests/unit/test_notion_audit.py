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

from kumc_agent.domain.models.source import AccessScope, SourceRawItem
from kumc_agent.usecases.ingestion.notion_audit import (
    NotionQualityThresholds,
    annotate_notion_raw_items,
    audit_notion_raw_dir,
)


class NotionAuditTests(unittest.TestCase):
    def test_audit_reports_repository_and_index_coverage_under_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "notion"
            raw_dir.mkdir()
            page_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            page_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            for page_id, title, body in (
                (page_a, "Page A", "# Page A\n\n本文があります。" * 20),
                (page_b, "Page B", "# Page B\n"),
            ):
                path = raw_dir / f"{page_id}__{title}.md"
                path.write_text(body, encoding="utf-8")
                path.with_suffix(path.suffix + ".meta.json").write_text(
                    json.dumps(
                        {
                            "source_type": "notion",
                            "notion_page_id": page_id,
                            "notion_title": title,
                            "notion_url": f"https://notion.so/{page_id}",
                            "notion_created_time": "2026-01-01T00:00:00Z",
                            "notion_last_edited_time": "2026-01-02T00:00:00Z",
                            "access_scope": {"visibility": "public"},
                            "notion_page_path": title,
                            "notion_asset_count": 1 if page_id == page_a else 0,
                            "notion_unsupported_block_types": ["image"] if page_id == page_a else [],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            repository_dir = root / "ingestion"
            repository_dir.mkdir()
            repository_dir.joinpath("source_items.jsonl").write_text(
                json.dumps(
                    {
                        "source_kind": "notion",
                        "external_id": page_a,
                        "index_status": "active",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            stage_dir = root / "chunks" / "first_rec_chunk"
            (stage_dir / "notion").mkdir(parents=True)
            (stage_dir / "notion" / f"{page_a}.jsonl").write_text("", encoding="utf-8")

            report = audit_notion_raw_dir(
                raw_dir=raw_dir,
                repository_dir=repository_dir,
                index_page_ids={page_a},
                stage_dirs=(stage_dir,),
                thresholds=NotionQualityThresholds(policy="warn"),
            )

        payload = report.to_payload()
        self.assertEqual(payload["source"], "notion")
        self.assertNotIn("repository_coverage_ratio", payload)
        self.assertEqual(payload["metadata"]["unique_page_ids"], 2)
        self.assertEqual(payload["metadata"]["repository_unique_page_ids"], 1)
        self.assertEqual(payload["metadata"]["repository_coverage_ratio"], 0.5)
        self.assertEqual(payload["metadata"]["index_coverage_ratio"], 0.5)
        self.assertEqual(payload["metadata"]["asset_block_count"], 1)
        self.assertEqual(
            payload["metadata"]["stage_layout"]["first_rec_chunk"]["source_directory_files"],
            1,
        )
        self.assertIn("repository_coverage_too_low", payload["metadata"]["critical_failures"])
        self.assertIn("Notion Raw/Indexing品質監査", report.to_markdown())

    def test_annotation_quarantines_low_information_and_marks_duplicates(self) -> None:
        items = [
            SourceRawItem(
                source_kind="notion",
                external_id="a" * 32,
                title="A",
                text="# Same\n",
                access_scope=AccessScope(visibility="public"),
            ),
            SourceRawItem(
                source_kind="notion",
                external_id="b" * 32,
                title="B",
                text="# Same\n",
                access_scope=AccessScope(visibility="public"),
            ),
        ]

        annotated = annotate_notion_raw_items(
            items,
            min_text_bytes=200,
            min_nonempty_characters=50,
            quarantine_low_information=True,
        )

        for item in annotated:
            self.assertEqual(item.metadata["index_status"], "quarantined")
            self.assertIn("low_information", item.metadata["quality_flags"])
            self.assertIn("duplicate_text", item.metadata["quality_flags"])
            self.assertEqual(item.metadata["duplicate_group_size"], 2)


if __name__ == "__main__":
    unittest.main()
