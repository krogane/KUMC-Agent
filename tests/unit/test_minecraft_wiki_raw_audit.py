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

from kumc_agent.usecases.ingestion.minecraft_wiki_audit import (
    MinecraftWikiQualityThresholds,
    audit_minecraft_wiki_raw_dir,
)


class MinecraftWikiRawAuditTests(unittest.TestCase):
    def test_audit_reports_redirect_ratio_and_metadata_under_payload_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            (raw_dir / "Alias.md").write_text(
                "#転送 [[丸石]]",
                encoding="utf-8",
            )
            (raw_dir / "Alias.md.meta.json").write_text(
                json.dumps(
                    {
                        "minecraft_wiki_title": "Alias",
                        "minecraft_wiki_page_id": "1",
                        "minecraft_wiki_revision_id": "10",
                        "canonical_url": "https://ja.minecraft.wiki/w/Alias",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (raw_dir / "丸石.md").write_text(
                "丸石は石を採掘すると得られる。用途はクラフト素材。" * 20,
                encoding="utf-8",
            )
            (raw_dir / "丸石.md.meta.json").write_text(
                json.dumps(
                    {
                        "minecraft_wiki_title": "丸石",
                        "minecraft_wiki_page_id": "2",
                        "minecraft_wiki_revision_id": "11",
                        "canonical_url": "https://ja.minecraft.wiki/w/%E4%B8%B8%E7%9F%B3",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = audit_minecraft_wiki_raw_dir(
                raw_dir=raw_dir,
                thresholds=MinecraftWikiQualityThresholds(
                    min_article_characters=100,
                    max_redirect_ratio=0.8,
                    min_indexable_pages=1,
                    min_chunk_count=1,
                    policy="fail",
                ),
                chunk_count=2,
            )

        payload = report.to_payload()
        self.assertEqual(payload["status"], "passed")
        self.assertNotIn("redirect_ratio", payload)
        self.assertEqual(payload["metadata"]["redirect_count"], 1)
        self.assertEqual(payload["metadata"]["indexable_page_count"], 1)
        self.assertEqual(payload["metadata"]["canonical_hosts"], {"ja.minecraft.wiki": 2})
        self.assertIn("Minecraft Wiki Raw品質監査", report.to_markdown())

    def test_audit_can_fail_when_publish_policy_is_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            (raw_dir / "Alias.md").write_text(
                "#REDIRECT [[丸石]]",
                encoding="utf-8",
            )
            (raw_dir / "Alias.md.meta.json").write_text(
                "{}",
                encoding="utf-8",
            )

            report = audit_minecraft_wiki_raw_dir(
                raw_dir=raw_dir,
                thresholds=MinecraftWikiQualityThresholds(
                    min_article_characters=100,
                    max_redirect_ratio=0.1,
                    min_indexable_pages=1,
                    min_chunk_count=1,
                    policy="fail",
                ),
                chunk_count=0,
            )

        self.assertEqual(report.status, "failed")
        self.assertFalse(report.can_continue)
        self.assertIn("redirect_ratio_too_high", report.critical_failures)
        self.assertIn("revision_id_missing", report.critical_failures)


if __name__ == "__main__":
    unittest.main()
