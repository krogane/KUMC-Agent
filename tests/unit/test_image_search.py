from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.operations import Asset
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.features.image_search import (
    ImageAssetBuildService,
    ImageSearchConfig,
    ImageSearchRequest,
    ImageSearchService,
)
from kumc_agent.infra.embeddings.local import LocalEmbedder
from kumc_agent.infra.operations import FileOperationsRepository


class ImageSearchTests(unittest.TestCase):
    def test_builder_indexes_discord_attachment_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "poster.png"
            image_path.write_bytes(_tiny_png())
            messages_dir = root / "raw" / "messages" / "guild"
            messages_dir.mkdir(parents=True)
            (messages_dir / "channel.jsonl").write_text(
                json.dumps(
                    {
                        "text": "2026 新歓ポスターです",
                        "metadata": {
                            "guild_id": "guild-1",
                            "channel_id": "channel-1",
                            "channel_name": "広報",
                            "message_id": "message-1",
                            "message_timestamp": "2026-04-01T00:00:00+00:00",
                            "attachments": [
                                {
                                    "id": "att-1",
                                    "filename": "poster.png",
                                    "url": str(image_path),
                                    "content_type": "image/png",
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            repository = FileOperationsRepository(root_dir=root / "operations")
            embedder = LocalEmbedder(model_name="", dimensions=32)
            config = ImageSearchConfig(limit=5, dense_top_k=5, feature_top_k=5)
            builder = ImageAssetBuildService(
                repository=repository,
                raw_dir=root / "raw",
                image_dir=root / "image_search" / "images",
                index_dir=root / "image_search",
                embedder=embedder,
                config=config,
            )

            run = builder.build_from_raw_sources()
            service = ImageSearchService(
                repository=repository,
                embedder=embedder,
                index_dir=root / "image_search",
                config=config,
                allowed_guild_ids=("guild-1",),
            )
            result = service.search(
                ImageSearchRequest(
                    query="新歓",
                    access_context=AccessContext(user_id="u", guild_id="guild-1"),
                )
            )

            self.assertEqual(run.status, "succeeded")
            self.assertEqual(len(result.assets), 1)
            asset = result.assets[0]
            self.assertEqual(asset.source_kind, "discord")
            self.assertEqual(asset.metadata["source_label"], "広報")
            self.assertIn("ocr_text", asset.metadata)
            self.assertIn("surrounding_text", asset.metadata)
            self.assertIn("search", asset.metadata)

    def test_access_filter_hides_protected_images_and_allows_public_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = FileOperationsRepository(root_dir=root / "operations")
            repository.save_asset(
                Asset(
                    id="discord-asset",
                    source_kind="discord",
                    title="内部写真",
                    description="内部イベント",
                    access_scope={"visibility": "guild", "guild_id": "guild-1"},
                    metadata={"surrounding_text": "内部イベント", "index_status": "active"},
                )
            )
            repository.save_asset(
                Asset(
                    id="hatena-asset",
                    source_kind="hatena",
                    title="公開記事画像",
                    description="公開イベント",
                    access_scope={"visibility": "public"},
                    metadata={"surrounding_text": "公開イベント", "index_status": "active"},
                )
            )
            embedder = LocalEmbedder(model_name="", dimensions=32)
            config = ImageSearchConfig(limit=5, dense_top_k=5, feature_top_k=5)
            builder = ImageAssetBuildService(
                repository=repository,
                raw_dir=root / "raw",
                image_dir=root / "image_search" / "images",
                index_dir=root / "image_search",
                embedder=embedder,
                config=config,
            )
            builder.build_from_raw_sources()
            service = ImageSearchService(
                repository=repository,
                embedder=embedder,
                index_dir=root / "image_search",
                config=config,
                allowed_guild_ids=("guild-1",),
            )

            denied = service.search(
                ImageSearchRequest(
                    query="イベント",
                    access_context=AccessContext(user_id="u", guild_id="other"),
                )
            )
            allowed = service.search(
                ImageSearchRequest(
                    query="イベント",
                    access_context=AccessContext(user_id="u", guild_id="guild-1"),
                )
            )

            self.assertEqual([asset.id for asset in denied.assets], ["hatena-asset"])
            self.assertEqual({asset.id for asset in allowed.assets}, {"discord-asset", "hatena-asset"})


def _tiny_png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
        b"\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )


if __name__ == "__main__":
    unittest.main()
