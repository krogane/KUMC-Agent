from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
from kumc_agent.infra.operations import FileOperationsRepository, PostgresOperationsRepository


class _FakePostgresCursor:
    def __init__(self, connection: "_FakePostgresConnection") -> None:
        self.connection = connection
        self.result: list[tuple[object, ...]] = []

    def __enter__(self) -> "_FakePostgresCursor":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def execute(self, sql: str, values: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(sql.split()).lower()
        if normalized.startswith("insert into assets") and values is not None:
            columns = [column.strip() for column in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
            payload = dict(zip(columns, values))
            self.connection.assets[str(payload["id"])] = payload
            return
        if normalized.startswith("select") and "from assets where id" in normalized and values is not None:
            columns = _select_columns(sql)
            payload = self.connection.assets.get(str(values[0]))
            self.result = [tuple(payload.get(column) for column in columns)] if payload else []
            return
        if normalized.startswith("select") and "from assets" in normalized:
            columns = _select_columns(sql)
            self.result = [
                tuple(payload.get(column) for column in columns)
                for payload in self.connection.assets.values()
            ]

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self.result)

    def fetchone(self) -> tuple[object, ...] | None:
        return self.result[0] if self.result else None


class _FakePostgresConnection:
    def __init__(self) -> None:
        self.assets: dict[str, dict[str, object]] = {}

    def __enter__(self) -> "_FakePostgresConnection":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def cursor(self) -> _FakePostgresCursor:
        return _FakePostgresCursor(self)

    def commit(self) -> None:
        return None


class _FakePostgres:
    def __init__(self) -> None:
        self.connection = _FakePostgresConnection()

    def connect(self) -> _FakePostgresConnection:
        return self.connection


def _select_columns(sql: str) -> list[str]:
    return [column.strip() for column in sql.lower().split("select", 1)[1].split("from", 1)[0].split(",")]


class ImageSearchTests(unittest.TestCase):
    def test_image_search_eval_set_is_valid_jsonl(self) -> None:
        path = ROOT / "docs" / "evals" / "image-search.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertGreaterEqual(len(records), 6)
        self.assertTrue({record["category"] for record in records} >= {
            "ocr_only",
            "caption_semantic",
            "similar_image",
            "protected_source_leakage",
            "source_attribution",
            "reuse_disclaimer",
        })

    def test_postgres_assets_can_be_read_and_searched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = PostgresOperationsRepository(
                root_dir=Path(tmp) / "operations",
                postgres=_FakePostgres(),  # type: ignore[arg-type]
            )
            asset = Asset(
                id="asset-1",
                source_kind="hatena",
                title="新歓画像",
                description="公開イベント",
                access_scope={"visibility": "public"},
                metadata={"surrounding_text": "イベント告知"},
            )

            repo.save_asset(asset)

            self.assertEqual(repo.get_asset("asset-1").id, "asset-1")  # type: ignore[union-attr]
            self.assertEqual(repo.list_assets(query="イベント")[0].id, "asset-1")

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
            config = ImageSearchConfig(
                limit=5,
                dense_top_k=5,
                feature_top_k=5,
                feature_model="local_hash",
                feature_dimensions=32,
            )
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

    def test_builder_falls_back_to_discord_proxy_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            messages_dir = root / "raw" / "messages" / "guild"
            messages_dir.mkdir(parents=True)
            (messages_dir / "channel.jsonl").write_text(
                json.dumps(
                    {
                        "text": "期限切れURLの画像です",
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
                                    "url": "https://cdn.discordapp.com/expired/poster.png",
                                    "proxy_url": "https://media.discordapp.net/proxy/poster.png",
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
            config = ImageSearchConfig(
                limit=5,
                dense_top_k=5,
                feature_top_k=5,
                feature_model="local_hash",
                feature_dimensions=32,
            )
            builder = ImageAssetBuildService(
                repository=repository,
                raw_dir=root / "raw",
                image_dir=root / "image_search" / "images",
                index_dir=root / "image_search",
                embedder=embedder,
                config=config,
            )

            def _fake_download(url: str, *, max_bytes: int) -> tuple[bytes, str]:
                del max_bytes
                if "expired" in url:
                    raise RuntimeError("expired")
                return _tiny_png(), "image/png"

            with patch(
                "kumc_agent.features.image_search.service._download_bytes",
                side_effect=_fake_download,
            ):
                run = builder.build_from_raw_sources()

            assets = repository.list_assets(query="")
            self.assertEqual("succeeded", run.status)
            self.assertEqual(1, len(assets))
            self.assertEqual("succeeded", assets[0].metadata["download_status"])
            self.assertTrue(assets[0].metadata["download_fallback_used"])
            self.assertEqual(
                "https://media.discordapp.net/proxy/poster.png",
                assets[0].metadata["downloaded_image_ref"],
            )

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
            config = ImageSearchConfig(
                limit=5,
                dense_top_k=5,
                feature_top_k=5,
                feature_model="local_hash",
                feature_dimensions=32,
            )
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

    def test_source_filter_limit_matched_fields_and_duplicate_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = FileOperationsRepository(root_dir=root / "operations")
            for asset_id, source_kind in (("hatena-1", "hatena"), ("hatena-2", "hatena")):
                repository.save_asset(
                    Asset(
                        id=asset_id,
                        source_kind=source_kind,
                        title="新歓ポスター",
                        description="新歓イベントの告知画像",
                        access_scope={"visibility": "public"},
                        metadata={
                            "ocr_text": "KUMC 新歓",
                            "surrounding_text": "新歓イベント",
                            "source_label": "公開記事",
                            "duplicate_group_id": "dup-1",
                            "index_status": "active",
                        },
                    )
                )
            repository.save_asset(
                Asset(
                    id="x-1",
                    source_kind="x",
                    title="新歓写真",
                    description="別画像",
                    access_scope={"visibility": "public"},
                    metadata={"surrounding_text": "新歓イベント", "index_status": "active"},
                )
            )
            embedder = LocalEmbedder(model_name="", dimensions=32)
            config = ImageSearchConfig(
                limit=5,
                dense_top_k=5,
                feature_top_k=5,
                feature_model="local_hash",
                feature_dimensions=32,
                duplicate_group_limit=1,
            )
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
            )

            result = service.search(
                ImageSearchRequest(
                    query="新歓",
                    source_filter=("hatena",),
                    limit=5,
                )
            )

            self.assertEqual(1, len(result.assets))
            self.assertEqual("hatena", result.assets[0].source_kind)
            self.assertIn("matched_fields", result.metadata["search_results"][0])
            self.assertIn("search", result.assets[0].metadata)


def _tiny_png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
        b"\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )


if __name__ == "__main__":
    unittest.main()
