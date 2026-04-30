from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.config.schema import ObjectStorageSection
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.source import (
    AccessScope,
    BackfillScope,
    NormalizedDocument,
    SourceRawItem,
    SyncCursor,
)
from kumc_agent.features.ingestion.chunking import IngestionChunker
from kumc_agent.features.ingestion.service import IngestionService
from kumc_agent.infra.audit.repository import FileAuditLogRepository
from kumc_agent.infra.database.postgres import PostgresClient
from kumc_agent.config.schema import DatabaseSection
from kumc_agent.infra.ingestion.repository import FileIngestionRepository
from kumc_agent.infra.object_storage.raw_snapshot import RawSnapshotStore
from kumc_agent.infra.object_storage.s3 import S3ObjectStorageClient
from kumc_agent.infra.secret_finding import SecretFindingDetector
from kumc_agent.infra.storage.filesystem import FileSystemStorage
from kumc_agent.utils.hashing import stable_hash


class DummyConnector:
    source_kind = "dummy"

    def __init__(self, items: list[SourceRawItem]) -> None:
        self.items = items

    async def backfill(self, scope: BackfillScope) -> AsyncIterator[SourceRawItem]:
        for item in self.items[: scope.limit]:
            yield item

    async def poll_changes(self, cursor: SyncCursor):
        for item in self.items:
            yield item

    async def fetch_item(self, external_id: str) -> SourceRawItem:
        for item in self.items:
            if item.external_id == external_id:
                return item
        raise KeyError(external_id)

    async def normalize(self, raw: SourceRawItem) -> NormalizedDocument:
        source_item_id = stable_hash(f"{raw.source_kind}:{raw.external_id}")
        return NormalizedDocument(
            id=stable_hash(f"doc:{source_item_id}:{raw.checksum}"),
            source_item_id=source_item_id,
            source_kind=raw.source_kind,
            external_id=raw.external_id,
            version=1,
            title=raw.title,
            normalized_text=raw.text,
            normalized_format="markdown",
            language="ja",
            access_scope=raw.access_scope,
            checksum=raw.checksum,
            metadata={},
        )


class FullScanOnlyConnector(DummyConnector):
    supports_incremental = False

    async def poll_changes(self, cursor: SyncCursor):
        raise AssertionError("unsupported connector must not poll changes")


class IncrementalConnector(DummyConnector):
    supports_incremental = True

    def __init__(self, items: list[SourceRawItem]) -> None:
        super().__init__(items)
        self.polled = False

    async def poll_changes(self, cursor: SyncCursor):
        self.polled = True
        for item in self.items:
            yield item


class IngestionServiceTests(unittest.TestCase):
    def test_raw_snapshot_key_uses_file_tool_safe_hatenablog_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ObjectStorageSection(
                endpoint_url="",
                bucket="",
                region="ap-northeast-1",
                access_key_id="",
                secret_access_key="",
                prefix="test",
                use_ssl=True,
            )
            store = RawSnapshotStore(
                config=config,
                local_root=root / "objects",
                s3=S3ObjectStorageClient(config),
            )
            key = store.put(
                SourceRawItem(
                    source_kind="hatenablog",
                    external_id="hatenablog:/entry/2025/04/21/160000",
                    title="Hatena",
                    text="body",
                    canonical_url="https://kumc.hatenablog.com/entry/2025/04/21/160000",
                    checksum="abcdef0123456789",
                    metadata={
                        "hatenablog_url": (
                            "https://kumc.hatenablog.com/entry/2025/04/21/160000"
                        )
                    },
                )
            )

            self.assertEqual(
                key,
                "test/raw/hatenablog/entry/2025/04/21/160000/abcdef0123456789.txt",
            )
            self.assertNotIn(":", key)

    def test_backfill_detects_checksum_and_secret_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = SourceRawItem(
                source_kind="dummy",
                external_id="item-1",
                title="Sample",
                text="hello\nsecret key: sk-abcdefghijklmnopqrstuvwxyz",
                access_scope=AccessScope(visibility="admin"),
                checksum=stable_hash("hello\nsecret key: sk-abcdefghijklmnopqrstuvwxyz"),
            )
            service = IngestionService(
                connectors={"dummy": DummyConnector([raw])},
                repository=FileIngestionRepository(root / "ingestion"),
                raw_snapshots=RawSnapshotStore(
                    config=ObjectStorageSection(
                        endpoint_url="",
                        bucket="",
                        region="ap-northeast-1",
                        access_key_id="",
                        secret_access_key="",
                        prefix="test",
                        use_ssl=True,
                    ),
                    local_root=root / "objects",
                    s3=S3ObjectStorageClient(
                        ObjectStorageSection(
                            endpoint_url="",
                            bucket="",
                            region="ap-northeast-1",
                            access_key_id="",
                            secret_access_key="",
                            prefix="test",
                            use_ssl=True,
                        )
                    ),
                ),
                chunker=IngestionChunker(),
                secret_detector=SecretFindingDetector(),
                audit_log=FileAuditLogRepository(root / "audit.jsonl"),
            )

            first = asyncio.run(service.backfill(source_kind="dummy"))
            second = asyncio.run(service.backfill(source_kind="dummy"))

            self.assertEqual(first.changed, 1)
            self.assertGreaterEqual(first.secret_findings, 1)
            self.assertEqual(second.skipped, 1)
            findings = (root / "ingestion" / "secret_findings.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn('"redaction_policy": "deny"', findings)
            source_item = json.loads(
                (root / "ingestion" / "source_items.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(source_item["metadata"]["terms_review_status"], "pending")
            self.assertFalse(source_item["metadata"]["external_reuse_allowed"])

    def test_existing_cursor_uses_full_scan_when_connector_has_no_incremental_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = FileIngestionRepository(root / "ingestion")
            repository.save_sync_cursor(
                SyncCursor(source_kind="dummy", cursor="old-cursor", metadata={})
            )
            raw = SourceRawItem(
                source_kind="dummy",
                external_id="item-1",
                title="Sample",
                text="hello",
                access_scope=AccessScope(visibility="public"),
                checksum=stable_hash("hello"),
            )
            service = IngestionService(
                connectors={"dummy": FullScanOnlyConnector([raw])},
                repository=repository,
                raw_snapshots=RawSnapshotStore(
                    config=ObjectStorageSection(
                        endpoint_url="",
                        bucket="",
                        region="ap-northeast-1",
                        access_key_id="",
                        secret_access_key="",
                        prefix="test",
                        use_ssl=True,
                    ),
                    local_root=root / "objects",
                    s3=S3ObjectStorageClient(
                        ObjectStorageSection(
                            endpoint_url="",
                            bucket="",
                            region="ap-northeast-1",
                            access_key_id="",
                            secret_access_key="",
                            prefix="test",
                            use_ssl=True,
                        )
                    ),
                ),
                chunker=IngestionChunker(),
                secret_detector=SecretFindingDetector(),
                audit_log=FileAuditLogRepository(root / "audit.jsonl"),
            )

            result = asyncio.run(service.backfill(source_kind="dummy"))
            cursor = repository.load_sync_cursor("dummy")

            self.assertEqual(result.changed, 1)
            self.assertIsNotNone(cursor)
            self.assertEqual(cursor.metadata["mode"], "full_scan_cursor_unsupported")  # type: ignore[union-attr]
            self.assertFalse(cursor.metadata["cursor_supported"])  # type: ignore[union-attr]
            self.assertTrue(cursor.metadata["previous_cursor_present"])  # type: ignore[union-attr]

    def test_existing_cursor_uses_poll_changes_when_connector_supports_incremental(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = FileIngestionRepository(root / "ingestion")
            repository.save_sync_cursor(
                SyncCursor(source_kind="dummy", cursor="old-cursor", metadata={})
            )
            raw = SourceRawItem(
                source_kind="dummy",
                external_id="item-1",
                title="Sample",
                text="hello",
                access_scope=AccessScope(visibility="public"),
                checksum=stable_hash("hello"),
            )
            connector = IncrementalConnector([raw])
            service = IngestionService(
                connectors={"dummy": connector},
                repository=repository,
                raw_snapshots=RawSnapshotStore(
                    config=ObjectStorageSection(
                        endpoint_url="",
                        bucket="",
                        region="ap-northeast-1",
                        access_key_id="",
                        secret_access_key="",
                        prefix="test",
                        use_ssl=True,
                    ),
                    local_root=root / "objects",
                    s3=S3ObjectStorageClient(
                        ObjectStorageSection(
                            endpoint_url="",
                            bucket="",
                            region="ap-northeast-1",
                            access_key_id="",
                            secret_access_key="",
                            prefix="test",
                            use_ssl=True,
                        )
                    ),
                ),
                chunker=IngestionChunker(),
                secret_detector=SecretFindingDetector(),
                audit_log=FileAuditLogRepository(root / "audit.jsonl"),
            )

            result = asyncio.run(service.backfill(source_kind="dummy"))
            cursor = repository.load_sync_cursor("dummy")

            self.assertEqual(result.changed, 1)
            self.assertTrue(connector.polled)
            self.assertIsNotNone(cursor)
            self.assertEqual(cursor.metadata["mode"], "poll_changes")  # type: ignore[union-attr]
            self.assertTrue(cursor.metadata["cursor_supported"])  # type: ignore[union-attr]

    def test_filesystem_storage_excludes_deny_chunks_from_answer_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = FileSystemStorage(
                chunks_path=root / "chunks.jsonl",
                index_documents_path=root / "ingestion" / "index_documents.jsonl",
            )
            storage.save_chunks(
                [
                    Chunk(id="ok", document_id="doc", text="public", index=0, metadata={}),
                    Chunk(
                        id="deny",
                        document_id="doc",
                        text="secret",
                        index=1,
                        metadata={"redaction_policy": "deny", "index_status": "quarantined"},
                    ),
                ]
            )
            self.assertEqual([chunk.id for chunk in storage.load_chunks()], ["ok"])

if __name__ == "__main__":
    unittest.main()
