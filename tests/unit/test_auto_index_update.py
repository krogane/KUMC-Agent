from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.operations import IndexingRun
from kumc_agent.domain.models.source import AccessScope, SourceRawItem
from kumc_agent.features.indexing.change_detection import SourceItemState, detect_source_change
from kumc_agent.features.indexing.lock import FileIndexingLock
from kumc_agent.features.indexing.quality import IndexQualitySmokeChecker
from kumc_agent.features.indexing.snapshot import IndexSnapshotPublisher
from kumc_agent.features.ingestion.service import IngestionResult
from kumc_agent.infra.ingestion.repository import FileIngestionRepository
from kumc_agent.usecases.indexing.auto_update import (
    AutoIndexUpdateRequest,
    AutoIndexUpdateUsecase,
)


class _Operations:
    def __init__(self) -> None:
        self.runs: list[IndexingRun] = []

    def save_indexing_run(self, run: IndexingRun) -> IndexingRun:
        self.runs.append(run)
        return run


class _Build:
    def execute(self, request):
        index_dir = request.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "dense_chunks.jsonl").write_text(
            json.dumps(
                {
                    "id": "chunk-1",
                    "document_id": "doc-1",
                    "text": "KUMC 自動インデックス更新",
                    "index": 0,
                    "metadata": {"index_status": "active"},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        np.save(index_dir / "dense_vectors.npy", np.ones((1, 3), dtype=np.float32))
        (index_dir / "bm25_tokens.json").write_text('[["kumc"]]', encoding="utf-8")
        (index_dir / "bm25_chunks.jsonl").write_text(
            (index_dir / "dense_chunks.jsonl").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return SimpleNamespace(loaded_sources=0, documents=1, chunks=1, index_dir=index_dir)


class _Ingestion:
    def __init__(self, result: IngestionResult) -> None:
        self.result = result

    async def backfill_many(self, *, source_kinds, scope):
        return (self.result,)


def _config(root: Path):
    return SimpleNamespace(
        app=SimpleNamespace(index_dir=root / "data" / "index"),
        infrastructure=SimpleNamespace(
            database=SimpleNamespace(url="", connect_timeout_seconds=1.0, application_name="test"),
            redis=SimpleNamespace(url="", socket_timeout_seconds=1.0),
        ),
        scheduler=SimpleNamespace(
            auto_index_enabled=True,
            auto_index_time="06:00",
            auto_index_weekdays=[0, 1, 2, 3, 4, 5, 6],
            auto_index_lock_ttl_minutes=5,
            quality_min_chunk_ratio=0.5,
            quality_smoke_queries=[],
            rollback_keep_snapshots=2,
        ),
    )


class AutoIndexUpdateTests(unittest.TestCase):
    def test_change_detection_uses_checksum_revision_and_acl_hash(self) -> None:
        item = SourceRawItem(
            source_kind="drive",
            external_id="a",
            title="A",
            text="body",
            checksum="new",
            metadata={"revision": "2"},
            access_scope=AccessScope(source_acl_hash="acl-2"),
        )
        previous = SourceItemState(
            source_kind="drive",
            external_id="a",
            checksum="old",
            revision="1",
            acl_hash="acl-1",
        )
        self.assertEqual(detect_source_change(item=item, previous=previous).change_kind, "updated")
        same_text = SourceRawItem(
            source_kind="drive",
            external_id="a",
            title="A",
            text="body",
            checksum="old",
            metadata={"revision": "1"},
            access_scope=AccessScope(source_acl_hash="acl-2"),
        )
        self.assertEqual(
            detect_source_change(item=same_text, previous=previous).change_kind,
            "permission_changed",
        )

    def test_file_lock_skips_second_run_and_recovers_from_stale_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / ".auto_index.lock"
            first = FileIndexingLock(path=lock_path, ttl_minutes=1)
            self.assertTrue(first.acquire(run_id="one").acquired)
            second = FileIndexingLock(path=lock_path, ttl_minutes=1)
            self.assertFalse(second.acquire(run_id="two").acquired)
            first.release()
            self.assertTrue(second.acquire(run_id="two").acquired)
            second.release()

    def test_snapshot_publish_and_quality_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "index"
            staging = root / "staging" / "run"
            _Build().execute(SimpleNamespace(index_dir=staging))
            quality = IndexQualitySmokeChecker().check(staging_dir=staging, current_dir=root)
            self.assertTrue(quality.passed)
            result = IndexSnapshotPublisher(index_dir=root).publish(run_id="run", staging_dir=staging)
            self.assertEqual(result.snapshot_id, "run")
            self.assertTrue((root / "dense_chunks.jsonl").exists())
            self.assertTrue((root / "current.json").exists())

    def test_auto_update_saves_runs_and_publishes_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            operations = _Operations()
            result = AutoIndexUpdateUsecase(
                config=config,
                build_usecase=_Build(),
                operations=operations,
                ingestion_service=_Ingestion(
                    IngestionResult(
                        source_kind="dummy",
                        seen=1,
                        changed=1,
                        skipped=0,
                        deleted=0,
                        documents=1,
                        chunks=1,
                        secret_findings=0,
                    )
                ),
            ).execute(AutoIndexUpdateRequest(trigger="manual"))
            self.assertEqual(result.status, "succeeded")
            self.assertTrue((config.app.index_dir / "dense_chunks.jsonl").exists())
            self.assertEqual(operations.runs[-1].metadata["source_results"][0]["changed"], 1)

    def test_ingestion_repository_loads_revision_and_acl_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = SourceRawItem(
                source_kind="drive",
                external_id="file-1",
                title="File",
                text="body",
                checksum="abc",
                metadata={"revision": "rev-1"},
                access_scope=AccessScope(source_acl_hash="acl-1"),
            )
            repo = FileIngestionRepository(root)
            repo.save_item(raw=raw, document=_doc(raw), chunks=[], findings=[], raw_object_key="raw/key")
            state = repo.load_item_states("drive")["file-1"]
            self.assertEqual(state.revision, "rev-1")
            self.assertEqual(state.acl_hash, "acl-1")


def _doc(raw: SourceRawItem):
    from kumc_agent.domain.models.source import NormalizedDocument

    return NormalizedDocument(
        id="doc-1",
        source_item_id="source-item-1",
        source_kind=raw.source_kind,
        external_id=raw.external_id,
        version=1,
        title=raw.title,
        normalized_text=raw.text,
        normalized_format="markdown",
        language="ja",
        access_scope=raw.access_scope,
        checksum=raw.checksum,
        metadata=raw.metadata,
    )


if __name__ == "__main__":
    unittest.main()
