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

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.operations import IndexingRun
from kumc_agent.domain.models.source import AccessScope, SourceRawItem
from kumc_agent.features.indexing.change_detection import SourceItemState, detect_source_change
from kumc_agent.features.indexing.lock import FileIndexingLock
from kumc_agent.features.indexing.quality import IndexQualitySmokeChecker
from kumc_agent.features.indexing.snapshot import IndexSnapshotPublisher
from kumc_agent.features.indexing.task_event import TaskEventIndexBuildService
from kumc_agent.features.ingestion.service import IngestionResult
from kumc_agent.infra.ingestion.repository import FileIngestionRepository
from kumc_agent.usecases.indexing.auto_update import (
    AutoIndexUpdateRequest,
    AutoIndexUpdateUsecase,
)
from kumc_agent.domain.models.workflow import Event, Task


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


class _BuildWithEmbeddingCache(_Build):
    def __init__(self) -> None:
        self.compacted: list[tuple[object, ...]] = []

    def execute(self, request):
        result = super().execute(request)
        result.stage_results = {
            "embedding": {
                "enabled": True,
                "embedded_chunks": 1,
                "reused_chunks": 0,
            }
        }
        result.embedding_cache_keys = ("cache-key-1",)
        return result

    def compact_embedding_cache(self, active_keys):
        keys = tuple(active_keys)
        self.compacted.append(keys)
        return {
            "status": "succeeded",
            "kept_records": len(keys),
        }


class _Ingestion:
    def __init__(self, result: IngestionResult) -> None:
        self.result = result

    async def backfill_many(self, *, source_kinds, scope):
        return (self.result,)


class _ExplodingIngestion:
    async def backfill_many(self, *, source_kinds, scope):
        raise AssertionError("ingestion should not run")


class _MemberProfileBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def rebuild_guild(self, *, guild_id: str, index_dir: Path | None = None) -> IndexingRun:
        assert index_dir is not None
        self.calls.append((guild_id, index_dir))
        marker = index_dir / "member_profiles" / f"{guild_id}.marker"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok", encoding="utf-8")
        return IndexingRun(
            id=f"member-{guild_id}",
            source_kind="member_profiles",
            status="succeeded",
            seen=2,
            changed=2,
            metadata={"guild_id": guild_id},
        )


class _Embedder:
    def embed_documents(self, texts):
        return np.ones((len(texts), 3), dtype=np.float32)

    def embed_query(self, text):
        return np.ones(3, dtype=np.float32)


class _WorkflowRepository:
    def list_tasks(self, **kwargs):
        return [
            Task(id="task-1", title="Active task", status="todo"),
            Task(id="task-2", title="Deleted task", status="deleted"),
        ]

    def list_events(self, **kwargs):
        return [
            Event(id="event-1", title="Active event", status="planning"),
            Event(id="event-2", title="Canceled event", status="canceled"),
        ]


class _EventDeltaExtractor:
    def __init__(self) -> None:
        self.calls = []

    def event_extract_from_delta(self, *, text, evidence, access, metadata):
        self.calls.append(
            {
                "text": text,
                "evidence": evidence,
                "access": access,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(
            metadata={"extraction": {"candidate_count": 1, "change_candidate_count": 0}},
            event_candidates=(object(),),
            event_change_candidates=tuple(),
        )


class _EventDeltaChunkSource:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.source_kinds = []

    def load_active_chunks(self, *, source_kinds: tuple[str, ...] = tuple()) -> list[Chunk]:
        self.source_kinds.append(source_kinds)
        return self.chunks


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
        event_management=SimpleNamespace(auto_extract_after_index_update=True),
        indexing=SimpleNamespace(
            embedding_cache=SimpleNamespace(compact_after_publish=True),
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
            build = _BuildWithEmbeddingCache()
            result = AutoIndexUpdateUsecase(
                config=config,
                build_usecase=build,
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
            self.assertEqual(build.compacted, [("cache-key-1",)])
            self.assertEqual(
                operations.runs[-1].metadata["stage_results"]["embedding"][
                    "cache_compaction"
                ]["status"],
                "succeeded",
            )

    def test_auto_update_extracts_events_from_ingestion_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            operations = _Operations()
            extractor = _EventDeltaExtractor()
            chunk_source = _EventDeltaChunkSource(
                [
                    Chunk(
                        id="chunk-1",
                        document_id="doc-1",
                        text="5月5日に新歓会を部室で開催します。",
                        index=0,
                        metadata={
                            "source_item_id": "source-1",
                            "source_kind": "discord",
                            "external_id": "message-1",
                            "source_title": "告知",
                        },
                    )
                ]
            )
            result = AutoIndexUpdateUsecase(
                config=config,
                build_usecase=_Build(),
                operations=operations,
                ingestion_service=_Ingestion(
                    IngestionResult(
                        source_kind="discord",
                        seen=1,
                        changed=1,
                        skipped=0,
                        deleted=0,
                        documents=1,
                        chunks=1,
                        secret_findings=0,
                    )
                ),
                event_delta_extractor=extractor,
                event_delta_chunk_source=chunk_source,
            ).execute(AutoIndexUpdateRequest(trigger="manual"))

            self.assertEqual(result.status, "succeeded")
            self.assertEqual(len(extractor.calls), 1)
            self.assertEqual(extractor.calls[0]["metadata"]["source"], "auto_index_update")
            self.assertEqual(extractor.calls[0]["access"].user_id, "auto_index_update")
            self.assertEqual(chunk_source.source_kinds[0], ("discord",))
            self.assertEqual(operations.runs[-1].metadata["event_extraction"]["candidate_count"], 1)

    def test_auto_update_refreshes_member_profiles_as_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            operations = _Operations()
            builder = _MemberProfileBuilder()
            result = AutoIndexUpdateUsecase(
                config=config,
                build_usecase=_Build(),
                operations=operations,
                ingestion_service=_ExplodingIngestion(),
                member_profile_builder=builder,
                member_profile_guild_ids=("guild-1",),
            ).execute(
                AutoIndexUpdateRequest(
                    trigger="manual",
                    source_filter=("member_profiles",),
                    quality_check_enabled=False,
                )
            )

            self.assertEqual(result.status, "succeeded")
            self.assertEqual(len(builder.calls), 1)
            self.assertTrue((config.app.index_dir / "member_profiles" / "guild-1.marker").exists())
            self.assertEqual(result.seen, 2)
            self.assertEqual(result.changed, 2)
            self.assertEqual(
                operations.runs[-1].metadata["stage_results"]["member_profiles"]["guild_ids"],
                ["guild-1"],
            )

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

    def test_file_ingestion_repository_excludes_deleted_chunks_from_active_index_source(self) -> None:
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
            source_item_id = _source_item_id(raw)
            repo.save_item(
                raw=raw,
                document=_doc(raw),
                chunks=[
                    Chunk(
                        id="chunk-1",
                        document_id="doc-1",
                        text="body",
                        index=0,
                        metadata={
                            "source_item_id": source_item_id,
                            "source_kind": "drive",
                            "index_status": "active",
                        },
                    )
                ],
                findings=[],
                raw_object_key="raw/key",
            )
            self.assertEqual(len(repo.load_active_chunks()), 1)
            repo.mark_deleted(source_kind="drive", external_id="file-1")
            self.assertEqual(repo.load_active_chunks(), [])

    def test_task_event_index_includes_only_canonical_active_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = TaskEventIndexBuildService(
                repository=_WorkflowRepository(),
                embedder=_Embedder(),
            ).rebuild(index_dir=root)

            docs = (root / "task_event" / "task_event_documents.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertEqual(run.seen, 4)
            self.assertEqual(run.changed, 2)
            self.assertEqual(run.deleted, 2)
            self.assertIn("task:task-1", docs)
            self.assertIn("event:event-1", docs)
            self.assertNotIn("task:task-2", docs)
            self.assertNotIn("event:event-2", docs)


def _doc(raw: SourceRawItem):
    from kumc_agent.domain.models.source import NormalizedDocument

    return NormalizedDocument(
        id="doc-1",
        source_item_id=_source_item_id(raw),
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


def _source_item_id(raw: SourceRawItem) -> str:
    from kumc_agent.utils.hashing import stable_hash

    return stable_hash(f"{raw.source_kind}:{raw.external_id}")


if __name__ == "__main__":
    unittest.main()
