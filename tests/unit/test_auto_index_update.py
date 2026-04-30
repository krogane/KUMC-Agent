from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
from kumc_agent.domain.models.source import AccessScope, NormalizedDocument, SourceRawItem
from kumc_agent.features.indexing.change_detection import SourceItemState, detect_source_change
from kumc_agent.features.indexing.lock import FileIndexingLock
from kumc_agent.features.indexing.paths import resolve_current_index_dir
from kumc_agent.features.indexing.quality import IndexQualitySmokeChecker
from kumc_agent.features.indexing.snapshot import IndexSnapshotPublisher
from kumc_agent.features.indexing.task_event import TaskEventIndexBuildService
from kumc_agent.features.ingestion.service import IngestionResult
from kumc_agent.infra.ingestion.repository import FileIngestionRepository
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever
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


class _BuildWithFailingStagedCommit(_Build):
    def __init__(self) -> None:
        self.commit_index_dir: Path | None = None

    def commit_staged_side_effects(self, index_dir: Path):
        self.commit_index_dir = index_dir
        raise RuntimeError("commit failed")


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


class _TaskDeltaExtractor:
    def __init__(self) -> None:
        self.calls = []

    def task_extract_from_delta(self, *, text, evidence, access, metadata):
        self.calls.append(
            {
                "text": text,
                "evidence": evidence,
                "access": access,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(
            metadata={
                "extraction": {
                    "schema_version": "workflow_extraction.v1",
                    "candidate_count": 1,
                    "change_candidate_count": 1,
                }
            },
            task_candidates=(object(),),
            task_change_candidates=(object(),),
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
        app=SimpleNamespace(
            data_dir=root / "data",
            index_dir=root / "data" / "index",
        ),
        infrastructure=SimpleNamespace(
            database=SimpleNamespace(url="", connect_timeout_seconds=1.0, application_name="test"),
            redis=SimpleNamespace(url="", socket_timeout_seconds=1.0),
        ),
        scheduler=SimpleNamespace(
            auto_index_enabled=True,
            auto_index_time="06:00",
            auto_index_weekdays=[0, 1, 2, 3, 4, 5, 6],
            auto_index_max_runtime_minutes=120,
            auto_index_lock_ttl_minutes=5,
            quality_min_chunk_ratio=0.5,
            quality_smoke_queries=[],
            rollback_keep_snapshots=2,
        ),
        event_management=SimpleNamespace(auto_extract_after_index_update=True),
        workflow_extraction=SimpleNamespace(lookback_days=1),
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
            (root / ".auto_index.lock").write_text("held", encoding="utf-8")
            result = IndexSnapshotPublisher(index_dir=root).publish(run_id="run", staging_dir=staging)
            self.assertEqual(result.snapshot_id, "run")
            self.assertTrue((result.release_dir / "dense_chunks.jsonl").exists())
            self.assertTrue((root / "current.json").exists())
            self.assertTrue((root / ".auto_index.lock").exists())
            self.assertEqual(resolve_current_index_dir(root), result.release_dir)
            (root / "current.json").write_text(
                json.dumps({"snapshot_id": "bad", "path": str(Path(tmp).parent)}),
                encoding="utf-8",
            )
            self.assertEqual(resolve_current_index_dir(root), root)

    def test_search_indexes_follow_current_release_pointer_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "index"
            publisher = IndexSnapshotPublisher(index_dir=root)
            runtime_dense = FaissLikeIndex(index_dir=root)
            runtime_sparse = SudachiBM25Retriever(index_dir=root)

            staging_one = publisher.staging_dir("run-one")
            chunks_one = [
                Chunk(
                    id="old",
                    document_id="doc-old",
                    text="alpha release",
                    index=0,
                    metadata={},
                )
            ]
            FaissLikeIndex(index_dir=staging_one).build(
                chunks=chunks_one,
                embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            )
            SudachiBM25Retriever(index_dir=staging_one).build(chunks_one)
            publisher.publish(run_id="run-one", staging_dir=staging_one)

            self.assertEqual(
                runtime_dense.search(
                    query_vector=np.array([1.0, 0.0], dtype=np.float32),
                    top_k=1,
                )[0].chunk.id,
                "old",
            )
            self.assertEqual(runtime_sparse.search("alpha", top_k=1)[0].id, "old")

            staging_two = publisher.staging_dir("run-two")
            chunks_two = [
                Chunk(
                    id="new",
                    document_id="doc-new",
                    text="beta release",
                    index=0,
                    metadata={},
                )
            ]
            FaissLikeIndex(index_dir=staging_two).build(
                chunks=chunks_two,
                embeddings=np.array([[0.0, 1.0]], dtype=np.float32),
            )
            SudachiBM25Retriever(index_dir=staging_two).build(chunks_two)
            publisher.publish(run_id="run-two", staging_dir=staging_two)

            self.assertEqual(
                runtime_dense.search(
                    query_vector=np.array([0.0, 1.0], dtype=np.float32),
                    top_k=1,
                )[0].chunk.id,
                "new",
            )
            self.assertEqual(runtime_sparse.search("beta", top_k=1)[0].id, "new")

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
            self.assertNotIn("running", [run.status for run in operations.runs])
            self.assertTrue((resolve_current_index_dir(config.app.index_dir) / "dense_chunks.jsonl").exists())
            self.assertEqual(operations.runs[-1].metadata["source_results"][0]["changed"], 1)
            self.assertEqual(build.compacted, [("cache-key-1",)])
            self.assertEqual(
                operations.runs[-1].metadata["stage_results"]["embedding"][
                    "cache_compaction"
                ]["status"],
                "succeeded",
            )

    def test_auto_update_does_not_rollback_published_index_when_side_effect_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            operations = _Operations()
            build = _BuildWithFailingStagedCommit()
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
            self.assertIsNotNone(build.commit_index_dir)
            self.assertEqual(resolve_current_index_dir(config.app.index_dir), build.commit_index_dir)
            self.assertTrue(
                (resolve_current_index_dir(config.app.index_dir) / "dense_chunks.jsonl").exists()
            )
            self.assertTrue(operations.runs[-1].metadata["degraded"])
            self.assertEqual(
                operations.runs[-1].metadata["stage_results"]["staged_side_effect_commit"]["status"],
                "failed",
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
                            "updated_at": datetime.now(UTC).isoformat(),
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
            self.assertEqual(
                operations.runs[-1].metadata["workflow_extraction"]["event"]["candidate_count"],
                1,
            )

    def test_auto_update_uses_unified_task_and_event_delta_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            config.task_management = SimpleNamespace(auto_extract_after_index_update=True)
            operations = _Operations()
            task_extractor = _TaskDeltaExtractor()
            event_extractor = _EventDeltaExtractor()
            chunk_source = _EventDeltaChunkSource(
                [
                    Chunk(
                        id="chunk-1",
                        document_id="doc-1",
                        text="5月5日に新歓会を部室で開催します。新歓資料を作成します。",
                        index=0,
                        metadata={
                            "source_item_id": "source-1",
                            "source_kind": "discord",
                            "updated_at": datetime.now(UTC).isoformat(),
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
                task_delta_extractor=task_extractor,
                event_delta_extractor=event_extractor,
                event_delta_chunk_source=chunk_source,
            ).execute(AutoIndexUpdateRequest(trigger="manual"))

            self.assertEqual(result.status, "succeeded")
            self.assertEqual(len(task_extractor.calls), 1)
            self.assertEqual(len(event_extractor.calls), 1)
            extraction = operations.runs[-1].metadata["workflow_extraction"]
            self.assertEqual(extraction["task"]["candidate_count"], 1)
            self.assertEqual(extraction["task"]["change_candidate_count"], 1)
            self.assertEqual(
                extraction["task"]["metadata"]["schema_version"],
                "workflow_extraction.v1",
            )
            self.assertEqual(extraction["event"]["candidate_count"], 1)

    def test_auto_update_filters_workflow_extraction_to_recent_timestamped_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            config.task_management = SimpleNamespace(auto_extract_after_index_update=True)
            operations = _Operations()
            task_extractor = _TaskDeltaExtractor()
            event_extractor = _EventDeltaExtractor()
            now = datetime.now(UTC)
            old = now - timedelta(days=2)
            chunk_source = _EventDeltaChunkSource(
                [
                    Chunk(
                        id="chunk-recent",
                        document_id="doc-recent",
                        text="最近の新歓会を開催します。最近の資料を作ります。",
                        index=0,
                        metadata={
                            "source_item_id": "source-recent",
                            "source_kind": "discord",
                            "updated_at": now.isoformat(),
                        },
                    ),
                    Chunk(
                        id="chunk-old",
                        document_id="doc-old",
                        text="古いイベントを開催します。",
                        index=0,
                        metadata={
                            "source_item_id": "source-old",
                            "source_kind": "discord",
                            "updated_at": old.isoformat(),
                        },
                    ),
                    Chunk(
                        id="chunk-missing-time",
                        document_id="doc-missing-time",
                        text="時刻不明のイベントを開催します。",
                        index=0,
                        metadata={
                            "source_item_id": "source-missing",
                            "source_kind": "discord",
                        },
                    ),
                ]
            )

            result = AutoIndexUpdateUsecase(
                config=config,
                build_usecase=_Build(),
                operations=operations,
                ingestion_service=_Ingestion(
                    IngestionResult(
                        source_kind="discord",
                        seen=3,
                        changed=3,
                        skipped=0,
                        deleted=0,
                        documents=3,
                        chunks=3,
                        secret_findings=0,
                    )
                ),
                task_delta_extractor=task_extractor,
                event_delta_extractor=event_extractor,
                event_delta_chunk_source=chunk_source,
            ).execute(AutoIndexUpdateRequest(trigger="manual"))

            self.assertEqual(result.status, "succeeded")
            self.assertEqual(len(task_extractor.calls), 1)
            self.assertEqual(len(event_extractor.calls), 1)
            self.assertIn("最近の新歓会", task_extractor.calls[0]["text"])
            self.assertNotIn("古いイベント", task_extractor.calls[0]["text"])
            self.assertNotIn("時刻不明", task_extractor.calls[0]["text"])
            self.assertEqual(task_extractor.calls[0]["evidence"][0].chunk_id, "chunk-recent")
            extraction = operations.runs[-1].metadata["workflow_extraction"]["task"]
            self.assertEqual(extraction["lookback_days"], 1)
            self.assertEqual(extraction["selected_chunks"], 1)
            self.assertEqual(extraction["excluded_older_chunks"], 1)
            self.assertEqual(extraction["excluded_missing_timestamp_chunks"], 1)
            self.assertEqual(extraction["metadata"]["selected_chunks"], 1)

    def test_auto_update_skips_workflow_extraction_when_recent_chunks_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            config.task_management = SimpleNamespace(auto_extract_after_index_update=True)
            operations = _Operations()
            task_extractor = _TaskDeltaExtractor()
            event_extractor = _EventDeltaExtractor()
            old = datetime.now(UTC) - timedelta(days=2)
            chunk_source = _EventDeltaChunkSource(
                [
                    Chunk(
                        id="chunk-old",
                        document_id="doc-old",
                        text="古い資料です。",
                        index=0,
                        metadata={
                            "source_item_id": "source-old",
                            "source_kind": "discord",
                            "updated_at": old.isoformat(),
                        },
                    ),
                    Chunk(
                        id="chunk-missing-time",
                        document_id="doc-missing-time",
                        text="時刻不明の資料です。",
                        index=0,
                        metadata={"source_item_id": "source-missing", "source_kind": "discord"},
                    ),
                ]
            )

            result = AutoIndexUpdateUsecase(
                config=config,
                build_usecase=_Build(),
                operations=operations,
                ingestion_service=_Ingestion(
                    IngestionResult(
                        source_kind="discord",
                        seen=2,
                        changed=2,
                        skipped=0,
                        deleted=0,
                        documents=2,
                        chunks=2,
                        secret_findings=0,
                    )
                ),
                task_delta_extractor=task_extractor,
                event_delta_extractor=event_extractor,
                event_delta_chunk_source=chunk_source,
            ).execute(AutoIndexUpdateRequest(trigger="manual"))

            self.assertEqual(result.status, "succeeded")
            self.assertEqual(task_extractor.calls, [])
            self.assertEqual(event_extractor.calls, [])
            extraction = operations.runs[-1].metadata["workflow_extraction"]
            self.assertEqual(extraction["task"]["status"], "skipped")
            self.assertEqual(extraction["task"]["reason"], "no_recent_chunks")
            self.assertEqual(extraction["task"]["selected_chunks"], 0)
            self.assertEqual(extraction["task"]["excluded_older_chunks"], 1)
            self.assertEqual(extraction["task"]["excluded_missing_timestamp_chunks"], 1)
            self.assertEqual(extraction["event"]["status"], "skipped")

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
            self.assertTrue(
                (
                    resolve_current_index_dir(config.app.index_dir)
                    / "member_profiles"
                    / "guild-1.marker"
                ).exists()
            )
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
            repo = FileIngestionRepository(root, auto_compact=False)
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
            repo = FileIngestionRepository(root, auto_compact=False)
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

    def test_file_ingestion_repository_compacts_append_only_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = FileIngestionRepository(root, auto_compact=False)
            raw_old = SourceRawItem(
                source_kind="crafters_colony",
                external_id="12345",
                title="Old",
                text="old body",
                checksum="old-checksum",
                metadata={
                    "crafters_colony_article_id": "12345",
                    "crafters_colony_updated_at": "2024-01-01T00:00:00+09:00",
                },
                access_scope=AccessScope(visibility="public"),
            )
            raw_new = SourceRawItem(
                source_kind="crafters_colony",
                external_id="12345",
                title="New",
                text="new body",
                checksum="new-checksum",
                metadata={
                    "crafters_colony_article_id": "12345",
                    "crafters_colony_updated_at": "2024-01-02T00:00:00+09:00",
                },
                access_scope=AccessScope(visibility="public"),
            )
            source_item_id = _source_item_id(raw_new)
            repo.save_item(
                raw=raw_old,
                document=_doc_with_id(raw_old, document_id="doc-old"),
                chunks=[
                    Chunk(
                        id="chunk-old",
                        document_id="doc-old",
                        text="old body",
                        index=0,
                        metadata={
                            "source_item_id": source_item_id,
                            "source_kind": "crafters_colony",
                            "index_status": "active",
                        },
                    )
                ],
                findings=[],
                raw_object_key="raw/old",
            )
            repo.save_item(
                raw=raw_new,
                document=_doc_with_id(raw_new, document_id="doc-new"),
                chunks=[
                    Chunk(
                        id="chunk-new",
                        document_id="doc-new",
                        text="new body",
                        index=0,
                        metadata={
                            "source_item_id": source_item_id,
                            "source_kind": "crafters_colony",
                            "index_status": "active",
                        },
                    )
                ],
                findings=[],
                raw_object_key="raw/new",
            )

            self.assertEqual(
                len(
                    (root / "source_items.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                2,
            )
            self.assertEqual(repo.load_active_chunks()[0].text, "new body")

            result = repo.compact_history(source_kinds=("crafters_colony",))

            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["active_source_items"], 1)
            self.assertEqual(result["active_documents"], 1)
            self.assertEqual(result["active_chunks"], 1)
            self.assertEqual(
                len(
                    (root / "source_items.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )
            self.assertEqual(
                len(
                    (root / "documents.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )
            self.assertEqual(
                len(
                    (root / "chunks.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )
            active_chunks = repo.load_active_chunks(source_kinds=("crafters_colony",))
            self.assertEqual(len(active_chunks), 1)
            self.assertEqual(active_chunks[0].text, "new body")
            current_source_items = (root / "current_source_items.jsonl").read_text(
                encoding="utf-8"
            )
            current_documents = (root / "current_documents.jsonl").read_text(
                encoding="utf-8"
            )
            quality_report = json.loads(
                (root / "ingestion_quality_report.json").read_text(encoding="utf-8")
            )
            document_payload = json.loads(current_documents.splitlines()[0])
            self.assertEqual(len(current_source_items.splitlines()), 1)
            self.assertEqual(document_payload["source_kind"], "crafters_colony")
            self.assertEqual(document_payload["source_type"], "crafters_colony")
            before_source_items = quality_report["before_compaction"]["files"][
                "source_items.jsonl"
            ]
            self.assertEqual(before_source_items["duplicate_rows"], 1)

    def test_file_ingestion_repository_auto_compacts_after_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = FileIngestionRepository(root)
            raw_old = SourceRawItem(
                source_kind="hatenablog",
                external_id="entry-1",
                title="Old",
                text="old body",
                checksum="old-checksum",
                metadata={"hatenablog_entry_id": "entry-1"},
                access_scope=AccessScope(visibility="public"),
            )
            raw_new = SourceRawItem(
                source_kind="hatenablog",
                external_id="entry-1",
                title="New",
                text="new body",
                checksum="new-checksum",
                metadata={"hatenablog_entry_id": "entry-1"},
                access_scope=AccessScope(visibility="public"),
            )
            source_item_id = _source_item_id(raw_new)
            repo.save_item(
                raw=raw_old,
                document=_doc_with_id(raw_old, document_id="doc-old"),
                chunks=[
                    Chunk(
                        id="chunk-old",
                        document_id="doc-old",
                        text="old body",
                        index=0,
                        metadata={
                            "source_item_id": source_item_id,
                            "source_kind": "hatenablog",
                            "source_type": "hatenablog",
                            "index_status": "active",
                            "access_scope": {"visibility": "public"},
                        },
                    )
                ],
                findings=[],
                raw_object_key="raw/old",
            )
            repo.save_item(
                raw=raw_new,
                document=_doc_with_id(raw_new, document_id="doc-new"),
                chunks=[
                    Chunk(
                        id="chunk-new",
                        document_id="doc-new",
                        text="new body",
                        index=0,
                        metadata={
                            "source_item_id": source_item_id,
                            "source_kind": "hatenablog",
                            "source_type": "hatenablog",
                            "index_status": "active",
                            "access_scope": {"visibility": "public"},
                        },
                    )
                ],
                findings=[],
                raw_object_key="raw/new",
            )

            self.assertEqual(
                len(
                    (root / "source_items.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )
            self.assertEqual(
                len(
                    (root / "documents.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )
            self.assertEqual(
                len(
                    (root / "chunks.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )
            self.assertEqual(
                repo.load_active_chunks(source_kinds=("hatenablog",))[0].text,
                "new body",
            )
            self.assertEqual(
                len(
                    (root / "current_chunk_acl_entries.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )
            quality_report = json.loads(
                (root / "ingestion_quality_report.json").read_text(encoding="utf-8")
            )
            source_items_before = quality_report["before_compaction"]["files"][
                "source_items.jsonl"
            ]["by_source"]["hatenablog"]
            self.assertEqual(source_items_before["duplicate_rows"], 1)

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


def _doc_with_id(raw: SourceRawItem, *, document_id: str) -> NormalizedDocument:
    return NormalizedDocument(
        id=document_id,
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
