from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import logging
import os
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.document import Document
from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.features.indexing.embedding_cache import (
    IndexEmbeddingCache,
    IndexEmbeddingCacheKey,
    IndexEmbeddingRecord,
)
from kumc_agent.infra.ingestion.repository import IngestionRepository
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever
from kumc_agent.infra.storage.filesystem import FileSystemStorage
from kumc_agent.utils.hashing import stable_hash

KEYWORD_CORPUS_SPARSE = "sparse"
KEYWORD_CORPUS_SPARSE_SECOND_REC = "sparse_second_rec"
KEYWORD_CORPUS_SECOND_REC_SPARSE = "second_rec_sparse"
KEYWORD_CORPUS_MINECRAFT_WIKI_SPARSE = "minecraft_wiki_sparse"
KEYWORD_CORPUS_MINECRAFT_WIKI_SPARSE_SECOND_REC = "minecraft_wiki_sparse_second_rec"
KEYWORD_CORPUS_MINECRAFT_WIKI_SECOND_REC_SPARSE = "minecraft_wiki_second_rec_sparse"
KEYWORD_CORPUS_MATERIAL_NAMES = "material_names"
_MATERIAL_NAME_INDEX_EXCLUDED_SOURCE_TYPES = frozenset(
    {"messages", "discord_message", "discord", "x_posts", "x"}
)

logger = logging.getLogger(__name__)


def _stable_json_hash(value: object) -> str:
    return stable_hash(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )


@dataclass(frozen=True)
class IndexBuildResult:
    loaded_sources: int
    documents: int
    chunks: int
    index_dir: Path
    stage_results: dict[str, object] | None = None
    embedding_cache_keys: tuple[IndexEmbeddingCacheKey, ...] = tuple()


@dataclass(frozen=True)
class _RepositoryIndexArtifacts:
    first_chunks: list[Chunk]
    second_chunks: list[Chunk]
    sparse_chunks: list[Chunk]
    summary_chunks: list[Chunk]
    index_chunks: list[Chunk]


@dataclass(frozen=True)
class _DenseEmbeddingBuildResult:
    matrix: np.ndarray
    metadata: dict[str, object]
    cache_keys: tuple[IndexEmbeddingCacheKey, ...] = tuple()


class IndexingService:
    def __init__(
        self,
        *,
        storage: FileSystemStorage,
        embedder: EmbedderPort,
        faiss_index: FaissLikeIndex,
        bm25_index: SudachiBM25Retriever,
        ingestion_dir: Path,
        app_config: RuntimeConfig,
        summary_llm: LLMPort | None = None,
        minecraft_wiki_summary_llm: LLMPort | None = None,
        image_asset_builder: object | None = None,
        ingestion_repository: IngestionRepository | None = None,
        embedding_cache: IndexEmbeddingCache | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._faiss_index = faiss_index
        self._bm25_index = bm25_index
        self._ingestion_dir = ingestion_dir
        self._runtime = app_config
        self._summary_llm = summary_llm
        self._minecraft_wiki_summary_llm = minecraft_wiki_summary_llm
        self._image_asset_builder = image_asset_builder
        self._ingestion_repository = ingestion_repository
        self._embedding_cache = embedding_cache
        self._chunks_root = self._runtime.app.data_dir / "chunks"
        self._first_rec_dir = self._chunks_root / "first_rec_chunk"
        self._second_rec_dir = self._chunks_root / "second_rec_chunk"
        self._sparse_second_rec_dir = self._chunks_root / "sparse_second_rec_chunk"
        self._summary_dir = self._chunks_root / "summary_chunk"
        self._ingestion_docs_dir = self._ingestion_dir / "docs"
        self._ingestion_docs_normalized_dir = self._ingestion_dir / "docs_normalized"
        self._ingestion_sheets_dir = self._ingestion_dir / "sheets"
        self._ingestion_sheets_structured_dir = self._ingestion_dir / "sheets_structured"
        self._ingestion_messages_dir = self._ingestion_dir / "messages"
        self._ingestion_x_dir = self._ingestion_dir / "x"
        self._ingestion_vc_dir = self._ingestion_dir / "vc"
        self._ingestion_hatenablog_dir = self._ingestion_dir / "hatenablog"
        self._ingestion_crafters_colony_dir = self._ingestion_dir / "crafters_colony"
        self._ingestion_notion_dir = self._ingestion_dir / "notion"
        self._ingestion_minecraft_wiki_dir = self._ingestion_dir / "minecraft_wiki"

        self._first_rec_docs_dir = self._first_rec_dir / "docs"
        self._first_rec_sheets_dir = self._first_rec_dir / "sheets"
        self._first_rec_messages_dir = self._first_rec_dir / "messages"
        self._first_rec_x_dir = self._first_rec_dir / "x"
        self._first_rec_hatenablog_dir = self._first_rec_dir / "hatenablog"
        self._first_rec_crafters_colony_dir = self._first_rec_dir / "crafters_colony"
        self._first_rec_notion_dir = self._first_rec_dir / "notion"
        self._first_rec_minecraft_wiki_dir = self._first_rec_dir / "minecraft_wiki"

        self._second_rec_docs_dir = self._second_rec_dir / "docs"
        self._second_rec_sheets_dir = self._second_rec_dir / "sheets"
        self._second_rec_messages_dir = self._second_rec_dir / "messages"
        self._second_rec_x_dir = self._second_rec_dir / "x"
        self._second_rec_vc_dir = self._second_rec_dir / "vc"
        self._second_rec_hatenablog_dir = self._second_rec_dir / "hatenablog"
        self._second_rec_crafters_colony_dir = self._second_rec_dir / "crafters_colony"
        self._second_rec_notion_dir = self._second_rec_dir / "notion"
        self._second_rec_minecraft_wiki_dir = self._second_rec_dir / "minecraft_wiki"

        self._sparse_second_rec_docs_dir = self._sparse_second_rec_dir / "docs"
        self._sparse_second_rec_sheets_dir = self._sparse_second_rec_dir / "sheets"
        self._sparse_second_rec_messages_dir = self._sparse_second_rec_dir / "messages"
        self._sparse_second_rec_x_dir = self._sparse_second_rec_dir / "x"
        self._sparse_second_rec_vc_dir = self._sparse_second_rec_dir / "vc"
        self._sparse_second_rec_hatenablog_dir = self._sparse_second_rec_dir / "hatenablog"
        self._sparse_second_rec_crafters_colony_dir = (
            self._sparse_second_rec_dir / "crafters_colony"
        )
        self._sparse_second_rec_notion_dir = self._sparse_second_rec_dir / "notion"
        self._sparse_second_rec_minecraft_wiki_dir = (
            self._sparse_second_rec_dir / "minecraft_wiki"
        )

        self._summary_docs_dir = self._summary_dir / "docs"
        self._summary_sheets_dir = self._summary_dir / "sheets"
        self._summary_messages_dir = self._summary_dir / "messages"
        self._summary_x_dir = self._summary_dir / "x"
        self._summary_hatenablog_dir = self._summary_dir / "hatenablog"
        self._summary_crafters_colony_dir = self._summary_dir / "crafters_colony"
        self._summary_notion_dir = self._summary_dir / "notion"
        self._summary_minecraft_wiki_dir = self._summary_dir / "minecraft_wiki"


    def build(
        self,
        *,
        loaded_sources: int,
        full_rebuild: bool = False,
        stage_selection: tuple[str, ...] | None = None,
        allow_cancel: bool = False,
        cancel_event: threading.Event | None = None,
        index_dir: Path | None = None,
        prefer_ingestion_repository: bool = False,
    ) -> IndexBuildResult:
        if index_dir is not None and index_dir != self._runtime.app.index_dir:
            return self._build_with_index_dir(
                loaded_sources=loaded_sources,
                full_rebuild=full_rebuild,
                stage_selection=stage_selection,
                allow_cancel=allow_cancel,
                cancel_event=cancel_event,
                index_dir=index_dir,
                prefer_ingestion_repository=prefer_ingestion_repository,
            )
        selected = {
            value.strip()
            for value in (stage_selection or ())
            if value and value.strip()
        }

        self._apply_clear_flags(full_rebuild=full_rebuild)
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)
        self._ensure_ingestion_source_dirs()

        documents = self._parse_documents_from_ingestion()
        self._storage.save_documents(documents)

        legacy_cfg = self._build_legacy_app_config()
        minecraft_wiki_cfg = self._build_minecraft_wiki_app_config()
        self._ensure_legacy_prompt_env_defaults()
        repository_chunks = (
            self._ingestion_repository.load_active_chunks()
            if prefer_ingestion_repository and self._ingestion_repository is not None
            else []
        )
        repository_artifacts: _RepositoryIndexArtifacts | None = None
        if repository_chunks:
            repository_chunks = [
                chunk
                for chunk in repository_chunks
                if not self._is_minecraft_wiki_chunk(chunk)
            ]
            repository_artifacts = self._build_repository_index_artifacts(
                repository_chunks=repository_chunks,
                legacy_cfg=legacy_cfg,
                selected=selected,
            )
            self._run_minecraft_wiki_chunk_pipeline(
                minecraft_wiki_cfg=minecraft_wiki_cfg,
                selected=selected,
                allow_cancel=allow_cancel,
                cancel_event=cancel_event,
            )
            self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)
            minecraft_wiki_artifacts = self._load_minecraft_wiki_stage_artifacts(
                legacy_cfg=legacy_cfg,
            )
            index_chunks = [
                *repository_artifacts.index_chunks,
                *minecraft_wiki_artifacts.index_chunks,
            ]
        else:
            self._run_legacy_chunk_pipeline(
                legacy_cfg=legacy_cfg,
                selected=selected,
                allow_cancel=allow_cancel,
                cancel_event=cancel_event,
            )
            self._run_minecraft_wiki_chunk_pipeline(
                minecraft_wiki_cfg=minecraft_wiki_cfg,
                selected=selected,
                allow_cancel=allow_cancel,
                cancel_event=cancel_event,
            )
            self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)
            minecraft_wiki_artifacts = None
            index_chunks = self._load_index_chunks_from_legacy_dirs(
                legacy_cfg=legacy_cfg
            )
        docs_quality_payload = self._docs_quality_payload(index_chunks=index_chunks)
        minecraft_wiki_quality_payload = self._minecraft_wiki_quality_payload(
            index_chunks=index_chunks
        )
        if not bool(docs_quality_payload.get("can_continue", True)):
            metadata = docs_quality_payload.get("metadata")
            failures: list[str] = []
            if isinstance(metadata, dict):
                raw_failures = metadata.get("critical_failures")
                if isinstance(raw_failures, list):
                    failures = [str(item) for item in raw_failures]
            raise RuntimeError(
                "Docs quality gate failed: "
                + (", ".join(failures) if failures else "unknown")
            )
        from kumc_agent.usecases.indexing.sheets_quality import (
            build_sheets_quality_payload,
        )

        sheets_quality_cfg = self._runtime.indexing.sheets_quality
        sheets_quality_payload = build_sheets_quality_payload(
            sheets_dir=self._ingestion_sheets_dir,
            structured_sheets_dir=self._ingestion_sheets_structured_dir,
            fail_fast=sheets_quality_cfg.fail_fast,
            max_empty_row_ratio=sheets_quality_cfg.max_empty_row_ratio,
            min_non_empty_cells=sheets_quality_cfg.min_non_empty_cells,
        )
        if not bool(sheets_quality_payload.get("can_continue", True)):
            metadata = sheets_quality_payload.get("metadata")
            warnings: list[str] = []
            if isinstance(metadata, dict):
                raw_warnings = metadata.get("warnings")
                if isinstance(raw_warnings, list):
                    warnings = [str(item) for item in raw_warnings]
            raise RuntimeError(
                "Sheets quality gate failed: "
                + (", ".join(warnings) if warnings else "unknown")
            )
        if (
            minecraft_wiki_quality_payload is not None
            and not bool(minecraft_wiki_quality_payload.get("can_continue", True))
        ):
            metadata = minecraft_wiki_quality_payload.get("metadata")
            failures = []
            if isinstance(metadata, dict):
                raw_failures = metadata.get("critical_failures")
                if isinstance(raw_failures, list):
                    failures = [str(item) for item in raw_failures]
            raise RuntimeError(
                "Minecraft Wiki quality gate failed: "
                + (", ".join(failures) if failures else "unknown")
            )
        self._storage.save_chunks(index_chunks)

        dense_embeddings = self._embed_index_chunks(
            index_chunks=index_chunks,
            full_rebuild=full_rebuild,
        )
        self._faiss_index.build(chunks=index_chunks, embeddings=dense_embeddings.matrix)
        self._bm25_index.build(index_chunks)

        if repository_artifacts is not None:
            self._build_material_catalog_from_repository_chunks(
                chunks=repository_artifacts.first_chunks or repository_artifacts.second_chunks
            )
            self._build_keyword_inverted_indexes_from_repository_artifacts(
                artifacts=repository_artifacts,
                legacy_cfg=legacy_cfg,
            )
            self._build_minecraft_wiki_keyword_inverted_indexes(
                minecraft_wiki_cfg=minecraft_wiki_cfg,
                artifacts=minecraft_wiki_artifacts,
            )
        else:
            self._build_material_catalog_legacy(legacy_cfg=legacy_cfg)
            self._build_keyword_inverted_indexes(legacy_cfg=legacy_cfg)
            self._build_minecraft_wiki_keyword_inverted_indexes(
                minecraft_wiki_cfg=minecraft_wiki_cfg,
                artifacts=None,
            )
        self._build_material_name_keyword_index(legacy_cfg=legacy_cfg)
        stage_results: dict[str, object] = {
            "source_of_chunks": (
                "ingestion_repository_plus_minecraft_wiki_ingestion_chunk_pipeline"
                if repository_artifacts is not None
                else "raw_chunk_pipeline"
            )
        }
        stage_results["docs_quality"] = docs_quality_payload
        if minecraft_wiki_quality_payload is not None:
            stage_results["minecraft_wiki_quality"] = minecraft_wiki_quality_payload
        stage_results["sheets_quality"] = sheets_quality_payload
        stage_results["embedding"] = dense_embeddings.metadata
        if self._image_asset_builder is not None:
            build_from_ingestion_sources = getattr(
                self._image_asset_builder,
                "build_from_ingestion_sources",
                None,
            )
            if callable(build_from_ingestion_sources):
                image_run = build_from_ingestion_sources(
                    index_dir=self._runtime.app.index_dir,
                    commit_repository=False,
                )
                stage_results["image"] = getattr(image_run, "__dict__", {"status": "unknown"})

        return IndexBuildResult(
            loaded_sources=loaded_sources,
            documents=len(documents),
            chunks=len(index_chunks),
            index_dir=self._faiss_index._index_dir,  # noqa: SLF001
            stage_results=stage_results,
            embedding_cache_keys=dense_embeddings.cache_keys,
        )

    def compact_embedding_cache(
        self,
        active_keys: tuple[IndexEmbeddingCacheKey, ...],
    ) -> dict[str, object]:
        if self._embedding_cache is None:
            return {"status": "skipped", "reason": "cache_not_configured"}
        if not self._embedding_cache_enabled():
            return {"status": "skipped", "reason": "cache_disabled"}
        return self._embedding_cache.compact(active_keys)

    def commit_staged_side_effects(self, index_dir: Path) -> dict[str, object]:
        results: dict[str, object] = {}
        if self._image_asset_builder is not None:
            commit = getattr(self._image_asset_builder, "commit_staged_assets", None)
            if callable(commit):
                results["image"] = commit(index_dir=index_dir)
        return results

    def update(
        self,
        *,
        loaded_sources: int,
        full_rebuild: bool = False,
        stage_selection: tuple[str, ...] | None = None,
        allow_cancel: bool = False,
        cancel_event: threading.Event | None = None,
        index_dir: Path | None = None,
        prefer_ingestion_repository: bool = False,
    ) -> IndexBuildResult:
        return self.build(
            loaded_sources=loaded_sources,
            full_rebuild=full_rebuild,
            stage_selection=stage_selection,
            allow_cancel=allow_cancel,
            cancel_event=cancel_event,
            index_dir=index_dir,
            prefer_ingestion_repository=prefer_ingestion_repository,
        )

    def _embed_index_chunks(
        self,
        *,
        index_chunks: list[Chunk],
        full_rebuild: bool,
    ) -> _DenseEmbeddingBuildResult:
        embedding_section = getattr(getattr(self._runtime, "providers", None), "embeddings", None)
        provider = str(getattr(embedding_section, "provider", "unknown") or "unknown")
        model = str(getattr(embedding_section, "model", "unknown") or "unknown")
        configured_dimensions = int(getattr(embedding_section, "dimensions", 0) or 0)
        dense_texts = [self._chunk_embedding_text_for_dense(chunk) for chunk in index_chunks]
        text_hashes = [stable_hash(text) for text in dense_texts]
        cache_enabled = (
            self._embedding_cache is not None
            and self._embedding_cache_enabled()
            and configured_dimensions > 0
        )
        force_reembed = bool(
            full_rebuild and self._embedding_cache_force_reembed_on_full_rebuild()
        )
        if not cache_enabled:
            matrix = self._embed_texts(
                dense_texts,
                expected_rows=len(index_chunks),
                expected_dimensions=None,
            )
            dimensions = int(matrix.shape[1]) if matrix.ndim == 2 else configured_dimensions
            manifest_path = self._write_dense_embedding_manifest(
                chunks=index_chunks,
                text_hashes=text_hashes,
                provider=provider,
                model=model,
                dimensions=dimensions,
            )
            return _DenseEmbeddingBuildResult(
                matrix=matrix,
                metadata={
                    "enabled": False,
                    "reason": (
                        "cache_disabled"
                        if self._embedding_cache is not None
                        else "cache_not_configured"
                    ),
                    "total_chunks": len(index_chunks),
                    "embedded_chunks": len(index_chunks),
                    "reused_chunks": 0,
                    "cache_misses": len(index_chunks),
                    "cache_invalid": 0,
                    "provider": provider,
                    "model": model,
                    "dimensions": dimensions,
                    "manifest_path": str(manifest_path),
                },
            )

        assert self._embedding_cache is not None
        cached = self._embedding_cache.load(
            provider=provider,
            model=model,
            dimensions=configured_dimensions,
        )
        matrix_rows: list[np.ndarray | None] = [None] * len(index_chunks)
        cache_keys: list[IndexEmbeddingCacheKey] = []
        missing_indices: list[int] = []
        cache_misses = 0
        reused_chunks = 0
        for index, (chunk, text_hash) in enumerate(zip(index_chunks, text_hashes)):
            key = IndexEmbeddingCacheKey(
                provider=provider,
                model=model,
                dimensions=configured_dimensions,
                chunk_id=chunk.id,
                embedding_text_hash=text_hash,
            )
            cache_keys.append(key)
            record = cached.records.get(key)
            if record is not None and not force_reembed:
                matrix_rows[index] = np.asarray(record.vector, dtype=np.float32)
                reused_chunks += 1
                continue
            if record is None:
                cache_misses += 1
            missing_indices.append(index)

        records_to_save: list[IndexEmbeddingRecord] = []
        if missing_indices:
            new_vectors = self._embed_texts(
                [dense_texts[index] for index in missing_indices],
                expected_rows=len(missing_indices),
                expected_dimensions=configured_dimensions,
            )
            for row_index, chunk_index in enumerate(missing_indices):
                vector = new_vectors[row_index]
                matrix_rows[chunk_index] = vector
                chunk = index_chunks[chunk_index]
                records_to_save.append(
                    IndexEmbeddingRecord(
                        provider=provider,
                        model=model,
                        dimensions=configured_dimensions,
                        chunk_id=chunk.id,
                        embedding_text_hash=text_hashes[chunk_index],
                        vector=vector,
                        chunk_metadata_hash=_stable_json_hash(chunk.metadata),
                        source_kind=str(
                            chunk.metadata.get("source_kind")
                            or chunk.metadata.get("source_type")
                            or ""
                        ),
                        source_item_id=str(chunk.metadata.get("source_item_id") or ""),
                    )
                )
        if records_to_save:
            self._embedding_cache.save(records_to_save)

        if matrix_rows:
            missing_rows = [index for index, row in enumerate(matrix_rows) if row is None]
            if missing_rows:
                raise ValueError(f"missing embedding rows: {missing_rows[:5]}")
            matrix = np.vstack([np.asarray(row, dtype=np.float32) for row in matrix_rows if row is not None])
        else:
            matrix = np.empty((0, configured_dimensions), dtype=np.float32)
        manifest_path = self._write_dense_embedding_manifest(
            chunks=index_chunks,
            text_hashes=text_hashes,
            provider=provider,
            model=model,
            dimensions=configured_dimensions,
        )
        return _DenseEmbeddingBuildResult(
            matrix=matrix,
            metadata={
                "enabled": True,
                "total_chunks": len(index_chunks),
                "embedded_chunks": len(missing_indices),
                "reused_chunks": reused_chunks,
                "cache_misses": cache_misses,
                "cache_invalid": cached.invalid_records,
                "forced_reembed_chunks": len(missing_indices) if force_reembed else 0,
                "force_reembed": force_reembed,
                "provider": provider,
                "model": model,
                "dimensions": configured_dimensions,
                "manifest_path": str(manifest_path),
            },
            cache_keys=tuple(cache_keys),
        )

    def _embed_texts(
        self,
        texts: list[str],
        *,
        expected_rows: int,
        expected_dimensions: int | None,
    ) -> np.ndarray:
        if not texts:
            dimensions = max(0, int(expected_dimensions or 0))
            return np.empty((0, dimensions), dtype=np.float32)
        matrix = np.asarray(self._embedder.embed_documents(texts), dtype=np.float32)
        if matrix.ndim == 1 and expected_rows == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("embedding result must be 2D")
        if int(matrix.shape[0]) != int(expected_rows):
            raise ValueError("embedding row count mismatch")
        if expected_dimensions and int(matrix.shape[1]) != int(expected_dimensions):
            raise ValueError("embedding dimension mismatch")
        return matrix

    def _write_dense_embedding_manifest(
        self,
        *,
        chunks: list[Chunk],
        text_hashes: list[str],
        provider: str,
        model: str,
        dimensions: int,
    ) -> Path:
        path = self._runtime.app.index_dir / "dense_embedding_manifest.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fw:
            for chunk, text_hash in zip(chunks, text_hashes):
                metadata = dict(chunk.metadata or {})
                fw.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.id,
                            "embedding_text_hash": text_hash,
                            "provider": provider,
                            "model": model,
                            "dimensions": int(dimensions),
                            "chunk_metadata_hash": _stable_json_hash(metadata),
                            "source_kind": str(
                                metadata.get("source_kind")
                                or metadata.get("source_type")
                                or ""
                            ),
                            "source_item_id": str(metadata.get("source_item_id") or ""),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        return path

    def _embedding_cache_enabled(self) -> bool:
        cache_config = getattr(getattr(self._runtime, "indexing", None), "embedding_cache", None)
        return bool(getattr(cache_config, "enabled", False))

    def _embedding_cache_force_reembed_on_full_rebuild(self) -> bool:
        cache_config = getattr(getattr(self._runtime, "indexing", None), "embedding_cache", None)
        return bool(getattr(cache_config, "force_reembed_on_full_rebuild", True))

    def _build_with_index_dir(
        self,
        *,
        loaded_sources: int,
        full_rebuild: bool,
        stage_selection: tuple[str, ...] | None,
        allow_cancel: bool,
        cancel_event: threading.Event | None,
        index_dir: Path,
        prefer_ingestion_repository: bool,
    ) -> IndexBuildResult:
        from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
        from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever

        previous_runtime = self._runtime
        previous_faiss = self._faiss_index
        previous_bm25 = self._bm25_index
        index_dir.mkdir(parents=True, exist_ok=True)
        self._runtime = replace(
            self._runtime,
            app=replace(self._runtime.app, index_dir=index_dir),
        )
        self._faiss_index = FaissLikeIndex(index_dir=index_dir)
        self._bm25_index = SudachiBM25Retriever(
            index_dir=index_dir,
            sudachi_mode=self._runtime.features.retrieval.sudachi_mode,
            bm25_k1=self._runtime.features.retrieval.sparse_bm25_k1,
            bm25_b=self._runtime.features.retrieval.sparse_bm25_b,
            use_normalized_form=self._runtime.features.retrieval.sparse_use_normalized_form,
            remove_symbols=self._runtime.features.retrieval.sparse_remove_symbols,
        )
        try:
            return self.build(
                loaded_sources=loaded_sources,
                full_rebuild=full_rebuild,
                stage_selection=stage_selection,
                allow_cancel=allow_cancel,
                cancel_event=cancel_event,
                prefer_ingestion_repository=prefer_ingestion_repository,
            )
        finally:
            self._runtime = previous_runtime
            self._faiss_index = previous_faiss
            self._bm25_index = previous_bm25

    def _apply_clear_flags(self, *, full_rebuild: bool) -> None:
        refresh = self._runtime.indexing.refresh
        clear_all = full_rebuild
        if clear_all or refresh.clear_ingestion_source_data:
            self._clear_ingestion_source_contents()
        if clear_all or refresh.clear_first_recursive_chunk_data:
            self._clear_dir_contents(self._first_rec_dir)
        if clear_all or refresh.clear_second_recursive_chunk_data:
            self._clear_dir_contents(self._second_rec_dir)
            self._clear_dir_contents(self._sparse_second_rec_dir)
        if clear_all or refresh.clear_summary_chunk_data:
            self._clear_dir_contents(self._summary_dir)

    def _ensure_ingestion_source_dirs(self) -> None:
        for path in (
            self._ingestion_docs_dir,
            self._ingestion_docs_normalized_dir,
            self._ingestion_sheets_dir,
            self._ingestion_sheets_structured_dir,
            self._ingestion_messages_dir,
            self._ingestion_x_dir,
            self._ingestion_vc_dir,
            self._ingestion_hatenablog_dir,
            self._ingestion_crafters_colony_dir,
            self._ingestion_notion_dir,
            self._ingestion_minecraft_wiki_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _build_legacy_app_config(self):
        from kumc_agent.infra.indexing.config import AppConfig as LegacyAppConfig

        chunking = self._runtime.indexing.chunking
        stages = self._runtime.indexing.stages
        refresh = self._runtime.indexing.refresh
        providers_llm = self._runtime.providers.llm
        retrieval = self._runtime.features.retrieval
        integrations = self._runtime.integrations

        return LegacyAppConfig(
            base_dir=self._runtime.base_dir,
            ingestion_data_dir=self._ingestion_dir,
            first_rec_chunk_dir=self._first_rec_dir,
            second_rec_chunk_dir=self._second_rec_dir,
            sparse_second_rec_chunk_dir=self._sparse_second_rec_dir,
            summery_chunk_dir=self._summary_dir,
            index_dir=self._runtime.app.index_dir,
            discord_bot_token=self._runtime.integrations.discord.bot_token,
            discord_guild_allow_list=tuple(self._runtime.security.discord_guild_allow_list),
            gemini_api_key=integrations.gemini_api_key,
            gemini_requests_per_minute=integrations.gemini_requests_per_minute,
            gemini_summary_requests_per_minute=(
                integrations.gemini_summary_requests_per_minute
            ),
            drive_folder_id=integrations.drive.folder_id,
            google_application_credentials=(
                integrations.drive.google_application_credentials
            ),
            drive_max_files=integrations.drive.max_files,
            crafters_colony_author_url=integrations.crafters_colony.author_url,
            crafters_colony_max_pages=integrations.crafters_colony.max_pages,
            crafters_colony_max_articles=integrations.crafters_colony.max_articles,
            pdf_ocr_model_path=integrations.drive.pdf_ocr_model_path,
            embedding_model=self._runtime.providers.embeddings.model,
            first_rec_chunk_size=chunking.first_recursive_chunk_size,
            first_rec_chunk_overlap=chunking.first_recursive_chunk_overlap,
            second_rec_enabled=stages.second_recursive_enabled,
            second_rec_chunk_size=chunking.second_recursive_chunk_size,
            second_rec_chunk_overlap=chunking.second_recursive_chunk_overlap,
            summery_enabled=stages.summary_enabled,
            summery_characters=chunking.summary_characters,
            summery_provider=chunking.summary_llm_provider,
            summery_gemini_model=chunking.summary_gemini_model,
            summery_temperature=chunking.summary_temperature,
            summery_max_output_tokens=chunking.summary_max_output_tokens,
            summery_max_retries=2,
            summery_batch_size=chunking.summary_batch_size,
            llm_provider=providers_llm.provider,
            genai_model=providers_llm.gemini_model,
            temperature=providers_llm.temperature,
            max_output_tokens=providers_llm.max_output_tokens,
            clear_ingestion_source_data=refresh.clear_ingestion_source_data,
            clear_first_rec_chunk_data=refresh.clear_first_recursive_chunk_data,
            clear_second_rec_chunk_data=refresh.clear_second_recursive_chunk_data,
            clear_summery_chunk_data=refresh.clear_summary_chunk_data,
            update_ingestion_source_data=refresh.update_ingestion_source_data,
            update_first_rec_chunk_data=refresh.update_first_recursive_chunk_data,
            update_second_rec_chunk_data=refresh.update_second_recursive_chunk_data,
            update_sparse_second_rec_chunk_data=(
                refresh.update_sparse_second_recursive_chunk_data
            ),
            update_summery_chunk_data=refresh.update_summary_chunk_data,
            sudachi_mode=retrieval.sudachi_mode,
            sparse_bm25_k1=retrieval.sparse_bm25_k1,
            sparse_bm25_b=retrieval.sparse_bm25_b,
            sparse_use_normalized_form=retrieval.sparse_use_normalized_form,
            sparse_remove_symbols=retrieval.sparse_remove_symbols,
        )

    def _build_minecraft_wiki_app_config(self):
        from kumc_agent.infra.indexing.config import AppConfig as LegacyAppConfig

        chunking = self._runtime.minecraft_wiki_rag.chunking
        retrieval = self._runtime.minecraft_wiki_rag.retrieval
        refresh = self._runtime.indexing.refresh
        integrations = self._runtime.integrations
        providers_llm = self._runtime.providers.llm

        return LegacyAppConfig(
            base_dir=self._runtime.base_dir,
            ingestion_data_dir=self._ingestion_dir,
            first_rec_chunk_dir=self._first_rec_dir,
            second_rec_chunk_dir=self._second_rec_dir,
            sparse_second_rec_chunk_dir=self._sparse_second_rec_dir,
            summery_chunk_dir=self._summary_dir,
            index_dir=self._runtime.app.index_dir,
            gemini_api_key=integrations.gemini_api_key,
            gemini_requests_per_minute=integrations.gemini_requests_per_minute,
            gemini_summary_requests_per_minute=(
                integrations.gemini_summary_requests_per_minute
            ),
            embedding_model=self._runtime.providers.embeddings.model,
            first_rec_chunk_size=chunking.first_recursive_chunk_size,
            first_rec_chunk_overlap=chunking.first_recursive_chunk_overlap,
            second_rec_enabled=self._runtime.indexing.stages.second_recursive_enabled,
            second_rec_chunk_size=chunking.second_recursive_chunk_size,
            second_rec_chunk_overlap=chunking.second_recursive_chunk_overlap,
            summery_enabled=self._runtime.indexing.stages.summary_enabled,
            summery_characters=chunking.summary_characters,
            summery_provider=chunking.summary_llm_provider,
            summery_gemini_model=chunking.summary_gemini_model,
            summery_temperature=chunking.summary_temperature,
            summery_max_output_tokens=chunking.summary_max_output_tokens,
            summery_max_retries=2,
            summery_batch_size=chunking.summary_batch_size,
            llm_provider=providers_llm.provider,
            genai_model=providers_llm.gemini_model,
            temperature=providers_llm.temperature,
            max_output_tokens=providers_llm.max_output_tokens,
            clear_ingestion_source_data=refresh.clear_ingestion_source_data,
            clear_first_rec_chunk_data=refresh.clear_first_recursive_chunk_data,
            clear_second_rec_chunk_data=refresh.clear_second_recursive_chunk_data,
            clear_summery_chunk_data=refresh.clear_summary_chunk_data,
            update_ingestion_source_data=refresh.update_ingestion_source_data,
            update_first_rec_chunk_data=refresh.update_first_recursive_chunk_data,
            update_second_rec_chunk_data=refresh.update_second_recursive_chunk_data,
            update_sparse_second_rec_chunk_data=(
                refresh.update_sparse_second_recursive_chunk_data
            ),
            update_summery_chunk_data=refresh.update_summary_chunk_data,
            sudachi_mode=retrieval.sudachi_mode,
            sparse_bm25_k1=retrieval.sparse_bm25_k1,
            sparse_bm25_b=retrieval.sparse_bm25_b,
            sparse_use_normalized_form=retrieval.sparse_use_normalized_form,
            sparse_remove_symbols=retrieval.sparse_remove_symbols,
        )

    def _ensure_legacy_prompt_env_defaults(self) -> None:
        defaults = {
            "PROMPT_LLM_CHUNK_SYSTEM_PROMPT": (
                "You are a careful Japanese document chunking assistant."
            ),
            "PROMPT_SUMMERY_CHUNK_MESSAGES_TEMPLATE": (
                "次の会話ログを{target_characters}文字以内で要約してください。"
                "\\n重要な人物名・日時・数値は残してください。"
                "\\n\\n本文:\\n{text}"
            ),
            "PROMPT_SUMMERY_CHUNK_SHEETS_TEMPLATE": (
                "次の表データ由来本文を{target_characters}文字以内で要約してください。"
                "\\n元ファイル: {drive_path_display}"
                "\\n\\n本文:\\n{text}"
            ),
            "PROMPT_SUMMERY_CHUNK_DEFAULT_TEMPLATE": (
                "次の本文を{target_characters}文字以内で要約してください。"
                "\\n元ファイル: {drive_path_display}"
                "\\n\\n本文:\\n{text}"
            ),
        }
        for key, value in defaults.items():
            if not os.getenv(key):
                os.environ[key] = value
        try:
            from kumc_agent.infra.indexing import config as legacy_config

            legacy_config.get_required_prompt_env.cache_clear()
        except Exception:
            logger.exception("Failed to clear legacy prompt env cache.")

    @staticmethod
    def _should_run_stage(*, stage_name: str, selected: set[str]) -> bool:
        if not selected:
            return True
        return stage_name in selected

    def _run_legacy_chunk_pipeline(
        self,
        *,
        legacy_cfg,
        selected: set[str],
        allow_cancel: bool,
        cancel_event: threading.Event | None,
    ) -> None:
        from kumc_agent.infra.indexing.chunking import (
            docs_chunk_dir,
            message_chunk_jsonl_dir,
            recursive_chunk_dir,
            recursive_chunk_jsonl_dir,
            sheets_chunk_dir,
            sparse_chunk_jsonl_dir,
            summery_chunk_jsonl_dir,
        )
        from kumc_agent.infra.indexing.constants import (
            DOCS_SEPARATORS,
            MESSAGE_SEPARATORS,
            SHEETS_SEPARATORS,
        )
        refresh = self._runtime.indexing.refresh
        stages = self._runtime.indexing.stages
        chunking = self._runtime.indexing.chunking

        if self._should_run_stage(stage_name="first_recursive", selected=selected):
            if self._ingestion_messages_dir.exists():
                message_chunk_jsonl_dir(
                    raw_messages_dir=self._ingestion_messages_dir,
                    chunk_dir=self._first_rec_messages_dir,
                    chunk_size=chunking.first_recursive_chunk_size,
                    chunk_overlap=chunking.first_recursive_chunk_overlap,
                    stage="first_recursive",
                    skip_existing=not refresh.clear_first_recursive_chunk_data,
                    update_existing=refresh.update_first_recursive_chunk_data,
                    sync_deleted=refresh.update_first_recursive_chunk_data,
                )
            if self._ingestion_x_dir.exists():
                message_chunk_jsonl_dir(
                    raw_messages_dir=self._ingestion_x_dir,
                    chunk_dir=self._first_rec_x_dir,
                    chunk_size=chunking.first_recursive_chunk_size,
                    chunk_overlap=chunking.first_recursive_chunk_overlap,
                    stage="first_recursive",
                    skip_existing=not refresh.clear_first_recursive_chunk_data,
                    update_existing=refresh.update_first_recursive_chunk_data,
                    sync_deleted=refresh.update_first_recursive_chunk_data,
                )
            docs_chunk_dir(
                ingestion_data_dir=self._ingestion_docs_dir,
                structured_data_dir=self._ingestion_docs_normalized_dir,
                chunk_dir=self._first_rec_docs_dir,
                chunk_size=chunking.first_recursive_chunk_size,
                chunk_overlap=chunking.first_recursive_chunk_overlap,
                separators=DOCS_SEPARATORS,
                stage="first_recursive",
                skip_existing=not refresh.clear_first_recursive_chunk_data,
                update_existing=refresh.update_first_recursive_chunk_data,
                sync_deleted=refresh.update_first_recursive_chunk_data,
            )
            sheets_chunk_dir(
                ingestion_data_dir=self._ingestion_sheets_dir,
                structured_data_dir=self._ingestion_sheets_structured_dir,
                chunk_dir=self._first_rec_sheets_dir,
                chunk_size=chunking.first_recursive_chunk_size,
                chunk_overlap=chunking.first_recursive_chunk_overlap,
                separators=SHEETS_SEPARATORS,
                stage="first_recursive",
                skip_existing=not refresh.clear_first_recursive_chunk_data,
                update_existing=refresh.update_first_recursive_chunk_data,
                sync_deleted=refresh.update_first_recursive_chunk_data,
            )
            recursive_chunk_dir(
                ingestion_data_dir=self._ingestion_hatenablog_dir,
                chunk_dir=self._first_rec_hatenablog_dir,
                chunk_size=chunking.first_recursive_chunk_size,
                chunk_overlap=chunking.first_recursive_chunk_overlap,
                separators=DOCS_SEPARATORS,
                source_type="hatenablog",
                stage="first_recursive",
                file_extensions=(".md",),
                skip_existing=not refresh.clear_first_recursive_chunk_data,
                update_existing=refresh.update_first_recursive_chunk_data,
                sync_deleted=refresh.update_first_recursive_chunk_data,
            )
            recursive_chunk_dir(
                ingestion_data_dir=self._ingestion_crafters_colony_dir,
                chunk_dir=self._first_rec_crafters_colony_dir,
                chunk_size=chunking.first_recursive_chunk_size,
                chunk_overlap=chunking.first_recursive_chunk_overlap,
                separators=DOCS_SEPARATORS,
                source_type="crafters_colony",
                stage="first_recursive",
                file_extensions=(".md",),
                skip_existing=not refresh.clear_first_recursive_chunk_data,
                update_existing=refresh.update_first_recursive_chunk_data,
                sync_deleted=refresh.update_first_recursive_chunk_data,
            )
            recursive_chunk_dir(
                ingestion_data_dir=self._ingestion_notion_dir,
                chunk_dir=self._first_rec_notion_dir,
                chunk_size=chunking.first_recursive_chunk_size,
                chunk_overlap=chunking.first_recursive_chunk_overlap,
                separators=DOCS_SEPARATORS,
                source_type="notion",
                stage="first_recursive",
                file_extensions=(".md",),
                skip_existing=not refresh.clear_first_recursive_chunk_data,
                update_existing=refresh.update_first_recursive_chunk_data,
                sync_deleted=refresh.update_first_recursive_chunk_data,
            )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        if (
            stages.second_recursive_enabled
            and self._should_run_stage(stage_name="second_recursive", selected=selected)
        ):
            if self._first_rec_docs_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_docs_dir,
                    output_chunk_dir=self._second_rec_docs_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=DOCS_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_sheets_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_sheets_dir,
                    output_chunk_dir=self._second_rec_sheets_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=SHEETS_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_messages_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_messages_dir,
                    output_chunk_dir=self._second_rec_messages_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=MESSAGE_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_x_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_x_dir,
                    output_chunk_dir=self._second_rec_x_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=MESSAGE_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_hatenablog_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_hatenablog_dir,
                    output_chunk_dir=self._second_rec_hatenablog_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=DOCS_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_crafters_colony_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_crafters_colony_dir,
                    output_chunk_dir=self._second_rec_crafters_colony_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=DOCS_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._first_rec_notion_dir.exists():
                recursive_chunk_jsonl_dir(
                    input_chunk_dir=self._first_rec_notion_dir,
                    output_chunk_dir=self._second_rec_notion_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=DOCS_SEPARATORS,
                    stage="second_recursive",
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
            if self._ingestion_vc_dir.exists():
                recursive_chunk_dir(
                    ingestion_data_dir=self._ingestion_vc_dir,
                    chunk_dir=self._second_rec_vc_dir,
                    chunk_size=chunking.second_recursive_chunk_size,
                    chunk_overlap=chunking.second_recursive_chunk_overlap,
                    separators=MESSAGE_SEPARATORS,
                    source_type="vc_transcript",
                    stage="second_recursive",
                    file_extensions=(".txt",),
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_second_recursive_chunk_data,
                    sync_deleted=refresh.update_second_recursive_chunk_data,
                )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        if (
            stages.sparse_second_recursive_enabled
            and self._should_run_stage(
                stage_name="sparse_second_recursive",
                selected=selected,
            )
        ):
            if stages.second_recursive_enabled:
                for input_dir, output_dir in (
                    (self._second_rec_docs_dir, self._sparse_second_rec_docs_dir),
                    (self._second_rec_sheets_dir, self._sparse_second_rec_sheets_dir),
                    (self._second_rec_messages_dir, self._sparse_second_rec_messages_dir),
                    (self._second_rec_x_dir, self._sparse_second_rec_x_dir),
                    (self._second_rec_hatenablog_dir, self._sparse_second_rec_hatenablog_dir),
                    (
                        self._second_rec_crafters_colony_dir,
                        self._sparse_second_rec_crafters_colony_dir,
                    ),
                    (self._second_rec_notion_dir, self._sparse_second_rec_notion_dir),
                ):
                    if not input_dir.exists():
                        continue
                    sparse_chunk_jsonl_dir(
                        input_chunk_dir=input_dir,
                        output_chunk_dir=output_dir,
                        config=legacy_cfg,
                        skip_existing=not refresh.clear_second_recursive_chunk_data,
                        update_existing=(
                            refresh.update_sparse_second_recursive_chunk_data
                        ),
                        sync_deleted=(
                            refresh.update_sparse_second_recursive_chunk_data
                        ),
                    )
            if self._second_rec_vc_dir.exists():
                sparse_chunk_jsonl_dir(
                    input_chunk_dir=self._second_rec_vc_dir,
                    output_chunk_dir=self._sparse_second_rec_vc_dir,
                    config=legacy_cfg,
                    skip_existing=not refresh.clear_second_recursive_chunk_data,
                    update_existing=refresh.update_sparse_second_recursive_chunk_data,
                    sync_deleted=refresh.update_sparse_second_recursive_chunk_data,
                )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        if stages.summary_enabled and self._should_run_stage(
            stage_name="summary",
            selected=selected,
        ):
            for input_dir, output_dir, second_dir in (
                (
                    self._first_rec_docs_dir,
                    self._summary_docs_dir,
                    self._second_rec_docs_dir if stages.second_recursive_enabled else None,
                ),
                (
                    self._first_rec_sheets_dir,
                    self._summary_sheets_dir,
                    self._second_rec_sheets_dir if stages.second_recursive_enabled else None,
                ),
                (
                    self._first_rec_messages_dir,
                    self._summary_messages_dir,
                    self._second_rec_messages_dir if stages.second_recursive_enabled else None,
                ),
                (
                    self._first_rec_x_dir,
                    self._summary_x_dir,
                    self._second_rec_x_dir if stages.second_recursive_enabled else None,
                ),
                (
                    self._first_rec_hatenablog_dir,
                    self._summary_hatenablog_dir,
                    self._second_rec_hatenablog_dir
                    if stages.second_recursive_enabled
                    else None,
                ),
                (
                    self._first_rec_crafters_colony_dir,
                    self._summary_crafters_colony_dir,
                    self._second_rec_crafters_colony_dir
                    if stages.second_recursive_enabled
                    else None,
                ),
                (
                    self._first_rec_notion_dir,
                    self._summary_notion_dir,
                    self._second_rec_notion_dir
                    if stages.second_recursive_enabled
                    else None,
                ),
            ):
                if not input_dir.exists():
                    continue
                summery_chunk_jsonl_dir(
                    input_chunk_dir=input_dir,
                    output_chunk_dir=output_dir,
                    second_chunk_dir=second_dir,
                    config=legacy_cfg,
                    skip_existing=not refresh.clear_summary_chunk_data,
                    update_existing=refresh.update_summary_chunk_data,
                    sync_deleted=refresh.update_summary_chunk_data,
                )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

    def _run_minecraft_wiki_chunk_pipeline(
        self,
        *,
        minecraft_wiki_cfg,
        selected: set[str],
        allow_cancel: bool,
        cancel_event: threading.Event | None,
    ) -> None:
        from kumc_agent.infra.indexing.chunking import (
            recursive_chunk_dir,
            recursive_chunk_jsonl_dir,
            sparse_chunk_jsonl_dir,
        )
        from kumc_agent.infra.indexing.constants import DOCS_SEPARATORS

        if not self._ingestion_minecraft_wiki_dir.exists():
            return

        refresh = self._runtime.indexing.refresh
        stages = self._runtime.indexing.stages
        chunking = self._runtime.minecraft_wiki_rag.chunking

        if self._should_run_stage(stage_name="first_recursive", selected=selected):
            recursive_chunk_dir(
                ingestion_data_dir=self._ingestion_minecraft_wiki_dir,
                chunk_dir=self._first_rec_minecraft_wiki_dir,
                chunk_size=chunking.first_recursive_chunk_size,
                chunk_overlap=chunking.first_recursive_chunk_overlap,
                separators=DOCS_SEPARATORS,
                source_type="minecraft_wiki",
                stage="first_recursive",
                file_extensions=(".md",),
                skip_existing=not refresh.clear_first_recursive_chunk_data,
                update_existing=refresh.update_first_recursive_chunk_data,
                sync_deleted=refresh.update_first_recursive_chunk_data,
            )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        if (
            stages.second_recursive_enabled
            and self._should_run_stage(stage_name="second_recursive", selected=selected)
            and self._first_rec_minecraft_wiki_dir.exists()
        ):
            recursive_chunk_jsonl_dir(
                input_chunk_dir=self._first_rec_minecraft_wiki_dir,
                output_chunk_dir=self._second_rec_minecraft_wiki_dir,
                chunk_size=chunking.second_recursive_chunk_size,
                chunk_overlap=chunking.second_recursive_chunk_overlap,
                separators=DOCS_SEPARATORS,
                stage="second_recursive",
                skip_existing=not refresh.clear_second_recursive_chunk_data,
                update_existing=refresh.update_second_recursive_chunk_data,
                sync_deleted=refresh.update_second_recursive_chunk_data,
            )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        if (
            stages.sparse_second_recursive_enabled
            and self._should_run_stage(
                stage_name="sparse_second_recursive",
                selected=selected,
            )
            and self._second_rec_minecraft_wiki_dir.exists()
        ):
            sparse_chunk_jsonl_dir(
                input_chunk_dir=self._second_rec_minecraft_wiki_dir,
                output_chunk_dir=self._sparse_second_rec_minecraft_wiki_dir,
                config=minecraft_wiki_cfg,
                skip_existing=not refresh.clear_second_recursive_chunk_data,
                update_existing=refresh.update_sparse_second_recursive_chunk_data,
                sync_deleted=refresh.update_sparse_second_recursive_chunk_data,
            )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

        if (
            stages.summary_enabled
            and self._should_run_stage(stage_name="summary", selected=selected)
            and self._first_rec_minecraft_wiki_dir.exists()
        ):
            self._build_minecraft_wiki_summary_chunks(
                input_chunk_dir=self._first_rec_minecraft_wiki_dir,
                output_chunk_dir=self._summary_minecraft_wiki_dir,
                skip_existing=not refresh.clear_summary_chunk_data,
                update_existing=refresh.update_summary_chunk_data,
                sync_deleted=refresh.update_summary_chunk_data,
            )
        self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)

    def _docs_quality_payload(
        self,
        *,
        index_chunks: list[Chunk],
    ) -> dict[str, object]:
        from kumc_agent.usecases.ingestion.google_drive_docs_audit import (
            GoogleDriveDocsQualityThresholds,
            build_google_drive_docs_quality_payload,
        )

        cfg = self._runtime.indexing.docs_quality
        docs_chunk_count = sum(
            1
            for chunk in index_chunks
            if str(chunk.metadata.get("source_type") or "").strip().lower() == "docs"
        )
        return build_google_drive_docs_quality_payload(
            raw_dir=self._ingestion_docs_dir,
            normalized_dir=self._ingestion_docs_normalized_dir,
            image_dir=self._ingestion_dir / "images" / "google_drive",
            chunk_count=docs_chunk_count,
            thresholds=GoogleDriveDocsQualityThresholds(
                enabled=cfg.enabled,
                policy="fail" if cfg.fail_fast else "warn",
                min_text_bytes=cfg.min_text_bytes,
                min_nonempty_characters=cfg.min_nonempty_characters,
                max_short_document_ratio=cfg.max_short_document_ratio,
                max_source_date_unknown_ratio=cfg.max_source_date_unknown_ratio,
            ),
        )

    def _minecraft_wiki_quality_payload(
        self,
        *,
        index_chunks: list[Chunk],
    ) -> dict[str, object] | None:
        if (
            not self._ingestion_minecraft_wiki_dir.exists()
            and not self._runtime.features.sources.minecraft_wiki
        ):
            return None
        from kumc_agent.usecases.ingestion.minecraft_wiki_audit import (
            MinecraftWikiQualityThresholds,
            audit_minecraft_wiki_raw_dir,
        )

        gate = self._runtime.integrations.minecraft_wiki.quality_gate
        wiki_chunk_count = sum(
            1
            for chunk in index_chunks
            if str(
                chunk.metadata.get("source_type")
                or chunk.metadata.get("source_kind")
                or ""
            ).strip().lower()
            == "minecraft_wiki"
        )
        report = audit_minecraft_wiki_raw_dir(
            raw_dir=self._ingestion_minecraft_wiki_dir,
            thresholds=MinecraftWikiQualityThresholds(
                enabled=gate.enabled,
                min_article_characters=gate.min_article_characters,
                max_redirect_ratio=gate.max_redirect_ratio,
                min_indexable_pages=gate.min_indexable_pages,
                min_chunk_count=gate.min_chunk_count,
                required_canonical_host=gate.required_canonical_host,
                policy=gate.policy,
            ),
            chunk_count=wiki_chunk_count,
        )
        return report.to_payload()

    def _build_minecraft_wiki_summary_chunks(
        self,
        *,
        input_chunk_dir: Path,
        output_chunk_dir: Path,
        skip_existing: bool,
        update_existing: bool,
        sync_deleted: bool,
    ) -> None:
        from kumc_agent.infra.indexing.chunks import (
            Chunk as LegacyChunk,
            load_chunks,
            write_chunks,
        )

        output_chunk_dir.mkdir(parents=True, exist_ok=True)
        expected_output_names: set[str] = set()
        target_chars = max(
            80,
            self._runtime.minecraft_wiki_rag.chunking.summary_characters,
        )
        batch_size = max(
            1,
            int(self._runtime.minecraft_wiki_rag.chunking.summary_batch_size),
        )
        summary_jobs: list[tuple[Path, int, str, dict[str, object]]] = []
        processed_paths: list[Path] = []
        for path in sorted(input_chunk_dir.glob("*.jsonl")):
            out_path = output_chunk_dir / path.name
            expected_output_names.add(out_path.name)
            if (
                skip_existing
                and out_path.exists()
                and (not update_existing or out_path.stat().st_mtime >= path.stat().st_mtime)
            ):
                continue
            processed_paths.append(out_path)
            chunks = load_chunks(path)
            output_index = 0
            for chunk in chunks:
                text = str(chunk.text or "").strip()
                if not text:
                    continue
                metadata = dict(chunk.metadata)
                parent_chunk_id = metadata.get("chunk_id")
                if parent_chunk_id is not None:
                    metadata["parent_chunk_id"] = parent_chunk_id
                metadata["chunk_id"] = output_index
                metadata["chunk_stage"] = "summary"
                summary_jobs.append((out_path, output_index, text, metadata))
                output_index += 1

        output_chunks_by_path: dict[Path, list[LegacyChunk]] = {
            path: [] for path in processed_paths
        }
        for batch_start in range(0, len(summary_jobs), batch_size):
            batch = summary_jobs[batch_start : batch_start + batch_size]
            max_workers = max(1, min(len(batch), batch_size))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: dict[
                    Future[str],
                    tuple[Path, int, str, dict[str, object]],
                ] = {}
                for out_path, output_index, text, metadata in batch:
                    futures[
                        executor.submit(
                            self._build_minecraft_wiki_summary_text,
                            text=text,
                            metadata=metadata,
                            target_chars=target_chars,
                        )
                    ] = (out_path, output_index, text, metadata)
                for future in as_completed(futures):
                    out_path, _output_index, text, metadata = futures[future]
                    try:
                        summary_text = future.result()
                    except Exception:
                        logger.exception(
                            "Minecraft Wiki summary chunk worker failed. Fallback to truncation."
                        )
                        summary_text = self._fallback_minecraft_wiki_summary(
                            text=text,
                            metadata=metadata,
                            target_chars=target_chars,
                        )
                    output_chunks_by_path.setdefault(out_path, []).append(
                        LegacyChunk(
                            text=summary_text,
                            metadata=metadata,
                        )
                    )

        for out_path in processed_paths:
            output_chunks = output_chunks_by_path.get(out_path, [])
            output_chunks.sort(
                key=lambda chunk: int(chunk.metadata.get("chunk_id", 0) or 0)
            )
            write_chunks(out_path, output_chunks)
        if sync_deleted:
            expected = expected_output_names
            for stale in output_chunk_dir.glob("*.jsonl"):
                if stale.name in expected:
                    continue
                try:
                    stale.unlink()
                except Exception:
                    logger.warning("Failed to remove stale Minecraft Wiki summary: %s", stale)

    def _build_minecraft_wiki_summary_text(
        self,
        *,
        text: str,
        metadata: dict[str, object],
        target_chars: int,
    ) -> str:
        fallback = self._fallback_minecraft_wiki_summary(
            text=text,
            metadata=metadata,
            target_chars=target_chars,
        )
        provider = (
            self._runtime.minecraft_wiki_rag.chunking.summary_llm_provider or ""
        ).strip().lower()
        if provider in {"", "none", "off", "disabled", "false", "0"}:
            return fallback
        if self._minecraft_wiki_summary_llm is None:
            return fallback

        title = str(metadata.get("minecraft_wiki_title") or "").strip()
        heading_path = metadata.get("heading_path")
        if isinstance(heading_path, list):
            heading = " > ".join(
                str(value).strip()
                for value in heading_path
                if str(value).strip()
            )
        else:
            heading = str(heading_path or "").strip()
        prompt = (
            f"日本語版Minecraft Wikiの記事本文を{target_chars}文字以内で要約してください。"
            "\n本文にある仕様条件、数値、クラフト素材、入手条件、例外は落とさないでください。"
            "\n表や箇条書きの情報は文章として保持してください。"
            "\n推測や本文外のバージョン判断は加えないでください。"
            f"\n\n記事名: {title or '不明'}"
            f"\n見出し: {heading or '不明'}"
            f"\n\n本文:\n{text}"
        )
        try:
            summary = self._minecraft_wiki_summary_llm.generate(
                system_prompt=(
                    "You summarize Japanese Minecraft Wiki articles faithfully."
                ),
                user_prompt=prompt,
                temperature=self._runtime.minecraft_wiki_rag.chunking.summary_temperature,
                max_output_tokens=(
                    self._runtime.minecraft_wiki_rag.chunking.summary_max_output_tokens
                ),
            )
        except Exception:
            logger.exception(
                "Minecraft Wiki summary chunk generation failed. Fallback to truncation."
            )
            return fallback
        normalized = (summary or "").strip()
        if not normalized or self._is_llm_error_text(normalized):
            return fallback
        if len(normalized) > target_chars:
            normalized = normalized[:target_chars].rstrip() + "..."
        prefix = ""
        if title and not normalized.startswith("記事名:"):
            prefix += f"記事名: {title}\n"
        if heading and "見出し:" not in normalized[:80]:
            prefix += f"見出し: {heading}\n"
        return (prefix + normalized).strip()

    @staticmethod
    def _fallback_minecraft_wiki_summary(
        *,
        text: str,
        metadata: dict[str, object],
        target_chars: int,
    ) -> str:
        title = str(metadata.get("minecraft_wiki_title") or "").strip()
        heading_path = metadata.get("heading_path")
        heading = ""
        if isinstance(heading_path, list) and heading_path:
            heading = " > ".join(str(value) for value in heading_path if str(value).strip())
        prefix = ""
        if title:
            prefix += f"記事名: {title}\n"
        if heading:
            prefix += f"見出し: {heading}\n"
        body = " ".join((text or "").split())
        limit = max(0, int(target_chars) - len(prefix))
        if limit and len(body) > limit:
            body = body[:limit].rstrip() + "..."
        return (prefix + body).strip()

    def _load_index_chunks_from_legacy_dirs(self, *, legacy_cfg) -> list[Chunk]:
        base_dirs: list[Path] = []
        if legacy_cfg.second_rec_enabled:
            base_dirs.extend(
                [
                    self._second_rec_docs_dir,
                    self._second_rec_sheets_dir,
                    self._second_rec_hatenablog_dir,
                    self._second_rec_crafters_colony_dir,
                    self._second_rec_notion_dir,
                    self._second_rec_minecraft_wiki_dir,
                ]
            )
            if self._second_rec_messages_dir.exists():
                base_dirs.append(self._second_rec_messages_dir)
            if self._second_rec_x_dir.exists():
                base_dirs.append(self._second_rec_x_dir)
        else:
            base_dirs.extend(
                [
                    self._first_rec_docs_dir,
                    self._first_rec_sheets_dir,
                    self._first_rec_hatenablog_dir,
                    self._first_rec_crafters_colony_dir,
                    self._first_rec_notion_dir,
                    self._first_rec_minecraft_wiki_dir,
                ]
            )
            if self._first_rec_messages_dir.exists():
                base_dirs.append(self._first_rec_messages_dir)
            if self._first_rec_x_dir.exists():
                base_dirs.append(self._first_rec_x_dir)

        chunks: list[Chunk] = []
        chunks.extend(self._load_legacy_chunks_from_dirs(base_dirs))
        chunks.extend(
            self._load_legacy_chunks_from_dirs(
                [
                    self._summary_docs_dir,
                    self._summary_sheets_dir,
                    self._summary_messages_dir,
                    self._summary_x_dir,
                    self._summary_hatenablog_dir,
                    self._summary_crafters_colony_dir,
                    self._summary_notion_dir,
                ]
            )
        )
        if self._second_rec_vc_dir.exists():
            chunks.extend(self._load_legacy_chunks_from_dirs([self._second_rec_vc_dir]))
        if self._summary_minecraft_wiki_dir.exists():
            chunks.extend(
                self._load_legacy_chunks_from_dirs([self._summary_minecraft_wiki_dir])
            )
        return chunks

    def _load_minecraft_wiki_stage_artifacts(self, *, legacy_cfg) -> _RepositoryIndexArtifacts:
        first_chunks = self._load_legacy_chunks_from_dirs(
            [self._first_rec_minecraft_wiki_dir]
        )
        second_base = (
            self._second_rec_minecraft_wiki_dir
            if legacy_cfg.second_rec_enabled
            else self._first_rec_minecraft_wiki_dir
        )
        second_chunks = self._load_legacy_chunks_from_dirs([second_base])
        sparse_chunks = self._load_legacy_chunks_from_dirs(
            [self._sparse_second_rec_minecraft_wiki_dir]
        )
        summary_chunks = self._load_legacy_chunks_from_dirs(
            [self._summary_minecraft_wiki_dir]
        )
        index_chunks = [*second_chunks, *summary_chunks]
        return _RepositoryIndexArtifacts(
            first_chunks=first_chunks,
            second_chunks=second_chunks,
            sparse_chunks=sparse_chunks,
            summary_chunks=summary_chunks,
            index_chunks=index_chunks,
        )

    @staticmethod
    def _is_minecraft_wiki_chunk(chunk: Chunk) -> bool:
        metadata = chunk.metadata or {}
        source_type = str(
            metadata.get("source_type") or metadata.get("source_kind") or ""
        ).strip().lower()
        return source_type == "minecraft_wiki"

    def _load_legacy_chunks_from_dirs(self, chunk_dirs: list[Path]) -> list[Chunk]:
        from kumc_agent.infra.indexing.chunks import load_chunks_from_dirs

        existing = [path for path in chunk_dirs if path.exists()]
        if not existing:
            return []
        legacy_chunks = load_chunks_from_dirs(existing)
        out: list[Chunk] = []
        for idx, legacy_chunk in enumerate(legacy_chunks):
            converted = self._legacy_chunk_to_domain_chunk(
                legacy_chunk=legacy_chunk,
                fallback_index=idx,
            )
            if converted is not None:
                out.append(converted)
        return out

    @staticmethod
    def _to_int(value: object, *, fallback: int) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return fallback
        return fallback

    def _legacy_chunk_uid(
        self,
        *,
        metadata: dict[str, object],
        text: str,
        fallback_index: int,
    ) -> str:
        source_type = str(metadata.get("source_type") or "").strip()
        source_name = str(
            metadata.get("drive_file_id")
            or metadata.get("minecraft_wiki_page_id")
            or metadata.get("source_file_name")
            or metadata.get("path")
            or ""
        ).strip()
        stage = str(metadata.get("chunk_stage") or "").strip()
        chunk_id = self._to_int(metadata.get("chunk_id"), fallback=fallback_index)
        return stable_hash(f"{source_type}|{source_name}|{stage}|{chunk_id}|{text[:256]}")

    def _legacy_chunk_to_domain_chunk(
        self,
        *,
        legacy_chunk,
        fallback_index: int,
    ) -> Chunk | None:
        text = str(getattr(legacy_chunk, "text", "") or "").strip()
        if not text:
            return None
        metadata = dict(getattr(legacy_chunk, "metadata", {}) or {})
        index = self._to_int(metadata.get("chunk_id"), fallback=fallback_index)
        source_type = str(metadata.get("source_type") or "").strip()
        source_name = str(
            metadata.get("drive_file_id")
            or metadata.get("minecraft_wiki_page_id")
            or metadata.get("source_file_name")
            or metadata.get("path")
            or ""
        ).strip()
        if not source_name:
            source_name = f"legacy:{source_type}:{index}"
        document_id = stable_hash(f"{source_type}:{source_name}")
        chunk_uid = self._legacy_chunk_uid(
            metadata=metadata,
            text=text,
            fallback_index=index,
        )
        metadata["chunk_uid"] = chunk_uid
        metadata.setdefault("chunk_id", index)
        return Chunk(
            id=chunk_uid,
            document_id=document_id,
            text=text,
            index=index,
            metadata=metadata,
        )

    def _chunk_embedding_text_for_dense(self, chunk: Chunk) -> str:
        metadata = chunk.metadata or {}
        source_type = str(metadata.get("source_type") or "").strip().lower()
        if source_type == "minecraft_wiki":
            title = str(metadata.get("minecraft_wiki_title") or metadata.get("source_title") or "").strip()
            heading_path = metadata.get("heading_path")
            if isinstance(heading_path, list):
                heading = " > ".join(
                    str(value).strip()
                    for value in heading_path
                    if str(value).strip()
                )
            else:
                heading = str(heading_path or "").strip()
            prefix_lines = []
            if title:
                prefix_lines.append(f"記事名: {title}")
            if heading:
                prefix_lines.append(f"見出し: {heading}")
            if prefix_lines:
                return "\n".join(prefix_lines) + "\n\n" + chunk.text
            return chunk.text
        from kumc_agent.infra.indexing.chunks import (
            Chunk as LegacyChunk,
            chunk_embedding_text,
        )

        try:
            return chunk_embedding_text(
                LegacyChunk(text=chunk.text, metadata=dict(chunk.metadata))
            )
        except Exception:
            logger.exception("Failed to build legacy embedding text. Falling back to chunk text.")
            return chunk.text

    def _build_repository_index_artifacts(
        self,
        *,
        repository_chunks: list[Chunk],
        legacy_cfg,
        selected: set[str],
    ) -> _RepositoryIndexArtifacts:
        first_chunks: list[Chunk] = []
        second_chunks: list[Chunk] = []
        sparse_chunks: list[Chunk] = []
        summary_chunks: list[Chunk] = []

        for idx, chunk in enumerate(repository_chunks):
            normalized = self._normalize_repository_chunk(chunk, fallback_index=idx)
            first_metadata = dict(normalized.metadata)
            first_metadata["chunk_stage"] = "first_recursive"
            first_metadata["chunk_id"] = normalized.index
            first_chunk = Chunk(
                id=stable_hash(f"{normalized.id}:first:{normalized.text[:256]}"),
                document_id=normalized.document_id,
                text=normalized.text,
                index=normalized.index,
                metadata=first_metadata,
            )
            first_chunks.append(first_chunk)

            second_metadata = dict(normalized.metadata)
            second_metadata["chunk_stage"] = "second_recursive"
            second_metadata["parent_chunk_id"] = first_chunk.metadata.get("chunk_id", first_chunk.index)
            second_metadata["chunk_id"] = normalized.index
            second_metadata["skip_parent_context"] = True
            second_chunk = replace(
                normalized,
                metadata=second_metadata,
            )
            second_chunks.append(second_chunk)

            sparse_text = self._repository_sparse_text(
                second_chunk.text,
                legacy_cfg=legacy_cfg,
            )
            if sparse_text:
                sparse_metadata = dict(second_metadata)
                sparse_metadata["chunk_stage"] = "second_recursive_sparse"
                sparse_chunks.append(
                    Chunk(
                        id=stable_hash(f"{second_chunk.id}:sparse:{sparse_text[:256]}"),
                        document_id=second_chunk.document_id,
                        text=sparse_text,
                        index=second_chunk.index,
                        metadata=sparse_metadata,
                    )
                )

        stages = self._runtime.indexing.stages
        chunking = self._runtime.indexing.chunking
        if stages.summary_enabled and (not selected or "summary" in selected):
            summary_chunks = self._build_summary_chunks_for_first_chunks(
                first_chunks=first_chunks,
                target_characters=int(chunking.summary_characters),
                include_index_in_hash=False,
            )

        self._clear_dir_contents(self._first_rec_dir)
        self._clear_dir_contents(self._second_rec_dir)
        self._clear_dir_contents(self._sparse_second_rec_dir)
        self._clear_dir_contents(self._summary_dir)
        self._write_stage_chunks(self._first_rec_dir, first_chunks)
        self._write_stage_chunks(self._second_rec_dir, second_chunks)
        self._write_stage_chunks(self._sparse_second_rec_dir, sparse_chunks)
        if summary_chunks:
            self._write_stage_chunks(self._summary_dir, summary_chunks)

        return _RepositoryIndexArtifacts(
            first_chunks=first_chunks,
            second_chunks=second_chunks,
            sparse_chunks=sparse_chunks,
            summary_chunks=summary_chunks,
            index_chunks=[*second_chunks, *summary_chunks],
        )

    def _normalize_repository_chunk(self, chunk: Chunk, *, fallback_index: int) -> Chunk:
        metadata = dict(chunk.metadata or {})
        source_type = str(
            metadata.get("source_type") or metadata.get("source_kind") or ""
        ).strip().lower()
        if not source_type:
            source_type = "misc"
        source_key = str(
            metadata.get("drive_file_id")
            or metadata.get("source_file_name")
            or metadata.get("external_id")
            or metadata.get("source_item_id")
            or chunk.document_id
            or ""
        ).strip()
        if not source_key:
            source_key = f"{source_type}:{fallback_index}"
        metadata["source_type"] = source_type
        metadata.setdefault("source_kind", source_type)
        metadata.setdefault("source_file_name", source_key)
        metadata.setdefault("external_id", source_key)
        source_title = str(metadata.get("source_title") or "").strip()
        if source_title:
            metadata.setdefault("drive_file_name", source_title)
        if "path" not in metadata:
            metadata["path"] = source_key
        metadata.setdefault("chunk_uid", chunk.id)
        metadata.setdefault("index_status", "active")
        index = self._to_int(metadata.get("chunk_id"), fallback=chunk.index)
        if index < 0:
            index = fallback_index
        return replace(chunk, index=index, metadata=metadata)

    @staticmethod
    def _repository_sparse_text(text: str, *, legacy_cfg) -> str:
        source_text = str(text or "").strip()
        if not source_text:
            return ""
        try:
            from kumc_agent.infra.indexing.sparse_normalizer import (
                SparseNormalizer,
                SparseNormalizerConfig,
            )

            normalizer = SparseNormalizer(
                config=SparseNormalizerConfig(
                    sudachi_mode=legacy_cfg.sudachi_mode,
                    use_normalized_form=legacy_cfg.sparse_use_normalized_form,
                    remove_symbols=legacy_cfg.sparse_remove_symbols,
                    remove_stopwords=False,
                )
            )
            tokens = normalizer.normalize_tokens(source_text)
        except Exception:
            tokens = [token for token in source_text.lower().split() if token]
        return " ".join(token for token in tokens if token)

    def _build_material_catalog_from_repository_chunks(
        self,
        *,
        chunks: list[Chunk],
    ) -> None:
        try:
            from kumc_agent.infra.indexing.material_catalog import (
                MaterialCatalogEntry,
                save_material_catalog,
            )
        except Exception:
            logger.exception("Material catalog dependencies are unavailable.")
            return

        grouped: dict[tuple[str, str], dict[str, object]] = {}
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            source_type = str(metadata.get("source_type") or "").strip().lower()
            source_key = str(
                metadata.get("drive_file_id")
                or metadata.get("source_file_name")
                or metadata.get("external_id")
                or metadata.get("source_item_id")
                or chunk.document_id
                or ""
            ).strip()
            if not source_type or not source_key:
                continue
            key = (source_type, source_key)
            row = grouped.setdefault(
                key,
                {
                    "chunks": [],
                    "aliases": [],
                    "metadata": metadata,
                },
            )
            row_chunks = row["chunks"]
            if isinstance(row_chunks, list):
                row_chunks.append(chunk.text)
            aliases = row["aliases"]
            if isinstance(aliases, list):
                aliases.extend(self._material_aliases_from_metadata(metadata, source_key))

        entries: list[MaterialCatalogEntry] = []
        material_text_dir = self._runtime.app.data_dir / "material_raw"
        material_text_dir.mkdir(parents=True, exist_ok=True)
        for (source_type, source_key), row in grouped.items():
            metadata = row.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            canonical = self._material_canonical_name(metadata, source_key=source_key)
            aliases = self._dedupe_texts(
                [canonical, source_key, *list(row.get("aliases") or [])]
            )
            material_id = f"{source_type}:{source_key}"
            raw_path = material_text_dir / f"{stable_hash(material_id)[:24]}.txt"
            chunk_texts = [
                str(value).strip()
                for value in row.get("chunks") or []
                if str(value).strip()
            ]
            raw_path.write_text("\n\n".join(chunk_texts), encoding="utf-8")
            entries.append(
                MaterialCatalogEntry(
                    material_id=material_id,
                    source_type=source_type,
                    source_key=source_key,
                    canonical_name=canonical,
                    aliases=tuple(aliases),
                    raw_path=str(raw_path),
                )
            )
        save_material_catalog(index_dir=self._runtime.app.index_dir, entries=entries)

    @staticmethod
    def _material_aliases_from_metadata(
        metadata: dict[str, object],
        source_key: str,
    ) -> list[str]:
        values = [
            source_key,
            metadata.get("source_title"),
            metadata.get("drive_file_name"),
            metadata.get("drive_file_path"),
            metadata.get("path"),
            metadata.get("canonical_url"),
            metadata.get("notion_title"),
            metadata.get("notion_url"),
            metadata.get("hatenablog_title"),
            metadata.get("hatenablog_url"),
            metadata.get("crafters_colony_title"),
            metadata.get("crafters_colony_article_url"),
        ]
        return [str(value).strip() for value in values if str(value or "").strip()]

    @staticmethod
    def _material_canonical_name(
        metadata: dict[str, object],
        *,
        source_key: str,
    ) -> str:
        for key in (
            "source_title",
            "drive_file_name",
            "notion_title",
            "hatenablog_title",
            "crafters_colony_title",
            "path",
        ):
            value = str(metadata.get(key) or "").strip()
            if value:
                return Path(value).stem or value
        return Path(source_key).stem or source_key

    @staticmethod
    def _dedupe_texts(values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return deduped

    def _build_keyword_inverted_indexes_from_repository_artifacts(
        self,
        *,
        artifacts: _RepositoryIndexArtifacts,
        legacy_cfg,
    ) -> None:
        try:
            from langchain_core.documents import Document as LangDocument
            from kumc_agent.infra.indexing.keyword_inverted_index import (
                KEYWORD_CORPUS_SECOND_REC_SPARSE,
                KEYWORD_CORPUS_SPARSE,
                KEYWORD_CORPUS_SPARSE_SECOND_REC,
                build_and_save_keyword_index,
                tokenize_sparse_doc,
            )
            from kumc_agent.infra.indexing.sparse_normalizer import (
                SparseNormalizer,
                SparseNormalizerConfig,
            )
        except Exception:
            logger.exception(
                "Keyword inverted index dependencies are unavailable. "
                "Falling back to lightweight keyword payload."
            )
            self._build_keyword_indexes_payload(
                sparse_chunks=artifacts.sparse_chunks,
                second_chunks=artifacts.second_chunks,
            )
            return

        normalizer = SparseNormalizer(
            config=SparseNormalizerConfig(
                sudachi_mode=legacy_cfg.sudachi_mode,
                use_normalized_form=legacy_cfg.sparse_use_normalized_form,
                remove_symbols=legacy_cfg.sparse_remove_symbols,
                remove_stopwords=False,
            )
        )

        def _doc(chunk: Chunk) -> LangDocument:
            metadata = dict(chunk.metadata or {})
            metadata.setdefault("chunk_uid", chunk.id)
            return LangDocument(page_content=chunk.text, metadata=metadata)

        def _tokenize(doc: LangDocument) -> list[str]:
            return tokenize_sparse_doc(
                doc,
                sparse_stage="second_recursive_sparse",
                sudachi_tokenize=normalizer.normalize_tokens,
            )

        sparse_docs = [_doc(chunk) for chunk in artifacts.sparse_chunks if chunk.text.strip()]
        second_docs = [_doc(chunk) for chunk in artifacts.second_chunks if chunk.text.strip()]
        for corpus_name, docs in (
            (KEYWORD_CORPUS_SPARSE, sparse_docs),
            (KEYWORD_CORPUS_SPARSE_SECOND_REC, sparse_docs),
            (KEYWORD_CORPUS_SECOND_REC_SPARSE, second_docs),
        ):
            build_and_save_keyword_index(
                index_dir=legacy_cfg.index_dir,
                corpus_name=corpus_name,
                docs=docs,
                tokenize_doc=_tokenize,
                k1=legacy_cfg.sparse_bm25_k1,
                b=legacy_cfg.sparse_bm25_b,
            )

    def _build_keyword_inverted_indexes(self, *, legacy_cfg) -> None:
        try:
            from langchain_core.documents import Document as LangDocument
            from kumc_agent.infra.indexing.chunks import load_chunks_from_dirs
            from kumc_agent.infra.indexing.keyword_inverted_index import (
                KEYWORD_CORPUS_MATERIAL_NAMES,
                KEYWORD_CORPUS_SECOND_REC_SPARSE,
                KEYWORD_CORPUS_SPARSE,
                KEYWORD_CORPUS_SPARSE_SECOND_REC,
                build_and_save_keyword_index,
                tokenize_sparse_doc,
            )
            from kumc_agent.infra.indexing.sparse_sources import (
                second_rec_chunk_dirs,
                sparse_chunk_dirs,
                sparse_second_rec_chunk_dirs,
            )
            from kumc_agent.infra.indexing.sparse_normalizer import (
                SparseNormalizer,
                SparseNormalizerConfig,
            )
        except Exception:
            logger.exception(
                "Legacy keyword inverted index dependencies are unavailable. "
                "Falling back to lightweight keyword payload."
            )
            self._build_keyword_indexes_payload(
                sparse_chunks=self._load_legacy_chunks_from_dirs(
                    [
                        self._sparse_second_rec_docs_dir,
                        self._sparse_second_rec_sheets_dir,
                        self._sparse_second_rec_messages_dir,
                        self._sparse_second_rec_x_dir,
                        self._sparse_second_rec_vc_dir,
                        self._sparse_second_rec_hatenablog_dir,
                        self._sparse_second_rec_crafters_colony_dir,
                        self._sparse_second_rec_notion_dir,
                        self._sparse_second_rec_minecraft_wiki_dir,
                    ]
                ),
                second_chunks=self._load_legacy_chunks_from_dirs(
                    [
                        self._second_rec_docs_dir,
                        self._second_rec_sheets_dir,
                        self._second_rec_messages_dir,
                        self._second_rec_x_dir,
                        self._second_rec_vc_dir,
                        self._second_rec_hatenablog_dir,
                        self._second_rec_crafters_colony_dir,
                        self._second_rec_notion_dir,
                        self._second_rec_minecraft_wiki_dir,
                    ]
                ),
            )
            return

        normalizer = SparseNormalizer(
            config=SparseNormalizerConfig(
                sudachi_mode=legacy_cfg.sudachi_mode,
                use_normalized_form=legacy_cfg.sparse_use_normalized_form,
                remove_symbols=legacy_cfg.sparse_remove_symbols,
                remove_stopwords=False,
            )
        )

        def _tokenize(doc: LangDocument) -> list[str]:
            return tokenize_sparse_doc(
                doc,
                sparse_stage="second_recursive_sparse",
                sudachi_tokenize=normalizer.normalize_tokens,
            )

        def _docs_for_dirs(chunk_dirs: list[Path]) -> list[LangDocument]:
            existing = [value for value in chunk_dirs if value.exists()]
            if not existing:
                return []
            chunks = load_chunks_from_dirs(existing)
            docs: list[LangDocument] = []
            for idx, chunk in enumerate(chunks):
                text = str(chunk.text or "").strip()
                if not text:
                    continue
                metadata = dict(chunk.metadata or {})
                metadata["chunk_uid"] = self._legacy_chunk_uid(
                    metadata=metadata,
                    text=text,
                    fallback_index=idx,
                )
                docs.append(LangDocument(page_content=text, metadata=metadata))
            return docs

        corpus_to_dirs = (
            (KEYWORD_CORPUS_SPARSE, sparse_chunk_dirs(legacy_cfg)),
            (KEYWORD_CORPUS_SPARSE_SECOND_REC, sparse_second_rec_chunk_dirs(legacy_cfg)),
            (KEYWORD_CORPUS_SECOND_REC_SPARSE, second_rec_chunk_dirs(legacy_cfg)),
        )
        for corpus_name, chunk_dirs in corpus_to_dirs:
            docs = _docs_for_dirs(chunk_dirs)
            build_and_save_keyword_index(
                index_dir=legacy_cfg.index_dir,
                corpus_name=corpus_name,
                docs=docs,
                tokenize_doc=_tokenize,
                k1=legacy_cfg.sparse_bm25_k1,
                b=legacy_cfg.sparse_bm25_b,
            )

    def _build_minecraft_wiki_keyword_inverted_indexes(
        self,
        *,
        minecraft_wiki_cfg,
        artifacts: _RepositoryIndexArtifacts | None,
    ) -> None:
        try:
            from langchain_core.documents import Document as LangDocument
            from kumc_agent.infra.indexing.keyword_inverted_index import (
                KEYWORD_CORPUS_MINECRAFT_WIKI_SECOND_REC_SPARSE,
                KEYWORD_CORPUS_MINECRAFT_WIKI_SPARSE,
                KEYWORD_CORPUS_MINECRAFT_WIKI_SPARSE_SECOND_REC,
                build_and_save_keyword_index,
                tokenize_sparse_doc,
            )
            from kumc_agent.infra.indexing.sparse_normalizer import (
                SparseNormalizer,
                SparseNormalizerConfig,
            )
        except Exception:
            logger.exception(
                "Minecraft Wiki keyword inverted index dependencies are unavailable. "
                "Falling back to lightweight keyword payload."
            )
            loaded = artifacts or self._load_minecraft_wiki_stage_artifacts(
                legacy_cfg=minecraft_wiki_cfg,
            )
            self._write_keyword_corpus(
                corpus_name=KEYWORD_CORPUS_MINECRAFT_WIKI_SPARSE,
                chunks=loaded.sparse_chunks,
            )
            self._write_keyword_corpus(
                corpus_name=KEYWORD_CORPUS_MINECRAFT_WIKI_SPARSE_SECOND_REC,
                chunks=loaded.sparse_chunks,
            )
            self._write_keyword_corpus(
                corpus_name=KEYWORD_CORPUS_MINECRAFT_WIKI_SECOND_REC_SPARSE,
                chunks=loaded.second_chunks,
            )
            return

        loaded = artifacts or self._load_minecraft_wiki_stage_artifacts(
            legacy_cfg=minecraft_wiki_cfg,
        )
        normalizer = SparseNormalizer(
            config=SparseNormalizerConfig(
                sudachi_mode=minecraft_wiki_cfg.sudachi_mode,
                use_normalized_form=minecraft_wiki_cfg.sparse_use_normalized_form,
                remove_symbols=minecraft_wiki_cfg.sparse_remove_symbols,
                remove_stopwords=False,
            )
        )

        def _doc(chunk: Chunk) -> LangDocument:
            metadata = dict(chunk.metadata or {})
            metadata.setdefault("chunk_uid", chunk.id)
            return LangDocument(page_content=chunk.text, metadata=metadata)

        def _tokenize(doc: LangDocument) -> list[str]:
            return tokenize_sparse_doc(
                doc,
                sparse_stage="second_recursive_sparse",
                sudachi_tokenize=normalizer.normalize_tokens,
            )

        sparse_docs = [
            _doc(chunk) for chunk in loaded.sparse_chunks if chunk.text.strip()
        ]
        second_docs = [
            _doc(chunk) for chunk in loaded.second_chunks if chunk.text.strip()
        ]
        for corpus_name, docs in (
            (KEYWORD_CORPUS_MINECRAFT_WIKI_SPARSE, sparse_docs),
            (KEYWORD_CORPUS_MINECRAFT_WIKI_SPARSE_SECOND_REC, sparse_docs),
            (KEYWORD_CORPUS_MINECRAFT_WIKI_SECOND_REC_SPARSE, second_docs),
        ):
            build_and_save_keyword_index(
                index_dir=minecraft_wiki_cfg.index_dir,
                corpus_name=corpus_name,
                docs=docs,
                tokenize_doc=_tokenize,
                k1=minecraft_wiki_cfg.sparse_bm25_k1,
                b=minecraft_wiki_cfg.sparse_bm25_b,
            )

    def _build_material_name_keyword_index(self, *, legacy_cfg) -> None:
        try:
            from langchain_core.documents import Document as LangDocument
            from kumc_agent.infra.indexing.keyword_inverted_index import (
                KEYWORD_CORPUS_MATERIAL_NAMES,
                build_and_save_keyword_index,
                tokenize_material_name_text,
            )
            from kumc_agent.infra.indexing.material_catalog import (
                load_material_catalog,
            )
        except Exception:
            logger.exception("Material-name keyword index dependencies are unavailable.")
            return

        docs: list[LangDocument] = []
        for entry in load_material_catalog(legacy_cfg.index_dir):
            source_type = str(entry.source_type or "").strip().lower()
            if source_type in _MATERIAL_NAME_INDEX_EXCLUDED_SOURCE_TYPES:
                continue
            texts: list[str] = []
            seen: set[str] = set()
            for value in (entry.canonical_name, entry.source_key, *entry.aliases):
                text = str(value or "").strip()
                if not text or text.casefold() in seen:
                    continue
                seen.add(text.casefold())
                texts.append(text)
            if not texts:
                continue
            docs.append(
                LangDocument(
                    page_content="\n".join(texts),
                    metadata={
                        "material_id": entry.material_id,
                        "source_type": entry.source_type,
                        "source_key": entry.source_key,
                        "canonical_name": entry.canonical_name,
                        "aliases": list(entry.aliases),
                    },
                )
            )

        build_and_save_keyword_index(
            index_dir=legacy_cfg.index_dir,
            corpus_name=KEYWORD_CORPUS_MATERIAL_NAMES,
            docs=docs,
            tokenize_doc=lambda doc: tokenize_material_name_text(doc.page_content),
            k1=legacy_cfg.sparse_bm25_k1,
            b=legacy_cfg.sparse_bm25_b,
        )

    def _build_material_catalog_legacy(self, *, legacy_cfg) -> None:
        try:
            from kumc_agent.infra.indexing.material_catalog import (
                build_and_save_material_catalog,
            )

            build_and_save_material_catalog(legacy_cfg)
        except Exception:
            logger.exception(
                "Legacy material catalog build failed. Falling back to basic catalog."
            )
            self._build_material_catalog(documents=self._parse_documents_from_ingestion())

    def _load_or_build_first_chunks(
        self,
        *,
        documents: list[Document],
        chunk_size: int,
        chunk_overlap: int,
        should_update: bool,
        force: bool,
        selected: set[str],
        allow_cancel: bool,
        cancel_event: threading.Event | None,
    ) -> list[Chunk]:
        stage_name = "first_recursive"
        if not force and not should_update and self._stage_exists(self._first_rec_dir):
            return self._load_stage_chunks(self._first_rec_dir)
        if selected and stage_name not in selected and self._stage_exists(self._first_rec_dir):
            return self._load_stage_chunks(self._first_rec_dir)
        chunks: list[Chunk] = []
        for doc in documents:
            self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)
            pieces = self._split_text(
                doc.text,
                chunk_size=max(1, chunk_size),
                chunk_overlap=max(0, chunk_overlap),
            )
            for idx, piece in enumerate(pieces):
                text = piece.strip()
                if not text:
                    continue
                chunk_id = stable_hash(f"{doc.id}:first:{idx}:{text[:64]}")
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=doc.id,
                        text=text,
                        index=idx,
                        metadata={
                            "source_type": doc.source_type,
                            "source_name": doc.source_name,
                            "source_uri": doc.source_uri,
                            "source_file_name": doc.source_name,
                            "chunk_stage": "first_recursive",
                            "chunk_id": idx,
                            "updated_at": (
                                doc.updated_at.astimezone(timezone.utc).isoformat()
                                if doc.updated_at is not None
                                else ""
                            ),
                            **doc.metadata,
                        },
                    )
                )
        self._write_stage_chunks(self._first_rec_dir, chunks)
        return chunks

    def _load_or_build_second_chunks(
        self,
        *,
        first_chunks: list[Chunk],
        chunk_size: int,
        chunk_overlap: int,
        enabled: bool,
        should_update: bool,
        force: bool,
        selected: set[str],
        allow_cancel: bool,
        cancel_event: threading.Event | None,
    ) -> list[Chunk]:
        if not enabled:
            return []
        stage_name = "second_recursive"
        if not force and not should_update and self._stage_exists(self._second_rec_dir):
            return self._load_stage_chunks(self._second_rec_dir)
        if selected and stage_name not in selected and self._stage_exists(self._second_rec_dir):
            return self._load_stage_chunks(self._second_rec_dir)
        chunks: list[Chunk] = []
        out_index = 0
        for parent in first_chunks:
            self._check_cancel(allow_cancel=allow_cancel, cancel_event=cancel_event)
            pieces = self._split_text(
                parent.text,
                chunk_size=max(1, chunk_size),
                chunk_overlap=max(0, chunk_overlap),
            )
            if len(pieces) == 1 and pieces[0].strip() == parent.text.strip():
                skip_parent_context = True
            else:
                skip_parent_context = False
            for piece in pieces:
                text = piece.strip()
                if not text:
                    continue
                metadata = dict(parent.metadata)
                metadata["chunk_stage"] = "second_recursive"
                metadata["parent_chunk_id"] = parent.metadata.get("chunk_id", parent.index)
                if skip_parent_context:
                    metadata["skip_parent_context"] = True
                metadata["chunk_id"] = out_index
                chunks.append(
                    Chunk(
                        id=stable_hash(f"{parent.id}:second:{out_index}:{text[:64]}"),
                        document_id=parent.document_id,
                        text=text,
                        index=out_index,
                        metadata=metadata,
                    )
                )
                out_index += 1
        self._write_stage_chunks(self._second_rec_dir, chunks)
        return chunks

    def _load_or_build_sparse_second_chunks(
        self,
        *,
        second_chunks: list[Chunk],
        enabled: bool,
        should_update: bool,
        force: bool,
        selected: set[str],
    ) -> list[Chunk]:
        if not enabled or not second_chunks:
            return []
        stage_name = "sparse_second_recursive"
        if not force and not should_update and self._stage_exists(self._sparse_second_rec_dir):
            return self._load_stage_chunks(self._sparse_second_rec_dir)
        if selected and stage_name not in selected and self._stage_exists(self._sparse_second_rec_dir):
            return self._load_stage_chunks(self._sparse_second_rec_dir)

        chunks: list[Chunk] = []
        for idx, chunk in enumerate(second_chunks):
            tokens = [token for token in (chunk.text or "").lower().split() if token]
            if not tokens:
                continue
            metadata = dict(chunk.metadata)
            metadata["chunk_stage"] = "second_recursive_sparse"
            metadata["chunk_id"] = idx
            text = " ".join(tokens)
            chunks.append(
                Chunk(
                    id=stable_hash(f"{chunk.id}:sparse:{idx}:{text[:64]}"),
                    document_id=chunk.document_id,
                    text=text,
                    index=idx,
                    metadata=metadata,
                )
            )
        self._write_stage_chunks(self._sparse_second_rec_dir, chunks)
        return chunks

    def _load_or_build_summary_chunks(
        self,
        *,
        first_chunks: list[Chunk],
        enabled: bool,
        target_characters: int,
        should_update: bool,
        force: bool,
        selected: set[str],
    ) -> list[Chunk]:
        if not enabled:
            return []
        stage_name = "summary"
        if not force and not should_update and self._stage_exists(self._summary_dir):
            return self._load_stage_chunks(self._summary_dir)
        if selected and stage_name not in selected and self._stage_exists(self._summary_dir):
            return self._load_stage_chunks(self._summary_dir)

        limit = max(32, target_characters)
        chunks = self._build_summary_chunks_for_first_chunks(
            first_chunks=first_chunks,
            target_characters=limit,
            include_index_in_hash=True,
        )
        self._write_stage_chunks(self._summary_dir, chunks)
        return chunks

    def _build_summary_chunks_for_first_chunks(
        self,
        *,
        first_chunks: list[Chunk],
        target_characters: int,
        include_index_in_hash: bool,
    ) -> list[Chunk]:
        limit = max(32, target_characters)
        batch_size = self._summary_batch_size()
        total_batches = (len(first_chunks) + batch_size - 1) // batch_size
        if total_batches > 1:
            logger.info(
                "Generating summary chunks in %d batches (batch_size=%d).",
                total_batches,
                batch_size,
            )
        summaries = [""] * len(first_chunks)
        for batch_start in range(0, len(first_chunks), batch_size):
            batch = first_chunks[batch_start : batch_start + batch_size]
            max_workers = max(1, min(len(batch), batch_size))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: dict[Future[str], tuple[int, Chunk]] = {}
                for offset, source_chunk in enumerate(batch):
                    idx = batch_start + offset
                    futures[
                        executor.submit(
                            self._build_summary_text,
                            text=source_chunk.text or "",
                            target_characters=limit,
                        )
                    ] = (idx, source_chunk)
                for future in as_completed(futures):
                    idx, source_chunk = futures[future]
                    fallback = (source_chunk.text or "").strip()[:limit]
                    try:
                        summary_text = (future.result() or "").strip()
                    except Exception:
                        logger.exception(
                            "Summary chunk worker failed. Fallback to truncation."
                        )
                        summary_text = fallback
                    summaries[idx] = summary_text or fallback

        chunks: list[Chunk] = []
        for idx, chunk in enumerate(first_chunks):
            summary = summaries[idx]
            if not summary:
                continue
            metadata = dict(chunk.metadata)
            metadata["chunk_stage"] = "summary"
            metadata["parent_chunk_id"] = chunk.metadata.get("chunk_id", chunk.index)
            metadata["chunk_id"] = idx
            hash_seed = (
                f"{chunk.id}:summary:{idx}:{summary[:64]}"
                if include_index_in_hash
                else f"{chunk.id}:summary:{summary[:256]}"
            )
            chunks.append(
                Chunk(
                    id=stable_hash(hash_seed),
                    document_id=chunk.document_id,
                    text=summary,
                    index=idx,
                    metadata=metadata,
                )
            )
        return chunks

    def _summary_batch_size(self) -> int:
        return max(1, int(self._runtime.indexing.chunking.summary_batch_size))

    def _build_summary_text(
        self,
        *,
        text: str,
        target_characters: int,
    ) -> str:
        source_text = (text or "").strip()
        if not source_text:
            return ""
        fallback = source_text[:target_characters]
        llm_provider = (
            self._runtime.indexing.chunking.summary_llm_provider or ""
        ).strip().lower()
        if llm_provider in {"", "none", "off", "disabled", "false", "0"}:
            return fallback
        if self._summary_llm is None:
            return fallback
        user_prompt = (
            f"次の本文を{target_characters}文字以内で要約してください。"
            "\n箇条書きにせず、重要な固有名詞・数値・日付は残してください。"
            f"\n\n本文:\n{source_text}"
        )
        try:
            summary = self._summary_llm.generate(
                system_prompt=(
                    "You summarize Japanese documents concisely and faithfully."
                ),
                user_prompt=user_prompt,
                temperature=self._runtime.indexing.chunking.summary_temperature,
                max_output_tokens=(
                    self._runtime.indexing.chunking.summary_max_output_tokens
                ),
            )
        except Exception:
            logger.exception(
                "Summary chunk generation failed. Fallback to truncation."
            )
            return fallback
        normalized = (summary or "").strip()
        if not normalized or self._is_llm_error_text(normalized):
            return fallback
        return normalized[:target_characters]

    @staticmethod
    def _is_llm_error_text(text: str) -> bool:
        return (
            "ローカルフォールバック回答" in text
            or text.endswith("回答生成に失敗しました。")
        )

    def _build_keyword_indexes_payload(
        self,
        *,
        sparse_chunks: list[Chunk],
        second_chunks: list[Chunk],
    ) -> None:
        self._write_keyword_corpus(
            corpus_name=KEYWORD_CORPUS_SPARSE,
            chunks=sparse_chunks,
        )
        self._write_keyword_corpus(
            corpus_name=KEYWORD_CORPUS_SPARSE_SECOND_REC,
            chunks=sparse_chunks,
        )
        self._write_keyword_corpus(
            corpus_name=KEYWORD_CORPUS_SECOND_REC_SPARSE,
            chunks=second_chunks,
        )

    def _write_keyword_corpus(
        self,
        *,
        corpus_name: str,
        chunks: list[Chunk],
    ) -> None:
        keyword_dir = self._runtime.app.index_dir / "keyword"
        keyword_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus_name": corpus_name,
            "docs": [
                {"page_content": chunk.text, "metadata": chunk.metadata}
                for chunk in chunks
            ],
        }
        path = keyword_dir / f"{corpus_name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    def _build_material_catalog(self, *, documents: list[Document]) -> None:
        materials: list[dict[str, object]] = []
        for doc in documents:
            source_key = str(doc.source_name or "").strip()
            if not source_key:
                continue
            canonical = Path(source_key).stem or source_key
            aliases = [canonical, source_key]
            materials.append(
                {
                    "material_id": f"{doc.source_type}:{source_key}",
                    "source_type": doc.source_type,
                    "source_key": source_key,
                    "canonical_name": canonical,
                    "aliases": aliases,
                    "raw_path": source_key,
                }
            )
        catalog = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "materials": materials,
        }
        path = self._runtime.app.index_dir / "material_catalog.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    def _parse_documents_from_ingestion(self) -> list[Document]:
        if not self._ingestion_dir.exists():
            return []

        source_specs: tuple[tuple[Path, set[str], str], ...] = (
            (self._ingestion_docs_dir, {".md"}, "docs"),
            (self._ingestion_sheets_dir, {".csv"}, "sheets"),
            (self._ingestion_messages_dir, {".jsonl"}, "messages"),
            (self._ingestion_x_dir, {".jsonl"}, "x_posts"),
            (self._ingestion_vc_dir, {".txt"}, "vc_transcript"),
            (self._ingestion_hatenablog_dir, {".md"}, "hatenablog"),
            (self._ingestion_crafters_colony_dir, {".md"}, "crafters_colony"),
            (self._ingestion_notion_dir, {".md"}, "notion"),
        )

        documents: list[Document] = []
        for root_dir, extensions, default_source_type in source_specs:
            if not root_dir.exists():
                continue
            for path in sorted(root_dir.rglob("*"), key=lambda value: str(value)):
                if not path.is_file():
                    continue
                if path.name.endswith((".meta.json", ".mtime.json", ".state.json")):
                    continue
                if path.suffix.lower() not in extensions:
                    continue
                text, extracted_meta, updated_at = self._read_ingestion_document(path)
                if not text.strip():
                    continue
                source_name = str(path.relative_to(self._ingestion_dir)).replace("\\", "/")
                source_type = str(extracted_meta.get("source_type") or "").strip().lower()
                if not source_type:
                    source_type = default_source_type
                doc_id = stable_hash(f"{source_type}:{source_name}")
                metadata = {"path": source_name, **extracted_meta}
                if "source_date" not in metadata:
                    inferred_source_date = self._derive_source_date(metadata)
                    if inferred_source_date:
                        metadata["source_date"] = inferred_source_date
                documents.append(
                    Document(
                        id=doc_id,
                        text=text,
                        source_type=source_type,
                        source_name=source_name,
                        source_uri="",
                        updated_at=updated_at,
                        metadata=metadata,
                    )
                )
        return documents

    @staticmethod
    def _read_ingestion_document(path: Path) -> tuple[str, dict[str, object], datetime | None]:
        sidecar_metadata = IndexingService._read_sidecar_metadata(path)
        file_updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            texts: list[str] = []
            latest_line_metadata: dict[str, object] = {}
            latest_line_updated_at: datetime | None = None
            with path.open("r", encoding="utf-8") as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(payload.get("text") or "").strip()
                    if text:
                        texts.append(text)
                    raw_metadata = payload.get("metadata")
                    if not isinstance(raw_metadata, dict):
                        continue
                    metadata = {
                        str(key): value
                        for key, value in raw_metadata.items()
                    }
                    candidate_updated_at = IndexingService._extract_updated_at(metadata)
                    if latest_line_updated_at is None:
                        latest_line_metadata = metadata
                        latest_line_updated_at = candidate_updated_at
                        continue
                    if (
                        candidate_updated_at is not None
                        and (
                            latest_line_updated_at is None
                            or candidate_updated_at >= latest_line_updated_at
                        )
                    ):
                        latest_line_metadata = metadata
                        latest_line_updated_at = candidate_updated_at

            merged_metadata = {
                **sidecar_metadata,
                **latest_line_metadata,
                "file_mtime": file_updated_at.isoformat(),
            }
            updated_at = latest_line_updated_at or IndexingService._extract_updated_at(merged_metadata) or file_updated_at
            merged_metadata["updated_at"] = updated_at.isoformat()
            return "\n".join(texts), merged_metadata, updated_at

        text = path.read_text(encoding="utf-8", errors="ignore")
        merged_metadata = {
            **sidecar_metadata,
            "file_mtime": file_updated_at.isoformat(),
        }
        updated_at = IndexingService._extract_updated_at(merged_metadata) or file_updated_at
        merged_metadata["updated_at"] = updated_at.isoformat()
        return text, merged_metadata, updated_at

    @staticmethod
    def _read_sidecar_metadata(path: Path) -> dict[str, object]:
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        if not sidecar.exists():
            return {}
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): value
            for key, value in payload.items()
        }

    @staticmethod
    def _extract_updated_at(metadata: dict[str, object]) -> datetime | None:
        for key in (
            "updated_at",
            "message_timestamp",
            "drive_modified_time",
            "hatenablog_updated_at",
            "hatenablog_created_at",
            "crafters_colony_published_at",
            "notion_last_edited_time",
            "created_at",
            "source_date",
            "first_message_date",
        ):
            value = str(metadata.get(key) or "").strip()
            if not value:
                continue
            parsed = IndexingService._parse_datetime(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _derive_source_date(metadata: dict[str, object]) -> str | None:
        updated_at = IndexingService._extract_updated_at(metadata)
        if updated_at is None:
            return None
        return updated_at.astimezone(timezone.utc).strftime("%Y/%m/%d")

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        raw = (value or "").strip()
        if not raw or raw == "不明":
            return None
        iso_value = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(iso_value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
            try:
                parsed = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                continue
        return None

    def _clear_ingestion_source_contents(self) -> None:
        for path in (
            self._ingestion_docs_dir,
            self._ingestion_docs_normalized_dir,
            self._ingestion_sheets_dir,
            self._ingestion_sheets_structured_dir,
            self._ingestion_messages_dir,
            self._ingestion_x_dir,
            self._ingestion_vc_dir,
            self._ingestion_hatenablog_dir,
            self._ingestion_crafters_colony_dir,
            self._ingestion_notion_dir,
            self._ingestion_minecraft_wiki_dir,
            self._ingestion_dir / "images",
        ):
            self._clear_dir_contents(path)
        try:
            self._runtime.app.index_documents_path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _clear_dir_contents(target: Path) -> None:
        if not target.exists():
            return
        for path in target.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted((p for p in target.rglob("*") if p.is_dir()), reverse=True):
            try:
                path.rmdir()
            except OSError:
                continue

    @staticmethod
    def _stage_exists(stage_dir: Path) -> bool:
        return stage_dir.exists() and any(stage_dir.rglob("*.jsonl"))

    @staticmethod
    def _write_stage_chunks(stage_dir: Path, chunks: list[Chunk]) -> None:
        stage_dir.mkdir(parents=True, exist_ok=True)
        by_source: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            source_type = str(chunk.metadata.get("source_type") or "misc").strip().lower()
            if not source_type:
                source_type = "misc"
            by_source[source_type].append(chunk)
        for source_type, grouped in by_source.items():
            out_path = stage_dir / f"{source_type}.jsonl"
            with out_path.open("w", encoding="utf-8") as fw:
                for chunk in grouped:
                    fw.write(
                        json.dumps(
                            {
                                "id": chunk.id,
                                "document_id": chunk.document_id,
                                "text": chunk.text,
                                "index": chunk.index,
                                "metadata": chunk.metadata,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    @staticmethod
    def _load_stage_chunks(stage_dir: Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        if not stage_dir.exists():
            return chunks
        for path in sorted(stage_dir.rglob("*.jsonl")):
            with path.open("r", encoding="utf-8") as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    chunks.append(
                        Chunk(
                            id=str(payload["id"]),
                            document_id=str(payload["document_id"]),
                            text=str(payload["text"]),
                            index=int(payload["index"]),
                            metadata=dict(payload.get("metadata", {})),
                        )
                    )
        return chunks

    @staticmethod
    def _check_cancel(
        *,
        allow_cancel: bool,
        cancel_event: threading.Event | None,
    ) -> None:
        if not allow_cancel:
            return
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Index build cancelled.")

    @staticmethod
    def _split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
        value = (text or "").strip()
        if not value:
            return []
        if len(value) <= chunk_size:
            return [value]
        out: list[str] = []
        start = 0
        overlap = max(0, min(chunk_overlap, chunk_size - 1))
        while start < len(value):
            end = min(len(value), start + chunk_size)
            piece = value[start:end].strip()
            if piece:
                out.append(piece)
            if end >= len(value):
                break
            start = max(start + 1, end - overlap)
        return out
