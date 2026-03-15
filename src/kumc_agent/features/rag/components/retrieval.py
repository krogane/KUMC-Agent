from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import re

import numpy as np

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever
from kumc_agent.utils.hashing import merge_unique, stable_hash

logger = logging.getLogger(__name__)

_YMD_SLASH_RE = re.compile(r"^(?P<y>\d{4})/(?P<m>\d{2})/(?P<d>\d{2})$")
_YMD_DASH_RE = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})$")
_YMD_COMPACT_RE = re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})$")
_RECENCY_KEYS = (
    "updated_at",
    "message_timestamp",
    "drive_modified_time",
    "hatenablog_updated_at",
    "hatenablog_created_at",
    "crafters_colony_published_at",
    "first_message_date",
    "source_date",
    "created_at",
)
_SECOND_REC_SPARSE_STAGE = "second_recursive_sparse"
_KEYWORD_CORPUS_SPARSE_SECOND_REC = "sparse_second_rec"
_KEYWORD_CORPUS_SECOND_REC_SPARSE = "second_rec_sparse"


@dataclass(frozen=True)
class _ScoredChunk:
    chunk: Chunk
    score: float


class RetrievalComponent:
    def __init__(
        self,
        *,
        embedder: EmbedderPort,
        dense_index: FaissLikeIndex,
        sparse_index: SudachiBM25Retriever,
        sparse_sudachi_mode: str = "B",
        sparse_use_normalized_form: bool = True,
        sparse_remove_symbols: bool = True,
    ) -> None:
        self._embedder = embedder
        self._dense_index = dense_index
        self._sparse_index = sparse_index
        self._sparse_sudachi_mode = sparse_sudachi_mode
        self._sparse_use_normalized_form = bool(sparse_use_normalized_form)
        self._sparse_remove_symbols = bool(sparse_remove_symbols)

    @property
    def index_dir(self):
        return self._dense_index._index_dir  # noqa: SLF001

    def retrieve(
        self,
        query: str,
        *,
        dense_top_k: int,
        sparse_top_k: int,
        sparse_initial_sparse_top_k: int | None = None,
        recency_mode: str = "off",
        recency_weight_soft: float = 0.20,
        recency_weight_hard: float = 0.45,
        recency_half_life_days: float = 45.0,
        mmr_lambda: float = 0.75,
    ) -> list[Chunk]:
        query_vector = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        dense_hits = self._dense_index.search(
            query_vector=query_vector,
            top_k=max(0, dense_top_k),
        )
        sparse_hits = self._search_sparse_mixed_sources(
            query=query,
            top_k=max(0, sparse_top_k),
            sparse_top_k=max(
                0,
                sparse_top_k
                if sparse_initial_sparse_top_k is None
                else sparse_initial_sparse_top_k,
            ),
        )
        if not sparse_hits:
            sparse_hits = self._sparse_index.search_with_scores(
                query,
                top_k=max(0, sparse_top_k),
            )
        scored = self._merge_scores(dense_hits=dense_hits, sparse_hits=sparse_hits)
        _ = (
            recency_mode,
            recency_weight_soft,
            recency_weight_hard,
            recency_half_life_days,
        )  # Recency is applied in RagService during rerank scoring.
        ranked = sorted(scored, key=lambda item: item.score, reverse=True)
        _ = mmr_lambda  # Applied explicitly in RagService after rerank stage.
        return [item.chunk for item in ranked]

    def reorder_with_mmr(
        self,
        *,
        query: str,
        chunks: list[Chunk],
        mmr_lambda: float,
    ) -> list[Chunk]:
        query_vector = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        return self._apply_mmr(
            query_vector=query_vector,
            chunks=chunks,
            mmr_lambda=mmr_lambda,
        )

    def rank_texts_by_dense(
        self,
        *,
        query: str,
        texts: Sequence[str],
        top_k: int = 1,
    ) -> list[tuple[int, float]]:
        if not texts:
            return []
        limit = max(0, int(top_k))
        if limit <= 0:
            return []

        values = [str(text or "").strip() for text in texts]
        valid_indices = [idx for idx, text in enumerate(values) if text]
        if not valid_indices:
            return []
        valid_texts = [values[idx] for idx in valid_indices]

        try:
            query_vector = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
            query_norm = _normalize_vector(query_vector.reshape(-1))
            if query_norm is None:
                return []

            doc_vectors = np.asarray(
                self._embedder.embed_documents(valid_texts),
                dtype=np.float32,
            )
            if doc_vectors.ndim == 1:
                doc_vectors = doc_vectors.reshape(1, -1)
            if doc_vectors.ndim != 2 or doc_vectors.shape[0] != len(valid_texts):
                return []
            if doc_vectors.shape[1] != query_norm.shape[0]:
                return []

            doc_norm = _normalize_matrix(doc_vectors)
            scores = doc_norm @ query_norm
            if scores.ndim != 1:
                return []

            ranked = np.argsort(scores)[::-1]
            results: list[tuple[int, float]] = []
            for pos in ranked:
                if len(results) >= limit:
                    break
                idx = valid_indices[int(pos)]
                score = float(scores[int(pos)])
                if not np.isfinite(score):
                    continue
                results.append((idx, score))
            return results
        except Exception:
            logger.exception("Dense text ranking failed.")
            return []

    def _search_sparse_mixed_sources(
        self,
        *,
        query: str,
        top_k: int,
        sparse_top_k: int,
    ) -> list[tuple[Chunk, float]]:
        total_k = max(0, top_k)
        if total_k <= 0:
            return []
        sparse_k = min(max(0, sparse_top_k), total_k)
        second_rec_k = total_k - sparse_k

        tokens = self._query_tokens(query)
        if not tokens:
            return []

        sparse_hits = self._search_keyword_index(
            tokens=tokens,
            top_k=total_k,
            corpus_name=_KEYWORD_CORPUS_SPARSE_SECOND_REC,
            restore_sparse_stage=True,
        )
        second_rec_hits = self._search_keyword_index(
            tokens=tokens,
            top_k=total_k,
            corpus_name=_KEYWORD_CORPUS_SECOND_REC_SPARSE,
            restore_sparse_stage=False,
        )

        selected_sparse = sparse_hits[:sparse_k]
        selected_second_rec = second_rec_hits[:second_rec_k]
        merged = self._merge_sparse_hits([selected_sparse, selected_second_rec])
        if len(merged) < total_k:
            merged = self._merge_sparse_hits(
                [merged, sparse_hits[sparse_k:], second_rec_hits[second_rec_k:]]
            )
        return merged[:total_k]

    def _search_keyword_index(
        self,
        *,
        tokens: Sequence[str],
        top_k: int,
        corpus_name: str,
        restore_sparse_stage: bool,
    ) -> list[tuple[Chunk, float]]:
        if top_k <= 0 or not tokens:
            return []
        try:
            from kumc_agent.infra.indexing.keyword_inverted_index import (
                load_keyword_index,
            )
        except Exception:
            logger.exception("Keyword index module import failed.")
            return []

        keyword_index = load_keyword_index(
            index_dir=self.index_dir,
            corpus_name=corpus_name,
        )
        if keyword_index is None or not keyword_index.docs:
            return []
        scores = keyword_index.get_scores(tokens)
        if scores is None or len(scores) == 0:
            return []

        ranked = sorted(range(len(keyword_index.docs)), key=lambda i: scores[i], reverse=True)
        hits: list[tuple[Chunk, float]] = []
        seen_ids: set[str] = set()
        second_rec_map = self._second_rec_chunk_map() if restore_sparse_stage else {}
        for idx in ranked:
            score = float(scores[idx])
            if score <= 0.0:
                break
            chunk = self._keyword_doc_to_chunk(keyword_index.docs[idx])
            if restore_sparse_stage:
                chunk = self._restore_sparse_hit_chunk(
                    chunk,
                    second_rec_map=second_rec_map,
                )
            if chunk.id in seen_ids:
                continue
            seen_ids.add(chunk.id)
            hits.append((chunk, score))
            if len(hits) >= top_k:
                break
        return hits

    def _query_tokens(self, query: str) -> list[str]:
        try:
            from kumc_agent.infra.indexing.sparse_normalizer import (
                SparseNormalizer,
                SparseNormalizerConfig,
            )
        except Exception:
            return self._sparse_index._tokenize(query)  # noqa: SLF001
        try:
            normalizer = SparseNormalizer(
                config=SparseNormalizerConfig(
                    sudachi_mode=self._sparse_sudachi_mode,
                    use_normalized_form=self._sparse_use_normalized_form,
                    remove_symbols=self._sparse_remove_symbols,
                    remove_stopwords=False,
                )
            )
            return normalizer.normalize_tokens(query)
        except Exception:
            logger.exception("Sparse query normalization failed.")
            return self._sparse_index._tokenize(query)  # noqa: SLF001

    def _keyword_doc_to_chunk(self, doc) -> Chunk:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        text = str(getattr(doc, "page_content", "") or "").strip()
        if not text:
            text = "(empty)"
        chunk_uid = str(metadata.get("chunk_uid") or "").strip()
        if not chunk_uid:
            chunk_uid = stable_hash(
                f"{metadata.get('source_type')}|{metadata.get('source_file_name')}|"
                f"{metadata.get('chunk_stage')}|{metadata.get('chunk_id')}|{text[:256]}"
            )
        source_type = str(metadata.get("source_type") or "").strip()
        source_name = str(
            metadata.get("drive_file_id")
            or metadata.get("source_file_name")
            or metadata.get("path")
            or ""
        ).strip()
        document_id = stable_hash(f"{source_type}:{source_name}")
        index = self._normalize_chunk_id(metadata.get("chunk_id")) or 0
        return Chunk(
            id=chunk_uid,
            document_id=document_id,
            text=text,
            index=index,
            metadata=metadata,
        )

    def _restore_sparse_hit_chunk(
        self,
        chunk: Chunk,
        *,
        second_rec_map: dict[tuple[object, ...], Chunk],
    ) -> Chunk:
        metadata = chunk.metadata or {}
        if str(metadata.get("chunk_stage") or "") != _SECOND_REC_SPARSE_STAGE:
            return chunk
        chunk_id = self._normalize_chunk_id(metadata.get("chunk_id"))
        if chunk_id is None:
            return chunk
        key = self._chunk_lookup_key(metadata, chunk_id=chunk_id)
        if key is None:
            return chunk
        resolved = second_rec_map.get(key)
        return resolved if resolved is not None else chunk

    def _second_rec_chunk_map(self) -> dict[tuple[object, ...], Chunk]:
        try:
            from kumc_agent.infra.indexing.keyword_inverted_index import (
                load_keyword_index,
            )
        except Exception:
            return {}
        keyword_index = load_keyword_index(
            index_dir=self.index_dir,
            corpus_name=_KEYWORD_CORPUS_SECOND_REC_SPARSE,
        )
        if keyword_index is None:
            return {}
        mapping: dict[tuple[object, ...], Chunk] = {}
        for doc in keyword_index.docs:
            chunk = self._keyword_doc_to_chunk(doc)
            chunk_id = self._normalize_chunk_id(chunk.metadata.get("chunk_id"))
            if chunk_id is None:
                continue
            key = self._chunk_lookup_key(chunk.metadata, chunk_id=chunk_id)
            if key is None:
                continue
            mapping[key] = chunk
        return mapping

    @staticmethod
    def _chunk_lookup_key(
        metadata: dict[str, object],
        *,
        chunk_id: int,
    ) -> tuple[object, ...] | None:
        source = metadata.get("drive_file_id") or metadata.get("source_file_name")
        if not source:
            return None
        source_type = metadata.get("source_type") or ""
        return (source_type, source, chunk_id)

    @staticmethod
    def _normalize_chunk_id(value: object) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _merge_sparse_hits(groups: list[list[tuple[Chunk, float]]]) -> list[tuple[Chunk, float]]:
        merged: list[tuple[Chunk, float]] = []
        seen: set[str] = set()
        for group in groups:
            for chunk, score in group:
                if chunk.id in seen:
                    continue
                seen.add(chunk.id)
                merged.append((chunk, score))
        return merged

    @staticmethod
    def _merge_scores(*, dense_hits, sparse_hits) -> list[_ScoredChunk]:
        dense_scores = {item.chunk.id: float(item.score) for item in dense_hits}
        sparse_scores = {chunk.id: float(score) for chunk, score in sparse_hits}
        dense_chunks = {item.chunk.id: item.chunk for item in dense_hits}
        sparse_chunks = {chunk.id: chunk for chunk, _ in sparse_hits}

        dense_norm = _normalize_score_map(dense_scores)
        sparse_norm = _normalize_score_map(sparse_scores)

        merged_ids = [
            str(value)
            for value in merge_unique(
                [item.chunk.id for item in dense_hits]
                + [chunk.id for chunk, _ in sparse_hits]
            )
        ]
        out: list[_ScoredChunk] = []
        for chunk_id in merged_ids:
            chunk = dense_chunks.get(chunk_id) or sparse_chunks.get(chunk_id)
            if chunk is None:
                continue
            score = dense_norm.get(chunk_id, 0.0) + sparse_norm.get(chunk_id, 0.0)
            out.append(_ScoredChunk(chunk=chunk, score=score))
        return out

    @staticmethod
    def _apply_recency(
        scored: list[_ScoredChunk],
        *,
        mode: str,
        recency_weight_soft: float,
        recency_weight_hard: float,
        recency_half_life_days: float,
    ) -> list[_ScoredChunk]:
        normalized_mode = (mode or "off").strip().lower()
        if normalized_mode not in {"soft", "hard"}:
            return scored
        if not scored:
            return scored

        soft_weight = max(0.0, min(1.0, float(recency_weight_soft)))
        hard_weight = max(0.0, min(1.0, float(recency_weight_hard)))
        weight = soft_weight if normalized_mode == "soft" else hard_weight
        half_life_days = max(0.1, float(recency_half_life_days))
        now = datetime.now(timezone.utc)
        adjusted: list[_ScoredChunk] = []
        for item in scored:
            updated_at = _chunk_updated_at(item.chunk)
            if updated_at is None:
                recency_score = 0.5
            else:
                age_days = max(0.0, (now - updated_at).total_seconds() / 86400.0)
                recency_score = 0.5 ** (age_days / half_life_days)
            blended = ((1.0 - weight) * item.score) + (weight * recency_score)
            adjusted.append(_ScoredChunk(chunk=item.chunk, score=blended))
        return adjusted

    def _apply_mmr(
        self,
        *,
        query_vector: np.ndarray,
        chunks: list[Chunk],
        mmr_lambda: float,
    ) -> list[Chunk]:
        if len(chunks) <= 1:
            return chunks
        lambda_mult = max(0.0, min(1.0, float(mmr_lambda)))
        if lambda_mult >= 0.999:
            return chunks
        try:
            doc_vectors = np.asarray(
                self._embedder.embed_documents([chunk.text for chunk in chunks]),
                dtype=np.float32,
            )
            if doc_vectors.ndim != 2 or doc_vectors.shape[0] != len(chunks):
                return chunks
            query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
            if query.ndim != 1 or query.size == 0:
                return chunks
            if doc_vectors.shape[1] != query.shape[0]:
                return chunks
            query_norm = _normalize_vector(query)
            if query_norm is None:
                return chunks
            doc_norm = _normalize_matrix(doc_vectors)

            sim_to_query = doc_norm @ query_norm
            fixed = min(3, len(chunks))
            selected: list[int] = list(range(fixed))
            remaining = list(range(fixed, len(chunks)))

            while remaining:
                selected_vecs = doc_norm[selected]
                sims = doc_norm[remaining] @ selected_vecs.T
                max_div = sims.max(axis=1)
                scores = (lambda_mult * sim_to_query[remaining]) - (
                    (1.0 - lambda_mult) * max_div
                )
                best_pos = int(np.argmax(scores))
                selected.append(remaining.pop(best_pos))

            return [chunks[idx] for idx in selected]
        except Exception:
            logger.exception("MMR ranking failed. Falling back to hybrid ranking.")
            return chunks


def _normalize_score_map(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    max_score = max(values)
    min_score = min(values)
    if max_score == min_score:
        if max_score <= 0.0:
            return {key: 0.0 for key in scores}
        return {key: 1.0 for key in scores}
    scale = max_score - min_score
    return {
        key: (value - min_score) / scale
        for key, value in scores.items()
    }


def _chunk_updated_at(chunk: Chunk) -> datetime | None:
    metadata = chunk.metadata or {}
    for key in _RECENCY_KEYS:
        raw = str(metadata.get(key) or "").strip()
        if not raw:
            continue
        parsed = _parse_datetime(raw)
        if parsed is not None:
            return parsed
    return None


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

    for pattern in (_YMD_SLASH_RE, _YMD_DASH_RE, _YMD_COMPACT_RE):
        match = pattern.match(raw)
        if match is None:
            continue
        try:
            parsed = datetime(
                int(match.group("y")),
                int(match.group("m")),
                int(match.group("d")),
                tzinfo=timezone.utc,
            )
            return parsed
        except ValueError:
            return None
    return None


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
