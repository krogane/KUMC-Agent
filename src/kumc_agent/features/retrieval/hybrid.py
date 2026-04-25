from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.retrieval import RetrievalQuery, ScoredChunk
from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.infra.retrieval.cross_encoder import CrossEncoderReranker
from kumc_agent.infra.retrieval_wave3 import RetrievalRepository
from kumc_agent.utils.hashing import cosine_similarity_matrix, stable_hash


@dataclass(frozen=True)
class HybridRetrievalConfig:
    dense_top_k: int = 40
    sparse_top_k: int = 40
    rerank_pool_size: int = 20
    top_k: int = 8
    rrf_k: int = 60
    doc_cap: int = 2
    mmr_lambda: float = 0.75
    embedding_model: str = "hashed"
    embedding_dimensions: int = 256


class HybridRetrievalService:
    def __init__(
        self,
        *,
        repository: RetrievalRepository,
        embedder: EmbedderPort,
        config: HybridRetrievalConfig,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._config = config
        self._reranker = reranker

    def embed_missing_chunks(self, chunks: list[Chunk]) -> dict[str, np.ndarray]:
        cached = self._repository.load_embeddings(
            model=self._config.embedding_model,
            dimensions=self._config.embedding_dimensions,
        )
        missing = [chunk for chunk in chunks if chunk.id not in cached]
        if missing:
            vectors = np.asarray(
                self._embedder.embed_documents([_embedding_text(chunk) for chunk in missing]),
                dtype=np.float32,
            )
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            to_save: dict[str, tuple[np.ndarray, str]] = {}
            for chunk, vector in zip(missing, vectors):
                checksum = str(chunk.metadata.get("checksum") or stable_hash(chunk.text))
                cached[chunk.id] = vector
                to_save[chunk.id] = (vector, checksum)
            self._repository.save_embeddings(
                model=self._config.embedding_model,
                dimensions=self._config.embedding_dimensions,
                embeddings=to_save,
            )
        return cached

    def retrieve(self, query: RetrievalQuery) -> list[ScoredChunk]:
        chunks = self._repository.load_chunks(query=query)
        if not chunks:
            self._repository.record_search_run(
                query=query,
                results=[],
                status="no_chunks",
                metadata={"reason": "no_visible_chunks"},
            )
            return []

        embeddings = self.embed_missing_chunks(chunks)
        dense = self._dense_hits(query.text, chunks=chunks, embeddings=embeddings)
        sparse = self._sparse_hits(query.text, chunks=chunks)
        fused = self._rrf(dense=dense, sparse=sparse)
        reranked = self._rerank(query=query.text, scored=fused)
        capped = self._apply_doc_cap(reranked)
        selected = self._apply_mmr(query=query.text, scored=capped, embeddings=embeddings)
        final = [
            ScoredChunk(
                chunk=item.chunk,
                score=item.score,
                rank=index + 1,
                score_breakdown=item.score_breakdown,
            )
            for index, item in enumerate(selected[: max(0, self._config.top_k)])
        ]
        self._repository.record_search_run(
            query=query,
            results=final,
            status="succeeded",
            metadata={
                "candidate_chunks": len(chunks),
                "dense_hits": len(dense),
                "sparse_hits": len(sparse),
            },
        )
        return final

    def _dense_hits(
        self,
        query: str,
        *,
        chunks: list[Chunk],
        embeddings: dict[str, np.ndarray],
    ) -> list[ScoredChunk]:
        if not chunks or self._config.dense_top_k <= 0:
            return []
        query_vector = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        matrix = np.vstack([embeddings[chunk.id] for chunk in chunks if chunk.id in embeddings])
        chunk_list = [chunk for chunk in chunks if chunk.id in embeddings]
        if matrix.size == 0 or not chunk_list:
            return []
        scores = cosine_similarity_matrix(query_vector, matrix)
        order = np.argsort(-scores)[: max(0, self._config.dense_top_k)]
        return [
            ScoredChunk(
                chunk=chunk_list[int(pos)],
                score=float(scores[int(pos)]),
                rank=rank + 1,
                score_breakdown={"dense": float(scores[int(pos)])},
            )
            for rank, pos in enumerate(order)
        ]

    def _sparse_hits(self, query: str, *, chunks: list[Chunk]) -> list[ScoredChunk]:
        if not chunks or self._config.sparse_top_k <= 0:
            return []
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        scored: list[ScoredChunk] = []
        for chunk in chunks:
            chunk_tokens = _tokens(_embedding_text(chunk))
            if not chunk_tokens:
                continue
            overlap = len(set(query_tokens) & set(chunk_tokens))
            if overlap <= 0:
                continue
            score = overlap / math.sqrt(max(1, len(set(chunk_tokens))))
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    rank=0,
                    score_breakdown={"sparse": score},
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return [
            ScoredChunk(
                chunk=item.chunk,
                score=item.score,
                rank=index + 1,
                score_breakdown=item.score_breakdown,
            )
            for index, item in enumerate(scored[: max(0, self._config.sparse_top_k)])
        ]

    def _rrf(
        self,
        *,
        dense: list[ScoredChunk],
        sparse: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        by_id: dict[str, ScoredChunk] = {}
        for source_name, results in (("dense", dense), ("sparse", sparse)):
            for result in results:
                score = 1.0 / (float(self._config.rrf_k) + float(result.rank))
                existing = by_id.get(result.chunk.id)
                if existing is None:
                    by_id[result.chunk.id] = ScoredChunk(
                        chunk=result.chunk,
                        score=score,
                        rank=0,
                        score_breakdown={source_name: score},
                    )
                else:
                    by_id[result.chunk.id] = ScoredChunk(
                        chunk=existing.chunk,
                        score=existing.score + score,
                        rank=0,
                        score_breakdown={
                            **existing.score_breakdown,
                            source_name: existing.score_breakdown.get(source_name, 0.0) + score,
                        },
                    )
        fused = list(by_id.values())
        fused.sort(key=lambda item: item.score, reverse=True)
        return [
            ScoredChunk(
                chunk=item.chunk,
                score=item.score,
                rank=index + 1,
                score_breakdown=item.score_breakdown,
            )
            for index, item in enumerate(fused)
        ]

    def _rerank(self, *, query: str, scored: list[ScoredChunk]) -> list[ScoredChunk]:
        if not scored:
            return []
        pool = scored[: max(1, self._config.rerank_pool_size)]
        if self._reranker is None:
            return pool
        try:
            reranked = self._reranker.score_documents(
                query=query,
                chunks=[item.chunk for item in pool],
            )
        except Exception:
            return pool
        score_by_id = {chunk.id: float(score) for score, _, chunk in reranked}
        out = [
            ScoredChunk(
                chunk=item.chunk,
                score=score_by_id.get(item.chunk.id, item.score),
                rank=0,
                score_breakdown={**item.score_breakdown, "rerank": score_by_id.get(item.chunk.id, item.score)},
            )
            for item in pool
        ]
        out.sort(key=lambda item: item.score, reverse=True)
        return [
            ScoredChunk(chunk=item.chunk, score=item.score, rank=index + 1, score_breakdown=item.score_breakdown)
            for index, item in enumerate(out)
        ]

    def _apply_doc_cap(self, scored: list[ScoredChunk]) -> list[ScoredChunk]:
        cap = max(1, self._config.doc_cap)
        counts: dict[str, int] = {}
        out: list[ScoredChunk] = []
        for item in scored:
            key = str(
                item.chunk.metadata.get("source_item_id")
                or item.chunk.document_id
                or item.chunk.id
            )
            if counts.get(key, 0) >= cap:
                continue
            counts[key] = counts.get(key, 0) + 1
            out.append(item)
        return out

    def _apply_mmr(
        self,
        *,
        query: str,
        scored: list[ScoredChunk],
        embeddings: dict[str, np.ndarray],
    ) -> list[ScoredChunk]:
        if not scored:
            return []
        selected: list[ScoredChunk] = []
        remaining = list(scored)
        lambda_ = max(0.0, min(1.0, float(self._config.mmr_lambda)))
        while remaining and len(selected) < max(0, self._config.top_k):
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_index = 0
            best_score = float("-inf")
            for idx, item in enumerate(remaining):
                relevance = float(item.score)
                similarity = max(
                    _cosine(
                        embeddings.get(item.chunk.id),
                        embeddings.get(selected_item.chunk.id),
                    )
                    for selected_item in selected
                )
                score = (lambda_ * relevance) - ((1.0 - lambda_) * similarity)
                if score > best_score:
                    best_index = idx
                    best_score = score
            selected.append(remaining.pop(best_index))
        return [
            ScoredChunk(
                chunk=item.chunk,
                score=item.score,
                rank=index + 1,
                score_breakdown=item.score_breakdown,
            )
            for index, item in enumerate(selected)
        ]


def _embedding_text(chunk: Chunk) -> str:
    metadata = dict(chunk.metadata or {})
    title = str(metadata.get("source_title") or metadata.get("title") or "")
    return f"{title}\n{chunk.text}".strip()


def _tokens(value: str) -> list[str]:
    return [
        token.lower()
        for token in re.split(r"[^0-9A-Za-zぁ-んァ-ン一-龠々ー]+", value or "")
        if token.strip()
    ]


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)
