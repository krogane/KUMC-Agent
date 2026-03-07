from __future__ import annotations

from functools import lru_cache
import logging

from kumc_agent.domain.models.chunk import Chunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, *, model_name: str) -> None:
        self._model_name = model_name

    def rerank(self, query: str, chunks: list[Chunk], *, top_k: int) -> list[Chunk]:
        limited_top_k = max(0, top_k)
        if limited_top_k <= 0 or not chunks:
            return []
        if len(chunks) == 1:
            return chunks[:1]
        try:
            cross_encoder = _cross_encoder_client(self._model_name)
            pairs = [(query, chunk.text) for chunk in chunks]
            scores = cross_encoder.predict(
                pairs,
                show_progress_bar=False,
            )
            scored = [
                (float(score), idx, chunk)
                for idx, (chunk, score) in enumerate(zip(chunks, scores))
            ]
            scored.sort(key=lambda item: (-item[0], item[1]))
            return [chunk for _, _, chunk in scored[:limited_top_k]]
        except Exception:
            logger.exception(
                "Cross-encoder rerank failed. Falling back to lexical overlap rerank."
            )
            return self._rerank_lexical(query=query, chunks=chunks, top_k=limited_top_k)

    @staticmethod
    def _rerank_lexical(query: str, chunks: list[Chunk], *, top_k: int) -> list[Chunk]:
        query_tokens = set((query or "").lower().split())
        scored = []
        for idx, chunk in enumerate(chunks):
            chunk_tokens = set((chunk.text or "").lower().split())
            score = float(len(query_tokens & chunk_tokens))
            scored.append((score, idx, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk for _, _, chunk in scored[:top_k]]


@lru_cache(maxsize=1)
def _cross_encoder_client(model_name: str):
    if not model_name:
        raise RuntimeError("Cross-encoder model name is required.")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for cross-encoder reranking."
        ) from exc
    return CrossEncoder(
        model_name,
        local_files_only=False,
        trust_remote_code=False,
    )
