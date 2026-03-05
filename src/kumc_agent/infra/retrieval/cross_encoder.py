from __future__ import annotations

from kumc_agent.domain.models.chunk import Chunk


class CrossEncoderReranker:
    def __init__(self, *, model_name: str) -> None:
        self._model_name = model_name

    def rerank(self, query: str, chunks: list[Chunk], *, top_k: int) -> list[Chunk]:
        # Lightweight deterministic rerank: lexical overlap score.
        query_tokens = set((query or "").lower().split())
        scored = []
        for chunk in chunks:
            chunk_tokens = set(chunk.text.lower().split())
            score = float(len(query_tokens & chunk_tokens))
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[: max(0, top_k)]]
