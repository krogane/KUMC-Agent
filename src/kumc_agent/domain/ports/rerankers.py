from __future__ import annotations

from typing import Protocol

from kumc_agent.domain.models.chunk import Chunk


class RerankerPort(Protocol):
    def rerank(self, query: str, chunks: list[Chunk], *, top_k: int) -> list[Chunk]:
        ...
