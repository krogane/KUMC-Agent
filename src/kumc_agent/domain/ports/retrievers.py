from __future__ import annotations

from typing import Protocol

from kumc_agent.domain.models.chunk import Chunk


class RetrieverPort(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> list[Chunk]:
        ...
