from __future__ import annotations

from typing import Protocol

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.document import Document


class ChunkerPort(Protocol):
    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Chunk parsed documents."""
