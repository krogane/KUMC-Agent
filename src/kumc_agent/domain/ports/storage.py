from __future__ import annotations

from typing import Protocol

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.document import Document


class StoragePort(Protocol):
    def save_documents(self, documents: list[Document]) -> None:
        ...

    def save_chunks(self, chunks: list[Chunk]) -> None:
        ...

    def load_chunks(self) -> list[Chunk]:
        ...
