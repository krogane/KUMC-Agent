from __future__ import annotations

from typing import Protocol

from kumc_agent.domain.models.document import Document


class ParserPort(Protocol):
    def parse(self) -> list[Document]:
        """Parse local raw files into domain documents."""
