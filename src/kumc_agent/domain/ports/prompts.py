from __future__ import annotations

from typing import Protocol


class PromptRepositoryPort(Protocol):
    def get(self, name: str) -> str:
        ...
