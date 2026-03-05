from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SummarizationConfig:
    target_characters: int = 200
