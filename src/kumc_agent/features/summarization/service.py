from __future__ import annotations

from kumc_agent.features.summarization.config import SummarizationConfig


class SummarizationService:
    def __init__(self, *, config: SummarizationConfig) -> None:
        self._config = config

    def summarize(self, text: str) -> str:
        normalized = (text or "").strip()
        if len(normalized) <= self._config.target_characters:
            return normalized
        return normalized[: self._config.target_characters].rstrip() + "..."
