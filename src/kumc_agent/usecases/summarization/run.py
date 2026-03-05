from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.features.summarization.service import SummarizationService


@dataclass(frozen=True)
class SummarizationRequest:
    text: str


class SummarizationUsecase:
    def __init__(self, *, service: SummarizationService) -> None:
        self._service = service

    def execute(self, request: SummarizationRequest) -> str:
        return self._service.summarize(request.text)
