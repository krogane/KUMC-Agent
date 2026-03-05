from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.domain.models.answer import Answer
from kumc_agent.features.rag.service import RagService


@dataclass(frozen=True)
class ChatRequest:
    query: str


class ChatAnswerUsecase:
    def __init__(self, *, rag_service: RagService) -> None:
        self._rag_service = rag_service

    def execute(self, request: ChatRequest) -> Answer:
        return self._rag_service.answer(request.query)
