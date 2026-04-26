from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.features.rag.service import RagService

ChatHistoryEntry = tuple[str, str, Sequence[str]]


@dataclass(frozen=True)
class ChatRequest:
    query: str
    question_author: str | None = None
    history_scope: str | int | None = None
    force_fast_mode: bool = False
    force_disable_additional_memory: bool = False
    routing_history_override: Sequence[ChatHistoryEntry] | None = None
    generation_history_override: Sequence[ChatHistoryEntry] | None = None
    append_sources_to_response: bool = True
    extra_mode_instruction: str | None = None
    disable_history: bool = False
    access_context: AccessContext | None = None
    route_override: str | None = None


class ChatAnswerUsecase:
    def __init__(self, *, rag_service: RagService) -> None:
        self._rag_service = rag_service

    def execute(self, request: ChatRequest) -> Answer:
        return self._rag_service.answer(
            query=request.query,
            question_author=request.question_author,
            history_scope=request.history_scope,
            force_fast_mode=request.force_fast_mode,
            force_disable_additional_memory=request.force_disable_additional_memory,
            routing_history_override=request.routing_history_override,
            generation_history_override=request.generation_history_override,
            append_sources_to_response=request.append_sources_to_response,
            extra_mode_instruction=request.extra_mode_instruction,
            disable_history=request.disable_history,
            access_context=request.access_context,
            route_override=request.route_override,
        )
