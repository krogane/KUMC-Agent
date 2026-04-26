from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Sequence

from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.entry_routing import EntryRoutingDecision
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.policies.source_format import format_sources
from kumc_agent.domain.models.source import Source
from kumc_agent.features.rag.components.entry_routing import EntryQueryRouter
from kumc_agent.infra.openclaw.client import OpenClawClient
from kumc_agent.usecases.chat.answer import ChatAnswerUsecase, ChatHistoryEntry, ChatRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatEntryRequest:
    query: str
    question_author: str | None = None
    history_scope: str | int | None = None
    force_fast_mode: bool = False
    force_disable_additional_memory: bool = False
    routing_history_override: Sequence[ChatHistoryEntry] | None = None
    generation_history_override: Sequence[ChatHistoryEntry] | None = None
    append_sources_to_response: bool = True
    extra_mode_instruction: str | None = None
    access_context: AccessContext | None = None


class ChatEntryUsecase:
    def __init__(
        self,
        *,
        chat_usecase: ChatAnswerUsecase,
        openclaw_client: OpenClawClient,
        entry_router: EntryQueryRouter,
    ) -> None:
        self._chat_usecase = chat_usecase
        self._openclaw_client = openclaw_client
        self._entry_router = entry_router

    def execute(self, request: ChatEntryRequest) -> Answer:
        cleaned_query = (request.query or "").strip()
        if not cleaned_query:
            return Answer(text="", route="none", metadata={"reason": "empty_query"})

        openclaw_mode = self._openclaw_client.enabled
        if not openclaw_mode:
            fallback_request = self._build_rag_request(
                cleaned_query=cleaned_query,
                request=request,
                disable_history=False,
            )
            return self._chat_usecase.execute(fallback_request)

        classifier_fallback = False
        try:
            entry_decision = self._entry_router.decide(cleaned_query)
        except Exception:
            logger.exception(
                "Entry route classifier failed. Falling back to OpenClaw route."
            )
            entry_decision = EntryRoutingDecision(
                route="openclaw",
                reason="fallback:classifier_error",
            )
            classifier_fallback = True
        else:
            classifier_fallback = str(entry_decision.reason).strip().lower().startswith(
                "fallback:"
            )

        if entry_decision.route == "direct_rag":
            direct_answer = self._chat_usecase.execute(
                self._build_rag_request(
                    cleaned_query=cleaned_query,
                    request=request,
                    disable_history=False,
                )
            )
            return self._attach_entry_routing_metadata(
                direct_answer,
                decision=entry_decision,
                fallback=classifier_fallback,
            )

        if entry_decision.route == "openclaw":
            session_id = self._normalize_history_scope(request.history_scope)
            response = self._openclaw_client.run_turn(
                query=cleaned_query,
                session_id=session_id,
                user_context={
                    "question_author": request.question_author or "",
                    "history_scope": session_id,
                    "force_fast_mode": bool(request.force_fast_mode),
                },
            )
            if response.ok and response.result is not None:
                return self._attach_entry_routing_metadata(
                    self._to_answer(
                        response.result.text,
                        payload=response.result.payload,
                        session_id=session_id,
                        append_sources_to_response=request.append_sources_to_response,
                    ),
                    decision=entry_decision,
                    fallback=classifier_fallback,
                )
            if response.failure is not None:
                stderr_preview = " ".join((response.failure.stderr or "").strip().splitlines()[:2]).strip()
                logger.warning(
                    "OpenClaw unavailable. Falling back to local chat path. reason=%s detail=%s stderr=%s",
                    response.failure.reason,
                    response.failure.detail,
                    stderr_preview,
                )

        fallback_answer = self._chat_usecase.execute(
            self._build_rag_request(
                cleaned_query=cleaned_query,
                request=request,
                disable_history=True,
            )
        )
        metadata = dict(fallback_answer.metadata)
        metadata["openclaw_fallback"] = True
        return self._attach_entry_routing_metadata(
            replace(fallback_answer, metadata=metadata),
            decision=entry_decision,
            fallback=classifier_fallback,
        )

    def _attach_entry_routing_metadata(
        self,
        answer: Answer,
        *,
        decision: EntryRoutingDecision,
        fallback: bool,
    ) -> Answer:
        metadata = dict(answer.metadata)
        metadata["entry_route"] = decision.route
        metadata["entry_route_reason"] = decision.reason
        metadata["entry_route_model"] = self._entry_router.model_label
        metadata["entry_route_fallback"] = bool(fallback)
        if decision.payload:
            metadata["entry_route_payload"] = dict(decision.payload)
        return replace(answer, metadata=metadata)

    def _build_rag_request(
        self,
        *,
        cleaned_query: str,
        request: ChatEntryRequest,
        disable_history: bool,
    ) -> ChatRequest:
        return ChatRequest(
            query=cleaned_query,
            question_author=request.question_author,
            history_scope=request.history_scope,
            force_fast_mode=request.force_fast_mode,
            force_disable_additional_memory=request.force_disable_additional_memory,
            routing_history_override=request.routing_history_override,
            generation_history_override=request.generation_history_override,
            append_sources_to_response=request.append_sources_to_response,
            extra_mode_instruction=request.extra_mode_instruction,
            disable_history=disable_history,
            access_context=request.access_context,
        )

    def _to_answer(
        self,
        text: str,
        *,
        payload: dict[str, object],
        session_id: str,
        append_sources_to_response: bool,
    ) -> Answer:
        route = "openclaw"
        metadata = dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {}
        metadata.pop("routing_decision", None)
        if "fast_mode" not in metadata and "fastmode" in metadata:
            metadata["fast_mode"] = metadata.get("fastmode")
        fast_mode_value = payload.get("fast_mode")
        if fast_mode_value is None:
            fast_mode_value = payload.get("fastmode")
        if fast_mode_value is not None and "fast_mode" not in metadata:
            metadata["fast_mode"] = fast_mode_value
        if "rag_query" in payload and "rag_query" not in metadata:
            metadata["rag_query"] = payload.get("rag_query")
        if "rag_iterations" in payload and "rag_iterations" not in metadata:
            metadata["rag_iterations"] = payload.get("rag_iterations")
        openclaw_payload = dict(payload)
        openclaw_payload.pop("route", None)
        openclaw_payload.pop("routing_decision", None)
        metadata["openclaw_payload"] = openclaw_payload
        metadata["openclaw_session_id"] = session_id

        parsed_sources: list[Source] = []
        raw_sources = payload.get("sources")
        if isinstance(raw_sources, list):
            for item in raw_sources:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "").strip()
                source_id = str(item.get("id") or label).strip()
                uri = str(item.get("uri") or "").strip()
                if not label and not source_id:
                    continue
                parsed_sources.append(
                    Source(
                        id=source_id or label,
                        label=label or source_id,
                        uri=uri,
                    )
                )

        final_text = text
        if append_sources_to_response and "主な情報源:" not in final_text:
            final_text = final_text + format_sources(parsed_sources)

        return Answer(
            text=final_text,
            route=route,
            sources=parsed_sources,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_history_scope(history_scope: str | int | None) -> str:
        if history_scope is None:
            return "default"
        value = str(history_scope).strip()
        return value or "default"
