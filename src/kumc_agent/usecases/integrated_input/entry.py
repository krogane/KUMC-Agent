from __future__ import annotations

from dataclasses import replace
from time import monotonic
from typing import Any
from uuid import uuid4

from kumc_agent.domain.models.agentic import AgentBudget, ComprehensiveAgentRequest
from kumc_agent.domain.models.integrated_input import (
    IntegratedInputDecision,
    IntegratedInputRequest,
    IntegratedInputResponse,
)
from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.retrieval import AccessContext, AskResponse, Citation, RetrievalQuery
from kumc_agent.domain.models.workflow import WorkRequest, WorkResponse
from kumc_agent.features.foundation.payload_sanitizer import sanitize_payload, sanitize_payload_metadata
from kumc_agent.features.rag.components.integrated_input_routing import (
    IntegratedInputRouter,
    IntegratedRoutingPolicy,
)
from kumc_agent.usecases.chat.answer import ChatRequest


class IntegratedInputUsecase:
    def __init__(
        self,
        *,
        ask_service: object,
        workflow_service: object | None,
        comprehensive_agent: object | None,
        router: IntegratedInputRouter,
        chat_answer_service: object | None = None,
        routing_policy: IntegratedRoutingPolicy | None = None,
    ) -> None:
        self.ask_service = ask_service
        self.chat_answer_service = chat_answer_service
        self.workflow_service = workflow_service
        self.comprehensive_agent = comprehensive_agent
        self.router = router
        self.routing_policy = routing_policy or IntegratedRoutingPolicy()

    def execute(self, request: IntegratedInputRequest) -> IntegratedInputResponse:
        started = monotonic()
        trace_id = str(uuid4())
        normalized = self._normalize_request(request, trace_id=trace_id)
        if not normalized.text:
            return self._finalize(
                IntegratedInputResponse(
                    text="入力内容を指定してください。",
                    confidence="low",
                    metadata={"trace_id": trace_id, "route": "clarify", "intent": "unknown"},
                ),
                started=started,
            )
        access = normalized.normalized_access()
        try:
            decision = self.router.decide(
                normalized.text,
                source=normalized.source,
                mode=normalized.mode,
                depth=normalized.depth,
                metadata=normalized.metadata,
            )
            decision = self.routing_policy.apply(
                decision,
                text=normalized.text,
                source=normalized.source,
                access_is_admin=access.is_admin,
            )
            response = self._dispatch(normalized, decision, access)
        except Exception as exc:
            response = IntegratedInputResponse(
                text="処理中にエラーが発生しました。入力内容を少し変えて再試行してください。",
                confidence="low",
                warnings=("route handler failed",),
                metadata={
                    "trace_id": trace_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "fallback": True,
                },
            )
        return self._finalize(
            response,
            started=started,
            request=normalized,
            decision=locals().get("decision"),
        )

    def _normalize_request(self, request: IntegratedInputRequest, *, trace_id: str) -> IntegratedInputRequest:
        source = (request.source or "all").strip() or "all"
        mode = (request.mode or "answer").strip() or "answer"
        depth = (request.depth or "normal").strip() or "normal"
        access = request.normalized_access()
        return replace(
            request,
            text=(request.text or "").strip(),
            source=source,
            mode=mode,
            depth=depth,
            access=access,
            metadata={
                **sanitize_payload_metadata(request.metadata),
                "trace_id": trace_id,
                "frontend": request.frontend,
            },
        )

    def _dispatch(
        self,
        request: IntegratedInputRequest,
        decision: IntegratedInputDecision,
        access: AccessContext,
    ) -> IntegratedInputResponse:
        if decision.route == "clarify":
            return IntegratedInputResponse(
                text=decision.clarification_question or "実行に必要な情報をもう少し具体的に指定してください。",
                confidence="low",
                metadata=self._decision_metadata(decision, handler="clarify"),
            )
        if decision.route == "deny":
            return IntegratedInputResponse(
                text="権限がないため、この操作は実行できません。",
                confidence="low",
                warnings=("permission denied",),
                metadata=self._decision_metadata(decision, handler="deny"),
            )
        if decision.risk == "admin_only" and not access.is_admin:
            denied = replace(decision, route="deny")
            return IntegratedInputResponse(
                text="権限がないため、この操作は実行できません。",
                confidence="low",
                warnings=("permission denied",),
                metadata=self._decision_metadata(denied, handler="deny"),
            )
        if decision.route == "circle_rag" and self.chat_answer_service is not None:
            return self._from_rag_answer(
                self.chat_answer_service.execute(
                    ChatRequest(
                        query=request.text,
                        question_author=access.user_id or None,
                        history_scope=request.history_scope or self._history_scope(request, access),
                        force_fast_mode=request.mode == "fast",
                        access_context=access,
                    )
                ),
                decision,
                handler="circle_rag",
            )
        if decision.route in {"circle_rag", "minecraft_wiki_rag"}:
            source_filter = "minecraft_wiki" if decision.route == "minecraft_wiki_rag" else self._rag_source(request, decision)
            return self._from_ask_response(
                self.ask_service.ask(
                    RetrievalQuery(
                        text=request.text,
                        source_filter=source_filter,
                        mode=request.mode,
                        depth=request.depth,
                        access=access,
                    )
                ),
                decision,
                handler=decision.route,
            )
        if decision.route == "comprehensive_agent":
            if self.comprehensive_agent is None:
                return IntegratedInputResponse(
                    text="総合エージェントが未設定です。",
                    confidence="low",
                    warnings=("comprehensive_agent is not configured",),
                    metadata=self._decision_metadata(decision, handler="comprehensive_agent"),
                )
            response = self.comprehensive_agent.run(
                ComprehensiveAgentRequest(
                    query=request.text,
                    source_filter=request.source,
                    access=access,
                    required_features=decision.required_features,
                    source_filters={"values": list(decision.source_filters)},
                    attribute_filters=decision.attribute_filters,
                    risk=decision.risk,
                    budget=AgentBudget(
                        allow_write_tools=decision.risk in {"candidate_only", "approval_required"} and access.is_admin
                    ),
                    metadata={
                        "depth": request.depth,
                        "frontend": request.frontend,
                        "routing": self._decision_payload(decision),
                    },
                )
            )
            return self._from_comprehensive_response(response, decision)
        return self._run_workflow_route(request, decision, access)

    def _run_workflow_route(
        self,
        request: IntegratedInputRequest,
        decision: IntegratedInputDecision,
        access: AccessContext,
    ) -> IntegratedInputResponse:
        if self.workflow_service is None:
            return IntegratedInputResponse(
                text="Workflow service が未設定です。",
                confidence="low",
                warnings=("workflow_service is not configured",),
                metadata=self._decision_metadata(decision, handler=decision.route),
            )
        work_type = self._work_type(request, decision)
        if decision.risk == "read_only" and work_type in _CANDIDATE_WORK_TYPES:
            return IntegratedInputResponse(
                text="候補作成が必要な依頼として解釈されました。作成内容をもう少し具体的に指定してください。",
                confidence="low",
                warnings=("read_only route blocked candidate creation",),
                metadata=self._decision_metadata(decision, handler="side_effect_guard"),
            )
        response = self.workflow_service.run(
            WorkRequest(
                work_type=work_type,
                instruction=request.text,
                access=access,
            )
        )
        integrated = self._from_work_response(response, decision, handler=work_type)
        return self._guard_workflow_response(integrated, decision, work_type)

    def _work_type(self, request: IntegratedInputRequest, decision: IntegratedInputDecision) -> str:
        if decision.route == "member_search":
            return "member_search"
        if decision.route == "image_search":
            return "image_search"
        if decision.route == "server_management":
            return "mc_status" if decision.intent in {"question", "search", "list"} and not _server_operation_requested(request.text) else "mc_request"
        if decision.route == "task_management":
            return _task_work_type(decision.intent, request.text)
        if decision.route == "event_management":
            return _event_work_type(decision.intent, request.text)
        return "task_list"

    def _rag_source(self, request: IntegratedInputRequest, decision: IntegratedInputDecision) -> str:
        if decision.source_filters:
            return decision.source_filters[0]
        return request.source if request.source not in {"member", "image", "task", "event"} else "all"

    def _from_ask_response(
        self,
        response: AskResponse,
        decision: IntegratedInputDecision,
        *,
        handler: str,
    ) -> IntegratedInputResponse:
        return IntegratedInputResponse(
            text=response.text,
            detail_markdown=response.detail_markdown,
            citations=response.citations,
            confidence=response.confidence,  # type: ignore[arg-type]
            warnings=response.warnings,
            metadata={
                **sanitize_payload_metadata(response.metadata),
                **self._decision_metadata(decision, handler=handler),
            },
        )

    def _from_rag_answer(
        self,
        answer: Answer,
        decision: IntegratedInputDecision,
        *,
        handler: str,
    ) -> IntegratedInputResponse:
        citations = tuple(
            Citation(
                source_item_id=source.id,
                chunk_id=source.id,
                label=source.label,
                url=source.uri,
            )
            for source in answer.sources
        )
        return IntegratedInputResponse(
            text=answer.text,
            detail_markdown="",
            citations=citations,
            confidence="medium" if answer.sources else "low",
            metadata={
                **sanitize_payload_metadata(answer.metadata),
                **self._decision_metadata(decision, handler=handler),
            },
        )

    @staticmethod
    def _history_scope(request: IntegratedInputRequest, access: AccessContext) -> str:
        if request.frontend == "discord" and access.guild_id:
            return f"discord:{access.guild_id}"
        if request.frontend and access.user_id:
            return f"{request.frontend}:{access.user_id}"
        return request.frontend or "integrated_input"

    def _from_work_response(
        self,
        response: WorkResponse,
        decision: IntegratedInputDecision,
        *,
        handler: str,
    ) -> IntegratedInputResponse:
        return IntegratedInputResponse(
            text=response.text,
            detail_markdown=response.detail_markdown,
            task_candidates=response.task_candidates,
            task_change_candidates=response.task_change_candidates,
            task_approval_batches=response.task_approval_batches,
            event_candidates=response.event_candidates,
            event_change_candidates=response.event_change_candidates,
            event_approval_batches=response.event_approval_batches,
            schedule_candidates=response.schedule_candidates,
            workflow_candidates=response.workflow_candidates,
            assets=response.assets,
            member_profiles=response.member_profiles,
            tasks=response.tasks,
            events=response.events,
            schedules=response.schedules,
            approvals=response.approvals,
            server_operations=response.server_operations,
            warnings=response.warnings,
            metadata={
                **sanitize_payload_metadata(response.metadata),
                **self._decision_metadata(decision, handler=handler),
            },
        )

    def _from_comprehensive_response(
        self,
        response: object,
        decision: IntegratedInputDecision,
    ) -> IntegratedInputResponse:
        return IntegratedInputResponse(
            text=str(getattr(response, "text", "")),
            detail_markdown=str(getattr(response, "detail_markdown", "")),
            citations=tuple(getattr(response, "citations", tuple())),
            confidence=str(getattr(response, "confidence", "low")),  # type: ignore[arg-type]
            task_candidates=tuple(getattr(response, "task_candidates", tuple())),
            event_candidates=tuple(getattr(response, "event_candidates", tuple())),
            server_operations=tuple(getattr(response, "server_operations", tuple())),
            assets=tuple(getattr(response, "assets", tuple())),
            member_profiles=tuple(getattr(response, "member_profiles", tuple())),
            warnings=tuple(getattr(response, "warnings", tuple())),
            metadata={
                **sanitize_payload_metadata(getattr(response, "metadata", {}) or {}),
                **self._decision_metadata(decision, handler="comprehensive_agent"),
            },
        )

    def _guard_workflow_response(
        self,
        response: IntegratedInputResponse,
        decision: IntegratedInputDecision,
        work_type: str,
    ) -> IntegratedInputResponse:
        changed = bool(response.tasks or response.events or response.schedules)
        if work_type in _READ_WORK_TYPES:
            return response
        if changed and decision.risk in {"candidate_only", "approval_required"}:
            return replace(
                response,
                text="副作用境界に反する実行済み結果を検出したため、出力を停止しました。",
                detail_markdown="",
                tasks=tuple(),
                events=tuple(),
                schedules=tuple(),
                warnings=tuple(dict.fromkeys([*response.warnings, "side effect boundary violation"])),
                metadata={
                    **response.metadata,
                    "side_effect_boundary_violation": True,
                    "blocked_work_type": work_type,
                },
            )
        return response

    def _finalize(
        self,
        response: IntegratedInputResponse,
        *,
        started: float,
        request: IntegratedInputRequest | None = None,
        decision: IntegratedInputDecision | None = None,
    ) -> IntegratedInputResponse:
        metadata = {
            **sanitize_payload_metadata(response.metadata),
            "latency_seconds": round(monotonic() - started, 3),
        }
        if request is not None:
            metadata.setdefault("frontend", request.frontend)
            metadata.setdefault("source", request.source)
            metadata.setdefault("mode", request.mode)
            metadata.setdefault("depth", request.depth)
        if decision is not None:
            metadata.update(self._decision_metadata(decision, handler=str(metadata.get("handler") or "")))
        payload = sanitize_payload(
            replace(response, metadata=metadata).to_payload()
        )
        return IntegratedInputResponse(
            text=str(payload.get("text", "")) if isinstance(payload, dict) else response.text,
            detail_markdown=str(payload.get("detail_markdown", "")) if isinstance(payload, dict) else response.detail_markdown,
            citations=tuple(payload.get("citations", ())) if isinstance(payload, dict) else tuple(),
            confidence=str(payload.get("confidence", "low")) if isinstance(payload, dict) else response.confidence,  # type: ignore[arg-type]
            task_candidates=tuple(payload.get("task_candidates", ())) if isinstance(payload, dict) else tuple(),
            task_change_candidates=tuple(payload.get("task_change_candidates", ())) if isinstance(payload, dict) else tuple(),
            task_approval_batches=tuple(payload.get("task_approval_batches", ())) if isinstance(payload, dict) else tuple(),
            event_candidates=tuple(payload.get("event_candidates", ())) if isinstance(payload, dict) else tuple(),
            event_change_candidates=tuple(payload.get("event_change_candidates", ())) if isinstance(payload, dict) else tuple(),
            event_approval_batches=tuple(payload.get("event_approval_batches", ())) if isinstance(payload, dict) else tuple(),
            schedule_candidates=tuple(payload.get("schedule_candidates", ())) if isinstance(payload, dict) else tuple(),
            workflow_candidates=tuple(payload.get("workflow_candidates", ())) if isinstance(payload, dict) else tuple(),
            assets=tuple(payload.get("assets", ())) if isinstance(payload, dict) else tuple(),
            member_profiles=tuple(payload.get("member_profiles", ())) if isinstance(payload, dict) else tuple(),
            tasks=tuple(payload.get("tasks", ())) if isinstance(payload, dict) else tuple(),
            events=tuple(payload.get("events", ())) if isinstance(payload, dict) else tuple(),
            schedules=tuple(payload.get("schedules", ())) if isinstance(payload, dict) else tuple(),
            approvals=tuple(payload.get("approvals", ())) if isinstance(payload, dict) else tuple(),
            server_operations=tuple(payload.get("server_operations", ())) if isinstance(payload, dict) else tuple(),
            warnings=tuple(payload.get("warnings", ())) if isinstance(payload, dict) else response.warnings,
            metadata=dict(payload.get("metadata", {})) if isinstance(payload, dict) else metadata,
        )

    def _decision_metadata(self, decision: IntegratedInputDecision, *, handler: str) -> dict[str, object]:
        return {
            "route": decision.route,
            "intent": decision.intent,
            "required_features": list(decision.required_features),
            "risk": decision.risk,
            "freshness_required": decision.freshness_required,
            "needs_clarification": decision.needs_clarification,
            "handler": handler,
            "routing_decision": self._decision_payload(decision),
        }

    def _decision_payload(self, decision: IntegratedInputDecision) -> dict[str, object]:
        return {
            "route": decision.route,
            "intent": decision.intent,
            "required_features": list(decision.required_features),
            "source_filters": list(decision.source_filters),
            "attribute_filters": sanitize_payload(decision.attribute_filters),
            "risk": decision.risk,
            "freshness_required": decision.freshness_required,
            "needs_clarification": decision.needs_clarification,
            "clarification_question": decision.clarification_question,
            "reason": decision.reason,
            "metadata": sanitize_payload_metadata(decision.metadata),
        }


_READ_WORK_TYPES = {"member_search", "image_search", "task_list", "event_list", "event_brief", "schedule_list", "mc_status"}
_CANDIDATE_WORK_TYPES = {
    "task_extract",
    "task_add",
    "task_done",
    "task_update",
    "task_delete",
    "task_notify_due",
    "task_batch_approval",
    "event_extract",
    "event_add",
    "event_update",
    "event_delete",
    "event_notify",
    "event_batch_approval",
    "event_complete",
    "schedule_add",
    "mc_request",
}


def _task_work_type(intent: str, text: str) -> str:
    if intent == "list":
        return "task_list"
    if intent == "extract":
        return "task_extract"
    if intent == "complete":
        return "task_done"
    if intent == "notify":
        return "task_notify_due"
    if intent == "approval":
        return "task_batch_approval"
    if intent == "delete_candidate" or any(token in text for token in ("削除", "delete")):
        return "task_delete"
    if intent == "update_candidate" or any(token in text for token in ("更新", "変更", "update")):
        return "task_update"
    if intent == "create_candidate" or any(token in text for token in ("追加", "作成", "候補", "add", "create")):
        return "task_add"
    return "task_list"


def _event_work_type(intent: str, text: str) -> str:
    if "日程" in text and intent in {"create_candidate", "extract"}:
        return "schedule_add"
    if "日程" in text and intent in {"list", "question", "search"}:
        return "schedule_list"
    if intent == "list":
        return "event_list"
    if intent == "extract":
        return "event_extract"
    if intent == "complete":
        return "event_complete"
    if intent == "notify":
        return "event_notify"
    if intent == "approval":
        return "event_batch_approval"
    if intent == "delete_candidate" or any(token in text for token in ("削除", "delete")):
        return "event_delete"
    if intent == "update_candidate" or any(token in text for token in ("更新", "変更", "update")):
        return "event_update"
    if intent == "create_candidate" or any(token in text for token in ("追加", "作成", "候補", "開催", "add", "create")):
        return "event_add"
    if any(token in text for token in ("概要", "brief", "要約")):
        return "event_brief"
    return "event_list"


def _server_operation_requested(text: str) -> bool:
    return any(token in text for token in ("再起動", "停止", "起動", "バックアップ", "ホワイトリスト", "追加", "削除", "request"))
