from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import re
from time import monotonic
from typing import Any
from uuid import uuid4

from kumc_agent.domain.models.agentic import (
    AgentBudget,
    AgentPlan,
    AgentRun,
    AgentStep,
    AgentTask,
    AgentToolResult,
    ComprehensiveAgentRequest,
    ComprehensiveAgentResponse,
    ToolCallPlan,
    VerificationResult,
)
from kumc_agent.domain.models.retrieval import Citation, RetrievalQuery
from kumc_agent.domain.models.workflow import WorkRequest, WorkResponse
from kumc_agent.features.agentic.tools import ToolSchemaRegistry
from kumc_agent.infra.agentic.repository import AgentTraceRepository


_READ_TOOLS = {
    "circle_rag_search",
    "minecraft_wiki_rag_search",
    "member_search",
    "image_search",
    "task_search",
    "event_search",
}
_WRITE_TOOLS = {
    "task_candidate_create",
    "event_candidate_create",
    "server_operation_candidate_create",
    "approval_candidate_create",
}


class ComprehensiveAgentService:
    def __init__(
        self,
        *,
        ask_service: object,
        repository: AgentTraceRepository,
        workflow_service: object | None = None,
        registry: ToolSchemaRegistry | None = None,
    ) -> None:
        self.ask_service = ask_service
        self.repository = repository
        self.workflow_service = workflow_service
        self.registry = registry or ToolSchemaRegistry()
        self.planner = ComprehensiveAgentPlanner(registry=self.registry)
        self.adapters = ComprehensiveToolAdapters(
            ask_service=ask_service,
            workflow_service=workflow_service,
        )
        self.verifier = ComprehensiveAgentVerifier(registry=self.registry)
        self.answer_builder = ComprehensiveAgentAnswerBuilder()

    def run(self, request: ComprehensiveAgentRequest) -> ComprehensiveAgentResponse:
        started = monotonic()
        run = self.repository.save_run(
            AgentRun(
                id=str(uuid4()),
                query=request.query,
                status="running",
                access=request.access,
                budget=request.budget,
                metadata={
                    "route": "comprehensive_agent",
                    "required_features": list(request.required_features),
                    "risk": request.risk,
                },
            )
        )
        steps: list[AgentStep] = []

        def add_step(
            state: str,
            *,
            input_payload: dict[str, object],
            output_payload: dict[str, object],
            status: str = "succeeded",
            cost_usd: float = 0.0,
        ) -> AgentStep:
            step = self.repository.save_step(
                AgentStep(
                    id=str(uuid4()),
                    run_id=run.id,
                    state=state,
                    input=sanitize_payload(input_payload, limit=1200),
                    output=sanitize_payload(output_payload, limit=1200),
                    status=status,
                    cost_usd=cost_usd,
                )
            )
            steps.append(step)
            return step

        warnings: list[str] = []
        tool_results: list[AgentToolResult] = []
        replan_count = 0
        total_cost = 0.0
        search_calls = 0
        plan = self.planner.plan(request, previous_results=tuple(), previous_verification=None)
        add_step("PLAN", input_payload={"query": request.query}, output_payload=_plan_payload(plan))

        if plan.needs_clarification:
            answer = plan.clarification_question or "追加情報を確認してください。"
            run = self._finish_run(
                run,
                steps=steps,
                status="insufficient_input",
                answer=answer,
                confidence="low",
                citations=tuple(),
                metadata={"needs_clarification": True},
            )
            return ComprehensiveAgentResponse(
                text=answer,
                detail_markdown=self.answer_builder.detail_markdown(run, plan, tuple(), None),
                citations=tuple(),
                confidence="low",
                run=run,
                warnings=tuple(warnings),
                metadata={"agent_run_id": run.id},
            )

        while True:
            for tool_call in plan.tool_sequence:
                if len(steps) + 2 > request.budget.max_steps:
                    warnings.append("総合エージェントのstep予算に達しました。")
                    break
                schema = self.registry.get(tool_call.tool_name)
                if not schema.read_only and not request.budget.allow_write_tools:
                    result = AgentToolResult(
                        tool_name=tool_call.tool_name,
                        status="needs_approval",
                        text="書き込み系toolは無効です。候補作成予定のみを記録しました。",
                        warnings=("allow_write_tools=False のため候補作成は実行していません。",),
                        metadata={"planned_input": tool_call.input},
                    )
                else:
                    projected_search_calls = search_calls + (
                        1 if tool_call.tool_name in {"circle_rag_search", "minecraft_wiki_rag_search"} else 0
                    )
                    if not _within_budget(
                        request.budget,
                        steps=len(steps) + 2,
                        search_calls=projected_search_calls,
                        cost_usd=total_cost + 0.01,
                        elapsed=monotonic() - started,
                    ):
                        warnings.append("総合エージェントの検索・コスト・時間予算に達しました。")
                        break
                    result = self.adapters.run(tool_call, request=request)
                    if tool_call.tool_name in {"circle_rag_search", "minecraft_wiki_rag_search"}:
                        search_calls += 1
                        total_cost += 0.01
                tool_results.append(result)
                add_step(
                    "TOOL",
                    input_payload={
                        "tool_name": tool_call.tool_name,
                        "read_only": schema.read_only,
                        "side_effect_boundary": tool_call.side_effect_boundary,
                        "input": tool_call.input,
                    },
                    output_payload=_tool_result_payload(result),
                    status=result.status,
                    cost_usd=0.01 if tool_call.tool_name in {"circle_rag_search", "minecraft_wiki_rag_search"} else 0.0,
                )

            verification = self.verifier.verify(
                plan=plan,
                results=tuple(tool_results),
                budget=request.budget,
            )
            add_step(
                "VERIFY",
                input_payload={
                    "success_criteria": list(plan.success_criteria),
                    "result_count": len(tool_results),
                },
                output_payload=_verification_payload(verification),
                status=verification.status,
            )
            if verification.status in {"succeeded", "needs_approval"}:
                break
            if replan_count >= request.budget.max_replans:
                break
            replan_count += 1
            plan = self.planner.plan(
                request,
                previous_results=tuple(tool_results),
                previous_verification=verification,
            )
            add_step(
                "PLAN",
                input_payload={"query": request.query, "replan_count": replan_count},
                output_payload=_plan_payload(plan),
            )

        answer, confidence, status = self.answer_builder.build(
            plan=plan,
            results=tuple(tool_results),
            verification=verification,
            warnings=tuple(warnings),
        )
        add_step(
            "ANSWER",
            input_payload={"query": request.query},
            output_payload={"status": status, "confidence": confidence},
            status=status,
        )

        citations = _unique_citations(
            citation
            for result in tool_results
            for citation in result.citations
        )[: request.budget.max_read_chunks]
        metadata = {
            "agent_run_id": run.id,
            "route": "comprehensive_agent",
            "tool_summary": _tool_summary(tool_results),
            "search_calls": search_calls,
            "cost_usd": round(total_cost, 4),
            "elapsed_seconds": round(monotonic() - started, 3),
            "replan_count": replan_count,
            "verification": verification.status,
        }
        run = self._finish_run(
            run,
            steps=steps,
            status=status,
            answer=answer,
            confidence=confidence,
            citations=tuple(citations),
            metadata=metadata,
        )
        detail = self.answer_builder.detail_markdown(run, plan, tuple(tool_results), verification)
        return ComprehensiveAgentResponse(
            text=answer,
            detail_markdown=detail,
            citations=tuple(citations),
            confidence=confidence,
            run=run,
            task_candidates=tuple(_candidate_dicts(tool_results, "task")),
            event_candidates=tuple(_candidate_dicts(tool_results, "event")),
            server_operations=tuple(_candidate_dicts(tool_results, "server_operation")),
            assets=tuple(_metadata_items(tool_results, "assets")),
            member_profiles=tuple(_metadata_items(tool_results, "member_profiles")),
            warnings=tuple(dict.fromkeys([*warnings, *(w for result in tool_results for w in result.warnings)])),
            metadata=metadata,
        )

    def _finish_run(
        self,
        run: AgentRun,
        *,
        steps: list[AgentStep],
        status: str,
        answer: str,
        confidence: str,
        citations: tuple[Citation, ...],
        metadata: dict[str, Any],
    ) -> AgentRun:
        return self.repository.save_run(
            replace(
                run,
                status=status,
                steps=tuple(steps),
                citations=citations,
                answer=sanitize_text(answer, limit=4000),
                confidence=confidence,
                metadata=sanitize_payload({**run.metadata, **metadata}, limit=1200),
            )
        )


class ComprehensiveAgentPlanner:
    def __init__(self, *, registry: ToolSchemaRegistry) -> None:
        self.registry = registry

    def plan(
        self,
        request: ComprehensiveAgentRequest,
        *,
        previous_results: tuple[AgentToolResult, ...],
        previous_verification: VerificationResult | None,
    ) -> AgentPlan:
        query = request.query.strip()
        if len(query) < 2:
            return AgentPlan(
                needs_clarification=True,
                clarification_question="何について調べるか、もう少し具体的に指定してください。",
            )
        features = list(request.required_features or detect_required_features(query, request.source_filter))
        if request.metadata.get("depth") == "deep" and "circle_rag" not in features:
            features.insert(0, "circle_rag")
        if not features:
            features.append("circle_rag")
        tools = self._tools_for_features(features, query=query)
        if previous_verification and previous_verification.missing:
            if "circle_rag_search" not in tools:
                tools.insert(0, "circle_rag_search")
        tool_sequence = tuple(
            ToolCallPlan(
                tool_name=tool,
                input=self._tool_input(tool, request),
                reason=f"{tool} is required for {', '.join(features)}",
                read_only=self.registry.get(tool).read_only,
                side_effect_boundary=(
                    "read_only" if self.registry.get(tool).read_only else "candidate_only"
                ),
            )
            for tool in tools
        )
        boundary = "read_only" if all(self.registry.get(tool).read_only for tool in tools) else "candidate_only"
        return AgentPlan(
            tasks=tuple(
                AgentTask(
                    id=f"task-{index + 1}",
                    description=f"{tool} を実行する",
                    tool_name=tool,
                    input=dict(call.input),
                    success_criteria=("tool result is available",),
                )
                for index, (tool, call) in enumerate(zip(tools, tool_sequence, strict=False))
            ),
            required_tools=tuple(tools),
            tool_sequence=tool_sequence,
            success_criteria=self._success_criteria(tools),
            side_effect_boundary=boundary,
            retry_policy={"max_replans": request.budget.max_replans},
            answer_requirements=("結論", "根拠", "使用した機能", "未確認事項", "承認待ち候補"),
            metadata={"required_features": features},
        )

    def _tools_for_features(self, features: list[str], *, query: str) -> list[str]:
        tools: list[str] = []
        create_words = ("候補", "作って", "作成", "追加", "抽出", "申請")
        for feature in features:
            if feature in {"circle_rag", "circle_rag_search"}:
                tools.append("circle_rag_search")
            elif feature in {"minecraft_wiki", "minecraft_wiki_rag"}:
                tools.append("minecraft_wiki_rag_search")
            elif feature == "member_search":
                tools.append("member_search")
            elif feature == "image_search":
                tools.append("image_search")
            elif feature == "task_management":
                tools.append("task_candidate_create" if any(word in query for word in create_words) else "task_search")
            elif feature == "event_management":
                tools.append("event_candidate_create" if any(word in query for word in create_words) else "event_search")
            elif feature == "server_management":
                tools.append("server_operation_candidate_create")
        return list(dict.fromkeys(tools))

    def _tool_input(self, tool: str, request: ComprehensiveAgentRequest) -> dict[str, object]:
        if tool.endswith("_candidate_create") or tool == "approval_candidate_create":
            return {"instruction": request.query, "target": request.attribute_filters.get("target", "")}
        return {"query": request.query, "source_filter": request.source_filter}

    @staticmethod
    def _success_criteria(tools: list[str]) -> tuple[str, ...]:
        criteria = ["tool result is available"]
        if any(tool in {"circle_rag_search", "minecraft_wiki_rag_search"} for tool in tools):
            criteria.append("引用可能な根拠がある")
        if any(tool in _WRITE_TOOLS for tool in tools):
            criteria.append("副作用は候補作成または承認待ちに限定されている")
        return tuple(criteria)


class ComprehensiveToolAdapters:
    def __init__(self, *, ask_service: object, workflow_service: object | None) -> None:
        self.ask_service = ask_service
        self.workflow_service = workflow_service

    def run(self, call: ToolCallPlan, *, request: ComprehensiveAgentRequest) -> AgentToolResult:
        try:
            if call.tool_name == "circle_rag_search":
                return self._rag(call, request=request, source_filter=request.source_filter or "all")
            if call.tool_name == "minecraft_wiki_rag_search":
                return self._rag(call, request=request, source_filter="minecraft_wiki")
            return self._workflow(call, request=request)
        except Exception as exc:  # pragma: no cover - defensive boundary
            return AgentToolResult(
                tool_name=call.tool_name,
                status="failed",
                text=f"tool failed: {type(exc).__name__}",
                warnings=(str(exc),),
            )

    def _rag(
        self,
        call: ToolCallPlan,
        *,
        request: ComprehensiveAgentRequest,
        source_filter: str,
    ) -> AgentToolResult:
        response = self.ask_service.ask(
            RetrievalQuery(
                text=str(call.input.get("query") or request.query),
                source_filter=source_filter,
                mode="search_only",
                depth="normal",
                access=request.access,
            )
        )
        return AgentToolResult(
            tool_name=call.tool_name,
            status="succeeded" if response.citations else "insufficient_evidence",
            text=sanitize_text(response.detail_markdown or response.text, limit=1200),
            citations=tuple(response.citations),
            warnings=tuple(response.warnings),
            metadata={"confidence": response.confidence, "citation_count": len(response.citations)},
        )

    def _workflow(self, call: ToolCallPlan, *, request: ComprehensiveAgentRequest) -> AgentToolResult:
        if self.workflow_service is None:
            return AgentToolResult(
                tool_name=call.tool_name,
                status="insufficient_input",
                text="workflow service is not configured.",
                warnings=("workflow service is required for this tool.",),
            )
        work_type = _work_type_for_tool(call.tool_name, instruction=request.query)
        response = self.workflow_service.run(
            WorkRequest(
                work_type=work_type,
                instruction=str(call.input.get("instruction") or call.input.get("query") or request.query),
                target=str(call.input.get("target") or ""),
                source_filter=_tool_source_filters(call.input, request=request),
                access=request.access,
            )
        )
        return _tool_result_from_work_response(call.tool_name, response)


class ComprehensiveAgentVerifier:
    def __init__(self, *, registry: ToolSchemaRegistry) -> None:
        self.registry = registry

    def verify(
        self,
        *,
        plan: AgentPlan,
        results: tuple[AgentToolResult, ...],
        budget: AgentBudget,
    ) -> VerificationResult:
        missing: list[str] = []
        conflicts: list[str] = []
        warnings: list[str] = []
        succeeded_tools = {result.tool_name for result in results if result.status in {"succeeded", "needs_approval"}}
        for tool in plan.required_tools:
            if tool not in succeeded_tools:
                missing.append(f"{tool} の成功結果")
        if budget.require_citations and any(tool in {"circle_rag_search", "minecraft_wiki_rag_search"} for tool in plan.required_tools):
            if not any(result.citations for result in results):
                missing.append("引用可能な根拠")
        for result in results:
            warnings.extend(result.warnings)
            if result.tool_name in _WRITE_TOOLS:
                for candidate in result.candidates:
                    if str(candidate.get("status") or "").lower() in {"done", "completed", "executed"}:
                        conflicts.append(f"{result.tool_name} returned executed candidate {candidate.get('id')}")
                if result.metadata.get("execution_allowed") is True:
                    conflicts.append(f"{result.tool_name} attempted execution before approval")
        if conflicts:
            return VerificationResult(status="failed", missing=tuple(missing), conflicts=tuple(conflicts), warnings=tuple(warnings))
        if missing:
            return VerificationResult(status="needs_more_evidence", missing=tuple(dict.fromkeys(missing)), warnings=tuple(warnings))
        if any(result.status == "needs_approval" or result.candidates for result in results):
            return VerificationResult(status="needs_approval", satisfied=tuple(plan.success_criteria), warnings=tuple(warnings))
        return VerificationResult(status="succeeded", satisfied=tuple(plan.success_criteria), warnings=tuple(warnings))


class ComprehensiveAgentAnswerBuilder:
    def build(
        self,
        *,
        plan: AgentPlan,
        results: tuple[AgentToolResult, ...],
        verification: VerificationResult,
        warnings: tuple[str, ...],
    ) -> tuple[str, str, str]:
        candidate_count = sum(len(result.candidates) for result in results)
        if verification.status == "failed":
            status = "failed"
            confidence = "low"
            conclusion = "検証で問題が見つかったため、実行結果を確定できませんでした。"
        elif verification.status == "needs_more_evidence":
            status = "insufficient_evidence"
            confidence = "low"
            conclusion = "十分な根拠を確認できませんでした。"
        elif candidate_count:
            status = "needs_approval"
            confidence = "medium"
            conclusion = "必要な候補を作成しました。承認前のため正本変更や実行はしていません。"
        else:
            status = "succeeded"
            confidence = "high" if any(result.citations for result in results) else "medium"
            conclusion = "確認できた範囲で回答します。"
        evidence = [
            f"- {result.tool_name}: {result.text}"
            for result in results
            if result.text
        ] or ["- 根拠として使えるtool結果はありません。"]
        used = [f"- {tool}" for tool in plan.required_tools] or ["- なし"]
        candidates = [
            f"- {candidate.get('type')}: {candidate.get('id')} {candidate.get('title') or candidate.get('operation') or ''}".rstrip()
            for result in results
            for candidate in result.candidates
        ] or ["- なし"]
        missing = [f"- {item}" for item in verification.missing] or ["- なし"]
        all_warnings = tuple(dict.fromkeys([*warnings, *verification.warnings]))
        warning_lines = [f"- {warning}" for warning in all_warnings] or ["- なし"]
        text = "\n".join(
            [
                "結論:",
                conclusion,
                "",
                "根拠:",
                *evidence[:8],
                "",
                "使用した機能:",
                *used,
                "",
                "承認待ち候補:",
                *candidates,
                "",
                "未確認事項:",
                *missing,
                "",
                "警告:",
                *warning_lines,
            ]
        )
        return sanitize_text(text, limit=4000), confidence, status

    def detail_markdown(
        self,
        run: AgentRun,
        plan: AgentPlan,
        results: tuple[AgentToolResult, ...],
        verification: VerificationResult | None,
    ) -> str:
        return "\n".join(
            [
                "# Comprehensive Agent Trace",
                "",
                f"- run_id: `{run.id}`",
                f"- status: `{run.status}`",
                f"- confidence: `{run.confidence}`",
                "",
                "## Plan",
                *[f"- {call.tool_name}: {call.side_effect_boundary}" for call in plan.tool_sequence],
                "",
                "## Tool Results",
                *[
                    f"- {result.tool_name}: {result.status}, citations={len(result.citations)}, candidates={len(result.candidates)}"
                    for result in results
                ],
                "",
                "## Verification",
                f"- status: `{verification.status if verification else 'not_run'}`",
                *([f"- missing: {item}" for item in verification.missing] if verification else []),
                *([f"- conflict: {item}" for item in verification.conflicts] if verification else []),
            ]
        )


def _work_type_for_tool(tool_name: str, *, instruction: str) -> str:
    if tool_name == "member_search":
        return "member_search"
    if tool_name == "image_search":
        return "image_search"
    if tool_name == "task_search":
        return "task_list"
    if tool_name == "event_search":
        return "event_list"
    if tool_name == "task_candidate_create":
        text = instruction.lower()
        if any(token in text for token in ("更新", "変更", "update")):
            return "task_update"
        if any(token in text for token in ("削除", "delete")):
            return "task_delete"
        if any(token in text for token in ("抽出", "過去", "資料", "議事録")):
            return "task_extract"
        return "task_add"
    if tool_name == "event_candidate_create":
        text = instruction.lower()
        if any(token in text for token in ("更新", "変更", "update")):
            return "event_update"
        if any(token in text for token in ("削除", "delete")):
            return "event_delete"
        if any(token in text for token in ("抽出", "過去", "資料", "告知")):
            return "event_extract"
        return "event_add"
    if tool_name == "server_operation_candidate_create":
        return "mc_request"
    return "task_list"


def _tool_source_filters(payload: dict[str, object], *, request: ComprehensiveAgentRequest) -> tuple[str, ...]:
    raw = payload.get("source_filter") or payload.get("source_filters") or request.source_filter
    if isinstance(raw, str):
        return (raw,) if raw.strip() and raw != "all" else tuple()
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw if str(item).strip() and str(item) != "all")
    return tuple()


def detect_required_features(query: str, source_filter: str = "all") -> tuple[str, ...]:
    text = query.lower()
    features: list[str] = []
    if source_filter == "minecraft_wiki" or any(token in text for token in ("minecraft", "マイクラ", "redstone", "サーバー", "whitelist")):
        features.append("minecraft_wiki")
    if source_filter == "member" or any(token in query for token in ("メンバー", "担当候補", "誰", "スキル", "得意")):
        features.append("member_search")
    if source_filter == "image" or any(token in query for token in ("画像", "写真", "素材", "asset", "サムネ")):
        features.append("image_search")
    if source_filter == "task" or any(token in query for token in ("タスク", "todo", "ToDo", "担当タスク", "やること")):
        features.append("task_management")
    if source_filter == "event" or any(token in query for token in ("イベント", "予定", "新歓", "日時", "開催")):
        features.append("event_management")
    if any(token in query for token in ("再起動", "停止", "起動", "ホワイトリスト", "サーバー操作", "server")):
        features.append("server_management")
    if source_filter in {"all", "drive", "discord", "notion", "hatena", "x", "crafters_colony"}:
        if any(token in query for token in ("資料", "過去", "根拠", "確認", "調べ", "KUMC", "サークル")):
            features.insert(0, "circle_rag")
    return tuple(dict.fromkeys(features))


def _tool_result_from_work_response(tool_name: str, response: WorkResponse) -> AgentToolResult:
    candidates = _work_candidates(response)
    citations = tuple(
        citation
        for item in (
            *response.task_candidates,
            *response.task_change_candidates,
            *response.event_candidates,
            *response.event_change_candidates,
            *response.schedule_candidates,
        )
        for citation in getattr(item, "evidence", tuple())
    )
    status = "needs_approval" if candidates else "succeeded"
    if response.metadata.get("authorized") is False or response.metadata.get("configured") is False:
        status = "failed"
    if response.metadata.get("candidate_created") is False:
        status = "insufficient_input"
    metadata = sanitize_payload(dict(response.metadata), limit=1200)
    metadata.update(
        {
            "assets": [_dump_item(item) for item in response.assets],
            "member_profiles": [_dump_item(item) for item in response.member_profiles],
            "tasks": [_dump_item(item) for item in response.tasks],
            "events": [_dump_item(item) for item in response.events],
            "execution_allowed": False,
        }
    )
    return AgentToolResult(
        tool_name=tool_name,
        status=status,
        text=sanitize_text(response.detail_markdown or response.text, limit=1200),
        citations=tuple(citations),
        candidates=tuple(candidates),
        warnings=tuple(response.warnings),
        metadata=metadata,
    )


def _work_candidates(response: WorkResponse) -> list[dict[str, Any]]:
    items: list[tuple[str, object]] = []
    items.extend(("task", item) for item in response.task_candidates)
    items.extend(("task_change", item) for item in response.task_change_candidates)
    items.extend(("event", item) for item in response.event_candidates)
    items.extend(("event_change", item) for item in response.event_change_candidates)
    items.extend(("server_operation", item) for item in response.server_operations)
    payload: list[dict[str, Any]] = []
    for item_type, item in items:
        dumped = _dump_item(item)
        payload.append(
            {
                "type": item_type,
                "id": str(dumped.get("id") or ""),
                "title": str(dumped.get("title") or ""),
                "operation": str(dumped.get("operation") or ""),
                "status": str(dumped.get("status") or "proposed"),
                "summary": str(dumped.get("description") or dumped.get("summary") or ""),
                "metadata": sanitize_payload(dumped.get("metadata") or {}, limit=800),
            }
        )
    return payload


def _candidate_dicts(results: list[AgentToolResult], candidate_type: str) -> list[dict[str, Any]]:
    return [
        candidate
        for result in results
        for candidate in result.candidates
        if candidate.get("type") == candidate_type
    ]


def _metadata_items(results: list[AgentToolResult], key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for result in results:
        value = result.metadata.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items


def _plan_payload(plan: AgentPlan) -> dict[str, object]:
    return {
        "tasks": [_dump_item(task) for task in plan.tasks],
        "required_tools": list(plan.required_tools),
        "tool_sequence": [_dump_item(call) for call in plan.tool_sequence],
        "success_criteria": list(plan.success_criteria),
        "side_effect_boundary": plan.side_effect_boundary,
        "retry_policy": dict(plan.retry_policy),
        "answer_requirements": list(plan.answer_requirements),
        "needs_clarification": plan.needs_clarification,
        "clarification_question": plan.clarification_question,
        "metadata": dict(plan.metadata),
    }


def _tool_result_payload(result: AgentToolResult) -> dict[str, object]:
    return {
        "tool_name": result.tool_name,
        "status": result.status,
        "text": result.text,
        "citation_count": len(result.citations),
        "candidates": list(result.candidates),
        "warnings": list(result.warnings),
        "metadata": dict(result.metadata),
    }


def _verification_payload(verification: VerificationResult) -> dict[str, object]:
    return {
        "status": verification.status,
        "satisfied": list(verification.satisfied),
        "missing": list(verification.missing),
        "conflicts": list(verification.conflicts),
        "warnings": list(verification.warnings),
        "metadata": dict(verification.metadata),
    }


def _tool_summary(results: list[AgentToolResult]) -> list[dict[str, object]]:
    return [
        {
            "tool_name": result.tool_name,
            "status": result.status,
            "citation_count": len(result.citations),
            "candidate_count": len(result.candidates),
        }
        for result in results
    ]


def _within_budget(
    budget: AgentBudget,
    *,
    steps: int,
    search_calls: int,
    cost_usd: float,
    elapsed: float,
) -> bool:
    return (
        steps <= budget.max_steps
        and search_calls <= budget.max_search_calls
        and cost_usd <= budget.max_cost_usd
        and elapsed <= budget.max_latency_seconds
    )


def _unique_citations(citations: Any) -> list[Citation]:
    seen: set[tuple[str, str]] = set()
    unique: list[Citation] = []
    for citation in citations:
        key = (citation.source_item_id, citation.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(citation)
    return unique


def _dump_item(item: object) -> dict[str, Any]:
    if is_dataclass(item):
        raw = asdict(item)
    elif isinstance(item, dict):
        raw = dict(item)
    else:
        raw = dict(getattr(item, "__dict__", {}) or {})
    return sanitize_payload(raw, limit=1200)


def sanitize_payload(value: object, *, limit: int = 2000) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in {
                "secret",
                "password",
                "token",
                "api_key",
                "raw",
                "contexts",
                "context",
                "llm_prompt",
                "executor_args",
                "server_state_before",
                "server_state_after",
                "container_state_before",
                "container_state_after",
            }:
                continue
            sanitized[key_text] = sanitize_payload(item, limit=limit)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize_payload(item, limit=limit) for item in value[:50]]
    if is_dataclass(value):
        return sanitize_payload(asdict(value), limit=limit)
    if isinstance(value, str):
        return sanitize_text(value, limit=limit)
    return value


def sanitize_text(text: str, *, limit: int = 2000) -> str:
    masked = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        str(text or ""),
    )
    masked = re.sub(
        r"\b(?:10|172\.(?:1[6-9]|2\d|3[0-1])|192\.168)\.\d{1,3}\.\d{1,3}\b",
        "[internal-ip]",
        masked,
    )
    masked = re.sub(
        r"(?i)(network[_-]?key|pin|unlock(?:ing)?[_ -]?steps?)\s*[:=]\s*[^\n]+",
        r"\1=[REDACTED]",
        masked,
    )
    normalized = re.sub(r"[ \t]+", " ", masked).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."
