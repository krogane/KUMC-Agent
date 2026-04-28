from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
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
from kumc_agent.domain.models.workflow import (
    ApprovalRecord,
    EventApprovalBatch,
    TaskApprovalBatch,
    WorkRequest,
    WorkResponse,
)
from kumc_agent.features.agentic.tools import ToolSchemaRegistry
from kumc_agent.infra.agentic.repository import AgentTraceRepository
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class ComprehensiveLLMConfig:
    enabled: bool = False
    prompt_name: str = ""
    prompts_dir: Path | None = None
    temperature: float = 0.0
    max_output_tokens: int = 1024
    max_retries: int = 1


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
        planner_llm: object | None = None,
        verifier_llm: object | None = None,
        planner_config: ComprehensiveLLMConfig | None = None,
        verifier_config: ComprehensiveLLMConfig | None = None,
        default_budget: AgentBudget | None = None,
    ) -> None:
        self.ask_service = ask_service
        self.repository = repository
        self.workflow_service = workflow_service
        self.registry = registry or ToolSchemaRegistry()
        self.default_budget = default_budget
        self.planner = ComprehensiveAgentPlanner(
            registry=self.registry,
            llm=planner_llm,
            config=planner_config or ComprehensiveLLMConfig(),
        )
        self.adapters = ComprehensiveToolAdapters(
            ask_service=ask_service,
            workflow_service=workflow_service,
        )
        self.verifier = ComprehensiveAgentVerifier(
            registry=self.registry,
            llm=verifier_llm,
            config=verifier_config or ComprehensiveLLMConfig(),
        )
        self.answer_builder = ComprehensiveAgentAnswerBuilder()

    def run(self, request: ComprehensiveAgentRequest) -> ComprehensiveAgentResponse:
        if self.default_budget is not None:
            request = replace(request, budget=_merge_budget(self.default_budget, request.budget))
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

        if plan.direct_route:
            answer = f"単一機能で処理できるため、`{plan.direct_route}` へ直接ルーティングしてください。"
            run = self._finish_run(
                run,
                steps=steps,
                status="direct_route",
                answer=answer,
                confidence="low",
                citations=tuple(),
                metadata={"direct_route": plan.direct_route},
            )
            return ComprehensiveAgentResponse(
                text=answer,
                detail_markdown=self.answer_builder.detail_markdown(run, plan, tuple(), None),
                citations=tuple(),
                confidence="low",
                warnings=tuple(warnings),
                metadata={"agent_run_id": run.id, "direct_route": plan.direct_route},
            )

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
                warnings=tuple(warnings),
                metadata={"agent_run_id": run.id},
            )

        if any(not self.registry.get(tool).read_only for tool in plan.required_tools) and not request.access.is_admin:
            answer = "権限がないため、候補作成を含む総合エージェント依頼は実行できません。"
            warnings.append("permission denied")
            run = self._finish_run(
                run,
                steps=steps,
                status="failed",
                answer=answer,
                confidence="low",
                citations=tuple(),
                metadata={"permission_denied": True},
            )
            return ComprehensiveAgentResponse(
                text=answer,
                detail_markdown=self.answer_builder.detail_markdown(run, plan, tuple(), None),
                citations=tuple(),
                confidence="low",
                warnings=tuple(warnings),
                metadata={"agent_run_id": run.id, "permission_denied": True},
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
                        status="failed",
                        text="候補作成toolは許可されていないため実行しませんでした。",
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
            candidates=tuple(_all_candidates(tool_results)),
            task_candidates=tuple(_candidate_dicts(tool_results, "task")),
            task_change_candidates=tuple(_candidate_dicts(tool_results, "task_change")),
            event_candidates=tuple(_candidate_dicts(tool_results, "event")),
            event_change_candidates=tuple(_candidate_dicts(tool_results, "event_change")),
            schedule_candidates=tuple(_candidate_dicts(tool_results, "schedule")),
            server_operations=tuple(_candidate_dicts(tool_results, "server_operation")),
            approvals=tuple(_metadata_items(tool_results, "approvals")),
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
    def __init__(
        self,
        *,
        registry: ToolSchemaRegistry,
        llm: object | None = None,
        config: ComprehensiveLLMConfig | None = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.config = config or ComprehensiveLLMConfig()

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
        llm_plan = self._plan_with_llm(
            request,
            previous_results=previous_results,
            previous_verification=previous_verification,
        )
        if llm_plan is not None:
            return llm_plan
        return self._fallback_plan(
            request,
            previous_results=previous_results,
            previous_verification=previous_verification,
        )

    def _fallback_plan(
        self,
        request: ComprehensiveAgentRequest,
        *,
        previous_results: tuple[AgentToolResult, ...],
        previous_verification: VerificationResult | None,
    ) -> AgentPlan:
        query = request.query.strip()
        features = list(request.required_features or detect_required_features(query, request.source_filter))
        if request.metadata.get("depth") == "deep" and "circle_rag" not in features:
            features.insert(0, "circle_rag")
        if not features:
            features.append("circle_rag")
        direct_route = ""
        if len(features) == 1 and request.metadata.get("depth") != "deep" and not previous_verification:
            direct_route = _direct_route_for_feature(features[0])
            if direct_route:
                return AgentPlan(
                    required_tools=tuple(),
                    direct_route=direct_route,
                    metadata={"required_features": features, "planner": "fallback"},
                )
        tools = self._tools_for_features(features, query=query)
        if previous_verification and previous_verification.missing:
            if "circle_rag_search" not in tools:
                tools.insert(0, "circle_rag_search")
        replan_suffix = ""
        if previous_verification and previous_verification.missing:
            replan_suffix = " " + " ".join(previous_verification.missing[:2])
        tool_sequence = tuple(
            ToolCallPlan(
                tool_name=tool,
                input=self._tool_input(tool, request, replan_suffix=replan_suffix),
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
            metadata={
                "required_features": features,
                "planner": "fallback",
                "previous_result_count": len(previous_results),
            },
        )

    def _plan_with_llm(
        self,
        request: ComprehensiveAgentRequest,
        *,
        previous_results: tuple[AgentToolResult, ...],
        previous_verification: VerificationResult | None,
    ) -> AgentPlan | None:
        if not self.config.enabled or self.llm is None:
            return None
        system_prompt = _read_prompt(
            self.config.prompts_dir,
            self.config.prompt_name,
            fallback="Return comprehensive agent planning JSON only.",
        )
        payload = {
            "query": request.query,
            "source_filter": request.source_filter,
            "required_features": list(request.required_features),
            "source_filters": request.source_filters,
            "attribute_filters": sanitize_payload(request.attribute_filters, limit=800),
            "risk": request.risk,
            "metadata": sanitize_payload(request.metadata, limit=800),
            "budget": asdict(request.budget),
            "available_tools": [_dump_item(schema) for schema in self.registry.list()],
            "previous_results": [_tool_result_payload(result) for result in previous_results[-8:]],
            "previous_verification": (
                _verification_payload(previous_verification)
                if previous_verification is not None
                else None
            ),
        }
        for _ in range(max(1, self.config.max_retries)):
            raw = _llm_generate(
                self.llm,
                system_prompt=system_prompt,
                user_payload=payload,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
            parsed = _load_json_object(raw)
            if parsed is None:
                continue
            plan = self._plan_from_payload(parsed, request=request)
            if plan is not None:
                return plan
        return None

    def _plan_from_payload(
        self,
        payload: dict[str, Any],
        *,
        request: ComprehensiveAgentRequest,
    ) -> AgentPlan | None:
        if bool(payload.get("needs_clarification")):
            return AgentPlan(
                needs_clarification=True,
                clarification_question=str(payload.get("clarification_question") or "追加情報を確認してください。"),
                metadata={"planner": "llm"},
            )
        direct_route = str(payload.get("direct_route") or "")
        if direct_route and direct_route in {
            "circle_rag",
            "minecraft_wiki_rag",
            "member_search",
            "image_search",
            "task_management",
            "event_management",
            "server_management",
        }:
            return AgentPlan(direct_route=direct_route, metadata={"planner": "llm"})
        calls: list[ToolCallPlan] = []
        for raw_call in payload.get("tool_sequence") or []:
            if not isinstance(raw_call, dict):
                continue
            tool_name = str(raw_call.get("tool_name") or "")
            try:
                schema = self.registry.get(tool_name)
            except KeyError:
                continue
            call_input = raw_call.get("input") if isinstance(raw_call.get("input"), dict) else {}
            calls.append(
                ToolCallPlan(
                    tool_name=tool_name,
                    input=sanitize_payload(call_input, limit=1200),
                    reason=str(raw_call.get("reason") or ""),
                    read_only=schema.read_only,
                    side_effect_boundary=str(
                        raw_call.get("side_effect_boundary")
                        or ("read_only" if schema.read_only else "candidate_only")
                    ),
                )
            )
        if not calls:
            return None
        required_tools = tuple(dict.fromkeys(str(tool) for tool in payload.get("required_tools") or ()))
        if not required_tools:
            required_tools = tuple(dict.fromkeys(call.tool_name for call in calls))
        required_tools = tuple(tool for tool in required_tools if tool in {schema.name for schema in self.registry.list()})
        if not required_tools:
            required_tools = tuple(dict.fromkeys(call.tool_name for call in calls))
        tasks: list[AgentTask] = []
        for index, raw_task in enumerate(payload.get("tasks") or []):
            if not isinstance(raw_task, dict):
                continue
            tool_name = str(raw_task.get("tool_name") or (calls[min(index, len(calls) - 1)].tool_name))
            if tool_name not in required_tools:
                continue
            task_input = raw_task.get("input") if isinstance(raw_task.get("input"), dict) else {}
            criteria = raw_task.get("success_criteria") if isinstance(raw_task.get("success_criteria"), list) else []
            tasks.append(
                AgentTask(
                    id=str(raw_task.get("id") or f"task-{index + 1}"),
                    description=str(raw_task.get("description") or f"{tool_name} を実行する"),
                    tool_name=tool_name,
                    input=sanitize_payload(task_input, limit=1200),
                    success_criteria=tuple(str(item) for item in criteria),
                )
            )
        if not tasks:
            tasks = [
                AgentTask(
                    id=f"task-{index + 1}",
                    description=f"{call.tool_name} を実行する",
                    tool_name=call.tool_name,
                    input=dict(call.input),
                    success_criteria=("tool result is available",),
                )
                for index, call in enumerate(calls)
            ]
        success_criteria = payload.get("success_criteria")
        answer_requirements = payload.get("answer_requirements")
        return AgentPlan(
            tasks=tuple(tasks),
            required_tools=required_tools,
            tool_sequence=tuple(calls),
            success_criteria=tuple(str(item) for item in success_criteria) if isinstance(success_criteria, list) else self._success_criteria(list(required_tools)),
            side_effect_boundary=str(
                payload.get("side_effect_boundary")
                or ("read_only" if all(self.registry.get(tool).read_only for tool in required_tools) else "candidate_only")
            ),
            retry_policy=dict(payload.get("retry_policy") or {"max_replans": request.budget.max_replans}),
            answer_requirements=(
                tuple(str(item) for item in answer_requirements)
                if isinstance(answer_requirements, list)
                else ("結論", "根拠", "使用した機能", "未確認事項", "承認待ち候補")
            ),
            metadata={
                **(dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {}),
                "planner": "llm",
            },
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

    def _tool_input(
        self,
        tool: str,
        request: ComprehensiveAgentRequest,
        *,
        replan_suffix: str = "",
    ) -> dict[str, object]:
        if tool.endswith("_candidate_create") or tool == "approval_candidate_create":
            return {"instruction": request.query, "target": request.attribute_filters.get("target", "")}
        return {"query": (request.query + replan_suffix).strip(), "source_filter": request.source_filter}

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
            if call.tool_name == "approval_candidate_create":
                return self._approval_candidate_create(call, request=request)
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
                access=request.access,
            )
        )
        result = _tool_result_from_work_response(call.tool_name, response)
        if call.tool_name in {
            "task_candidate_create",
            "event_candidate_create",
            "server_operation_candidate_create",
        } and result.candidates:
            result = self._attach_approval_artifacts(result, response, request=request, tool_name=call.tool_name)
        return result

    def _approval_candidate_create(
        self,
        call: ToolCallPlan,
        *,
        request: ComprehensiveAgentRequest,
    ) -> AgentToolResult:
        if self.workflow_service is None:
            return AgentToolResult(
                tool_name=call.tool_name,
                status="insufficient_input",
                text="workflow service is not configured.",
                warnings=("workflow service is required for this tool.",),
            )
        target_type = str(call.input.get("target_type") or call.input.get("type") or "").strip()
        raw_ids = call.input.get("target_ids") or call.input.get("target_id") or call.input.get("id") or []
        if isinstance(raw_ids, str):
            target_ids = [raw_ids]
        elif isinstance(raw_ids, list):
            target_ids = [str(item) for item in raw_ids]
        else:
            target_ids = []
        if not target_type or not target_ids:
            return AgentToolResult(
                tool_name=call.tool_name,
                status="insufficient_input",
                text="承認対象の target_type と target_id が不足しています。",
                warnings=("approval target is missing.",),
            )
        artifacts = self._create_approval_records(
            targets=[{"type": target_type, "id": target_id} for target_id in target_ids],
            request=request,
            source_tool=call.tool_name,
        )
        return AgentToolResult(
            tool_name=call.tool_name,
            status="needs_approval",
            text=f"承認待ち対象を {len(target_ids)} 件作成しました。",
            candidates=tuple({"type": "approval", "id": str(item.get("id") or ""), "target_type": target_type} for item in artifacts.get("approvals", [])),
            metadata=artifacts,
        )

    def _attach_approval_artifacts(
        self,
        result: AgentToolResult,
        response: WorkResponse,
        *,
        request: ComprehensiveAgentRequest,
        tool_name: str,
    ) -> AgentToolResult:
        targets = _approval_targets(response)
        artifacts = self._create_approval_records(
            targets=targets,
            request=request,
            source_tool=tool_name,
        )
        artifacts.update(self._create_approval_batches(response, request=request, source_tool=tool_name))
        metadata = {**result.metadata, **artifacts}
        warnings = list(result.warnings)
        warnings.extend(str(item) for item in artifacts.get("warnings", []) if isinstance(item, str))
        return replace(
            result,
            status="needs_approval",
            text=sanitize_text(
                result.text + "\n承認待ちrecord/batchを自動作成しました。",
                limit=1200,
            ),
            warnings=tuple(dict.fromkeys(warnings)),
            metadata=sanitize_payload(metadata, limit=1200),
        )

    def _create_approval_records(
        self,
        *,
        targets: list[dict[str, str]],
        request: ComprehensiveAgentRequest,
        source_tool: str,
    ) -> dict[str, Any]:
        repository = getattr(self.workflow_service, "repository", None)
        if repository is None:
            return {"approval_targets": targets, "warnings": ["workflow repository is not available."]}
        approvals: list[dict[str, Any]] = []
        warnings: list[str] = []
        actor_id = request.access.user_id or "agent"
        for target in targets:
            target_type = str(target.get("type") or "")
            target_id = str(target.get("id") or "")
            if not target_type or not target_id:
                continue
            record = ApprovalRecord(
                id=stable_hash(f"approval-request:{source_tool}:{target_type}:{target_id}")[:32],
                target_type=target_type,
                target_id=target_id,
                action="requested",
                actor_id=actor_id,
                comment="comprehensive_agent auto approval request",
            )
            try:
                stored = repository.save_approval(record)
                approvals.append(_dump_item(stored))
            except Exception as exc:  # pragma: no cover - repository-specific
                warnings.append(f"approval record creation failed: {type(exc).__name__}")
        return {"approval_targets": targets, "approvals": approvals, "warnings": warnings}

    def _create_approval_batches(
        self,
        response: WorkResponse,
        *,
        request: ComprehensiveAgentRequest,
        source_tool: str,
    ) -> dict[str, Any]:
        repository = getattr(self.workflow_service, "repository", None)
        if repository is None:
            return {}
        payload: dict[str, Any] = {}
        warnings: list[str] = []
        task_ids = tuple(candidate.id for candidate in response.task_candidates)
        task_change_ids = tuple(candidate.id for candidate in response.task_change_candidates)
        if task_ids or task_change_ids:
            batch = TaskApprovalBatch(
                id=stable_hash(f"task-batch:{source_tool}:{':'.join(task_ids)}:{':'.join(task_change_ids)}")[:32],
                candidate_ids=task_ids,
                change_candidate_ids=task_change_ids,
                period_start=datetime.now(UTC) - timedelta(days=1),
                period_end=datetime.now(UTC),
                status="pending",
                metadata={"source": "comprehensive_agent", "auto_created": True},
            )
            try:
                payload["task_approval_batches"] = [_dump_item(repository.save_task_approval_batch(batch))]
            except Exception as exc:  # pragma: no cover - repository-specific
                warnings.append(f"task approval batch creation failed: {type(exc).__name__}")
        event_ids = tuple(candidate.id for candidate in response.event_candidates)
        event_change_ids = tuple(candidate.id for candidate in response.event_change_candidates)
        if event_ids or event_change_ids:
            batch = EventApprovalBatch(
                id=stable_hash(f"event-batch:{source_tool}:{':'.join(event_ids)}:{':'.join(event_change_ids)}")[:32],
                candidate_ids=event_ids,
                change_candidate_ids=event_change_ids,
                period_start=datetime.now(UTC) - timedelta(days=1),
                period_end=datetime.now(UTC),
                status="pending",
                metadata={"source": "comprehensive_agent", "auto_created": True},
            )
            try:
                payload["event_approval_batches"] = [_dump_item(repository.save_event_approval_batch(batch))]
            except Exception as exc:  # pragma: no cover - repository-specific
                warnings.append(f"event approval batch creation failed: {type(exc).__name__}")
        if warnings:
            payload["warnings"] = warnings
        return payload


class ComprehensiveAgentVerifier:
    def __init__(
        self,
        *,
        registry: ToolSchemaRegistry,
        llm: object | None = None,
        config: ComprehensiveLLMConfig | None = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.config = config or ComprehensiveLLMConfig()

    def verify(
        self,
        *,
        plan: AgentPlan,
        results: tuple[AgentToolResult, ...],
        budget: AgentBudget,
    ) -> VerificationResult:
        deterministic = self._deterministic_verify(plan=plan, results=results, budget=budget)
        llm_result = self._verify_with_llm(plan=plan, results=results, deterministic=deterministic)
        if llm_result is None:
            return deterministic
        missing = tuple(dict.fromkeys([*deterministic.missing, *llm_result.missing]))
        conflicts = tuple(dict.fromkeys([*deterministic.conflicts, *llm_result.conflicts]))
        warnings = tuple(dict.fromkeys([*deterministic.warnings, *llm_result.warnings]))
        if conflicts:
            status = "failed"
        elif missing:
            status = "needs_more_evidence"
        else:
            status = llm_result.status
        return VerificationResult(
            status=status,
            satisfied=tuple(dict.fromkeys([*deterministic.satisfied, *llm_result.satisfied])),
            missing=missing,
            conflicts=conflicts,
            warnings=warnings,
            metadata={**deterministic.metadata, **llm_result.metadata, "verifier": "llm"},
        )

    def _deterministic_verify(
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
            forbidden_paths = _find_forbidden_payload_paths(result.metadata)
            if forbidden_paths:
                conflicts.append(f"{result.tool_name} metadata contains forbidden payload keys: {', '.join(forbidden_paths[:5])}")
            if _contains_unmasked_secret(result.text) or _contains_unmasked_secret(json.dumps(result.candidates, ensure_ascii=False, default=str)):
                conflicts.append(f"{result.tool_name} output may contain unmasked secret")
            if result.tool_name in _WRITE_TOOLS:
                for candidate in result.candidates:
                    if str(candidate.get("status") or "").lower() in {"done", "completed", "executed"}:
                        conflicts.append(f"{result.tool_name} returned executed candidate {candidate.get('id')}")
                if result.metadata.get("execution_allowed") is True:
                    conflicts.append(f"{result.tool_name} attempted execution before approval")
                if result.metadata.get("tasks") or result.metadata.get("events") or result.metadata.get("schedules"):
                    conflicts.append(f"{result.tool_name} returned master records before approval")
                if result.candidates and not result.metadata.get("approvals") and not (
                    result.metadata.get("task_approval_batches") or result.metadata.get("event_approval_batches")
                ):
                    missing.append(f"{result.tool_name} の承認recordまたはbatch")
        conflicts.extend(_semantic_field_conflicts(results))
        if conflicts:
            return VerificationResult(status="failed", missing=tuple(dict.fromkeys(missing)), conflicts=tuple(dict.fromkeys(conflicts)), warnings=tuple(dict.fromkeys(warnings)), metadata={"verifier": "deterministic"})
        if missing:
            return VerificationResult(status="needs_more_evidence", missing=tuple(dict.fromkeys(missing)), warnings=tuple(dict.fromkeys(warnings)), metadata={"verifier": "deterministic"})
        if any(result.status == "needs_approval" or result.candidates for result in results):
            return VerificationResult(status="needs_approval", satisfied=tuple(plan.success_criteria), warnings=tuple(dict.fromkeys(warnings)), metadata={"verifier": "deterministic"})
        return VerificationResult(status="succeeded", satisfied=tuple(plan.success_criteria), warnings=tuple(dict.fromkeys(warnings)), metadata={"verifier": "deterministic"})

    def _verify_with_llm(
        self,
        *,
        plan: AgentPlan,
        results: tuple[AgentToolResult, ...],
        deterministic: VerificationResult,
    ) -> VerificationResult | None:
        if not self.config.enabled or self.llm is None:
            return None
        system_prompt = _read_prompt(
            self.config.prompts_dir,
            self.config.prompt_name,
            fallback="Return comprehensive agent verification JSON only.",
        )
        payload = {
            "plan": _plan_payload(plan),
            "tool_results": [_tool_result_payload(result) for result in results],
            "deterministic": _verification_payload(deterministic),
        }
        for _ in range(max(1, self.config.max_retries)):
            raw = _llm_generate(
                self.llm,
                system_prompt=system_prompt,
                user_payload=payload,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
            parsed = _load_json_object(raw)
            if parsed is None:
                continue
            status = str(parsed.get("status") or "")
            if status not in {"succeeded", "needs_approval", "needs_more_evidence", "failed"}:
                continue
            return VerificationResult(
                status=status,
                satisfied=_string_tuple(parsed.get("satisfied")),
                missing=_string_tuple(parsed.get("missing")),
                conflicts=_string_tuple(parsed.get("conflicts")),
                warnings=_string_tuple(parsed.get("warnings")),
                metadata=dict(parsed.get("metadata") or {}) if isinstance(parsed.get("metadata"), dict) else {},
            )
        return None


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


def _direct_route_for_feature(feature: str) -> str:
    return {
        "circle_rag": "circle_rag",
        "circle_rag_search": "circle_rag",
        "minecraft_wiki": "minecraft_wiki_rag",
        "minecraft_wiki_rag": "minecraft_wiki_rag",
        "member_search": "member_search",
        "image_search": "image_search",
        "task_management": "task_management",
        "event_management": "event_management",
        "server_management": "server_management",
    }.get(feature, "")


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
            "schedules": [_dump_item(item) for item in response.schedules],
            "approvals": [_dump_item(item) for item in response.approvals],
            "task_approval_batches": [_dump_item(item) for item in response.task_approval_batches],
            "event_approval_batches": [_dump_item(item) for item in response.event_approval_batches],
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
    items.extend(("schedule", item) for item in response.schedule_candidates)
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


def _approval_targets(response: WorkResponse) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    targets.extend({"type": "task", "id": item.id} for item in response.task_candidates)
    targets.extend({"type": "task", "id": item.id} for item in response.task_change_candidates)
    targets.extend({"type": "event", "id": item.id} for item in response.event_candidates)
    targets.extend({"type": "event", "id": item.id} for item in response.event_change_candidates)
    targets.extend({"type": "schedule", "id": item.id} for item in response.schedule_candidates)
    targets.extend({"type": "server_operation", "id": item.id} for item in response.server_operations)
    return targets


def _all_candidates(results: list[AgentToolResult]) -> list[dict[str, Any]]:
    return [candidate for result in results for candidate in result.candidates]


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
        "direct_route": plan.direct_route,
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


def _read_prompt(prompts_dir: Path | None, prompt_name: str, *, fallback: str) -> str:
    if prompts_dir is None or not prompt_name:
        return fallback
    path = prompts_dir / f"{prompt_name}.md"
    if not path.exists() and prompt_name.endswith(".md"):
        path = prompts_dir / prompt_name
    if not path.exists():
        return fallback
    return path.read_text(encoding="utf-8")


def _llm_generate(
    llm: object,
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    temperature: float,
    max_output_tokens: int,
) -> str:
    generate = getattr(llm, "generate")
    return str(
        generate(
            system_prompt=system_prompt,
            user_prompt=json.dumps(user_payload, ensure_ascii=False, default=str),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    )


def _load_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        stripped = match.group(1).strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return tuple()


_FORBIDDEN_TRACE_KEYS = {
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
}


def _find_forbidden_payload_paths(value: object, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.lower() in _FORBIDDEN_TRACE_KEYS:
                paths.append(path)
                continue
            paths.extend(_find_forbidden_payload_paths(item, prefix=path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value[:50]):
            paths.extend(_find_forbidden_payload_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def _contains_unmasked_secret(text: str) -> bool:
    normalized = str(text or "")
    if "[REDACTED]" in normalized:
        return False
    return bool(
        re.search(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+", normalized)
        or re.search(r"(?i)(network[_-]?key|pin|unlock(?:ing)?[_ -]?steps?)\s*[:=]\s*[^\n]+", normalized)
        or re.search(r"\b(?:10|172\.(?:1[6-9]|2\d|3[0-1])|192\.168)\.\d{1,3}\.\d{1,3}\b", normalized)
    )


def _semantic_field_conflicts(results: tuple[AgentToolResult, ...]) -> list[str]:
    seen: dict[tuple[str, str], str] = {}
    conflicts: list[str] = []
    for result in results:
        for candidate in result.candidates:
            candidate_id = str(candidate.get("id") or "")
            if not candidate_id:
                continue
            for field in ("title", "operation", "status"):
                value = str(candidate.get(field) or "")
                if not value:
                    continue
                key = (candidate_id, field)
                previous = seen.get(key)
                if previous is not None and previous != value:
                    conflicts.append(f"candidate {candidate_id} has conflicting {field}: {previous} / {value}")
                seen[key] = value
    return conflicts


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


def _merge_budget(default: AgentBudget, requested: AgentBudget) -> AgentBudget:
    baseline = AgentBudget()
    return AgentBudget(
        max_steps=requested.max_steps if requested.max_steps != baseline.max_steps else default.max_steps,
        max_search_calls=(
            requested.max_search_calls
            if requested.max_search_calls != baseline.max_search_calls
            else default.max_search_calls
        ),
        max_read_chunks=(
            requested.max_read_chunks
            if requested.max_read_chunks != baseline.max_read_chunks
            else default.max_read_chunks
        ),
        max_replans=(
            requested.max_replans if requested.max_replans != baseline.max_replans else default.max_replans
        ),
        max_cost_usd=(
            requested.max_cost_usd if requested.max_cost_usd != baseline.max_cost_usd else default.max_cost_usd
        ),
        max_latency_seconds=(
            requested.max_latency_seconds
            if requested.max_latency_seconds != baseline.max_latency_seconds
            else default.max_latency_seconds
        ),
        allow_write_tools=requested.allow_write_tools,
        require_citations=(
            requested.require_citations
            if requested.require_citations != baseline.require_citations
            else default.require_citations
        ),
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
