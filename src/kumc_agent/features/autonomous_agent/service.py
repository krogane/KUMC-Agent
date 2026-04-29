from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from kumc_agent.domain.models.agentic import AgentRun, AgentStep
from kumc_agent.domain.models.audit import AuditEvent
from kumc_agent.domain.models.automation import AutomationRun
from kumc_agent.domain.models.autonomous_agent import (
    AutonomousDecision,
    AutonomousAgentRequest,
    AutonomousAgentResponse,
    AutonomousPlan,
    AutonomousToolResult,
    autonomous_budget_from_config,
)
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.features.autonomous_agent.idempotency import (
    AutonomousIdempotencyInput,
    build_autonomous_idempotency_key,
)
from kumc_agent.features.autonomous_agent.integrated_input import AutonomousIntegratedInputAdapter
from kumc_agent.features.autonomous_agent.planner import AutonomousPlanner
from kumc_agent.features.autonomous_agent.sanitizer import (
    sanitize_autonomous_metadata,
    sanitize_autonomous_payload,
)
from kumc_agent.features.autonomous_agent.snapshot import AutonomousSnapshotCollector
from kumc_agent.features.autonomous_agent.verifier import AutonomousVerifier
from kumc_agent.infra.agentic import AgentTraceRepository
from kumc_agent.infra.audit.repository import AuditLogRepository
from kumc_agent.infra.automation import AutomationRepository


@dataclass(frozen=True)
class AutonomousAgentServiceConfig:
    enabled: bool = False
    schedule_times: tuple[str, ...] = ("08:00", "13:00", "20:00")
    timezone: str = "Asia/Tokyo"
    scopes: tuple[str, ...] = ("tasks", "events", "rag_delta", "server_ops", "automation")
    notification_channel_id: str = ""
    dry_run: bool = True
    lookahead_days: dict[str, int] | None = None
    duplicate_suppression_hours: int = 24
    budget: object | None = None
    system_user_id: str = "system"
    system_guild_id: str = ""
    system_role_ids: tuple[str, ...] = tuple()
    system_is_admin: bool = False


class AutonomousAgentService:
    def __init__(
        self,
        *,
        config: AutonomousAgentServiceConfig,
        trace_repository: AgentTraceRepository,
        automation_repository: AutomationRepository,
        snapshot_collector: AutonomousSnapshotCollector,
        planner: AutonomousPlanner,
        adapter: AutonomousIntegratedInputAdapter,
        verifier: AutonomousVerifier,
        audit_log: AuditLogRepository | None = None,
    ) -> None:
        self.config = config
        self.trace_repository = trace_repository
        self.automation_repository = automation_repository
        self.snapshot_collector = snapshot_collector
        self.planner = planner
        self.adapter = adapter
        self.verifier = verifier
        self.audit_log = audit_log

    def run(self, request: AutonomousAgentRequest) -> AutonomousAgentResponse:
        access = self._system_access(request.access)
        scopes = request.scopes or self.config.scopes
        dry_run = self.config.dry_run if request.dry_run is None else bool(request.dry_run)
        lookahead = dict(self.config.lookahead_days or {})
        idempotency_key = request.idempotency_key.strip() or build_autonomous_idempotency_key(
            AutonomousIdempotencyInput(
                slot=request.slot or request.trigger,
                scopes=scopes,
                timezone=self.config.timezone,
                guild_id=access.guild_id,
                channel_id=self.config.notification_channel_id,
                lookahead_days=lookahead,
            )
        )
        existing = self.automation_repository.get_run_by_idempotency_key(idempotency_key)
        if existing is not None:
            self._audit(
                "autonomous_agent.duplicate",
                access,
                "duplicate",
                existing.id,
                {"idempotency_key": idempotency_key, "existing_status": existing.status},
            )
            return AutonomousAgentResponse(
                status="duplicate",
                text=f"同じ自律エージェントrunは記録済みです: {existing.id}",
                warnings=("duplicate_idempotency_key",),
                metadata={
                    "duplicate": True,
                    "existing_run_id": existing.id,
                    "idempotency_key": idempotency_key,
                },
            )

        if not self.config.enabled and request.trigger in {"automation", "schedule"}:
            history_id = str(uuid4())
            self._record_history(
                request=request,
                status="blocked",
                idempotency_key=idempotency_key,
                run_id="",
                warnings=("autonomous_agent_disabled",),
                dry_run=dry_run,
                history_id=history_id,
                metadata={"blocked_reason": "autonomous_agent_disabled"},
            )
            self._audit(
                "autonomous_agent.blocked",
                access,
                "blocked",
                idempotency_key,
                {"reason": "autonomous_agent_disabled"},
            )
            return AutonomousAgentResponse(
                status="blocked",
                text="自律エージェントは設定で無効化されています。",
                warnings=("autonomous_agent_disabled",),
                metadata={
                    "idempotency_key": idempotency_key,
                    "dry_run": dry_run,
                    "blocked": True,
                },
            )

        run = self.trace_repository.save_run(
            AgentRun(
                id=str(uuid4()),
                query=f"autonomous_agent:{request.trigger}:{request.slot}",
                status="running",
                access=access,
                budget=autonomous_budget_from_config(self.config.budget),
                metadata={
                    "agent": "autonomous_agent",
                    "trigger": request.trigger,
                    "slot": request.slot,
                    "scopes": list(scopes),
                    "dry_run": dry_run,
                    "idempotency_key": idempotency_key,
                },
            )
        )
        history_id = str(uuid4())
        self._record_history(
            request=request,
            status="running",
            idempotency_key=idempotency_key,
            run_id=run.id,
            warnings=tuple(),
            dry_run=dry_run,
            history_id=history_id,
            metadata={"reservation": True},
        )
        try:
            response = self._run_started(
                run=run,
                request=request,
                access=access,
                scopes=scopes,
                dry_run=dry_run,
                idempotency_key=idempotency_key,
                history_id=history_id,
            )
        except Exception as exc:
            failed_run = self.trace_repository.save_run(
                replace(
                    run,
                    status="failed",
                    metadata={
                        **run.metadata,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            )
            self._record_history(
                request=request,
                status="failed",
                idempotency_key=idempotency_key,
                run_id=failed_run.id,
                warnings=(f"autonomous_agent_failed:{type(exc).__name__}",),
                dry_run=dry_run,
                history_id=history_id,
                metadata={"error_type": type(exc).__name__, "error": str(exc)},
            )
            self._audit("autonomous_agent.failed", access, "failed", failed_run.id, {"error": str(exc)})
            return AutonomousAgentResponse(
                status="failed",
                text="自律エージェントrunが失敗しました。",
                warnings=(f"autonomous_agent_failed:{type(exc).__name__}",),
                run=failed_run,
                metadata={"run_id": failed_run.id, "idempotency_key": idempotency_key},
            )
        return response

    def _run_started(
        self,
        *,
        run: AgentRun,
        request: AutonomousAgentRequest,
        access: AccessContext,
        scopes: tuple[str, ...],
        dry_run: bool,
        idempotency_key: str,
        history_id: str,
    ) -> AutonomousAgentResponse:
        started_at = datetime.now(UTC)
        snapshot = self.snapshot_collector.collect(scopes=scopes)
        max_steps = max(0, int(run.budget.max_steps))
        max_search_calls = max(0, int(run.budget.max_search_calls))
        max_replans = max(0, int(run.budget.max_replans))
        max_latency_seconds = max(0.0, float(run.budget.max_latency_seconds))
        replan_count = 0
        search_calls = 0
        tool_step_count = 0
        all_tool_results: list[AutonomousToolResult] = []
        plan: AutonomousPlan | None = None
        decision: AutonomousDecision | None = None
        budget_warnings: list[str] = []

        while True:
            plan = self.planner.plan(snapshot)
            self._save_step(
                run.id,
                "PLAN",
                input_payload={"snapshot": snapshot, "replan_count": replan_count},
                output_payload={"plan": plan},
            )
            self._audit(
                "autonomous_agent.plan",
                access,
                "succeeded",
                run.id,
                {
                    "target_refs": list(plan.target_refs),
                    "checks": len(plan.checks),
                    "queries": len(plan.required_queries),
                    "replan_count": replan_count,
                },
            )

            iteration_results: list[AutonomousToolResult] = []
            for query in plan.required_queries:
                if max_latency_seconds and (datetime.now(UTC) - started_at).total_seconds() >= max_latency_seconds:
                    budget_warnings.append("budget_max_latency_seconds_exceeded")
                    break
                if tool_step_count >= max_steps:
                    budget_warnings.append("budget_max_steps_exceeded")
                    break
                if search_calls >= max_search_calls:
                    budget_warnings.append("budget_max_search_calls_exceeded")
                    break
                result = self.adapter.run_query(query, access=access, dry_run=dry_run)
                iteration_results.append(result)
                all_tool_results.append(result)
                search_calls += 1
                tool_step_count += 1
                self._save_step(
                    run.id,
                    "TOOL",
                    input_payload={"query": query, "replan_count": replan_count},
                    output_payload={"result": result},
                    status=result.status,
                )
                self._audit(
                    "autonomous_agent.tool",
                    access,
                    result.status,
                    run.id,
                    {
                        "query_id": query.id,
                        "target_refs": list(query.target_refs),
                        "side_effects": result.metadata.get("side_effects", "none"),
                    },
                )

            decision = self.verifier.verify(plan=plan, tool_results=tuple(iteration_results))
            if budget_warnings:
                decision = replace(
                    decision,
                    warnings=tuple(dict.fromkeys([*decision.warnings, *budget_warnings])),
                )
            self._save_step(
                run.id,
                "VERIFY",
                input_payload={
                    "plan": plan,
                    "tool_results": tuple(iteration_results),
                    "replan_count": replan_count,
                },
                output_payload={"decision": decision},
            )
            self._audit_verify(access=access, run_id=run.id, decision=decision, replan_count=replan_count)

            if (
                decision.decision == "retry_search"
                and replan_count < max_replans
                and search_calls < max_search_calls
                and tool_step_count < max_steps
            ):
                replan_count += 1
                continue
            if decision.decision == "retry_search" and replan_count >= max_replans:
                decision = replace(
                    decision,
                    warnings=tuple(dict.fromkeys([*decision.warnings, "budget_max_replans_exceeded"])),
                )
            break

        assert plan is not None
        assert decision is not None

        status = _status_from_decision(decision.decision, warnings=decision.warnings)
        notification_target_refs = tuple(
            dict.fromkeys(
                ref
                for proposal in decision.notification_proposals
                for ref in proposal.target_refs
            )
        )
        elapsed_seconds = (datetime.now(UTC) - started_at).total_seconds()
        typed_refs = _typed_result_refs(all_tool_results)
        proposals = _proposal_payloads(decision)
        final_run = self.trace_repository.save_run(
            replace(
                run,
                status=status,
                answer=_response_text(status, decision),
                metadata=sanitize_autonomous_metadata(
                    {
                        **run.metadata,
                        "notification_proposals": decision.notification_proposals,
                        "approval_requests": decision.approval_requests,
                        "candidate_refs": list(decision.candidate_refs),
                        "notification_target_refs": list(notification_target_refs),
                        "task_candidates": typed_refs["task_candidates"],
                        "event_candidates": typed_refs["event_candidates"],
                        "automation_runs": typed_refs["automation_runs"],
                        "server_operations": typed_refs["server_operations"],
                        "replan_count": replan_count,
                        "search_calls": search_calls,
                        "elapsed_seconds": elapsed_seconds,
                        "cost_usd": 0.0,
                    }
                ),
            )
        )
        self._audit_proposals(access=access, run_id=final_run.id, decision=decision)
        self._record_history(
            request=request,
            status=status,
            idempotency_key=idempotency_key,
            run_id=final_run.id,
            warnings=decision.warnings,
            dry_run=dry_run,
            history_id=history_id,
            metadata={
                "notification_proposals": decision.notification_proposals,
                "approval_requests": decision.approval_requests,
                "candidate_refs": list(decision.candidate_refs),
                "task_candidates": typed_refs["task_candidates"],
                "event_candidates": typed_refs["event_candidates"],
                "automation_runs": typed_refs["automation_runs"],
                "server_operations": typed_refs["server_operations"],
                "replan_count": replan_count,
                "search_calls": search_calls,
                "elapsed_seconds": elapsed_seconds,
                "cost_usd": 0.0,
            },
        )
        return AutonomousAgentResponse(
            status=status,
            text=_response_text(status, decision),
            detail_markdown=_detail_markdown(plan, decision),
            proposals=tuple(proposals),
            notification_proposals=decision.notification_proposals,
            approval_requests=decision.approval_requests,
            candidate_refs=decision.candidate_refs,
            task_candidates=tuple(typed_refs["task_candidates"]),
            event_candidates=tuple(typed_refs["event_candidates"]),
            automation_runs=tuple(typed_refs["automation_runs"]),
            server_operations=tuple(typed_refs["server_operations"]),
            warnings=decision.warnings,
            run=final_run,
            metadata={
                "run_id": final_run.id,
                "trace_id": final_run.id,
                "idempotency_key": idempotency_key,
                "decision": decision.decision,
                "dry_run": dry_run,
                "search_calls": search_calls,
                "replan_count": replan_count,
                "elapsed_seconds": elapsed_seconds,
                "cost_usd": 0.0,
            },
        )

    def _save_step(
        self,
        run_id: str,
        state: str,
        *,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        status: str = "succeeded",
    ) -> AgentStep:
        return self.trace_repository.save_step(
            AgentStep(
                id=str(uuid4()),
                run_id=run_id,
                state=state,
                input=_dict_payload(input_payload),
                output=_dict_payload(output_payload),
                status=status,
            )
        )

    def _record_history(
        self,
        *,
        request: AutonomousAgentRequest,
        status: str,
        idempotency_key: str,
        run_id: str,
        warnings: tuple[str, ...],
        dry_run: bool,
        history_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.automation_repository.save_run(
            AutomationRun(
                id=history_id or str(uuid4()),
                rule_id="autonomous_agent",
                trigger_key=f"{request.trigger}:{request.slot}",
                mode="dry_run" if dry_run else "approval_required",
                status=status,
                idempotency_key=idempotency_key,
                warnings=warnings,
                metadata=sanitize_autonomous_metadata(
                    {
                        "agent_run_id": run_id,
                        "side_effects": "none",
                        **(metadata or {}),
                    }
                ),
            )
        )

    def _audit_verify(
        self,
        *,
        access: AccessContext,
        run_id: str,
        decision: AutonomousDecision,
        replan_count: int,
    ) -> None:
        self._audit(
            "autonomous_agent.verify",
            access,
            decision.decision,
            run_id,
            {
                "notification_ids": [proposal.id for proposal in decision.notification_proposals],
                "approval_request_ids": [proposal.id for proposal in decision.approval_requests],
                "candidate_refs": list(decision.candidate_refs),
                "missing": list(decision.missing),
                "conflicts": list(decision.conflicts),
                "replan_count": replan_count,
                "decision_metadata": decision.metadata,
            },
        )

    def _audit_proposals(
        self,
        *,
        access: AccessContext,
        run_id: str,
        decision: AutonomousDecision,
    ) -> None:
        for proposal in decision.notification_proposals:
            self._audit(
                "autonomous_agent.proposal",
                access,
                "notification_proposed",
                run_id,
                {
                    "proposal_id": proposal.id,
                    "target_refs": list(proposal.target_refs),
                    "risk": proposal.risk,
                },
            )
        for proposal in decision.approval_requests:
            self._audit(
                "autonomous_agent.proposal",
                access,
                "approval_proposed",
                run_id,
                {
                    "proposal_id": proposal.id,
                    "target_type": proposal.target_type,
                    "target_id": proposal.target_id,
                    "risk": proposal.risk,
                },
            )

    def _system_access(self, access: AccessContext) -> AccessContext:
        if access.user_id:
            return access
        return AccessContext(
            user_id=self.config.system_user_id or "system",
            guild_id=access.guild_id or self.config.system_guild_id,
            role_ids=access.role_ids or self.config.system_role_ids,
            is_admin=bool(access.is_admin or self.config.system_is_admin),
        )

    def _audit(
        self,
        action: str,
        access: AccessContext,
        outcome: str,
        target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.audit_log is None:
            return
        self.audit_log.append(
            AuditEvent(
                action=action,
                actor_id=access.user_id or "system",
                actor_type="system",
                outcome=outcome,
                target=target,
                risk_level="low",
                trace_id=target,
                metadata=sanitize_autonomous_metadata(metadata or {}),
            )
        )


def _dict_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_autonomous_payload(payload)
    return sanitized if isinstance(sanitized, dict) else {}


def _typed_result_refs(results: list[AutonomousToolResult]) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {
        "task_candidates": [],
        "event_candidates": [],
        "automation_runs": [],
        "server_operations": [],
    }
    for result in results:
        for key in payload:
            value = result.metadata.get(key)
            if isinstance(value, list):
                payload[key].extend(
                    item
                    for item in value
                    if isinstance(item, dict) and str(item.get("id") or "").strip()
                )
    for key, values in payload.items():
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for value in values:
            item_id = str(value.get("id") or "")
            if item_id in seen:
                continue
            seen.add(item_id)
            unique.append(dict(value))
        payload[key] = unique
    return payload


def _proposal_payloads(decision: AutonomousDecision) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for proposal in decision.notification_proposals:
        proposals.append(
            {
                "id": proposal.id,
                "type": "notification",
                "status": proposal.status,
                "target_refs": list(proposal.target_refs),
                "risk": proposal.risk,
            }
        )
    for proposal in decision.approval_requests:
        proposals.append(
            {
                "id": proposal.id,
                "type": "approval_request",
                "status": proposal.status,
                "target_type": proposal.target_type,
                "target_id": proposal.target_id,
                "risk": proposal.risk,
            }
        )
    return proposals


def _status_from_decision(decision: str, *, warnings: tuple[str, ...]) -> str:
    if any(warning.startswith("candidate_citations_missing") for warning in warnings):
        return "insufficient_evidence"
    if decision in {"request_approval", "create_candidates"}:
        return "needs_approval"
    if decision == "notify":
        return "succeeded"
    if decision == "retry_search":
        return "insufficient_evidence"
    return "noop"


def _response_text(status: str, decision) -> str:
    if status == "noop":
        return "自律エージェントの確認対象はありませんでした。"
    return (
        "自律エージェントrunを完了しました。"
        f" 通知候補 {len(decision.notification_proposals)} 件、"
        f"承認申請候補 {len(decision.approval_requests)} 件、"
        f"候補参照 {len(decision.candidate_refs)} 件。"
    )


def _detail_markdown(plan, decision) -> str:
    lines = [
        "## PLAN",
        f"- checks: {len(plan.checks)}",
        f"- queries: {len(plan.required_queries)}",
        f"- risk: {plan.risk}",
        "## VERIFY",
        f"- decision: {decision.decision}",
        f"- notifications: {len(decision.notification_proposals)}",
        f"- approvals: {len(decision.approval_requests)}",
        f"- candidates: {len(decision.candidate_refs)}",
    ]
    if decision.warnings:
        lines.extend(["## Warnings", *[f"- {warning}" for warning in decision.warnings]])
    return "\n".join(lines)
