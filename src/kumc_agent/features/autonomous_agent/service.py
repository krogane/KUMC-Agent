from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from kumc_agent.domain.models.agentic import AgentRun, AgentStep
from kumc_agent.domain.models.audit import AuditEvent
from kumc_agent.domain.models.automation import AutomationRun
from kumc_agent.domain.models.autonomous_agent import (
    AutonomousAgentRequest,
    AutonomousAgentResponse,
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
        access = _system_access(request.access)
        scopes = request.scopes or self.config.scopes
        dry_run = self.config.dry_run if request.dry_run is True else bool(request.dry_run)
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
        try:
            response = self._run_started(
                run=run,
                request=request,
                access=access,
                scopes=scopes,
                dry_run=dry_run,
                idempotency_key=idempotency_key,
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
    ) -> AutonomousAgentResponse:
        snapshot = self.snapshot_collector.collect(scopes=scopes)
        plan = self.planner.plan(snapshot)
        self._save_step(
            run.id,
            "PLAN",
            input_payload={"snapshot": snapshot},
            output_payload={"plan": plan},
        )
        self._audit(
            "autonomous_agent.plan",
            access,
            "succeeded",
            run.id,
            {"target_refs": list(plan.target_refs), "checks": len(plan.checks)},
        )

        tool_results = []
        max_steps = max(0, int(run.budget.max_steps))
        for query in plan.required_queries[:max_steps]:
            result = self.adapter.run_query(query, access=access, dry_run=dry_run)
            tool_results.append(result)
            self._save_step(
                run.id,
                "TOOL",
                input_payload={"query": query},
                output_payload={"result": result},
                status=result.status,
            )
            self._audit(
                "autonomous_agent.tool",
                access,
                result.status,
                run.id,
                {"query_id": query.id, "target_refs": list(query.target_refs)},
            )
        budget_warnings: tuple[str, ...] = tuple()
        if len(plan.required_queries) > max_steps:
            budget_warnings = ("budget_max_steps_exceeded",)

        decision = self.verifier.verify(plan=plan, tool_results=tuple(tool_results))
        if budget_warnings:
            decision = replace(
                decision,
                warnings=tuple(dict.fromkeys([*decision.warnings, *budget_warnings])),
            )
        self._save_step(
            run.id,
            "VERIFY",
            input_payload={"plan": plan, "tool_results": tuple(tool_results)},
            output_payload={"decision": decision},
        )
        self._audit(
            "autonomous_agent.verify",
            access,
            decision.decision,
            run.id,
            {
                "notification_ids": [proposal.id for proposal in decision.notification_proposals],
                "approval_request_ids": [proposal.id for proposal in decision.approval_requests],
                "candidate_refs": list(decision.candidate_refs),
            },
        )

        status = _status_from_decision(decision.decision, warnings=decision.warnings)
        notification_target_refs = tuple(
            dict.fromkeys(
                ref
                for proposal in decision.notification_proposals
                for ref in proposal.target_refs
            )
        )
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
                    }
                ),
            )
        )
        self._record_history(
            request=request,
            status=status,
            idempotency_key=idempotency_key,
            run_id=final_run.id,
            warnings=decision.warnings,
            dry_run=dry_run,
        )
        return AutonomousAgentResponse(
            status=status,
            text=_response_text(status, decision),
            detail_markdown=_detail_markdown(plan, decision),
            notification_proposals=decision.notification_proposals,
            approval_requests=decision.approval_requests,
            candidate_refs=decision.candidate_refs,
            warnings=decision.warnings,
            run=final_run,
            metadata={
                "run_id": final_run.id,
                "trace_id": final_run.id,
                "idempotency_key": idempotency_key,
                "decision": decision.decision,
                "dry_run": dry_run,
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
    ) -> None:
        self.automation_repository.save_run(
            AutomationRun(
                rule_id="autonomous_agent",
                trigger_key=f"{request.trigger}:{request.slot}",
                mode="dry_run" if dry_run else "approval_required",
                status=status,
                idempotency_key=idempotency_key,
                warnings=warnings,
                metadata={
                    "agent_run_id": run_id,
                    "side_effects": "none",
                },
            )
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


def _system_access(access: AccessContext) -> AccessContext:
    if access.user_id:
        return access
    return AccessContext(
        user_id="system",
        guild_id=access.guild_id,
        role_ids=access.role_ids,
        is_admin=access.is_admin,
    )


def _dict_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_autonomous_payload(payload)
    return sanitized if isinstance(sanitized, dict) else {}


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
