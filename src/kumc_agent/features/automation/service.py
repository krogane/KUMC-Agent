from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Any, Callable

from kumc_agent.domain.models.audit import AuditEvent
from kumc_agent.domain.models.automation import (
    ActionSpecRef,
    AutomationResponse,
    AutomationRule,
    AutomationRun,
    ConditionSpec,
    TriggerSpec,
)
from kumc_agent.domain.models.operations import ActionRun
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.features.foundation.feature_flags import FeatureFlagService
from kumc_agent.features.foundation.tracing import current_trace_id
from kumc_agent.infra.audit.repository import AuditLogRepository
from kumc_agent.infra.automation import AutomationRepository
from kumc_agent.infra.operations import OperationsRepository
from kumc_agent.utils.hashing import stable_hash

_VALID_MODES = {"dry_run", "approval_required", "auto_run"}
_HIGH_RISKS = {"high", "critical"}
_AUTO_RUN_ALLOWLIST = {
    "auto_index_update",
    "ingest_backfill",
    "task_due_reminder",
    "weekly_summary_draft",
}
_FORCED_APPROVAL_ACTIONS = {
    "external_post",
    "role_change",
    "server_operation",
    "accounting_finalize",
    "auto_reply",
}


class AutomationService:
    def __init__(
        self,
        *,
        repository: AutomationRepository,
        feature_flags: FeatureFlagService,
        audit_log: AuditLogRepository | None = None,
        operations: OperationsRepository | None = None,
        action_executor: Callable[[ActionSpecRef], dict[str, object]] | None = None,
        auto_index_cron: str = "0 6 * * MON-FRI",
        auto_index_enabled: bool = True,
    ) -> None:
        self.repository = repository
        self.feature_flags = feature_flags
        self.audit_log = audit_log
        self.operations = operations
        self.action_executor = action_executor
        self.auto_index_cron = auto_index_cron
        self.auto_index_enabled = auto_index_enabled

    def seed_defaults(self) -> tuple[AutomationRule, ...]:
        if self.repository.list_rules():
            return tuple(self.repository.list_rules())
        stored = [
            self.repository.save_rule(rule)
            for rule in _default_rules(
                auto_index_cron=self.auto_index_cron,
                auto_index_enabled=self.auto_index_enabled,
            )
        ]
        self._audit(
            "automation.seed_defaults",
            AccessContext(user_id="system", is_admin=True),
            "succeeded",
            "automation_rules",
            metadata={"count": len(stored)},
        )
        return tuple(stored)

    def list_rules(self) -> AutomationResponse:
        rules = tuple(self.repository.list_rules() or self.seed_defaults())
        return AutomationResponse(
            text=f"Automation rule は {len(rules)} 件です。",
            detail_markdown=_format_rules(rules),
            rules=rules,
        )

    def show(self, *, rule_id: str) -> AutomationResponse:
        rule = self._require_rule(rule_id)
        runs = tuple(self.repository.list_runs(rule_id=rule.id)[:5])
        return AutomationResponse(
            text=f"Automation rule: {rule.name}",
            detail_markdown="\n\n".join((_format_rules((rule,)), _format_runs(runs))),
            rules=(rule,),
            runs=runs,
        )

    def enable(self, *, rule_id: str, access: AccessContext) -> AutomationResponse:
        self._require_operator(access)
        rule = self.repository.save_rule(replace(self._require_rule(rule_id), enabled=True))
        self._audit("automation.enable", access, "succeeded", rule.id, rule.risk_level)
        return AutomationResponse(
            text=f"Automation rule を enable にしました: {rule.name}",
            detail_markdown=_format_rules((rule,)),
            rules=(rule,),
        )

    def disable(self, *, rule_id: str, access: AccessContext) -> AutomationResponse:
        self._require_operator(access)
        rule = self.repository.save_rule(replace(self._require_rule(rule_id), enabled=False))
        self._audit("automation.disable", access, "succeeded", rule.id, rule.risk_level)
        return AutomationResponse(
            text=f"Automation rule を disable にしました: {rule.name}",
            detail_markdown=_format_rules((rule,)),
            rules=(rule,),
        )

    def set_mode(
        self,
        *,
        rule_id: str,
        mode: str,
        access: AccessContext,
    ) -> AutomationResponse:
        self._require_operator(access)
        normalized = _normalize_mode(mode)
        rule = self._require_rule(rule_id)
        warnings = self._mode_warnings(rule, normalized)
        if normalized == "auto_run" and not access.is_admin:
            raise PermissionError("auto_run への変更には admin 権限が必要です。")
        if any(warning.startswith("blocked:") for warning in warnings):
            raise ValueError("; ".join(warnings))
        next_rule = replace(
            rule,
            mode=normalized,
            approved_by_user_id=access.user_id if normalized == "auto_run" else rule.approved_by_user_id,
        )
        stored = self.repository.save_rule(next_rule)
        self._audit(
            "automation.set_mode",
            access,
            "succeeded",
            stored.id,
            stored.risk_level,
            metadata={"mode": normalized, "warnings": warnings},
        )
        return AutomationResponse(
            text=f"Automation rule mode を {normalized} にしました: {stored.name}",
            detail_markdown=_format_rules((stored,)),
            rules=(stored,),
            warnings=tuple(warnings),
        )

    def dry_run(
        self,
        *,
        rule_id: str,
        trigger_key: str = "manual",
        idempotency_key: str = "",
        access: AccessContext = AccessContext(),
    ) -> AutomationResponse:
        return self._run(
            rule_id=rule_id,
            trigger_key=trigger_key,
            idempotency_key=idempotency_key,
            access=access,
            force_dry_run=True,
        )

    def run(
        self,
        *,
        rule_id: str,
        trigger_key: str = "manual",
        idempotency_key: str = "",
        access: AccessContext = AccessContext(),
    ) -> AutomationResponse:
        return self._run(
            rule_id=rule_id,
            trigger_key=trigger_key,
            idempotency_key=idempotency_key,
            access=access,
            force_dry_run=False,
        )

    def _run(
        self,
        *,
        rule_id: str,
        trigger_key: str,
        idempotency_key: str,
        access: AccessContext,
        force_dry_run: bool,
    ) -> AutomationResponse:
        self._require_operator(access)
        rule = self._require_rule(rule_id)
        resolved_key = (
            idempotency_key.strip()
            or stable_hash(f"automation:{rule.id}:{trigger_key}:{force_dry_run}")[:32]
        )
        existing = self.repository.get_run_by_idempotency_key(resolved_key)
        if existing is not None:
            return AutomationResponse(
                text=f"同じ idempotency_key の automation run は記録済みです: {existing.id}",
                detail_markdown=_format_runs((existing,)),
                rules=(rule,),
                runs=(existing,),
                warnings=("duplicate_idempotency_key",),
                metadata={"duplicate": True},
            )

        mode = "dry_run" if force_dry_run else rule.mode
        warnings = list(self._mode_warnings(rule, mode))
        status = self._resolve_run_status(rule, mode, warnings)
        action_plan = tuple(self._plan_action(action, status=status) for action in rule.actions)
        executor_results = self._execute_allowed_actions(
            rule=rule,
            actions=rule.actions,
            status=status,
            access=access,
            idempotency_key=resolved_key,
        )
        run = self.repository.save_run(
            AutomationRun(
                rule_id=rule.id,
                trigger_key=trigger_key,
                mode=mode,
                status=status,
                idempotency_key=resolved_key,
                action_plan=action_plan,
                warnings=tuple(warnings),
                metadata={
                    "enabled": rule.enabled,
                    "requested_by": access.user_id,
                    "side_effects": "none",
                    "executor_results": executor_results,
                },
            )
        )
        self.repository.save_rule(replace(rule, last_run_at=run.created_at))
        self._audit(
            f"automation.{mode}",
            access,
            status,
            rule.id,
            rule.risk_level,
            metadata={"run_id": run.id, "idempotency_key": resolved_key},
        )
        return AutomationResponse(
            text=_run_text(rule, run),
            detail_markdown="\n\n".join((_format_rules((rule,)), _format_runs((run,)))),
            rules=(rule,),
            runs=(run,),
            warnings=tuple(warnings),
        )

    def _resolve_run_status(
        self,
        rule: AutomationRule,
        mode: str,
        warnings: list[str],
    ) -> str:
        if not rule.enabled:
            warnings.append("rule_disabled")
            return "skipped"
        if mode == "dry_run":
            return "dry_run"
        if mode == "approval_required":
            return "waiting_approval"
        if self.feature_flags.mode_for("automation_auto_run") != "enabled":
            warnings.append("automation_auto_run_disabled")
            return "blocked"
        if any(warning.startswith("blocked:") for warning in warnings):
            return "blocked"
        return "executed_equivalent"

    def _mode_warnings(self, rule: AutomationRule, mode: str) -> tuple[str, ...]:
        normalized = _normalize_mode(mode)
        warnings: list[str] = []
        if normalized != "auto_run":
            return tuple(warnings)
        if rule.risk_level.lower() in _HIGH_RISKS:
            warnings.append("blocked:auto_run_high_or_critical_risk")
        for action in rule.actions:
            if action.risk_level.lower() in _HIGH_RISKS:
                warnings.append(f"blocked:auto_run_action_risk:{action.action_type}")
            if action.action_type in _FORCED_APPROVAL_ACTIONS:
                warnings.append(f"blocked:auto_run_forced_approval:{action.action_type}")
            if action.action_type not in _AUTO_RUN_ALLOWLIST:
                warnings.append(f"blocked:auto_run_not_allowlisted:{action.action_type}")
        return tuple(warnings)

    @staticmethod
    def _plan_action(action: ActionSpecRef, *, status: str) -> dict[str, Any]:
        return {
            "action_type": action.action_type,
            "target": action.target,
            "payload": dict(action.payload),
            "risk_level": action.risk_level,
            "approval_required": action.approval_required,
            "execution": status,
            "side_effects": "none",
        }

    def _execute_allowed_actions(
        self,
        *,
        rule: AutomationRule,
        actions: tuple[ActionSpecRef, ...],
        status: str,
        access: AccessContext,
        idempotency_key: str,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for index, action in enumerate(actions):
            allowed = (
                status == "executed_equivalent"
                and action.action_type in _AUTO_RUN_ALLOWLIST
                and action.risk_level.lower() not in _HIGH_RISKS
                and not action.approval_required
            )
            executor_payload: dict[str, object] = {}
            action_status = "executed_internal" if allowed else status
            side_effects = "none"
            if allowed and self.action_executor is not None:
                executor_payload = self.action_executor(action)
                action_status = str(executor_payload.get("status") or action_status)
                side_effects = str(executor_payload.get("side_effects") or "internal")
            result = {
                "action_type": action.action_type,
                "target": action.target,
                "status": action_status,
                "side_effects": side_effects,
            }
            if executor_payload:
                result["executor_payload"] = executor_payload
            results.append(result)
            if self.operations is not None:
                self.operations.save_action_run(
                    ActionRun(
                        id=stable_hash(f"action-run:{idempotency_key}:{index}")[:32],
                        action_type=action.action_type,
                        target=action.target,
                        actor_user_id=access.user_id,
                        status=action_status,
                        risk_level=action.risk_level,
                        idempotency_key=f"{idempotency_key}:{index}",
                        request_payload={
                            "rule_id": rule.id,
                            "payload": dict(action.payload),
                            "automation_status": status,
                        },
                        result_payload=result,
                        trace_id=current_trace_id(),
                        metadata={"external_side_effects": "none"},
                    )
                )
        return results

    def _require_rule(self, rule_id: str) -> AutomationRule:
        if not self.repository.list_rules():
            self.seed_defaults()
        rule = self.repository.get_rule(rule_id)
        if rule is None:
            raise KeyError(rule_id)
        return rule

    @staticmethod
    def _require_operator(access: AccessContext) -> None:
        roles = {role.lower() for role in access.role_ids}
        if access.is_admin or "admin" in roles or "organizer" in roles:
            return
        raise PermissionError("/automation は organizer / admin のみ実行できます。")

    def _audit(
        self,
        action: str,
        access: AccessContext,
        outcome: str,
        target: str,
        risk_level: str = "low",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.audit_log is None:
            return
        self.audit_log.append(
            AuditEvent(
                action=action,
                actor_id=access.user_id or "system",
                actor_type="discord_user" if access.user_id else "system",
                target=target,
                outcome=outcome,
                risk_level=risk_level,
                trace_id=current_trace_id(),
                metadata=dict(metadata or {}),
            )
        )


def _default_rules(
    *,
    auto_index_cron: str = "0 6 * * MON-FRI",
    auto_index_enabled: bool = True,
) -> tuple[AutomationRule, ...]:
    return (
        AutomationRule(
            id="auto_index_daily",
            name="自動インデックス日次更新",
            enabled=auto_index_enabled,
            trigger=TriggerSpec("schedule_cron", {"cron": auto_index_cron}),
            actions=(
                ActionSpecRef(
                    "auto_index_update",
                    target="index",
                    payload={"trigger": "automation"},
                    risk_level="low",
                ),
            ),
            mode="auto_run",
            risk_level="low",
            created_by_user_id="system",
            approved_by_user_id="system",
        ),
        AutomationRule(
            id="weekly_summary",
            name="週次まとめ draft",
            enabled=True,
            trigger=TriggerSpec("schedule_cron", {"cron": "0 9 * * MON"}),
            actions=(
                ActionSpecRef(
                    "weekly_summary_draft",
                    target="announcement",
                    payload={"format": "markdown"},
                    risk_level="low",
                ),
            ),
            mode="dry_run",
            risk_level="low",
            created_by_user_id="system",
        ),
        AutomationRule(
            id="drive_delta_sync",
            name="Drive 差分取り込み",
            enabled=True,
            trigger=TriggerSpec("drive_changed", {"source": "drive"}),
            actions=(
                ActionSpecRef(
                    "ingest_backfill",
                    target="drive",
                    payload={"source": "drive", "force": False},
                    risk_level="low",
                ),
            ),
            mode="auto_run",
            risk_level="low",
            created_by_user_id="system",
            approved_by_user_id="system",
        ),
        AutomationRule(
            id="notion_delta_sync",
            name="Notion 差分取り込み",
            enabled=False,
            trigger=TriggerSpec("notion_changed", {"source": "notion"}),
            actions=(
                ActionSpecRef(
                    "ingest_backfill",
                    target="notion",
                    payload={"source": "notion", "force": False},
                    risk_level="low",
                ),
            ),
            mode="auto_run",
            risk_level="low",
            created_by_user_id="system",
            approved_by_user_id="system",
        ),
        AutomationRule(
            id="task_due_reminder",
            name="Task 期限前 reminder",
            enabled=True,
            trigger=TriggerSpec("task_due_soon", {"hours": 24}),
            conditions=(ConditionSpec("task.status", "in", ["todo", "doing"]),),
            actions=(
                ActionSpecRef(
                    "task_due_reminder",
                    target="discord",
                    payload={"channel": "ops"},
                    risk_level="low",
                ),
            ),
            mode="auto_run",
            risk_level="low",
            created_by_user_id="system",
            approved_by_user_id="system",
        ),
        AutomationRule(
            id="task_approval_batch",
            name="Task まとめ承認",
            enabled=True,
            trigger=TriggerSpec("schedule_cron", {"cron": "0 9 */7 * *"}),
            actions=(
                ActionSpecRef(
                    "task_approval_batch",
                    target="discord",
                    payload={},
                    risk_level="low",
                ),
            ),
            mode="auto_run",
            risk_level="low",
            created_by_user_id="system",
            approved_by_user_id="system",
        ),
        AutomationRule(
            id="meeting_prep",
            name="Meeting prep 候補作成",
            enabled=True,
            trigger=TriggerSpec("schedule_cron", {"cron": "0 8 * * *"}),
            actions=(
                ActionSpecRef(
                    "meeting_prepare",
                    target="workflow",
                    risk_level="medium",
                    approval_required=True,
                ),
            ),
            mode="approval_required",
            risk_level="medium",
            created_by_user_id="system",
        ),
        AutomationRule(
            id="auto_reply_candidate",
            name="Discord 自動返信候補",
            enabled=False,
            trigger=TriggerSpec("discord_message_matched", {"pattern": "faq"}),
            actions=(
                ActionSpecRef(
                    "auto_reply",
                    target="discord",
                    risk_level="high",
                    approval_required=True,
                ),
            ),
            mode="approval_required",
            risk_level="high",
            created_by_user_id="system",
        ),
    )


def _normalize_mode(mode: str) -> str:
    normalized = (mode or "").strip().lower()
    if normalized not in _VALID_MODES:
        raise ValueError("mode must be one of: dry_run, approval_required, auto_run")
    return normalized


def _format_rules(rules: tuple[AutomationRule, ...]) -> str:
    if not rules:
        return "# Automation Rules\n- なし"
    lines = ["# Automation Rules"]
    for rule in rules:
        actions = ", ".join(action.action_type for action in rule.actions) or "none"
        lines.append(
            f"- `{rule.id}` {rule.name} / enabled={rule.enabled} / "
            f"mode={rule.mode} / risk={rule.risk_level} / trigger={rule.trigger.kind} / "
            f"actions={actions}"
        )
    return "\n".join(lines)


def _format_runs(runs: tuple[AutomationRun, ...]) -> str:
    if not runs:
        return "# Automation Runs\n- なし"
    lines = ["# Automation Runs"]
    for run in runs:
        created = run.created_at.isoformat() if run.created_at else "unknown"
        lines.append(
            f"- `{run.id}` rule={run.rule_id} / status={run.status} / "
            f"mode={run.mode} / key={run.idempotency_key} / created={created}"
        )
    return "\n".join(lines)


def _run_text(rule: AutomationRule, run: AutomationRun) -> str:
    if run.status == "executed_equivalent":
        return f"Automation run を記録しました（副作用なしの executed-equivalent）: {rule.name}"
    if run.status == "waiting_approval":
        return f"Automation run を承認待ちとして記録しました: {rule.name}"
    if run.status == "blocked":
        return f"Automation run は safety gate で blocked になりました: {rule.name}"
    if run.status == "skipped":
        return f"Automation run は skipped です: {rule.name}"
    return f"Automation dry-run を記録しました: {rule.name}"
