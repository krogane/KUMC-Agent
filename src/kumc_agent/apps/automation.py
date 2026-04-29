from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.domain.models.automation import ActionSpecRef, AutomationRule, TriggerSpec
from kumc_agent.features.automation import AutomationService
from kumc_agent.features.hardening import ProductionReadinessService
from kumc_agent.infra.automation import build_automation_repository
from kumc_agent.infra.operations import build_operations_repository


@dataclass(frozen=True)
class AutomationAppContext:
    automation: AutomationService
    readiness: ProductionReadinessService


def build_automation_app_context(
    *,
    base_dir: Path | None = None,
    seed_defaults: bool = True,
) -> AutomationAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    repository = build_automation_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "automation",
    )
    operations = build_operations_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "operations",
    )
    automation = AutomationService(
        repository=repository,
        feature_flags=foundation.feature_flags,
        audit_log=foundation.audit_log,
        operations=operations,
        action_executor=_build_action_executor(base_dir=foundation.config.base_dir),
        auto_index_cron=_auto_index_cron_from_config(foundation.config.scheduler),
        auto_index_enabled=foundation.config.scheduler.auto_index_enabled,
        auto_index_timezone=foundation.config.scheduler.auto_index_timezone,
        autonomous_agent_rules=_autonomous_agent_rules_from_config(foundation.config.autonomous_agent),
    )
    if seed_defaults:
        automation.seed_defaults()
    readiness = ProductionReadinessService(
        config=foundation.config,
        feature_flags=foundation.feature_flags,
        runbook_dir=foundation.config.base_dir / "docs" / "runbooks",
    )
    return AutomationAppContext(automation=automation, readiness=readiness)


def _auto_index_cron_from_config(scheduler) -> str:
    hour, minute = str(scheduler.auto_index_time or "00:00").split(":", 1)
    weekdays = sorted({int(value) for value in scheduler.auto_index_weekdays})
    day_names = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
    day_expr = "*" if weekdays == list(range(7)) else ",".join(day_names[value] for value in weekdays if 0 <= value <= 6)
    return f"{int(minute)} {int(hour)} * * {day_expr or '*'}"


def _autonomous_agent_rules_from_config(config) -> tuple[AutomationRule, ...]:
    rules: list[AutomationRule] = []
    for slot in config.schedule_times:
        cron = _daily_cron(str(slot))
        if not cron:
            continue
        rule_id = f"autonomous_agent_{str(slot).replace(':', '')}"
        rules.append(
            AutomationRule(
                id=rule_id,
                name=f"自律エージェント {slot}",
                enabled=bool(config.enabled),
                trigger=TriggerSpec(
                    "schedule_cron",
                    {
                        "cron": cron,
                        "timezone": config.timezone,
                        "slot": str(slot),
                    },
                ),
                actions=(
                    ActionSpecRef(
                        "autonomous_agent_run",
                        target="autonomous_agent",
                        payload={
                            "trigger": "automation",
                            "slot": str(slot),
                            "scopes": list(config.scopes),
                            "dry_run": bool(config.dry_run),
                        },
                        risk_level="low",
                        approval_required=False,
                    ),
                ),
                mode="auto_run",
                risk_level="low",
                created_by_user_id="system",
                approved_by_user_id="system",
                metadata={"source": "autonomous_agent.schedule_times"},
            )
        )
    return tuple(rules)


def _daily_cron(slot: str) -> str:
    try:
        hour, minute = slot.split(":", 1)
        return f"{int(minute)} {int(hour)} * * *"
    except (ValueError, TypeError):
        return ""


def _build_action_executor(*, base_dir: Path):
    def _execute(action: ActionSpecRef) -> dict[str, object]:
        worker_job_types = {
            "auto_index_update",
            "autonomous_agent_run",
            "task_due_reminder",
            "task_approval_batch",
            "event_reminder",
            "event_approval_batch",
            "workflow_prepare",
        }
        if action.action_type not in worker_job_types:
            return {
                "status": "executed_internal",
                "metadata": {"side_effects": "none"},
            }
        from kumc_agent.apps.worker.app import run_once

        payload = dict(action.payload)
        worker_result = run_once(
            base_dir=base_dir,
            job_type=action.action_type,
            payload=payload,
        )
        return {
            "status": "executed_internal",
            "metadata": {
                "side_effects": (
                    "indexing_snapshot_publish"
                    if action.action_type == "auto_index_update"
                    else "none"
                    if action.action_type == "autonomous_agent_run"
                    else "worker_action"
                )
            },
            "worker": worker_result,
        }

    return _execute
