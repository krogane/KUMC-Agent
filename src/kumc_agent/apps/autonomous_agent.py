from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.automation import build_automation_app_context
from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.workflow import build_workflow_app_context
from kumc_agent.features.autonomous_agent import AutonomousAgentService
from kumc_agent.features.autonomous_agent.integrated_input import AutonomousIntegratedInputAdapter
from kumc_agent.features.autonomous_agent.planner import AutonomousPlanner, PlannerConfig
from kumc_agent.features.autonomous_agent.service import AutonomousAgentServiceConfig
from kumc_agent.features.autonomous_agent.snapshot import (
    AutonomousSnapshotCollector,
    SnapshotCollectorConfig,
)
from kumc_agent.features.autonomous_agent.verifier import AutonomousVerifier, VerifierConfig
from kumc_agent.infra.agentic import build_agent_trace_repository
from kumc_agent.infra.minecraft import build_server_operation_repository


@dataclass(frozen=True)
class AutonomousAgentAppContext:
    autonomous_agent: AutonomousAgentService


def build_autonomous_agent_app_context(*, base_dir: Path | None = None) -> AutonomousAgentAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    workflow = build_workflow_app_context(base_dir=base_dir)
    automation = build_automation_app_context(base_dir=base_dir, seed_defaults=False)
    trace_repository = build_agent_trace_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "agentic",
    )
    server_repository = build_server_operation_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "minecraft",
    )
    cfg = foundation.config.autonomous_agent
    service_config = AutonomousAgentServiceConfig(
        enabled=cfg.enabled,
        schedule_times=tuple(cfg.schedule_times),
        timezone=cfg.timezone,
        scopes=tuple(cfg.scopes),
        notification_channel_id=cfg.notification_channel_id,
        dry_run=cfg.dry_run,
        lookahead_days={
            "tasks": cfg.lookahead_days.tasks,
            "events": cfg.lookahead_days.events,
        },
        duplicate_suppression_hours=cfg.duplicate_suppression_hours,
        budget=cfg.budget,
    )
    snapshot = AutonomousSnapshotCollector(
        workflow_repository=workflow.workflow.repository,
        automation_repository=automation.automation.repository,
        agent_trace_repository=trace_repository,
        server_operation_repository=server_repository,
        config=SnapshotCollectorConfig(
            task_lookahead_days=cfg.lookahead_days.tasks,
            event_lookahead_days=cfg.lookahead_days.events,
            recent_run_limit=20,
        ),
    )
    return AutonomousAgentAppContext(
        autonomous_agent=AutonomousAgentService(
            config=service_config,
            trace_repository=trace_repository,
            automation_repository=automation.automation.repository,
            snapshot_collector=snapshot,
            planner=AutonomousPlanner(
                config=PlannerConfig(
                    notification_channel_id=cfg.notification_channel_id,
                    max_replans=cfg.budget.max_replans,
                )
            ),
            adapter=AutonomousIntegratedInputAdapter(
                integrated_input=None,
                workflow_service=workflow.workflow,
            ),
            verifier=AutonomousVerifier(
                config=VerifierConfig(
                    notification_channel_id=cfg.notification_channel_id,
                    require_citations_for_candidates=True,
                )
            ),
            audit_log=foundation.audit_log,
        )
    )
