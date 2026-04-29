from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.automation import build_automation_app_context
from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.integrated_input import build_integrated_input_app_context
from kumc_agent.config.schema import AutonomousAgentLLMSection
from kumc_agent.features.autonomous_agent import AutonomousAgentService
from kumc_agent.features.autonomous_agent.integrated_input import AutonomousIntegratedInputAdapter
from kumc_agent.features.autonomous_agent.llm import AutonomousLLMConfig
from kumc_agent.features.autonomous_agent.planner import AutonomousPlanner, PlannerConfig
from kumc_agent.features.autonomous_agent.service import AutonomousAgentServiceConfig
from kumc_agent.features.autonomous_agent.snapshot import (
    AutonomousSnapshotCollector,
    SnapshotCollectorConfig,
)
from kumc_agent.features.autonomous_agent.verifier import AutonomousVerifier, VerifierConfig
from kumc_agent.infra.agentic import build_agent_trace_repository
from kumc_agent.infra.ingestion import build_ingestion_repository
from kumc_agent.infra.llm.gemini import GeminiLLM
from kumc_agent.infra.llm.openai import OpenAILLM
from kumc_agent.infra.minecraft import build_server_operation_repository


@dataclass(frozen=True)
class AutonomousAgentAppContext:
    autonomous_agent: AutonomousAgentService


def build_autonomous_agent_app_context(*, base_dir: Path | None = None) -> AutonomousAgentAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    integrated = build_integrated_input_app_context(base_dir=base_dir)
    automation = build_automation_app_context(base_dir=base_dir, seed_defaults=False)
    trace_repository = build_agent_trace_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "agentic",
    )
    ingestion_repository = build_ingestion_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "ingestion",
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
        system_user_id=cfg.access.system_user_id,
        system_guild_id=cfg.access.guild_id,
        system_role_ids=tuple(cfg.access.role_ids),
        system_is_admin=cfg.access.is_admin,
    )
    snapshot = AutonomousSnapshotCollector(
        workflow_repository=integrated.workflow.workflow.repository,
        automation_repository=automation.automation.repository,
        agent_trace_repository=trace_repository,
        server_operation_repository=server_repository,
        ingestion_repository=ingestion_repository,
        config=SnapshotCollectorConfig(
            task_lookahead_days=cfg.lookahead_days.tasks,
            event_lookahead_days=cfg.lookahead_days.events,
            rag_delta_lookback_hours=cfg.rag_delta_lookback_hours,
            recent_run_limit=20,
        ),
    )
    prompts_dir = foundation.config.base_dir / "assets" / "prompts"
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
                    duplicate_suppression_hours=cfg.duplicate_suppression_hours,
                ),
                llm=_build_llm(
                    cfg.planner,
                    gemini_api_key=foundation.config.integrations.gemini_api_key,
                    gemini_default_model=foundation.config.providers.llm.gemini_model,
                    gemini_requests_per_minute=foundation.config.integrations.gemini_requests_per_minute,
                    openai_api_key=foundation.config.integrations.openai_api_key,
                    limiter_name="autonomous_agent_planner",
                ),
                llm_config=_llm_config(cfg.planner, prompts_dir=prompts_dir),
            ),
            adapter=AutonomousIntegratedInputAdapter(
                integrated_input=integrated.integrated_input,
                workflow_service=integrated.workflow.workflow,
                automation_service=automation.automation,
                retrieval_service=integrated.retrieval.ask,
                server_operation_repository=server_repository,
            ),
            verifier=AutonomousVerifier(
                config=VerifierConfig(
                    notification_channel_id=cfg.notification_channel_id,
                    require_citations_for_candidates=True,
                ),
                llm=_build_llm(
                    cfg.verifier,
                    gemini_api_key=foundation.config.integrations.gemini_api_key,
                    gemini_default_model=foundation.config.providers.llm.gemini_model,
                    gemini_requests_per_minute=foundation.config.integrations.gemini_requests_per_minute,
                    openai_api_key=foundation.config.integrations.openai_api_key,
                    limiter_name="autonomous_agent_verifier",
                ),
                llm_config=_llm_config(cfg.verifier, prompts_dir=prompts_dir),
            ),
            audit_log=foundation.audit_log,
        )
    )


def _build_llm(
    section: AutonomousAgentLLMSection,
    *,
    gemini_api_key: str,
    gemini_default_model: str,
    gemini_requests_per_minute: int,
    openai_api_key: str,
    limiter_name: str,
) -> object | None:
    if not section.enabled:
        return None
    provider = (section.provider or "gemini").strip().lower()
    if provider == "openai":
        return OpenAILLM(
            api_key=openai_api_key,
            model=section.openai_model or "gpt-5.2",
        )
    if provider == "gemini":
        return GeminiLLM(
            api_key=gemini_api_key,
            model=section.gemini_model or gemini_default_model,
            requests_per_minute=gemini_requests_per_minute,
            limiter_name=limiter_name,
        )
    return None


def _llm_config(section: AutonomousAgentLLMSection, *, prompts_dir: Path) -> AutonomousLLMConfig:
    return AutonomousLLMConfig(
        enabled=section.enabled,
        prompt_name=section.prompt_name,
        prompts_dir=prompts_dir,
        temperature=section.temperature,
        max_output_tokens=section.max_output_tokens,
        max_retries=section.max_retries,
    )
