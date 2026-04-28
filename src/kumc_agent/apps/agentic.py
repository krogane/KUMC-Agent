from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.retrieval import build_retrieval_app_context
from kumc_agent.domain.models.agentic import AgentBudget
from kumc_agent.features.agentic import ComprehensiveAgentService
from kumc_agent.features.agentic.comprehensive import ComprehensiveLLMConfig
from kumc_agent.infra.agentic import build_agent_trace_repository
from kumc_agent.infra.llm.gemini import GeminiLLM


@dataclass(frozen=True)
class ComprehensiveAgentAppContext:
    comprehensive_agent: ComprehensiveAgentService
    trace_repository: object


def build_comprehensive_agent_app_context(
    *,
    base_dir: Path | None = None,
    workflow_service: object | None = None,
) -> ComprehensiveAgentAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    retrieval = build_retrieval_app_context(base_dir=base_dir)
    repository = build_agent_trace_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "agentic",
    )
    cfg = foundation.config.comprehensive_agent
    planner_llm = _build_llm(
        enabled=cfg.enabled and cfg.planner.enabled,
        provider=cfg.planner.provider,
        model=cfg.planner.gemini_model,
        api_key=foundation.config.integrations.gemini_api_key,
        requests_per_minute=foundation.config.integrations.gemini_requests_per_minute,
    )
    verifier_llm = _build_llm(
        enabled=cfg.enabled and cfg.verifier.enabled,
        provider=cfg.verifier.provider,
        model=cfg.verifier.gemini_model,
        api_key=foundation.config.integrations.gemini_api_key,
        requests_per_minute=foundation.config.integrations.gemini_requests_per_minute,
    )
    return ComprehensiveAgentAppContext(
        comprehensive_agent=ComprehensiveAgentService(
            ask_service=retrieval.ask,
            repository=repository,
            workflow_service=workflow_service,
            planner_llm=planner_llm,
            verifier_llm=verifier_llm,
            planner_config=ComprehensiveLLMConfig(
                enabled=bool(planner_llm is not None),
                prompt_name=cfg.planner.prompt_name,
                prompts_dir=foundation.config.base_dir / "assets" / "prompts",
                temperature=cfg.planner.temperature,
                max_output_tokens=cfg.planner.max_output_tokens,
                max_retries=cfg.planner.max_retries,
            ),
            verifier_config=ComprehensiveLLMConfig(
                enabled=bool(verifier_llm is not None),
                prompt_name=cfg.verifier.prompt_name,
                prompts_dir=foundation.config.base_dir / "assets" / "prompts",
                temperature=cfg.verifier.temperature,
                max_output_tokens=cfg.verifier.max_output_tokens,
                max_retries=cfg.verifier.max_retries,
            ),
            default_budget=AgentBudget(
                max_steps=cfg.budget.max_steps,
                max_search_calls=cfg.budget.max_search_calls,
                max_read_chunks=cfg.budget.max_read_chunks,
                max_replans=cfg.budget.max_replans,
                max_cost_usd=cfg.budget.max_cost_usd,
                max_latency_seconds=cfg.budget.max_latency_seconds,
                require_citations=cfg.budget.require_citations,
            ),
        ),
        trace_repository=repository,
    )


def _build_llm(
    *,
    enabled: bool,
    provider: str,
    model: str,
    api_key: str,
    requests_per_minute: int,
) -> object | None:
    if not enabled:
        return None
    if provider.lower() != "gemini":
        return None
    if not api_key or not model:
        return None
    return GeminiLLM(
        api_key=api_key,
        model=model,
        requests_per_minute=requests_per_minute,
    )


def build_agentic_app_context(
    *,
    base_dir: Path | None = None,
    workflow_service: object | None = None,
) -> ComprehensiveAgentAppContext:
    return build_comprehensive_agent_app_context(
        base_dir=base_dir,
        workflow_service=workflow_service,
    )
