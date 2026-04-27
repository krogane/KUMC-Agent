from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.retrieval import build_retrieval_app_context
from kumc_agent.features.agentic import ComprehensiveAgentService
from kumc_agent.infra.agentic import build_agent_trace_repository


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
    return ComprehensiveAgentAppContext(
        comprehensive_agent=ComprehensiveAgentService(
            ask_service=retrieval.ask,
            repository=repository,
            workflow_service=workflow_service,
        ),
        trace_repository=repository,
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
