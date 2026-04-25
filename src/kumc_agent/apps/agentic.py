from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.retrieval import build_retrieval_app_context
from kumc_agent.features.agentic import AgenticSearchService
from kumc_agent.infra.agentic import build_agent_trace_repository


@dataclass(frozen=True)
class AgenticAppContext:
    agentic_search: AgenticSearchService


def build_agentic_app_context(*, base_dir: Path | None = None) -> AgenticAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    retrieval = build_retrieval_app_context(base_dir=base_dir)
    repository = build_agent_trace_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "agentic",
    )
    return AgenticAppContext(
        agentic_search=AgenticSearchService(
            ask_service=retrieval.ask,
            repository=repository,
        )
    )
