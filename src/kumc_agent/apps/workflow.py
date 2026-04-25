from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.agentic import build_agentic_app_context
from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.retrieval import build_retrieval_app_context
from kumc_agent.features.announcement import AnnouncementDraftService
from kumc_agent.features.docgen.service import DocGenService
from kumc_agent.features.minecraft import MinecraftSupportService
from kumc_agent.features.workflow import WorkflowService
from kumc_agent.infra.announcement import build_announcement_repository
from kumc_agent.infra.minecraft import build_server_operation_repository
from kumc_agent.infra.workflow import build_workflow_repository


@dataclass(frozen=True)
class WorkflowAppContext:
    workflow: WorkflowService


def build_workflow_app_context(*, base_dir: Path | None = None) -> WorkflowAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    retrieval = build_retrieval_app_context(base_dir=base_dir)
    agentic = build_agentic_app_context(base_dir=base_dir)
    repository = build_workflow_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "workflow",
    )
    announcement_repository = build_announcement_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "announcement",
    )
    server_operation_repository = build_server_operation_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "minecraft",
    )
    docgen = DocGenService()
    return WorkflowAppContext(
        workflow=WorkflowService(
            repository=repository,
            ask_service=retrieval.ask,
            audit_log=foundation.audit_log,
            agentic_search=agentic.agentic_search,
            docgen=docgen,
            announcement=AnnouncementDraftService(
                repository=announcement_repository,
                docgen=docgen,
            ),
            minecraft=MinecraftSupportService(
                repository=server_operation_repository,
                feature_flags=foundation.feature_flags,
            ),
        )
    )
