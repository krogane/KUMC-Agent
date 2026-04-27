from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.agentic import build_agentic_app_context
from kumc_agent.apps.automation import build_automation_app_context
from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.ingestion import build_ingestion_app_context
from kumc_agent.apps.integrated_input import build_integrated_input_app_context
from kumc_agent.apps.retrieval import build_retrieval_app_context
from kumc_agent.apps.workflow import build_workflow_app_context
from kumc_agent.frontends.http.app import create_app as create_http_app


@dataclass(frozen=True)
class ApiAppContext:
    foundation: object
    retrieval: object
    agentic: object
    workflow: object
    automation: object
    ingestion: object
    integrated_input: object


def create_app(*, base_dir: Path | None = None):
    workflow = build_workflow_app_context(base_dir=base_dir)
    context = ApiAppContext(
        foundation=build_foundation_app_context(base_dir=base_dir),
        retrieval=build_retrieval_app_context(base_dir=base_dir),
        agentic=build_agentic_app_context(
            base_dir=base_dir,
            workflow_service=workflow.workflow,
        ),
        workflow=workflow,
        automation=build_automation_app_context(base_dir=base_dir),
        ingestion=build_ingestion_app_context(base_dir=base_dir),
        integrated_input=build_integrated_input_app_context(base_dir=base_dir),
    )
    return create_http_app(context)


def main(*, host: str = "127.0.0.1", port: int = 8000, base_dir: Path | None = None) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on deployment env
        raise RuntimeError("uvicorn is required to run the API app.") from exc

    uvicorn.run(create_app(base_dir=base_dir), host=host, port=port)
