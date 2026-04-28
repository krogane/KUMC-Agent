from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.agentic import build_agentic_app_context
from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.retrieval import build_retrieval_app_context
from kumc_agent.apps.workflow import build_workflow_app_context
from kumc_agent.features.rag.components.integrated_input_routing import (
    IntegratedInputRouter,
    IntegratedRoutingPolicy,
)
from kumc_agent.usecases.integrated_input import IntegratedInputUsecase


@dataclass(frozen=True)
class IntegratedInputAppContext:
    integrated_input: IntegratedInputUsecase
    retrieval: object
    workflow: object
    agentic: object


def build_integrated_input_app_context(
    *,
    base_dir: Path | None = None,
    chat_answer_service: object | None = None,
) -> IntegratedInputAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    retrieval = build_retrieval_app_context(base_dir=base_dir)
    workflow = build_workflow_app_context(base_dir=base_dir)
    agentic = build_agentic_app_context(
        base_dir=base_dir,
        workflow_service=workflow.workflow,
    )
    router = IntegratedInputRouter(
        provider=foundation.config.rag.routing.provider,
        gemini_model=foundation.config.rag.routing.gemini_model,
        temperature=foundation.config.rag.routing.temperature,
        max_new_tokens=foundation.config.rag.routing.max_new_tokens,
        max_retries=foundation.config.rag.routing.max_retries,
        gemini_api_key=foundation.config.integrations.gemini_api_key,
        gemini_requests_per_minute=foundation.config.integrations.gemini_requests_per_minute,
        prompts_dir=foundation.config.base_dir / "assets" / "prompts",
        prompt_name="integrated_input_routing",
        log_enabled=foundation.config.rag.routing.log_enabled,
    )
    return IntegratedInputAppContext(
        integrated_input=IntegratedInputUsecase(
            ask_service=retrieval.ask,
            workflow_service=workflow.workflow,
            comprehensive_agent=agentic.comprehensive_agent,
            router=router,
            chat_answer_service=chat_answer_service,
            routing_policy=IntegratedRoutingPolicy(),
        ),
        retrieval=retrieval,
        workflow=workflow,
        agentic=agentic,
    )
