from __future__ import annotations

from kumc_agent.usecases.eval.adapters.base import AdapterRunResult, EvalAdapter
from kumc_agent.usecases.eval.adapters.contract import (
    AgenticEvalAdapter,
    ContractEvalAdapter,
    IntegratedInputEvalAdapter,
    SearchEvalAdapter,
    ServerEvalAdapter,
    WorkflowEvalAdapter,
)
from kumc_agent.usecases.eval.adapters.ragas_adapter import RagasEvalAdapter
from kumc_agent.usecases.eval.ragas import EvaluateRagasUsecase


def build_default_adapter_registry(
    *,
    ragas_usecase: EvaluateRagasUsecase | None = None,
) -> dict[str, EvalAdapter]:
    registry: dict[str, EvalAdapter] = {
        "rag_circle": RagasEvalAdapter(target="rag_circle", ragas_usecase=ragas_usecase),
        "rag_minecraft": RagasEvalAdapter(target="rag_minecraft", ragas_usecase=ragas_usecase),
    }
    for target in ("member_search", "image_search"):
        registry[target] = SearchEvalAdapter(target=target)
    for target in ("task_management", "event_management", "message_posting", "automation"):
        registry[target] = WorkflowEvalAdapter(target=target)
    registry["server_management"] = ServerEvalAdapter(target="server_management")
    registry["integrated_input"] = IntegratedInputEvalAdapter(target="integrated_input")
    registry["agentic"] = AgenticEvalAdapter(target="agentic")
    registry["autonomous_agent"] = AgenticEvalAdapter(target="autonomous_agent")
    return registry


__all__ = [
    "AdapterRunResult",
    "EvalAdapter",
    "RagasEvalAdapter",
    "ContractEvalAdapter",
    "WorkflowEvalAdapter",
    "SearchEvalAdapter",
    "ServerEvalAdapter",
    "IntegratedInputEvalAdapter",
    "AgenticEvalAdapter",
    "build_default_adapter_registry",
]
