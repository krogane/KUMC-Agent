from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.usecases.chat.answer import ChatAnswerUsecase
from kumc_agent.usecases.chat.route import ChatRouteUsecase
from kumc_agent.usecases.eval.ragas import EvaluateRagasUsecase
from kumc_agent.usecases.indexing.build import BuildIndexUsecase
from kumc_agent.usecases.indexing.update import UpdateIndexUsecase
from kumc_agent.usecases.summarization.run import SummarizationUsecase
from kumc_agent.usecases.vc.run import VCUsecase
from kumc_agent.usecases.warmup.run import WarmupUsecase


@dataclass(frozen=True)
class RuntimeContext:
    config: RuntimeConfig
    chat_answer: ChatAnswerUsecase
    chat_route: ChatRouteUsecase
    warmup: WarmupUsecase
    build_index: BuildIndexUsecase
    update_index: UpdateIndexUsecase
    eval_ragas: EvaluateRagasUsecase
    summarize: SummarizationUsecase
    vc: VCUsecase
