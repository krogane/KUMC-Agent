from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.domain.models.retrieval import AskResponse, RetrievalQuery
from kumc_agent.features.retrieval.answering import ExtractiveAnswerBuilder
from kumc_agent.features.retrieval.context import ContextPacker
from kumc_agent.features.retrieval.hybrid import HybridRetrievalService


@dataclass(frozen=True)
class AskService:
    retrieval: HybridRetrievalService
    packer: ContextPacker
    answer_builder: ExtractiveAnswerBuilder

    def ask(self, query: RetrievalQuery) -> AskResponse:
        scored = self.retrieval.retrieve(query)
        context = self.packer.pack(scored)
        return self.answer_builder.build(query=query, context=context)
