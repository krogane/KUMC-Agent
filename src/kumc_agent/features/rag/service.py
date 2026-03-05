from __future__ import annotations

from kumc_agent.domain.models.answer import Answer
from kumc_agent.features.rag.config import RagConfig
from kumc_agent.features.rag.components.generation import GenerationComponent
from kumc_agent.features.rag.components.retrieval import RetrievalComponent
from kumc_agent.features.rag.components.routing import QueryRouter
from kumc_agent.infra.retrieval.cross_encoder import CrossEncoderReranker


class RagService:
    def __init__(
        self,
        *,
        config: RagConfig,
        router: QueryRouter,
        retrieval: RetrievalComponent,
        generation: GenerationComponent,
        reranker: CrossEncoderReranker | None,
    ) -> None:
        self._config = config
        self._router = router
        self._retrieval = retrieval
        self._generation = generation
        self._reranker = reranker

    def answer(self, query: str) -> Answer:
        decision = self._router.route(query)
        if decision.target_model == "refusal":
            return self._generation.generate_refusal(
                query=query,
                temperature=self._config.llm_temperature,
                max_output_tokens=self._config.llm_max_output_tokens,
                thinking_level=self._config.llm_thinking_level,
            )

        chunks = self._retrieval.retrieve(
            query,
            dense_top_k=self._config.dense_top_k,
            sparse_top_k=self._config.sparse_top_k,
        )

        if self._reranker is not None and chunks:
            chunks = self._reranker.rerank(
                query,
                chunks,
                top_k=min(self._config.rerank_pool_size, len(chunks)),
            )

        chunks = chunks[: self._config.top_k]
        if not chunks:
            return self._generation.generate_refusal(
                query=query,
                temperature=self._config.llm_temperature,
                max_output_tokens=self._config.llm_max_output_tokens,
                thinking_level=self._config.llm_thinking_level,
            )

        return self._generation.generate_rag_answer(
            query=query,
            chunks=chunks,
            temperature=self._config.llm_temperature,
            max_output_tokens=self._config.llm_max_output_tokens,
            thinking_level=self._config.llm_thinking_level,
        )
