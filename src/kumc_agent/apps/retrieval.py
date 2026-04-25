from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.features.retrieval.answering import ExtractiveAnswerBuilder
from kumc_agent.features.retrieval.ask import AskService
from kumc_agent.features.retrieval.citation import CitationValidator
from kumc_agent.features.retrieval.context import ContextPacker, ContextPackingConfig
from kumc_agent.features.retrieval.hybrid import HybridRetrievalConfig, HybridRetrievalService
from kumc_agent.infra.embeddings.gemini import GeminiEmbedder
from kumc_agent.infra.embeddings.local import LocalEmbedder
from kumc_agent.infra.retrieval.cross_encoder import CrossEncoderReranker
from kumc_agent.infra.retrieval_wave3 import build_retrieval_repository


@dataclass(frozen=True)
class RetrievalAppContext:
    ask: AskService


def build_retrieval_app_context(*, base_dir: Path | None = None) -> RetrievalAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    repository = build_retrieval_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "ingestion",
    )
    dimensions = foundation.config.providers.embeddings.dimensions
    if (
        foundation.config.providers.embeddings.provider == "gemini"
        and foundation.config.integrations.gemini_api_key
    ):
        embedder = GeminiEmbedder(
            api_key=foundation.config.integrations.gemini_api_key,
            model_name=foundation.config.providers.embeddings.model,
            dimensions=dimensions,
            requests_per_minute=foundation.config.integrations.gemini_embedding_requests_per_minute,
        )
    else:
        local_model = (
            foundation.config.providers.embeddings.model
            if foundation.config.providers.embeddings.provider != "gemini"
            else ""
        )
        embedder = LocalEmbedder(model_name=local_model, dimensions=dimensions)
    reranker = (
        CrossEncoderReranker(model_name=foundation.config.providers.reranker.model)
        if foundation.config.providers.reranker.enabled
        else None
    )
    hybrid = HybridRetrievalService(
        repository=repository,
        embedder=embedder,
        reranker=reranker,
        config=HybridRetrievalConfig(
            dense_top_k=foundation.config.features.retrieval.dense_top_k,
            sparse_top_k=foundation.config.features.retrieval.sparse_top_k,
            rerank_pool_size=foundation.config.features.retrieval.rerank_pool_size,
            top_k=foundation.config.features.retrieval.top_k,
            doc_cap=foundation.config.features.retrieval.parent_chunk_cap,
            mmr_lambda=foundation.config.features.retrieval.mmr_lambda,
            embedding_model=foundation.config.providers.embeddings.model,
            embedding_dimensions=dimensions,
        ),
    )
    ask = AskService(
        retrieval=hybrid,
        packer=ContextPacker(
            ContextPackingConfig(
                max_context_characters=8000,
                max_citations=foundation.config.app.source_max_count,
            )
        ),
        answer_builder=ExtractiveAnswerBuilder(citation_validator=CitationValidator()),
    )
    return RetrievalAppContext(ask=ask)
