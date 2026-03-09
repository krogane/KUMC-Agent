from __future__ import annotations

from pathlib import Path

from kumc_agent.config.load import load_runtime_config
from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.infra.embeddings.gemini import GeminiEmbedder
from kumc_agent.infra.embeddings.local import LocalEmbedder
from kumc_agent.infra.llm.gemini import GeminiLLM
from kumc_agent.infra.llm.gemini_rate_limit import index_summary_rate_limiter_name
from kumc_agent.infra.llm.llama_cpp import LlamaCppLLM
from kumc_agent.infra.loaders.crafters_colony import CraftersColonyLoader
from kumc_agent.infra.loaders.discord import DiscordLoader
from kumc_agent.infra.loaders.google_drive import GoogleDriveLoader
from kumc_agent.infra.loaders.hatenablog import HatenaBlogLoader
from kumc_agent.infra.loaders.x import XPostsLoader
from kumc_agent.infra.retrieval.cross_encoder import CrossEncoderReranker
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever
from kumc_agent.infra.storage.filesystem import FilePromptRepository, FileSystemStorage
from kumc_agent.features.indexing.service import IndexingService
from kumc_agent.features.rag.config import (
    RagConfig,
    RagGenerationSettings,
    RagIdeaGenerationSettings,
    RagPromptTextSettings,
)
from kumc_agent.features.rag.components.generation import GenerationComponent
from kumc_agent.features.rag.components.retrieval import RetrievalComponent
from kumc_agent.features.rag.components.routing import QueryRouter
from kumc_agent.features.rag.service import RagService
from kumc_agent.features.summarization.config import SummarizationConfig
from kumc_agent.features.summarization.service import SummarizationService
from kumc_agent.features.vc.config import VCManagerConfig
from kumc_agent.features.vc.service import VCService
from kumc_agent.runtime.context import RuntimeContext
from kumc_agent.usecases.chat.answer import ChatAnswerUsecase
from kumc_agent.usecases.chat.route import ChatRouteUsecase
from kumc_agent.usecases.eval.ragas import EvaluateRagasUsecase
from kumc_agent.usecases.indexing.build import BuildIndexUsecase
from kumc_agent.usecases.indexing.update import UpdateIndexUsecase
from kumc_agent.usecases.summarization.run import SummarizationUsecase
from kumc_agent.usecases.vc.run import VCUsecase
from kumc_agent.usecases.warmup.run import WarmupUsecase
from kumc_agent.utils.migrate_summary_dir import migrate_summery_chunk_dir


def _build_llm_for_route(
    *,
    provider: str,
    gemini_model: str,
    llama_model_path: str,
    gemini_api_key: str,
    gemini_requests_per_minute: int,
):
    normalized = (provider or "").strip().lower()
    if normalized in {"llama", "llama_cpp"}:
        return LlamaCppLLM(model_path=llama_model_path)
    return GeminiLLM(
        api_key=gemini_api_key,
        model=gemini_model,
        requests_per_minute=gemini_requests_per_minute,
    )


def _build_summary_chunk_llm(config: RuntimeConfig) -> LLMPort | None:
    chunking = config.indexing.chunking
    provider = (chunking.summary_llm_provider or "").strip().lower()
    if provider in {"", "none", "off", "disabled", "false", "0"}:
        return None
    if provider in {"llama", "llama_cpp"}:
        if not chunking.summary_llama_model_path:
            return None
        return LlamaCppLLM(model_path=chunking.summary_llama_model_path)
    if provider == "gemini":
        if not config.integrations.gemini_api_key:
            return None
        return GeminiLLM(
            api_key=config.integrations.gemini_api_key,
            model=chunking.summary_gemini_model,
            requests_per_minute=config.integrations.gemini_summary_requests_per_minute,
            limiter_name=index_summary_rate_limiter_name(),
        )
    raise ValueError(
        "Unsupported indexing.chunking.summary_llm_provider. Use 'none', 'gemini', or 'llama'."
    )


def build_runtime_context(*, base_dir: Path | None = None) -> RuntimeContext:
    config = load_runtime_config(base_dir=base_dir)
    migrate_summery_chunk_dir(base_dir=config.base_dir)

    if config.providers.embeddings.provider == "gemini":
        embedder = GeminiEmbedder(
            api_key=config.integrations.gemini_api_key,
            model_name=config.providers.embeddings.model,
            dimensions=config.providers.embeddings.dimensions,
            requests_per_minute=config.integrations.gemini_requests_per_minute,
        )
    else:
        embedder = LocalEmbedder(
            model_name=config.providers.embeddings.model,
            dimensions=config.providers.embeddings.dimensions,
        )

    rag_llm = _build_llm_for_route(
        provider=config.rag.generation.rag.provider,
        gemini_model=config.rag.generation.rag.gemini_model,
        llama_model_path=config.rag.generation.rag.llama_model_path,
        gemini_api_key=config.integrations.gemini_api_key,
        gemini_requests_per_minute=config.integrations.gemini_requests_per_minute,
    )
    no_rag_llm = _build_llm_for_route(
        provider=config.rag.generation.no_rag.provider,
        gemini_model=config.rag.generation.no_rag.gemini_model,
        llama_model_path=config.rag.generation.no_rag.llama_model_path,
        gemini_api_key=config.integrations.gemini_api_key,
        gemini_requests_per_minute=config.integrations.gemini_requests_per_minute,
    )
    refusal_llm = _build_llm_for_route(
        provider=config.rag.generation.refusal.provider,
        gemini_model=config.rag.generation.refusal.gemini_model,
        llama_model_path=config.rag.generation.refusal.llama_model_path,
        gemini_api_key=config.integrations.gemini_api_key,
        gemini_requests_per_minute=config.integrations.gemini_requests_per_minute,
    )
    summary_chunk_llm = _build_summary_chunk_llm(config)

    storage = FileSystemStorage(
        chunks_path=config.app.chunks_path,
        raw_dir=config.app.raw_dir,
    )
    dense_index = FaissLikeIndex(index_dir=config.app.index_dir)
    sparse_index = SudachiBM25Retriever(
        index_dir=config.app.index_dir,
        sudachi_mode=config.features.retrieval.sudachi_mode,
        bm25_k1=config.features.retrieval.sparse_bm25_k1,
        bm25_b=config.features.retrieval.sparse_bm25_b,
        use_normalized_form=config.features.retrieval.sparse_use_normalized_form,
        remove_symbols=config.features.retrieval.sparse_remove_symbols,
    )

    retrieval_component = RetrievalComponent(
        embedder=embedder,
        dense_index=dense_index,
        sparse_index=sparse_index,
    )

    reranker = (
        CrossEncoderReranker(model_name=config.providers.reranker.model)
        if config.providers.reranker.enabled
        else None
    )

    prompt_repo = FilePromptRepository(config.base_dir / "assets" / "prompts")
    generation_component = GenerationComponent(
        llm=rag_llm,
        no_rag_llm=no_rag_llm,
        refusal_llm=refusal_llm,
        prompts=prompt_repo,
        source_max_count=config.app.source_max_count,
        raw_dir=config.app.raw_dir,
        prompt_texts=RagPromptTextSettings(
            empty_context=config.rag.prompt_texts.empty_context,
            empty_history=config.rag.prompt_texts.empty_history,
            history_user_prefix=config.rag.prompt_texts.history_user_prefix,
            history_assistant_prefix=config.rag.prompt_texts.history_assistant_prefix,
            history_sources_label=config.rag.prompt_texts.history_sources_label,
            gemini_header_chat_history=config.rag.prompt_texts.gemini_header_chat_history,
            gemini_header_retry_history=config.rag.prompt_texts.gemini_header_retry_history,
            gemini_header_circle_info=config.rag.prompt_texts.gemini_header_circle_info,
            gemini_header_capabilities=config.rag.prompt_texts.gemini_header_capabilities,
            gemini_header_context=config.rag.prompt_texts.gemini_header_context,
            gemini_header_output_format=config.rag.prompt_texts.gemini_header_output_format,
            gemini_header_instructions=config.rag.prompt_texts.gemini_header_instructions,
            gemini_header_question=config.rag.prompt_texts.gemini_header_question,
            llama_header_question=config.rag.prompt_texts.llama_header_question,
            llama_header_previous_attempt=config.rag.prompt_texts.llama_header_previous_attempt,
            llama_header_circle_info=config.rag.prompt_texts.llama_header_circle_info,
            llama_header_capabilities=config.rag.prompt_texts.llama_header_capabilities,
            llama_header_context=config.rag.prompt_texts.llama_header_context,
            llama_header_output_format=config.rag.prompt_texts.llama_header_output_format,
            llama_header_instructions=config.rag.prompt_texts.llama_header_instructions,
        ),
    )

    router = QueryRouter(
        refusal_keywords=config.security.refusal_keywords,
        routing_enabled=config.rag.routing.enabled,
        provider=config.rag.routing.provider,
        gemini_model=config.rag.routing.gemini_model,
        llama_model_path=config.rag.routing.llama_model_path,
        temperature=config.rag.routing.temperature,
        max_new_tokens=config.rag.routing.max_new_tokens,
        max_retries=config.rag.routing.max_retries,
        log_enabled=config.rag.routing.log_enabled,
        material_search_max_names=config.rag.routing.material_search_max_names,
        llm_thinking_level=config.providers.llm.thinking_level,
        llm_threads=config.providers.llm.threads,
        llm_gpu_layers=config.providers.llm.gpu_layers,
        llm_ctx_size=4096,
        gemini_api_key=config.integrations.gemini_api_key,
        gemini_requests_per_minute=config.integrations.gemini_requests_per_minute,
    )

    rag_service = RagService(
        config=RagConfig(
            top_k=config.features.retrieval.top_k,
            dense_top_k=config.features.retrieval.dense_top_k,
            sparse_top_k=config.features.retrieval.sparse_top_k,
            rerank_pool_size=config.features.retrieval.rerank_pool_size,
            mmr_lambda=config.features.retrieval.mmr_lambda,
            recency_weight_soft=config.features.retrieval.recency_weight_soft,
            recency_weight_hard=config.features.retrieval.recency_weight_hard,
            recency_half_life_days=config.features.retrieval.recency_half_life_days,
            source_max_count=config.app.source_max_count,
            recency_mode=config.features.recency_mode,
            rag_generation=RagGenerationSettings(
                provider=config.rag.generation.rag.provider,
                temperature=config.rag.generation.rag.temperature,
                max_output_tokens=config.rag.generation.rag.max_output_tokens,
                thinking_level=config.rag.generation.rag.thinking_level,
                prompt_name=config.rag.generation.rag.prompt_name,
            ),
            no_rag_generation=RagGenerationSettings(
                provider=config.rag.generation.no_rag.provider,
                temperature=config.rag.generation.no_rag.temperature,
                max_output_tokens=config.rag.generation.no_rag.max_output_tokens,
                thinking_level=config.rag.generation.no_rag.thinking_level,
                prompt_name=config.rag.generation.no_rag.prompt_name,
            ),
            refusal_generation=RagGenerationSettings(
                provider=config.rag.generation.refusal.provider,
                temperature=config.rag.generation.refusal.temperature,
                max_output_tokens=config.rag.generation.refusal.max_output_tokens,
                thinking_level=config.rag.generation.refusal.thinking_level,
                prompt_name=config.rag.generation.refusal.prompt_name,
            ),
            idea_generation=RagIdeaGenerationSettings(
                prompt_name=config.rag.generation.idea_generation.prompt_name,
                temperature=config.rag.generation.idea_generation.temperature,
            ),
            parent_doc_enabled=config.features.retrieval.parent_doc_enabled,
            parent_chunk_cap=config.features.retrieval.parent_chunk_cap,
            answer_json_max_retries=config.rag.answer_json_max_retries,
            history_enabled=config.rag.history.enabled,
            history_max_turns=config.rag.history.max_turns,
            prompt_default_turns=config.rag.history.prompt_default_turns,
            prompt_additional_turns=config.rag.history.prompt_additional_turns,
            fast_model_notice=config.rag.fast_model_notice,
        ),
        router=router,
        retrieval=retrieval_component,
        generation=generation_component,
        reranker=reranker,
    )

    indexing_service = IndexingService(
        storage=storage,
        embedder=embedder,
        faiss_index=dense_index,
        bm25_index=sparse_index,
        raw_dir=config.app.raw_dir,
        app_config=config,
        summary_llm=summary_chunk_llm,
    )

    drive_loader = (
        GoogleDriveLoader(
            folder_id=config.integrations.drive.folder_id,
            credentials_path=config.integrations.drive.google_application_credentials,
            raw_dir=config.app.raw_dir,
            max_files=config.integrations.drive.max_files,
            batch_size=config.integrations.drive.batch_size,
            download_max_retries=config.integrations.drive.download_max_retries,
            download_retry_initial_delay_seconds=(
                config.integrations.drive.download_retry_initial_delay_seconds
            ),
            download_retry_max_delay_seconds=(
                config.integrations.drive.download_retry_max_delay_seconds
            ),
            download_retry_backoff_multiplier=(
                config.integrations.drive.download_retry_backoff_multiplier
            ),
            pdf_ocr_model_path=config.integrations.drive.pdf_ocr_model_path,
        )
        if config.features.sources.drive
        else None
    )
    discord_loader = (
        DiscordLoader(
            bot_token=config.integrations.discord.bot_token,
            raw_dir=config.app.raw_dir,
            allow_guild_ids=config.security.discord_guild_allow_list,
        )
        if config.features.sources.discord
        else None
    )
    hatena_loader = (
        HatenaBlogLoader(raw_dir=config.app.raw_dir)
        if config.features.sources.hatenablog
        else None
    )
    crafters_loader = (
        CraftersColonyLoader(
            raw_dir=config.app.raw_dir,
            author_url=config.integrations.crafters_colony.author_url,
            max_pages=config.integrations.crafters_colony.max_pages,
            max_articles=config.integrations.crafters_colony.max_articles,
        )
        if config.features.sources.crafters_colony
        else None
    )
    x_loader = (
        XPostsLoader(raw_dir=config.app.raw_dir)
        if config.features.sources.x
        else None
    )

    build_index_usecase = BuildIndexUsecase(
        indexing_service=indexing_service,
        drive_loader=drive_loader,
        discord_loader=discord_loader,
        hatenablog_loader=hatena_loader,
        crafters_colony_loader=crafters_loader,
        x_loader=x_loader,
    )

    chat_answer_usecase = ChatAnswerUsecase(rag_service=rag_service)
    chat_route_usecase = ChatRouteUsecase(router=router)
    warmup_usecase = WarmupUsecase(
        config=config,
        embedder=embedder,
        reranker=reranker,
        route_usecase=chat_route_usecase,
        rag_llm=rag_llm,
        no_rag_llm=no_rag_llm,
        refusal_llm=refusal_llm,
    )
    vc_usecase = VCUsecase(service=VCService(config=VCManagerConfig.from_runtime(config)))

    return RuntimeContext(
        config=config,
        chat_answer=chat_answer_usecase,
        chat_route=chat_route_usecase,
        warmup=warmup_usecase,
        build_index=build_index_usecase,
        update_index=UpdateIndexUsecase(
            build_usecase=build_index_usecase,
            indexing_service=indexing_service,
        ),
        eval_ragas=EvaluateRagasUsecase(
            chat_usecase=chat_answer_usecase,
            gemini_api_key=config.integrations.gemini_api_key,
            ragas_gemini_model=config.providers.llm.gemini_model,
            ragas_gemini_requests_per_minute=config.integrations.gemini_ragas_requests_per_minute,
            default_ragas_batch_size=config.ops.ragas_batch_size,
            eval_answer_relevancy_enabled=config.ops.ragas_metrics.answer_relevancy_enabled,
            eval_faithfulness_enabled=config.ops.ragas_metrics.faithfulness_enabled,
            eval_context_precision_enabled=config.ops.ragas_metrics.context_precision_enabled,
            eval_context_recall_enabled=config.ops.ragas_metrics.context_recall_enabled,
        ),
        summarize=SummarizationUsecase(
            service=SummarizationService(
                config=SummarizationConfig(
                    target_characters=int(
                        config.experiments.get("summarization", {}).get(
                            "target_characters",
                            200,
                        )
                    )
                )
            )
        ),
        vc=vc_usecase,
    )
