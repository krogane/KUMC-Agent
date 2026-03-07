from __future__ import annotations

from pathlib import Path

from kumc_agent.config.load import load_runtime_config
from kumc_agent.infra.embeddings.gemini import GeminiEmbedder
from kumc_agent.infra.embeddings.local import LocalEmbedder
from kumc_agent.infra.llm.gemini import GeminiLLM
from kumc_agent.infra.llm.llama_cpp import LlamaCppLLM
from kumc_agent.infra.loaders.crafters_colony import CraftersColonyLoader
from kumc_agent.infra.loaders.discord import DiscordLoader
from kumc_agent.infra.loaders.google_drive import GoogleDriveLoader
from kumc_agent.infra.loaders.hatenablog import HatenaBlogLoader
from kumc_agent.infra.retrieval.cross_encoder import CrossEncoderReranker
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever
from kumc_agent.infra.storage.filesystem import FilePromptRepository, FileSystemStorage
from kumc_agent.features.indexing.service import IndexingService
from kumc_agent.features.rag.config import RagConfig
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
from kumc_agent.utils.migrate_summary_dir import migrate_summery_chunk_dir


def build_runtime_context(*, base_dir: Path | None = None) -> RuntimeContext:
    config = load_runtime_config(base_dir=base_dir)
    migrate_summery_chunk_dir(base_dir=config.base_dir)

    if config.providers.embeddings.provider == "gemini":
        embedder = GeminiEmbedder(
            api_key=config.integrations.gemini_api_key,
            model_name=config.providers.embeddings.model,
            dimensions=config.providers.embeddings.dimensions,
        )
    else:
        embedder = LocalEmbedder(
            model_name=config.providers.embeddings.model,
            dimensions=config.providers.embeddings.dimensions,
        )

    if config.providers.llm.provider == "llama":
        llm = LlamaCppLLM(model_path=config.providers.llm.llama_model_path)
    else:
        llm = GeminiLLM(
            api_key=config.integrations.gemini_api_key,
            model=config.providers.llm.gemini_model,
        )

    storage = FileSystemStorage(
        chunks_path=config.app.chunks_path,
        raw_dir=config.app.raw_dir,
    )
    dense_index = FaissLikeIndex(index_dir=config.app.index_dir)
    sparse_index = SudachiBM25Retriever(index_dir=config.app.index_dir)

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
        llm=llm,
        prompts=prompt_repo,
        source_max_count=config.app.source_max_count,
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
    )

    rag_service = RagService(
        config=RagConfig(
            top_k=config.features.retrieval.top_k,
            dense_top_k=config.features.retrieval.dense_top_k,
            sparse_top_k=config.features.retrieval.sparse_top_k,
            rerank_pool_size=config.features.retrieval.rerank_pool_size,
            mmr_lambda=config.features.retrieval.mmr_lambda,
            source_max_count=config.app.source_max_count,
            recency_mode=config.features.recency_mode,
            llm_temperature=config.providers.llm.temperature,
            llm_max_output_tokens=config.providers.llm.max_output_tokens,
            llm_thinking_level=config.providers.llm.thinking_level,
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
    )

    drive_loader = (
        GoogleDriveLoader(
            folder_id=config.integrations.drive.folder_id,
            credentials_path=config.integrations.drive.google_application_credentials,
            raw_dir=config.app.raw_dir,
            max_files=config.integrations.drive.max_files,
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

    build_index_usecase = BuildIndexUsecase(
        indexing_service=indexing_service,
        drive_loader=drive_loader,
        discord_loader=discord_loader,
        hatenablog_loader=hatena_loader,
        crafters_colony_loader=crafters_loader,
    )

    chat_answer_usecase = ChatAnswerUsecase(rag_service=rag_service)
    chat_route_usecase = ChatRouteUsecase(router=router)
    vc_usecase = VCUsecase(service=VCService(config=VCManagerConfig.from_runtime(config)))

    return RuntimeContext(
        config=config,
        chat_answer=chat_answer_usecase,
        chat_route=chat_route_usecase,
        build_index=build_index_usecase,
        update_index=UpdateIndexUsecase(
            build_usecase=build_index_usecase,
            indexing_service=indexing_service,
        ),
        eval_ragas=EvaluateRagasUsecase(chat_usecase=chat_answer_usecase),
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
