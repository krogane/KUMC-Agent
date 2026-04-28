from __future__ import annotations

from pathlib import Path

from kumc_agent.config.load import load_runtime_config
from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.apps.retrieval import build_retrieval_app_context
from kumc_agent.infra.embeddings.gemini import GeminiEmbedder
from kumc_agent.infra.embeddings.local import LocalEmbedder
from kumc_agent.infra.llm.gemini import GeminiLLM
from kumc_agent.infra.llm.gemini_rate_limit import (
    embedding_rate_limiter_name,
    index_summary_rate_limiter_name,
)
from kumc_agent.infra.loaders.crafters_colony import CraftersColonyLoader
from kumc_agent.infra.loaders.discord import DiscordLoader
from kumc_agent.infra.loaders.google_drive import GoogleDriveLoader
from kumc_agent.infra.loaders.hatenablog import HatenaBlogLoader
from kumc_agent.infra.loaders.notion import NotionLoader
from kumc_agent.infra.loaders.x import XPostsLoader
from kumc_agent.infra.database.postgres import PostgresClient
from kumc_agent.infra.audit.repository import build_audit_repository
from kumc_agent.infra.connectors import build_source_connectors
from kumc_agent.infra.connectors.discord_members import DiscordMemberDirectoryConnector
from kumc_agent.infra.ingestion import build_ingestion_repository
from kumc_agent.infra.object_storage.raw_snapshot import RawSnapshotStore
from kumc_agent.infra.object_storage.s3 import S3ObjectStorageClient
from kumc_agent.infra.operations import build_operations_repository
from kumc_agent.infra.workflow import build_workflow_repository
from kumc_agent.infra.retrieval.cross_encoder import CrossEncoderReranker
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever
from kumc_agent.infra.storage.filesystem import FilePromptRepository, FileSystemStorage
from kumc_agent.features.image_search import (
    GeminiImageCaptioner,
    ImageAssetBuildService,
    ImageSearchConfig,
    LocalImageOcrExtractor,
)
from kumc_agent.features.member_search import MemberProfileBuildService, MemberSearchConfig
from kumc_agent.features.member_search.service import (
    AskServiceEvidenceSource,
    MemberProfileGenerator,
    MemberProfileIndexService,
)
from kumc_agent.features.indexing.service import IndexingService
from kumc_agent.features.indexing.task_event import TaskEventIndexBuildService
from kumc_agent.features.ingestion.chunking import ChunkingSettings, IngestionChunker
from kumc_agent.features.ingestion.service import IngestionService
from kumc_agent.features.rag.config import (
    RagConfig,
    RagGenerationSettings,
    RagPromptTextSettings,
)
from kumc_agent.features.rag.components.generation import GenerationComponent
from kumc_agent.features.rag.components.answer_filter import AnswerFilterComponent
from kumc_agent.features.rag.components.query_synthesis import QuerySynthesizer
from kumc_agent.features.rag.components.retrieval import RetrievalComponent
from kumc_agent.features.rag.components.routing import QueryRouter, RoutingTaskConfig
from kumc_agent.features.rag.service import RagService
from kumc_agent.features.summarization.config import SummarizationConfig
from kumc_agent.features.summarization.service import SummarizationService
from kumc_agent.features.vc.config import VCManagerConfig
from kumc_agent.features.vc.service import VCService
from kumc_agent.runtime.context import RuntimeContext
from kumc_agent.usecases.chat.answer import ChatAnswerUsecase
from kumc_agent.usecases.chat.route import ChatRouteUsecase
from kumc_agent.usecases.eval.ragas import EvaluateRagasUsecase
from kumc_agent.usecases.indexing.auto_update import AutoIndexUpdateUsecase
from kumc_agent.usecases.indexing.build import BuildIndexUsecase
from kumc_agent.usecases.indexing.update import UpdateIndexUsecase
from kumc_agent.usecases.summarization.run import SummarizationUsecase
from kumc_agent.usecases.vc.run import VCUsecase
from kumc_agent.utils.migrate_summary_dir import migrate_summery_chunk_dir
from kumc_agent.infra.secret_finding import SecretFindingDetector


def _build_llm_for_route(
    *,
    provider: str,
    gemini_model: str,
    gemini_api_key: str,
    gemini_requests_per_minute: int,
):
    normalized = (provider or "").strip().lower()
    if normalized != "gemini":
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'gemini'.")
    return GeminiLLM(
        api_key=gemini_api_key,
        model=gemini_model,
        requests_per_minute=gemini_requests_per_minute,
    )


def _build_indexing_stage_llm(
    *,
    provider: str,
    gemini_model: str,
    gemini_api_key: str,
    gemini_requests_per_minute: int,
    limiter_name: str = "",
) -> LLMPort | None:
    normalized = (provider or "").strip().lower()
    if normalized in {"", "none", "off", "disabled", "false", "0"}:
        return None
    if normalized == "gemini":
        if not gemini_api_key:
            return None
        kwargs: dict[str, object] = {
            "api_key": gemini_api_key,
            "model": gemini_model,
            "requests_per_minute": gemini_requests_per_minute,
        }
        if limiter_name:
            kwargs["limiter_name"] = limiter_name
        return GeminiLLM(**kwargs)
    raise ValueError("Unsupported indexing stage llm provider. Use 'none' or 'gemini'.")


def _build_summary_chunk_llm(config: RuntimeConfig) -> LLMPort | None:
    chunking = config.indexing.chunking
    return _build_indexing_stage_llm(
        provider=chunking.summary_llm_provider,
        gemini_model=chunking.summary_gemini_model,
        gemini_api_key=config.integrations.gemini_api_key,
        gemini_requests_per_minute=config.integrations.gemini_summary_requests_per_minute,
        limiter_name=index_summary_rate_limiter_name(),
    )


def build_runtime_context(*, base_dir: Path | None = None) -> RuntimeContext:
    config = load_runtime_config(base_dir=base_dir)
    migrate_summery_chunk_dir(base_dir=config.base_dir)

    if config.providers.embeddings.provider == "gemini":
        embedder = GeminiEmbedder(
            api_key=config.integrations.gemini_api_key,
            model_name=config.providers.embeddings.model,
            dimensions=config.providers.embeddings.dimensions,
            requests_per_minute=config.integrations.gemini_embedding_requests_per_minute,
            limiter_name=embedding_rate_limiter_name(),
        )
    else:
        embedder = LocalEmbedder(
            model_name=config.providers.embeddings.model,
            dimensions=config.providers.embeddings.dimensions,
        )

    rag_llm = _build_llm_for_route(
        provider=config.rag.generation.rag.provider,
        gemini_model=config.rag.generation.rag.gemini_model,
        gemini_api_key=config.integrations.gemini_api_key,
        gemini_requests_per_minute=config.integrations.gemini_requests_per_minute,
    )
    no_rag_llm = _build_llm_for_route(
        provider=config.rag.generation.no_rag.provider,
        gemini_model=config.rag.generation.no_rag.gemini_model,
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
    operations_repository = build_operations_repository(
        postgres=PostgresClient(config.infrastructure.database),
        fallback_dir=config.base_dir / "data" / "operations",
    )
    audit_log = build_audit_repository(
        postgres=PostgresClient(config.infrastructure.database),
        fallback_path=config.base_dir / "logs" / "audit.jsonl",
    )
    ingestion_repository = build_ingestion_repository(
        postgres=PostgresClient(config.infrastructure.database),
        fallback_dir=config.base_dir / "data" / "ingestion",
    )
    workflow_repository = build_workflow_repository(
        postgres=PostgresClient(config.infrastructure.database),
        fallback_dir=config.base_dir / "data" / "workflow",
    )

    retrieval_component = RetrievalComponent(
        embedder=embedder,
        dense_index=dense_index,
        sparse_index=sparse_index,
        sparse_sudachi_mode=config.features.retrieval.sudachi_mode,
        sparse_use_normalized_form=config.features.retrieval.sparse_use_normalized_form,
        sparse_remove_symbols=config.features.retrieval.sparse_remove_symbols,
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
        ),
    )
    query_synthesizer = QuerySynthesizer(
        llm=rag_llm,
        prompts=prompt_repo,
        temperature=config.rag.routing.temperature,
        max_output_tokens=min(512, config.rag.generation.rag.max_output_tokens),
        max_retries=config.rag.routing.max_retries,
    )
    answer_filter = AnswerFilterComponent(
        llm=rag_llm,
        prompts=prompt_repo,
        temperature=0.0,
        max_output_tokens=min(512, config.rag.generation.rag.max_output_tokens),
        max_retries=config.rag.routing.max_retries,
    )

    router = QueryRouter(
        routing_enabled=config.rag.routing.enabled,
        provider=config.rag.routing.provider,
        gemini_model=config.rag.routing.gemini_model,
        prompt_name=config.rag.routing.prompt_name,
        temperature=config.rag.routing.temperature,
        max_new_tokens=config.rag.routing.max_new_tokens,
        max_retries=config.rag.routing.max_retries,
        log_enabled=config.rag.routing.log_enabled,
        material_search_max_names=config.rag.routing.material_search_max_names,
        gemini_api_key=config.integrations.gemini_api_key,
        gemini_requests_per_minute=config.integrations.gemini_requests_per_minute,
        task_configs={
            "target_model": RoutingTaskConfig(
                provider=config.rag.routing.tasks.target_model.provider,
                gemini_model=config.rag.routing.tasks.target_model.gemini_model,
                prompt_name=config.rag.routing.tasks.target_model.prompt_name,
            ),
            "use_additional_memory": RoutingTaskConfig(
                provider=config.rag.routing.tasks.use_additional_memory.provider,
                gemini_model=config.rag.routing.tasks.use_additional_memory.gemini_model,
                prompt_name=config.rag.routing.tasks.use_additional_memory.prompt_name,
            ),
            "include_capabilities_info": RoutingTaskConfig(
                provider=config.rag.routing.tasks.include_capabilities_info.provider,
                gemini_model=(
                    config.rag.routing.tasks.include_capabilities_info.gemini_model
                ),
                prompt_name=(
                    config.rag.routing.tasks.include_capabilities_info.prompt_name
                ),
            ),
            "needs_additional_query": RoutingTaskConfig(
                provider=config.rag.routing.tasks.needs_additional_query.provider,
                gemini_model=(
                    config.rag.routing.tasks.needs_additional_query.gemini_model
                ),
                prompt_name=(
                    config.rag.routing.tasks.needs_additional_query.prompt_name
                ),
            ),
            "additional_queries": RoutingTaskConfig(
                provider=config.rag.routing.tasks.additional_queries.provider,
                gemini_model=config.rag.routing.tasks.additional_queries.gemini_model,
                prompt_name=config.rag.routing.tasks.additional_queries.prompt_name,
            ),
            "material_names": RoutingTaskConfig(
                provider=config.rag.routing.tasks.material_names.provider,
                gemini_model=config.rag.routing.tasks.material_names.gemini_model,
                prompt_name=config.rag.routing.tasks.material_names.prompt_name,
            ),
            "recency_mode": RoutingTaskConfig(
                provider=config.rag.routing.tasks.recency_mode.provider,
                gemini_model=config.rag.routing.tasks.recency_mode.gemini_model,
                prompt_name=config.rag.routing.tasks.recency_mode.prompt_name,
            ),
        },
    )

    rag_service = RagService(
        config=RagConfig(
            top_k=config.features.retrieval.top_k,
            dense_top_k=config.features.retrieval.dense_top_k,
            sparse_top_k=config.features.retrieval.sparse_top_k,
            sparse_initial_sparse_top_k=(
                config.features.retrieval.sparse_initial_sparse_top_k
            ),
            rerank_pool_size=config.features.retrieval.rerank_pool_size,
            rrf_k=config.features.retrieval.rrf_k,
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
                prompt_name=config.rag.generation.rag.prompt_name,
            ),
            no_rag_generation=RagGenerationSettings(
                provider=config.rag.generation.no_rag.provider,
                temperature=config.rag.generation.no_rag.temperature,
                max_output_tokens=config.rag.generation.no_rag.max_output_tokens,
                prompt_name=config.rag.generation.no_rag.prompt_name,
            ),
            material_search_max_names=config.rag.routing.material_search_max_names,
            parent_doc_enabled=config.features.retrieval.parent_doc_enabled,
            parent_chunk_cap=config.features.retrieval.parent_chunk_cap,
            sparse_normalized_ratio=(
                config.features.retrieval.sparse_normalized_ratio
            ),
            material_full_text_char_limit=(
                config.features.retrieval.material_full_text_char_limit
            ),
            answer_json_max_retries=config.rag.answer_json_max_retries,
            history_enabled=config.rag.history.enabled,
            history_max_turns=config.rag.history.max_turns,
            prompt_default_turns=config.rag.history.prompt_default_turns,
            prompt_additional_turns=config.rag.history.prompt_additional_turns,
            fast_model_notice=config.rag.fast_model_notice,
            allowed_guild_ids=tuple(
                str(value) for value in config.security.discord_guild_allow_list
            ),
            admin_user_ids=tuple(
                str(value) for value in config.security.maintenance_command_author_ids
            ),
            minecraft_wiki_top_k=config.minecraft_wiki_rag.retrieval.top_k,
            minecraft_wiki_dense_top_k=config.minecraft_wiki_rag.retrieval.dense_top_k,
            minecraft_wiki_sparse_top_k=config.minecraft_wiki_rag.retrieval.sparse_top_k,
            minecraft_wiki_sparse_initial_sparse_top_k=(
                config.minecraft_wiki_rag.retrieval.sparse_initial_sparse_top_k
            ),
            minecraft_wiki_sparse_normalized_ratio=(
                config.minecraft_wiki_rag.retrieval.sparse_normalized_ratio
            ),
            minecraft_wiki_rerank_pool_size=(
                config.minecraft_wiki_rag.retrieval.rerank_pool_size
            ),
            minecraft_wiki_rrf_k=config.minecraft_wiki_rag.retrieval.rrf_k,
            minecraft_wiki_mmr_lambda=config.minecraft_wiki_rag.retrieval.mmr_lambda,
            minecraft_wiki_parent_doc_enabled=(
                config.minecraft_wiki_rag.retrieval.parent_doc_enabled
            ),
            minecraft_wiki_parent_chunk_cap=(
                config.minecraft_wiki_rag.retrieval.parent_chunk_cap
            ),
        ),
        router=router,
        retrieval=retrieval_component,
        generation=generation_component,
        reranker=reranker,
        query_synthesizer=query_synthesizer,
        answer_filter=answer_filter,
    )

    image_search_config = ImageSearchConfig(
        limit=config.features.image_search.limit,
        dense_top_k=config.features.image_search.dense_top_k,
        feature_top_k=config.features.image_search.feature_top_k,
        rrf_k=config.features.image_search.rrf_k,
        ocr_text_char_limit=config.features.image_search.ocr_text_char_limit,
        surrounding_text_char_limit=config.features.image_search.surrounding_text_char_limit,
        ocr_model=config.features.image_search.ocr_model or config.integrations.drive.pdf_ocr_model_path,
        caption_model=config.features.image_search.caption_model or config.providers.llm.gemini_model,
    )
    image_asset_builder = ImageAssetBuildService(
        repository=operations_repository,
        raw_dir=config.app.raw_dir,
        image_dir=config.base_dir / "data" / "image_search" / "images",
        index_dir=config.app.index_dir / "image_search",
        embedder=embedder,
        config=image_search_config,
        captioner=GeminiImageCaptioner(
            api_key=config.integrations.gemini_api_key,
            model=image_search_config.caption_model,
            prompt_path=config.base_dir / "assets" / "prompts" / "image_caption.md",
        ),
        ocr=LocalImageOcrExtractor(
            model_path=image_search_config.ocr_model
        ),
    )

    indexing_service = IndexingService(
        storage=storage,
        embedder=embedder,
        faiss_index=dense_index,
        bm25_index=sparse_index,
        raw_dir=config.app.raw_dir,
        app_config=config,
        summary_llm=summary_chunk_llm,
        image_asset_builder=image_asset_builder,
        ingestion_repository=ingestion_repository,
    )

    drive_loader = GoogleDriveLoader(
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
    discord_loader = DiscordLoader(
        bot_token=config.integrations.discord.bot_token,
        raw_dir=config.app.raw_dir,
        allow_guild_ids=config.security.discord_guild_allow_list,
    )
    hatena_loader = HatenaBlogLoader(
        raw_dir=config.app.raw_dir,
        blog_url=config.integrations.hatenablog.blog_url,
    )
    crafters_loader = CraftersColonyLoader(
        raw_dir=config.app.raw_dir,
        author_url=config.integrations.crafters_colony.author_url,
        max_pages=config.integrations.crafters_colony.max_pages,
        max_articles=config.integrations.crafters_colony.max_articles,
    )
    x_loader = XPostsLoader(raw_dir=config.app.raw_dir)
    notion_loader = (
        NotionLoader(
            api_token=config.integrations.notion.api_token,
            database_ids=config.integrations.notion.database_ids,
            raw_dir=config.app.raw_dir,
        )
        if config.features.sources.notion
        else None
    )

    build_index_usecase = BuildIndexUsecase(
        indexing_service=indexing_service,
        drive_loader=drive_loader,
        discord_loader=discord_loader,
        hatenablog_loader=hatena_loader,
        crafters_colony_loader=crafters_loader,
        x_loader=x_loader,
        notion_loader=notion_loader,
    )
    ingestion_service = IngestionService(
        connectors=build_source_connectors(config),
        repository=ingestion_repository,
        raw_snapshots=RawSnapshotStore(
            config=config.infrastructure.object_storage,
            local_root=config.base_dir / "data" / "object_storage",
            s3=S3ObjectStorageClient(config.infrastructure.object_storage),
        ),
        chunker=IngestionChunker(
            ChunkingSettings(
                max_characters=config.indexing.chunking.second_recursive_chunk_size * 4,
                overlap_characters=config.indexing.chunking.second_recursive_chunk_overlap * 4,
            )
        ),
        secret_detector=SecretFindingDetector(),
        audit_log=audit_log,
    )
    member_profile_guild_ids = tuple(
        str(value) for value in config.security.effective_member_profile_guild_ids()
    )
    member_search_allowed_guild_ids = tuple(
        dict.fromkeys(
            str(value)
            for value in (
                config.security.discord_guild_allow_list
                + config.security.effective_member_profile_guild_ids()
            )
        )
    )
    member_search_config = MemberSearchConfig(
        allowed_guild_ids=member_search_allowed_guild_ids,
        admin_user_ids=tuple(str(value) for value in config.security.maintenance_command_author_ids),
        search_limit=config.features.retrieval.top_k,
        rrf_k=config.features.retrieval.rrf_k,
        sparse_bm25_k1=config.features.retrieval.sparse_bm25_k1,
        sparse_bm25_b=config.features.retrieval.sparse_bm25_b,
        sudachi_mode=config.features.retrieval.sudachi_mode,
        sparse_use_normalized_form=config.features.retrieval.sparse_use_normalized_form,
        sparse_remove_symbols=config.features.retrieval.sparse_remove_symbols,
    )
    retrieval_app = build_retrieval_app_context(base_dir=config.base_dir)
    member_profile_builder = MemberProfileBuildService(
        repository=operations_repository,
        directory=DiscordMemberDirectoryConnector(
            bot_token=config.integrations.discord.bot_token,
            allowed_guild_ids=member_profile_guild_ids,
        ),
        evidence_source=AskServiceEvidenceSource(
            ask_service=retrieval_app.ask,
            max_evidence=member_search_config.max_evidence,
        ),
        generator=MemberProfileGenerator(
            llm=rag_llm,
            prompts_dir=config.base_dir / "assets" / "prompts",
            prompt_name=member_search_config.profile_prompt_name,
            temperature=0.0,
        ),
        config=member_search_config,
        indexer=MemberProfileIndexService(
            index_dir=config.app.index_dir,
            embedder=embedder,
            config=member_search_config,
        ),
    )

    chat_answer_usecase = ChatAnswerUsecase(rag_service=rag_service)
    from kumc_agent.apps.integrated_input import build_integrated_input_app_context

    integrated_input = build_integrated_input_app_context(
        base_dir=config.base_dir,
        chat_answer_service=chat_answer_usecase,
    ).integrated_input
    chat_route_usecase = ChatRouteUsecase(router=router)
    vc_usecase = VCUsecase(service=VCService(config=VCManagerConfig.from_runtime(config)))

    return RuntimeContext(
        config=config,
        integrated_input=integrated_input,
        chat_answer=chat_answer_usecase,
        chat_route=chat_route_usecase,
        build_index=build_index_usecase,
        update_index=UpdateIndexUsecase(
            build_usecase=build_index_usecase,
            indexing_service=indexing_service,
        ),
        auto_index_update=AutoIndexUpdateUsecase(
            config=config,
            build_usecase=build_index_usecase,
            operations=operations_repository,
            ingestion_service=ingestion_service,
            member_profile_builder=member_profile_builder,
            member_profile_guild_ids=member_profile_guild_ids,
            task_event_indexer=TaskEventIndexBuildService(
                repository=workflow_repository,
                embedder=embedder,
            ),
        ),
        eval_ragas=EvaluateRagasUsecase(
            chat_usecase=chat_answer_usecase,
            gemini_api_key=config.integrations.gemini_api_key,
            ragas_gemini_model=config.providers.llm.gemini_model,
            ragas_gemini_requests_per_minute=config.integrations.gemini_ragas_requests_per_minute,
            ragas_gemini_embedding_requests_per_minute=(
                config.integrations.gemini_ragas_embedding_requests_per_minute
            ),
            default_answer_generation_batch_size=(
                config.ops.ragas_answer_generation_batch_size
            ),
            default_ragas_batch_size=config.ops.ragas_batch_size,
            default_ragas_max_workers=config.ops.ragas_max_workers,
            default_ragas_timeout_seconds=config.ops.ragas_timeout_seconds,
            default_ragas_max_retries=config.ops.ragas_max_retries,
            default_answer_cache_enabled=config.ops.ragas_answer_cache_enabled,
            default_answer_cache_path=config.ops.ragas_answer_cache_path,
            default_disable_history_for_eval=config.ops.ragas_disable_history_for_eval,
            eval_answer_relevancy_enabled=config.ops.ragas_metrics.answer_relevancy_enabled,
            eval_faithfulness_enabled=config.ops.ragas_metrics.faithfulness_enabled,
            eval_context_precision_enabled=config.ops.ragas_metrics.context_precision_enabled,
            eval_context_recall_enabled=config.ops.ragas_metrics.context_recall_enabled,
        ),
        summarize=SummarizationUsecase(
            service=SummarizationService(
                config=SummarizationConfig(
                    target_characters=config.summarization.target_characters,
                )
            )
        ),
        vc=vc_usecase,
    )
