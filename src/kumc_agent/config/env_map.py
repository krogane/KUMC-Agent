from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EnvBinding:
    env_name: str
    path: str
    parser: Callable[[str], object]


def _to_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(raw: str) -> int:
    return int(raw.strip())


def _to_float(raw: str) -> float:
    return float(raw.strip())


def _to_int_list(raw: str) -> list[int]:
    value = raw.strip()
    if not value:
        return []
    if value in {"*", "all", "every"}:
        return [0, 1, 2, 3, 4, 5, 6]
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            out.append(int(token))
            continue
        weekday = {
            "mon": 0,
            "tue": 1,
            "wed": 2,
            "thu": 3,
            "fri": 4,
            "sat": 5,
            "sun": 6,
        }.get(token[:3].lower())
        if weekday is None:
            raise ValueError(f"Invalid weekday token: {token}")
        out.append(weekday)
    return out


def _to_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _to_int_csv(raw: str) -> list[int]:
    values = _to_str_list(raw)
    return [int(value) for value in values]


ENV_BINDINGS: tuple[EnvBinding, ...] = (
    EnvBinding("KUMC_LOG_LEVEL", "app.log_level", str),
    EnvBinding("KUMC_COMMAND_PREFIX", "app.command_prefix", str),
    EnvBinding("KUMC_INDEX_COMMAND_PREFIX", "app.index_command_prefix", str),
    EnvBinding("KUMC_MAX_INPUT_CHARACTERS", "app.max_input_characters", _to_int),
    EnvBinding("KUMC_DISCORD_BOT_TOKEN", "integrations.discord.bot_token", str),
    EnvBinding("KUMC_OPENCLAW_ENABLED", "integrations.openclaw.enabled", _to_bool),
    EnvBinding("KUMC_OPENCLAW_AGENT", "integrations.openclaw.agent", str),
    EnvBinding("KUMC_OPENCLAW_MODEL", "integrations.openclaw.model", str),
    EnvBinding("KUMC_OPENCLAW_LITE_AGENT", "integrations.openclaw.lite_agent", str),
    EnvBinding("KUMC_OPENCLAW_LITE_MODEL", "integrations.openclaw.lite_model", str),
    EnvBinding("KUMC_OPENAI_API_KEY", "integrations.openai_api_key", str),
    EnvBinding("KUMC_GEMINI_API_KEY", "integrations.gemini_api_key", str),
    EnvBinding(
        "KUMC_GEMINI_REQUESTS_PER_MINUTE",
        "integrations.gemini_requests_per_minute",
        _to_int,
    ),
    EnvBinding(
        "KUMC_GEMINI_EMBEDDING_REQUESTS_PER_MINUTE",
        "integrations.gemini_embedding_requests_per_minute",
        _to_int,
    ),
    EnvBinding(
        "KUMC_GEMINI_SUMMARY_REQUESTS_PER_MINUTE",
        "integrations.gemini_summary_requests_per_minute",
        _to_int,
    ),
    EnvBinding(
        "KUMC_GEMINI_RAGAS_REQUESTS_PER_MINUTE",
        "integrations.gemini_ragas_requests_per_minute",
        _to_int,
    ),
    EnvBinding(
        "KUMC_GEMINI_RAGAS_EMBEDDING_REQUESTS_PER_MINUTE",
        "integrations.gemini_ragas_embedding_requests_per_minute",
        _to_int,
    ),
    EnvBinding("KUMC_DRIVE_FOLDER_ID", "integrations.drive.folder_id", str),
    EnvBinding(
        "KUMC_GOOGLE_APPLICATION_CREDENTIALS",
        "integrations.drive.google_application_credentials",
        str,
    ),
    EnvBinding("KUMC_DRIVE_MAX_FILES", "integrations.drive.max_files", _to_int),
    EnvBinding("KUMC_DRIVE_BATCH_SIZE", "integrations.drive.batch_size", _to_int),
    EnvBinding(
        "KUMC_DRIVE_DOWNLOAD_MAX_RETRIES",
        "integrations.drive.download_max_retries",
        _to_int,
    ),
    EnvBinding(
        "KUMC_DRIVE_DOWNLOAD_RETRY_INITIAL_DELAY_SECONDS",
        "integrations.drive.download_retry_initial_delay_seconds",
        _to_float,
    ),
    EnvBinding(
        "KUMC_DRIVE_DOWNLOAD_RETRY_MAX_DELAY_SECONDS",
        "integrations.drive.download_retry_max_delay_seconds",
        _to_float,
    ),
    EnvBinding(
        "KUMC_DRIVE_DOWNLOAD_RETRY_BACKOFF_MULTIPLIER",
        "integrations.drive.download_retry_backoff_multiplier",
        _to_float,
    ),
    EnvBinding(
        "KUMC_DRIVE_PDF_OCR_MODEL_PATH",
        "integrations.drive.pdf_ocr_model_path",
        str,
    ),
    EnvBinding(
        "KUMC_HATENA_BLOG_URL",
        "integrations.hatenablog.blog_url",
        str,
    ),
    EnvBinding(
        "KUMC_CRAFTERS_COLONY_AUTHOR_URL",
        "integrations.crafters_colony.author_url",
        str,
    ),
    EnvBinding(
        "KUMC_CRAFTERS_COLONY_MAX_PAGES",
        "integrations.crafters_colony.max_pages",
        _to_int,
    ),
    EnvBinding(
        "KUMC_CRAFTERS_COLONY_MAX_ARTICLES",
        "integrations.crafters_colony.max_articles",
        _to_int,
    ),
    EnvBinding("KUMC_NOTION_API_TOKEN", "integrations.notion.api_token", str),
    EnvBinding(
        "KUMC_NOTION_DATABASE_IDS",
        "integrations.notion.database_ids",
        _to_str_list,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_PAGES",
        "integrations.minecraft_wiki.page_titles",
        _to_str_list,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_API_URL",
        "integrations.minecraft_wiki.api_url",
        str,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_PAGE_URL_BASE",
        "integrations.minecraft_wiki.page_url_base",
        str,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_MAX_PAGES",
        "integrations.minecraft_wiki.max_pages",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_RATE_LIMIT_PER_MINUTE",
        "integrations.minecraft_wiki.rate_limit_per_minute",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_REQUEST_INTERVAL_SECONDS",
        "integrations.minecraft_wiki.request_interval_seconds",
        _to_float,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_NAMESPACES",
        "integrations.minecraft_wiki.namespaces",
        _to_int_csv,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_FULL_BACKFILL_ENABLED",
        "integrations.minecraft_wiki.full_backfill_enabled",
        _to_bool,
    ),
    EnvBinding("KUMC_LLM_PROVIDER", "providers.llm.provider", str),
    EnvBinding("KUMC_LLM_GEMINI_MODEL", "providers.llm.gemini_model", str),
    EnvBinding("KUMC_LLM_TEMPERATURE", "providers.llm.temperature", _to_float),
    EnvBinding(
        "KUMC_LLM_MAX_OUTPUT_TOKENS",
        "providers.llm.max_output_tokens",
        _to_int,
    ),
    EnvBinding("KUMC_LLM_THINKING_LEVEL", "providers.llm.thinking_level", str),
    EnvBinding("KUMC_EMBEDDING_PROVIDER", "providers.embeddings.provider", str),
    EnvBinding("KUMC_EMBEDDING_MODEL", "providers.embeddings.model", str),
    EnvBinding("KUMC_EMBEDDING_DIMENSIONS", "providers.embeddings.dimensions", _to_int),
    EnvBinding("KUMC_RERANKER_MODEL", "providers.reranker.model", str),
    EnvBinding("KUMC_RERANKER_ENABLED", "providers.reranker.enabled", _to_bool),
    EnvBinding("KUMC_FUNCTION_CALL_ENABLED", "providers.function_call.enabled", _to_bool),
    EnvBinding("KUMC_FUNCTION_CALL_PROVIDER", "providers.function_call.provider", str),
    EnvBinding(
        "KUMC_FUNCTION_CALL_GEMINI_MODEL",
        "providers.function_call.gemini_model",
        str,
    ),
    EnvBinding("KUMC_RAG_ROUTING_ENABLED", "rag.routing.enabled", _to_bool),
    EnvBinding("KUMC_RAG_ROUTING_PROVIDER", "rag.routing.provider", str),
    EnvBinding("KUMC_RAG_ROUTING_GEMINI_MODEL", "rag.routing.gemini_model", str),
    EnvBinding("KUMC_RAG_ROUTING_TEMPERATURE", "rag.routing.temperature", _to_float),
    EnvBinding("KUMC_RAG_ROUTING_MAX_NEW_TOKENS", "rag.routing.max_new_tokens", _to_int),
    EnvBinding("KUMC_RAG_ROUTING_MAX_RETRIES", "rag.routing.max_retries", _to_int),
    EnvBinding("KUMC_RAG_ROUTING_LOG_ENABLED", "rag.routing.log_enabled", _to_bool),
    EnvBinding(
        "KUMC_RAG_MATERIAL_SEARCH_MAX_NAMES",
        "rag.routing.material_search_max_names",
        _to_int,
    ),
    EnvBinding("KUMC_RAGAS_BATCH_SIZE", "ops.ragas_batch_size", _to_int),
    EnvBinding(
        "KUMC_RAGAS_ANSWER_GENERATION_BATCH_SIZE",
        "ops.ragas_answer_generation_batch_size",
        _to_int,
    ),
    EnvBinding("KUMC_RAGAS_MAX_WORKERS", "ops.ragas_max_workers", _to_int),
    EnvBinding(
        "KUMC_RAGAS_TIMEOUT_SECONDS",
        "ops.ragas_timeout_seconds",
        _to_float,
    ),
    EnvBinding("KUMC_RAGAS_MAX_RETRIES", "ops.ragas_max_retries", _to_int),
    EnvBinding(
        "KUMC_RAGAS_ANSWER_CACHE_ENABLED",
        "ops.ragas_answer_cache_enabled",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_RAGAS_ANSWER_CACHE_PATH",
        "ops.ragas_answer_cache_path",
        str,
    ),
    EnvBinding(
        "KUMC_RAGAS_DISABLE_HISTORY_FOR_EVAL",
        "ops.ragas_disable_history_for_eval",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_RAGAS_METRIC_ANSWER_RELEVANCY_ENABLED",
        "ops.ragas_metrics.answer_relevancy_enabled",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_RAGAS_METRIC_FAITHFULNESS_ENABLED",
        "ops.ragas_metrics.faithfulness_enabled",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_RAGAS_METRIC_CONTEXT_PRECISION_ENABLED",
        "ops.ragas_metrics.context_precision_enabled",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_RAGAS_METRIC_CONTEXT_RECALL_ENABLED",
        "ops.ragas_metrics.context_recall_enabled",
        _to_bool,
    ),
    EnvBinding("KUMC_CHAT_HISTORY_ENABLED", "rag.history.enabled", _to_bool),
    EnvBinding("KUMC_CHAT_HISTORY_MAX_TURNS", "rag.history.max_turns", _to_int),
    EnvBinding(
        "KUMC_PROMPT_HISTORY_DEFAULT_TURNS",
        "rag.history.prompt_default_turns",
        _to_int,
    ),
    EnvBinding(
        "KUMC_PROMPT_HISTORY_ADDITIONAL_TURNS",
        "rag.history.prompt_additional_turns",
        _to_int,
    ),
    EnvBinding(
        "KUMC_SPECIAL_CHANNEL_HISTORY_LIMIT",
        "rag.history.special_channel_history_limit",
        _to_int,
    ),
    EnvBinding(
        "KUMC_SPECIAL_CHANNEL_NAMES",
        "rag.history.special_channel_names",
        _to_str_list,
    ),
    EnvBinding(
        "KUMC_SPECIAL_CHANNEL_CUSTOM_INSTRUCTION",
        "rag.history.special_channel_custom_instruction",
        str,
    ),
    EnvBinding("KUMC_FAST_MODEL_NOTICE", "rag.fast_model_notice", str),
    EnvBinding("PROMPT_EMPTY_CONTEXT", "rag.prompt_texts.empty_context", str),
    EnvBinding("PROMPT_EMPTY_HISTORY", "rag.prompt_texts.empty_history", str),
    EnvBinding("PROMPT_HISTORY_USER_PREFIX", "rag.prompt_texts.history_user_prefix", str),
    EnvBinding(
        "PROMPT_HISTORY_ASSISTANT_PREFIX",
        "rag.prompt_texts.history_assistant_prefix",
        str,
    ),
    EnvBinding("PROMPT_HISTORY_SOURCES_LABEL", "rag.prompt_texts.history_sources_label", str),
    EnvBinding(
        "PROMPT_GEMINI_HEADER_CHAT_HISTORY",
        "rag.prompt_texts.gemini_header_chat_history",
        str,
    ),
    EnvBinding(
        "PROMPT_GEMINI_HEADER_RETRY_HISTORY",
        "rag.prompt_texts.gemini_header_retry_history",
        str,
    ),
    EnvBinding(
        "PROMPT_GEMINI_HEADER_CIRCLE_INFO",
        "rag.prompt_texts.gemini_header_circle_info",
        str,
    ),
    EnvBinding(
        "PROMPT_GEMINI_HEADER_CAPABILITIES",
        "rag.prompt_texts.gemini_header_capabilities",
        str,
    ),
    EnvBinding(
        "PROMPT_GEMINI_HEADER_CONTEXT",
        "rag.prompt_texts.gemini_header_context",
        str,
    ),
    EnvBinding(
        "PROMPT_GEMINI_HEADER_OUTPUT_FORMAT",
        "rag.prompt_texts.gemini_header_output_format",
        str,
    ),
    EnvBinding(
        "PROMPT_GEMINI_HEADER_INSTRUCTIONS",
        "rag.prompt_texts.gemini_header_instructions",
        str,
    ),
    EnvBinding(
        "PROMPT_GEMINI_HEADER_QUESTION",
        "rag.prompt_texts.gemini_header_question",
        str,
    ),
    EnvBinding("KUMC_RAG_GENERATION_RAG_PROVIDER", "rag.generation.rag.provider", str),
    EnvBinding(
        "KUMC_RAG_GENERATION_RAG_GEMINI_MODEL",
        "rag.generation.rag.gemini_model",
        str,
    ),
    EnvBinding(
        "KUMC_RAG_GENERATION_RAG_TEMPERATURE",
        "rag.generation.rag.temperature",
        _to_float,
    ),
    EnvBinding(
        "KUMC_RAG_GENERATION_RAG_MAX_OUTPUT_TOKENS",
        "rag.generation.rag.max_output_tokens",
        _to_int,
    ),
    EnvBinding(
        "KUMC_RAG_GENERATION_RAG_THINKING_LEVEL",
        "rag.generation.rag.thinking_level",
        str,
    ),
    EnvBinding(
        "KUMC_RAG_GENERATION_RAG_PROMPT_NAME",
        "rag.generation.rag.prompt_name",
        str,
    ),
    EnvBinding(
        "KUMC_RAG_GENERATION_NO_RAG_PROVIDER",
        "rag.generation.no_rag.provider",
        str,
    ),
    EnvBinding(
        "KUMC_RAG_GENERATION_NO_RAG_GEMINI_MODEL",
        "rag.generation.no_rag.gemini_model",
        str,
    ),
    EnvBinding(
        "KUMC_RAG_GENERATION_NO_RAG_TEMPERATURE",
        "rag.generation.no_rag.temperature",
        _to_float,
    ),
    EnvBinding(
        "KUMC_RAG_GENERATION_NO_RAG_MAX_OUTPUT_TOKENS",
        "rag.generation.no_rag.max_output_tokens",
        _to_int,
    ),
    EnvBinding(
        "KUMC_RAG_GENERATION_NO_RAG_THINKING_LEVEL",
        "rag.generation.no_rag.thinking_level",
        str,
    ),
    EnvBinding(
        "KUMC_RAG_GENERATION_NO_RAG_PROMPT_NAME",
        "rag.generation.no_rag.prompt_name",
        str,
    ),
    EnvBinding(
        "KUMC_RAG_IDEA_PROMPT_NAME",
        "rag.generation.idea_generation.prompt_name",
        str,
    ),
    EnvBinding(
        "KUMC_RAG_IDEA_TEMPERATURE",
        "rag.generation.idea_generation.temperature",
        _to_float,
    ),
    EnvBinding(
        "KUMC_RAG_IDEA_PROVIDER",
        "rag.generation.idea_generation.provider",
        str,
    ),
    EnvBinding(
        "KUMC_RAG_IDEA_GEMINI_MODEL",
        "rag.generation.idea_generation.gemini_model",
        str,
    ),
    EnvBinding(
        "KUMC_RAG_IDEA_MAX_OUTPUT_TOKENS",
        "rag.generation.idea_generation.max_output_tokens",
        _to_int,
    ),
    EnvBinding(
        "KUMC_RAG_IDEA_THINKING_LEVEL",
        "rag.generation.idea_generation.thinking_level",
        str,
    ),
    EnvBinding(
        "KUMC_INDEXING_FIRST_RECURSIVE_CHUNK_SIZE",
        "indexing.chunking.first_recursive_chunk_size",
        _to_int,
    ),
    EnvBinding(
        "KUMC_INDEXING_FIRST_RECURSIVE_CHUNK_OVERLAP",
        "indexing.chunking.first_recursive_chunk_overlap",
        _to_int,
    ),
    EnvBinding(
        "KUMC_INDEXING_SECOND_RECURSIVE_CHUNK_SIZE",
        "indexing.chunking.second_recursive_chunk_size",
        _to_int,
    ),
    EnvBinding(
        "KUMC_INDEXING_SECOND_RECURSIVE_CHUNK_OVERLAP",
        "indexing.chunking.second_recursive_chunk_overlap",
        _to_int,
    ),
    EnvBinding(
        "KUMC_INDEXING_SUMMARY_CHARACTERS",
        "indexing.chunking.summary_characters",
        _to_int,
    ),
    EnvBinding(
        "KUMC_INDEXING_SUMMARY_BATCH_SIZE",
        "indexing.chunking.summary_batch_size",
        _to_int,
    ),
    EnvBinding(
        "KUMC_INDEXING_SECOND_RECURSIVE_ENABLED",
        "indexing.stages.second_recursive_enabled",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_INDEXING_SPARSE_SECOND_RECURSIVE_ENABLED",
        "indexing.stages.sparse_second_recursive_enabled",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_INDEXING_SUMMARY_ENABLED",
        "indexing.stages.summary_enabled",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_CLEAR_INGESTION_SOURCE_DATA",
        "indexing.refresh.clear_ingestion_source_data",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_CLEAR_FIRST_RECURSIVE_CHUNK_DATA",
        "indexing.refresh.clear_first_recursive_chunk_data",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_CLEAR_SECOND_RECURSIVE_CHUNK_DATA",
        "indexing.refresh.clear_second_recursive_chunk_data",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_CLEAR_SUMMARY_CHUNK_DATA",
        "indexing.refresh.clear_summary_chunk_data",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_UPDATE_INGESTION_SOURCE_DATA",
        "indexing.refresh.update_ingestion_source_data",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_UPDATE_FIRST_RECURSIVE_CHUNK_DATA",
        "indexing.refresh.update_first_recursive_chunk_data",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_UPDATE_SECOND_RECURSIVE_CHUNK_DATA",
        "indexing.refresh.update_second_recursive_chunk_data",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_UPDATE_SPARSE_SECOND_RECURSIVE_CHUNK_DATA",
        "indexing.refresh.update_sparse_second_recursive_chunk_data",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_UPDATE_SUMMARY_CHUNK_DATA",
        "indexing.refresh.update_summary_chunk_data",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_INDEX_UPDATE_ESTIMATE_MIN_MINUTES",
        "ops.index_update_estimate_min_minutes",
        _to_int,
    ),
    EnvBinding(
        "KUMC_INDEX_UPDATE_ESTIMATE_MAX_MINUTES",
        "ops.index_update_estimate_max_minutes",
        _to_int,
    ),
    EnvBinding(
        "KUMC_ANSWER_RECORD_LOG_ENABLED",
        "ops.answer_record_log_enabled",
        _to_bool,
    ),
    EnvBinding("KUMC_ANSWER_RECORD_LOG_PATH", "ops.answer_record_log_path", str),
    EnvBinding("KUMC_AUTO_INDEX_ENABLED", "scheduler.auto_index_enabled", _to_bool),
    EnvBinding("KUMC_AUTO_INDEX_TIME", "scheduler.auto_index_time", str),
    EnvBinding("KUMC_AUTO_INDEX_WEEKDAYS", "scheduler.auto_index_weekdays", _to_int_list),
    EnvBinding("KUMC_DATABASE_URL", "infrastructure.database.url", str),
    EnvBinding(
        "KUMC_DATABASE_CONNECT_TIMEOUT_SECONDS",
        "infrastructure.database.connect_timeout_seconds",
        _to_float,
    ),
    EnvBinding(
        "KUMC_DATABASE_APPLICATION_NAME",
        "infrastructure.database.application_name",
        str,
    ),
    EnvBinding("KUMC_REDIS_URL", "infrastructure.redis.url", str),
    EnvBinding(
        "KUMC_REDIS_SOCKET_TIMEOUT_SECONDS",
        "infrastructure.redis.socket_timeout_seconds",
        _to_float,
    ),
    EnvBinding(
        "KUMC_OBJECT_STORAGE_ENDPOINT_URL",
        "infrastructure.object_storage.endpoint_url",
        str,
    ),
    EnvBinding(
        "KUMC_OBJECT_STORAGE_BUCKET",
        "infrastructure.object_storage.bucket",
        str,
    ),
    EnvBinding(
        "KUMC_OBJECT_STORAGE_REGION",
        "infrastructure.object_storage.region",
        str,
    ),
    EnvBinding(
        "KUMC_OBJECT_STORAGE_ACCESS_KEY_ID",
        "infrastructure.object_storage.access_key_id",
        str,
    ),
    EnvBinding(
        "KUMC_OBJECT_STORAGE_SECRET_ACCESS_KEY",
        "infrastructure.object_storage.secret_access_key",
        str,
    ),
    EnvBinding(
        "KUMC_OBJECT_STORAGE_PREFIX",
        "infrastructure.object_storage.prefix",
        str,
    ),
    EnvBinding(
        "KUMC_OBJECT_STORAGE_USE_SSL",
        "infrastructure.object_storage.use_ssl",
        _to_bool,
    ),
    EnvBinding("KUMC_MIGRATION_DIR", "infrastructure.migrations.directory", str),
    EnvBinding(
        "KUMC_MIGRATION_TABLE",
        "infrastructure.migrations.table_name",
        str,
    ),
    EnvBinding("KUMC_FEATURE_RAG", "features.rag", _to_bool),
    EnvBinding("KUMC_FEATURE_INDEXING", "features.indexing", _to_bool),
    EnvBinding("KUMC_FEATURE_EVAL", "features.eval", _to_bool),
    EnvBinding("KUMC_FEATURE_SUMMARIZATION", "features.summarization", _to_bool),
    EnvBinding("KUMC_FEATURE_VC", "features.vc", _to_bool),
    EnvBinding("KUMC_FEATURE_DOCGEN", "features.docgen", _to_bool),
    EnvBinding("KUMC_FEATURE_HTTP", "features.http", _to_bool),
    EnvBinding(
        "KUMC_FEATURE_SOURCE_MINECRAFT_WIKI",
        "features.sources.minecraft_wiki",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_FEATURE_ACTION_EXECUTION_MODE",
        "features.risk_flags.action_execution",
        str,
    ),
    EnvBinding(
        "KUMC_FEATURE_EXTERNAL_POSTING_MODE",
        "features.risk_flags.external_posting",
        str,
    ),
    EnvBinding(
        "KUMC_FEATURE_MINECRAFT_SERVER_OPS_MODE",
        "features.risk_flags.minecraft_server_ops",
        str,
    ),
    EnvBinding(
        "KUMC_FEATURE_ACCOUNTING_FINALIZE_MODE",
        "features.risk_flags.accounting_finalize",
        str,
    ),
    EnvBinding(
        "KUMC_FEATURE_AUTO_REPLY_MODE",
        "features.risk_flags.auto_reply",
        str,
    ),
    EnvBinding(
        "KUMC_FEATURE_AUTOMATION_AUTO_RUN_MODE",
        "features.risk_flags.automation_auto_run",
        str,
    ),
    EnvBinding(
        "KUMC_FEATURE_VC_RECORDING_MODE",
        "features.risk_flags.vc_recording",
        str,
    ),
    EnvBinding(
        "KUMC_FEATURE_IMAGE_GENERATION_MODE",
        "features.risk_flags.image_generation",
        str,
    ),
    EnvBinding("KUMC_RETRIEVAL_TOP_K", "features.retrieval.top_k", _to_int),
    EnvBinding("KUMC_RETRIEVAL_DENSE_TOP_K", "features.retrieval.dense_top_k", _to_int),
    EnvBinding("KUMC_RETRIEVAL_SPARSE_TOP_K", "features.retrieval.sparse_top_k", _to_int),
    EnvBinding(
        "KUMC_RETRIEVAL_SPARSE_INITIAL_SPARSE_TOP_K",
        "features.retrieval.sparse_initial_sparse_top_k",
        _to_int,
    ),
    EnvBinding(
        "KUMC_RETRIEVAL_RERANK_POOL_SIZE",
        "features.retrieval.rerank_pool_size",
        _to_int,
    ),
    EnvBinding("KUMC_RETRIEVAL_RRF_K", "features.retrieval.rrf_k", _to_int),
    EnvBinding("KUMC_RETRIEVAL_MMR_LAMBDA", "features.retrieval.mmr_lambda", _to_float),
    EnvBinding(
        "KUMC_RETRIEVAL_RECENCY_WEIGHT_SOFT",
        "features.retrieval.recency_weight_soft",
        _to_float,
    ),
    EnvBinding(
        "KUMC_RETRIEVAL_RECENCY_WEIGHT_HARD",
        "features.retrieval.recency_weight_hard",
        _to_float,
    ),
    EnvBinding(
        "KUMC_RETRIEVAL_RECENCY_HALF_LIFE_DAYS",
        "features.retrieval.recency_half_life_days",
        _to_float,
    ),
    EnvBinding(
        "KUMC_RETRIEVAL_MATERIAL_FULL_TEXT_CHAR_LIMIT",
        "features.retrieval.material_full_text_char_limit",
        _to_int,
    ),
    EnvBinding("SUDACHI_MODE", "features.retrieval.sudachi_mode", str),
    EnvBinding("SPARSE_BM25_K1", "features.retrieval.sparse_bm25_k1", _to_float),
    EnvBinding("SPARSE_BM25_B", "features.retrieval.sparse_bm25_b", _to_float),
    EnvBinding(
        "SPARSE_USE_NORMALIZED_FORM",
        "features.retrieval.sparse_use_normalized_form",
        _to_bool,
    ),
    EnvBinding(
        "SPARSE_REMOVE_SYMBOLS",
        "features.retrieval.sparse_remove_symbols",
        _to_bool,
    ),
    EnvBinding("KUMC_RETRIEVAL_SUDACHI_MODE", "features.retrieval.sudachi_mode", str),
    EnvBinding(
        "KUMC_RETRIEVAL_SPARSE_BM25_K1",
        "features.retrieval.sparse_bm25_k1",
        _to_float,
    ),
    EnvBinding(
        "KUMC_RETRIEVAL_SPARSE_BM25_B",
        "features.retrieval.sparse_bm25_b",
        _to_float,
    ),
    EnvBinding(
        "KUMC_RETRIEVAL_SPARSE_USE_NORMALIZED_FORM",
        "features.retrieval.sparse_use_normalized_form",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_RETRIEVAL_SPARSE_REMOVE_SYMBOLS",
        "features.retrieval.sparse_remove_symbols",
        _to_bool,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_RAG_FIRST_RECURSIVE_CHUNK_SIZE",
        "minecraft_wiki_rag.chunking.first_recursive_chunk_size",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_RAG_FIRST_RECURSIVE_CHUNK_OVERLAP",
        "minecraft_wiki_rag.chunking.first_recursive_chunk_overlap",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_RAG_SECOND_RECURSIVE_CHUNK_SIZE",
        "minecraft_wiki_rag.chunking.second_recursive_chunk_size",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_RAG_SECOND_RECURSIVE_CHUNK_OVERLAP",
        "minecraft_wiki_rag.chunking.second_recursive_chunk_overlap",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_RAG_SUMMARY_CHARACTERS",
        "minecraft_wiki_rag.chunking.summary_characters",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_RAG_TOP_K",
        "minecraft_wiki_rag.retrieval.top_k",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_RAG_DENSE_TOP_K",
        "minecraft_wiki_rag.retrieval.dense_top_k",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_RAG_SPARSE_TOP_K",
        "minecraft_wiki_rag.retrieval.sparse_top_k",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_RAG_RERANK_POOL_SIZE",
        "minecraft_wiki_rag.retrieval.rerank_pool_size",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MINECRAFT_WIKI_RAG_PARENT_CHUNK_CAP",
        "minecraft_wiki_rag.retrieval.parent_chunk_cap",
        _to_int,
    ),
    EnvBinding(
        "KUMC_MAINTENANCE_COMMAND_AUTHOR_IDS",
        "security.maintenance_command_author_ids",
        _to_int_csv,
    ),
    EnvBinding(
        "KUMC_DISCORD_GUILD_ALLOW_LIST",
        "security.discord_guild_allow_list",
        _to_int_csv,
    ),
    EnvBinding(
        "KUMC_DISCORD_MEMBER_PROFILE_GUILD_IDS",
        "security.discord_member_profile_guild_ids",
        _to_int_csv,
    ),
    EnvBinding("KUMC_MODEL_ROOT_DIR", "model.root_dir", str),
    EnvBinding("KUMC_MODEL_LLM_DIR", "model.llm_dir", str),
    EnvBinding("KUMC_MODEL_EMBEDDING_DIR", "model.embedding_dir", str),
    EnvBinding("KUMC_MODEL_CROSS_ENCODER_DIR", "model.cross_encoder_dir", str),
    EnvBinding("KUMC_MODEL_WHISPER_DIR", "model.whisper_dir", str),
    EnvBinding("KUMC_MODEL_OCR_DIR", "model.ocr_dir", str),
    EnvBinding("KUMC_VC_FEATURE_ENABLED", "vc.feature_enabled", _to_bool),
    EnvBinding("KUMC_VC_AUTO_JOIN_ENABLED", "vc.auto_join_enabled", _to_bool),
    EnvBinding("KUMC_VC_AUTO_JOIN_WEEKDAYS", "vc.auto_join_weekdays", _to_int_list),
    EnvBinding("KUMC_VC_AUTO_JOIN_TIME", "vc.auto_join_time", str),
    EnvBinding(
        "KUMC_VC_AUTO_JOIN_DURATION_MINUTES",
        "vc.auto_join_duration_minutes",
        _to_int,
    ),
    EnvBinding(
        "KUMC_VC_TARGET_VOICE_CHANNEL_NAME",
        "vc.target_voice_channel_name",
        str,
    ),
    EnvBinding(
        "KUMC_VC_AUTO_JOIN_MIN_PARTICIPANTS",
        "vc.auto_join_min_participants",
        _to_int,
    ),
    EnvBinding(
        "KUMC_VC_PARTICIPANT_CHECK_INTERVAL_SECONDS",
        "vc.participant_check_interval_seconds",
        _to_int,
    ),
    EnvBinding(
        "KUMC_VC_SUMMARY_TRANSCRIBE_INTERVAL_SECONDS",
        "vc.summary_transcribe_interval_seconds",
        _to_int,
    ),
    EnvBinding("KUMC_VC_TRANSCRIBE_MODEL", "vc.transcribe_model", str),
    EnvBinding("KUMC_VC_TRANSCRIBE_DEVICE", "vc.transcribe_device", str),
    EnvBinding("KUMC_VC_TRANSCRIBE_TORCH_DTYPE", "vc.transcribe_torch_dtype", str),
    EnvBinding("KUMC_VC_TRANSCRIBE_LANGUAGE", "vc.transcribe_language", str),
    EnvBinding("KUMC_VC_AUTO_QUIT_ENABLED", "vc.auto_quit_enabled", _to_bool),
    EnvBinding(
        "KUMC_VC_FINAL_SUMMARY_ENABLED",
        "vc.final_summary_enabled",
        _to_bool,
    ),
    EnvBinding("KUMC_VC_SUMMARY_PREVIOUS_MAX", "vc.summary_previous_max", _to_int),
    EnvBinding(
        "KUMC_VC_SUMMARY_TARGET_CHARACTERS",
        "vc.summary_target_characters",
        _to_int,
    ),
    EnvBinding("KUMC_VC_SUMMARY_LLM_PROVIDER", "vc.summary_llm_provider", str),
    EnvBinding("KUMC_VC_SUMMARY_GEMINI_MODEL", "vc.summary_gemini_model", str),
    EnvBinding("KUMC_VC_SUMMARY_TEMPERATURE", "vc.summary_temperature", _to_float),
    EnvBinding(
        "KUMC_VC_SUMMARY_MAX_OUTPUT_TOKENS",
        "vc.summary_max_output_tokens",
        _to_int,
    ),
    EnvBinding("KUMC_VC_SUMMARY_THINKING_LEVEL", "vc.summary_thinking_level", str),
    EnvBinding("KUMC_VC_MINUTES_ENABLED", "vc.minutes_enabled", _to_bool),
    EnvBinding("KUMC_VC_MINUTES_DRIVE_DIR", "vc.minutes_drive_dir", str),
    EnvBinding(
        "KUMC_VC_MINUTES_FETCH_MAX_RETRIES",
        "vc.minutes_fetch_max_retries",
        _to_int,
    ),
    EnvBinding(
        "KUMC_VC_MINUTES_APPLY_MAX_RETRIES",
        "vc.minutes_apply_max_retries",
        _to_int,
    ),
    EnvBinding(
        "KUMC_VC_MINUTES_LLM_MAX_RETRIES",
        "vc.minutes_llm_max_retries",
        _to_int,
    ),
    EnvBinding(
        "KUMC_VC_MINUTES_HISTORY_SUMMARY_MAX",
        "vc.minutes_history_summary_max",
        _to_int,
    ),
    EnvBinding(
        "KUMC_VC_MINUTES_IMAGE_BATCH_SIZE",
        "vc.minutes_image_batch_size",
        _to_int,
    ),
    EnvBinding(
        "KUMC_VC_MINUTES_EDIT_LLM_PROVIDER",
        "vc.minutes_edit_llm_provider",
        str,
    ),
    EnvBinding(
        "KUMC_VC_MINUTES_EDIT_GEMINI_MODEL",
        "vc.minutes_edit_gemini_model",
        str,
    ),
    EnvBinding(
        "KUMC_VC_MINUTES_EDIT_TEMPERATURE",
        "vc.minutes_edit_temperature",
        _to_float,
    ),
    EnvBinding(
        "KUMC_VC_MINUTES_EDIT_MAX_OUTPUT_TOKENS",
        "vc.minutes_edit_max_output_tokens",
        _to_int,
    ),
    EnvBinding(
        "KUMC_VC_MINUTES_EDIT_THINKING_LEVEL",
        "vc.minutes_edit_thinking_level",
        str,
    ),
    EnvBinding(
        "KUMC_VC_FINAL_SUMMARY_LLM_PROVIDER",
        "vc.final_summary_llm_provider",
        str,
    ),
    EnvBinding(
        "KUMC_VC_FINAL_SUMMARY_GEMINI_MODEL",
        "vc.final_summary_gemini_model",
        str,
    ),
    EnvBinding(
        "KUMC_VC_FINAL_SUMMARY_TEMPERATURE",
        "vc.final_summary_temperature",
        _to_float,
    ),
    EnvBinding(
        "KUMC_VC_FINAL_SUMMARY_MAX_OUTPUT_TOKENS",
        "vc.final_summary_max_output_tokens",
        _to_int,
    ),
    EnvBinding(
        "KUMC_VC_FINAL_SUMMARY_THINKING_LEVEL",
        "vc.final_summary_thinking_level",
        str,
    ),
)
