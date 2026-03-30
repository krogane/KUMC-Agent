from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppSection:
    command_prefix: str
    index_command_prefix: str
    max_input_characters: int
    log_level: str
    data_dir: Path
    raw_dir: Path
    chunks_path: Path
    index_dir: Path
    eval_dir: Path
    cache_dir: Path
    answer_record_log_path: Path
    source_max_count: int


@dataclass(frozen=True)
class ModelSection:
    root_dir: Path
    llm_dir: Path
    embedding_dir: Path
    cross_encoder_dir: Path
    whisper_dir: Path
    ocr_dir: Path


@dataclass(frozen=True)
class LLMSection:
    provider: str
    gemini_model: str
    llama_model_path: str
    temperature: float
    max_output_tokens: int
    thinking_level: str
    threads: int
    gpu_layers: int


@dataclass(frozen=True)
class EmbeddingSection:
    provider: str
    model: str
    dimensions: int


@dataclass(frozen=True)
class RerankerSection:
    model: str
    enabled: bool


@dataclass(frozen=True)
class FunctionCallSection:
    enabled: bool
    provider: str
    gemini_model: str
    llama_model_path: str


@dataclass(frozen=True)
class ProviderSection:
    llm: LLMSection
    embeddings: EmbeddingSection
    reranker: RerankerSection
    function_call: FunctionCallSection


@dataclass(frozen=True)
class SecuritySection:
    maintenance_command_author_ids: list[int]
    discord_guild_allow_list: list[int]
    refusal_keywords: list[str]


@dataclass(frozen=True)
class SchedulerSection:
    auto_index_enabled: bool
    auto_index_time: str
    auto_index_weekdays: list[int]


@dataclass(frozen=True)
class RetrievalSection:
    top_k: int
    dense_top_k: int
    sparse_top_k: int
    sparse_initial_sparse_top_k: int
    rerank_pool_size: int
    mmr_lambda: float
    recency_weight_soft: float
    recency_weight_hard: float
    recency_half_life_days: float
    parent_doc_enabled: bool
    parent_chunk_cap: int
    material_full_text_char_limit: int
    sudachi_mode: str
    sparse_bm25_k1: float
    sparse_bm25_b: float
    sparse_use_normalized_form: bool
    sparse_remove_symbols: bool


@dataclass(frozen=True)
class SourcesSection:
    drive: bool
    discord: bool
    hatenablog: bool
    crafters_colony: bool
    x: bool


@dataclass(frozen=True)
class FeatureSection:
    rag: bool
    indexing: bool
    eval: bool
    summarization: bool
    vc: bool
    docgen: bool
    http: bool
    recency_mode: str
    sources: SourcesSection
    retrieval: RetrievalSection


@dataclass(frozen=True)
class RagRoutingTaskSection:
    provider: str
    gemini_model: str
    llama_model_path: str
    prompt_name: str


@dataclass(frozen=True)
class RagRoutingTasksSection:
    target_model: RagRoutingTaskSection
    use_additional_memory: RagRoutingTaskSection
    include_capabilities_info: RagRoutingTaskSection
    needs_additional_query: RagRoutingTaskSection
    additional_queries: RagRoutingTaskSection
    material_names: RagRoutingTaskSection
    recency_mode: RagRoutingTaskSection


@dataclass(frozen=True)
class RagRoutingSection:
    enabled: bool
    provider: str
    gemini_model: str
    llama_model_path: str
    prompt_name: str
    temperature: float
    max_new_tokens: int
    max_retries: int
    log_enabled: bool
    material_search_max_names: int
    tasks: RagRoutingTasksSection


@dataclass(frozen=True)
class RagHistorySection:
    enabled: bool
    max_turns: int
    prompt_default_turns: int
    prompt_additional_turns: int
    special_channel_history_limit: int
    special_channel_names: list[str]
    special_channel_custom_instruction: str


@dataclass(frozen=True)
class RagGenerationProfileSection:
    provider: str
    gemini_model: str
    llama_model_path: str
    temperature: float
    max_output_tokens: int
    thinking_level: str
    prompt_name: str


@dataclass(frozen=True)
class RagGenerationSection:
    rag: RagGenerationProfileSection
    no_rag: RagGenerationProfileSection
    refusal: RagGenerationProfileSection


@dataclass(frozen=True)
class RagPromptTextSection:
    empty_context: str
    empty_history: str
    history_user_prefix: str
    history_assistant_prefix: str
    history_sources_label: str
    gemini_header_chat_history: str
    gemini_header_retry_history: str
    gemini_header_circle_info: str
    gemini_header_capabilities: str
    gemini_header_context: str
    gemini_header_output_format: str
    gemini_header_instructions: str
    gemini_header_question: str
    llama_header_question: str
    llama_header_previous_attempt: str
    llama_header_circle_info: str
    llama_header_capabilities: str
    llama_header_context: str
    llama_header_output_format: str
    llama_header_instructions: str


@dataclass(frozen=True)
class RagSection:
    routing: RagRoutingSection
    history: RagHistorySection
    generation: RagGenerationSection
    prompt_texts: RagPromptTextSection
    fast_model_notice: str
    answer_json_max_retries: int


@dataclass(frozen=True)
class IndexingChunkingSection:
    first_recursive_chunk_size: int
    first_recursive_chunk_overlap: int
    second_recursive_chunk_size: int
    second_recursive_chunk_overlap: int
    summary_characters: int
    summary_batch_size: int
    summary_llm_provider: str
    summary_gemini_model: str
    summary_llama_model_path: str
    summary_temperature: float
    summary_max_output_tokens: int
    summary_thinking_level: str
    proposition_llm_provider: str
    proposition_gemini_model: str
    proposition_llama_model_path: str
    proposition_temperature: float
    proposition_max_output_tokens: int
    proposition_thinking_level: str
    proposition_max_retries: int
    raptor_llm_provider: str
    raptor_gemini_model: str
    raptor_llama_model_path: str
    raptor_temperature: float
    raptor_max_output_tokens: int
    raptor_thinking_level: str
    raptor_max_retries: int
    raptor_cluster_max_tokens: int
    raptor_stop_chunk_count: int
    raptor_k_max: int
    raptor_k_selection: str


@dataclass(frozen=True)
class IndexingStagesSection:
    second_recursive_enabled: bool
    sparse_second_recursive_enabled: bool
    summary_enabled: bool
    proposition_enabled: bool
    raptor_enabled: bool


@dataclass(frozen=True)
class IndexingRefreshSection:
    clear_raw_data: bool
    clear_first_recursive_chunk_data: bool
    clear_second_recursive_chunk_data: bool
    clear_summary_chunk_data: bool
    clear_proposition_chunk_data: bool
    clear_raptor_chunk_data: bool
    update_raw_data: bool
    update_first_recursive_chunk_data: bool
    update_second_recursive_chunk_data: bool
    update_sparse_second_recursive_chunk_data: bool
    update_summary_chunk_data: bool
    update_proposition_chunk_data: bool
    update_raptor_chunk_data: bool


@dataclass(frozen=True)
class IndexingSection:
    chunking: IndexingChunkingSection
    stages: IndexingStagesSection
    refresh: IndexingRefreshSection


@dataclass(frozen=True)
class OpsRagasMetricsSection:
    answer_relevancy_enabled: bool
    faithfulness_enabled: bool
    context_precision_enabled: bool
    context_recall_enabled: bool


@dataclass(frozen=True)
class OpsSection:
    warmup_interval_minutes: int
    index_update_estimate_min_minutes: int
    index_update_estimate_max_minutes: int
    ragas_answer_generation_batch_size: int
    ragas_batch_size: int
    ragas_max_workers: int
    ragas_timeout_seconds: float
    ragas_max_retries: int
    ragas_answer_cache_enabled: bool
    ragas_answer_cache_path: Path
    ragas_disable_history_for_eval: bool
    ragas_metrics: OpsRagasMetricsSection
    answer_record_log_enabled: bool
    answer_record_log_path: Path


@dataclass(frozen=True)
class IntegrationDiscordSection:
    bot_token: str


@dataclass(frozen=True)
class IntegrationOpenClawSection:
    enabled: bool
    agent: str
    model: str
    lite_agent: str
    lite_model: str
    config_dir: Path


@dataclass(frozen=True)
class IntegrationDriveSection:
    folder_id: str
    google_application_credentials: str
    max_files: int
    batch_size: int
    download_max_retries: int
    download_retry_initial_delay_seconds: float
    download_retry_max_delay_seconds: float
    download_retry_backoff_multiplier: float
    pdf_ocr_model_path: str


@dataclass(frozen=True)
class IntegrationCraftersColonySection:
    author_url: str
    max_pages: int
    max_articles: int


@dataclass(frozen=True)
class IntegrationSection:
    discord: IntegrationDiscordSection
    openclaw: IntegrationOpenClawSection
    drive: IntegrationDriveSection
    crafters_colony: IntegrationCraftersColonySection
    openai_api_key: str
    gemini_api_key: str
    gemini_requests_per_minute: int
    gemini_embedding_requests_per_minute: int
    gemini_summary_requests_per_minute: int
    gemini_ragas_requests_per_minute: int
    gemini_ragas_embedding_requests_per_minute: int


@dataclass(frozen=True)
class VCSection:
    feature_enabled: bool
    auto_join_enabled: bool
    auto_join_weekdays: list[int]
    auto_join_time: str
    auto_join_duration_minutes: int
    target_voice_channel_name: str
    auto_join_min_participants: int
    participant_check_interval_seconds: int
    summary_transcribe_interval_seconds: int
    transcribe_model: str
    transcribe_device: str
    transcribe_torch_dtype: str
    transcribe_language: str
    auto_quit_enabled: bool
    final_summary_enabled: bool
    summary_previous_max: int
    summary_target_characters: int
    summary_llm_provider: str
    summary_gemini_model: str
    summary_llama_model_path: str
    summary_llama_ctx_size: int
    summary_temperature: float
    summary_max_output_tokens: int
    summary_thinking_level: str
    minutes_enabled: bool
    minutes_drive_dir: str
    minutes_fetch_max_retries: int
    minutes_apply_max_retries: int
    minutes_llm_max_retries: int
    minutes_history_summary_max: int
    minutes_image_batch_size: int
    minutes_edit_llm_provider: str
    minutes_edit_gemini_model: str
    minutes_edit_llama_model_path: str
    minutes_edit_llama_ctx_size: int
    minutes_edit_temperature: float
    minutes_edit_max_output_tokens: int
    minutes_edit_thinking_level: str
    final_summary_llm_provider: str
    final_summary_gemini_model: str
    final_summary_llama_model_path: str
    final_summary_llama_ctx_size: int
    final_summary_temperature: float
    final_summary_max_output_tokens: int
    final_summary_thinking_level: str


@dataclass(frozen=True)
class RuntimeConfig:
    base_dir: Path
    experiment_profile: str
    app: AppSection
    providers: ProviderSection
    security: SecuritySection
    scheduler: SchedulerSection
    features: FeatureSection
    rag: RagSection
    indexing: IndexingSection
    ops: OpsSection
    integrations: IntegrationSection
    model: ModelSection
    vc: VCSection
    experiments: dict[str, Any] = field(default_factory=dict)

    @property
    def is_discord_enabled(self) -> bool:
        return self.features.rag and bool(self.integrations.discord.bot_token)

    @property
    def required_env_missing(self) -> list[str]:
        missing: list[str] = []
        if not self.integrations.discord.bot_token:
            missing.append("KUMC_DISCORD_BOT_TOKEN")
        if not self.integrations.gemini_api_key and self.providers.llm.provider == "gemini":
            missing.append("KUMC_GEMINI_API_KEY")
        if not self.integrations.drive.folder_id:
            missing.append("KUMC_DRIVE_FOLDER_ID")
        return missing
