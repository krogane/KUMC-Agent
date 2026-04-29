from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AppSection:
    command_prefix: str
    index_command_prefix: str
    max_input_characters: int
    log_level: str
    data_dir: Path
    ingestion_dir: Path
    index_documents_path: Path
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
    temperature: float
    max_output_tokens: int
    thinking_level: str


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
    discord_member_profile_guild_ids: list[int]

    def effective_member_profile_guild_ids(self) -> list[int]:
        return list(self.discord_member_profile_guild_ids or self.discord_guild_allow_list)


@dataclass(frozen=True)
class SchedulerSection:
    auto_index_enabled: bool
    auto_index_time: str
    auto_index_weekdays: list[int]
    auto_index_timezone: str
    auto_index_max_runtime_minutes: int
    auto_index_lock_ttl_minutes: int
    quality_min_chunk_ratio: float
    quality_smoke_queries: list[str]
    rollback_keep_snapshots: int


@dataclass(frozen=True)
class AutonomousAgentLookaheadSection:
    tasks: int
    events: int


@dataclass(frozen=True)
class AutonomousAgentBudgetSection:
    max_steps: int
    max_search_calls: int
    max_replans: int
    max_cost_usd: float
    max_latency_seconds: float


@dataclass(frozen=True)
class AutonomousAgentLLMSection:
    enabled: bool
    provider: str
    gemini_model: str
    openai_model: str
    prompt_name: str
    temperature: float
    max_output_tokens: int
    max_retries: int


@dataclass(frozen=True)
class AutonomousAgentAccessSection:
    system_user_id: str
    guild_id: str
    role_ids: list[str]
    is_admin: bool


@dataclass(frozen=True)
class AutonomousAgentSection:
    enabled: bool
    schedule_times: list[str]
    timezone: str
    scopes: list[str]
    notification_channel_id: str
    dry_run: bool
    lookahead_days: AutonomousAgentLookaheadSection
    duplicate_suppression_hours: int
    budget: AutonomousAgentBudgetSection
    planner: AutonomousAgentLLMSection
    verifier: AutonomousAgentLLMSection
    access: AutonomousAgentAccessSection
    rag_delta_lookback_hours: int


@dataclass(frozen=True)
class ComprehensiveAgentLLMSection:
    enabled: bool
    provider: str
    gemini_model: str
    prompt_name: str
    temperature: float
    max_output_tokens: int
    max_retries: int


@dataclass(frozen=True)
class ComprehensiveAgentBudgetSection:
    max_steps: int
    max_search_calls: int
    max_read_chunks: int
    max_replans: int
    max_cost_usd: float
    max_latency_seconds: float
    require_citations: bool


@dataclass(frozen=True)
class ComprehensiveAgentSection:
    enabled: bool
    planner: ComprehensiveAgentLLMSection
    verifier: ComprehensiveAgentLLMSection
    budget: ComprehensiveAgentBudgetSection


@dataclass(frozen=True)
class TaskManagementSection:
    approval_batch_interval_days: int
    due_soon_notice_days: int
    notification_channel_id: str
    admin_user_ids: list[str]
    admin_role_ids: list[str]
    prompt_name: str
    auto_extract_after_index_update: bool


@dataclass(frozen=True)
class EventManagementSection:
    approval_batch_interval_days: int
    notification_before_days: int
    notification_channel_id: str
    admin_user_ids: list[str]
    admin_role_ids: list[str]
    prompt_name: str
    auto_extract_after_index_update: bool
    timezone: str


@dataclass(frozen=True)
class DatabaseSection:
    url: str
    connect_timeout_seconds: float
    application_name: str


@dataclass(frozen=True)
class RedisSection:
    url: str
    socket_timeout_seconds: float


@dataclass(frozen=True)
class ObjectStorageSection:
    endpoint_url: str
    bucket: str
    region: str
    access_key_id: str
    secret_access_key: str
    prefix: str
    use_ssl: bool


@dataclass(frozen=True)
class MigrationSection:
    directory: Path
    table_name: str


@dataclass(frozen=True)
class InfrastructureSection:
    database: DatabaseSection
    redis: RedisSection
    object_storage: ObjectStorageSection
    migrations: MigrationSection


@dataclass(frozen=True)
class RetrievalSection:
    top_k: int
    dense_top_k: int
    sparse_top_k: int
    sparse_initial_sparse_top_k: int
    sparse_normalized_ratio: float | None
    rerank_pool_size: int
    rrf_k: int
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
class MinecraftWikiRagChunkingSection:
    first_recursive_chunk_size: int
    first_recursive_chunk_overlap: int
    second_recursive_chunk_size: int
    second_recursive_chunk_overlap: int
    summary_characters: int
    summary_batch_size: int
    summary_llm_provider: str
    summary_gemini_model: str
    summary_temperature: float
    summary_max_output_tokens: int
    summary_thinking_level: str


@dataclass(frozen=True)
class MinecraftWikiRagRetrievalSection:
    top_k: int
    dense_top_k: int
    sparse_top_k: int
    sparse_initial_sparse_top_k: int
    sparse_normalized_ratio: float | None
    rerank_pool_size: int
    rrf_k: int
    mmr_lambda: float
    parent_doc_enabled: bool
    parent_chunk_cap: int
    sudachi_mode: str
    sparse_bm25_k1: float
    sparse_bm25_b: float
    sparse_use_normalized_form: bool
    sparse_remove_symbols: bool


@dataclass(frozen=True)
class MinecraftWikiRagSection:
    chunking: MinecraftWikiRagChunkingSection
    retrieval: MinecraftWikiRagRetrievalSection


@dataclass(frozen=True)
class SourcesSection:
    drive: bool
    discord: bool
    hatenablog: bool
    crafters_colony: bool
    x: bool
    notion: bool
    minecraft_wiki: bool


@dataclass(frozen=True)
class RiskFeatureFlagsSection:
    action_execution: str
    external_posting: str
    minecraft_server_ops: str
    accounting_finalize: str
    auto_reply: str
    automation_auto_run: str
    vc_recording: str
    image_generation: str


@dataclass(frozen=True)
class ImageSearchFeatureSection:
    enabled: bool = True
    limit: int = 8
    dense_top_k: int = 24
    feature_top_k: int = 16
    rrf_k: int = 60
    ocr_text_char_limit: int = 800
    surrounding_text_char_limit: int = 1200
    caption_model: str = ""
    ocr_model: str = ""
    feature_model: str = "openai/clip-vit-base-patch32"
    feature_dimensions: int = 512
    duplicate_group_limit: int = 1


@dataclass(frozen=True)
class MemberSearchFeatureSection:
    exclude_role_names: list[str] = field(default_factory=list)


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
    risk_flags: RiskFeatureFlagsSection
    image_search: ImageSearchFeatureSection = field(default_factory=ImageSearchFeatureSection)
    member_search: MemberSearchFeatureSection = field(default_factory=MemberSearchFeatureSection)


@dataclass(frozen=True)
class RagRoutingTaskSection:
    provider: str
    gemini_model: str
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
    temperature: float
    max_output_tokens: int
    thinking_level: str
    prompt_name: str


@dataclass(frozen=True)
class RagGenerationSection:
    rag: RagGenerationProfileSection
    no_rag: RagGenerationProfileSection
    idea_generation: RagGenerationProfileSection


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
    summary_temperature: float
    summary_max_output_tokens: int
    summary_thinking_level: str


@dataclass(frozen=True)
class IndexingStagesSection:
    second_recursive_enabled: bool
    sparse_second_recursive_enabled: bool
    summary_enabled: bool


@dataclass(frozen=True)
class IndexingRefreshSection:
    clear_ingestion_source_data: bool
    clear_first_recursive_chunk_data: bool
    clear_second_recursive_chunk_data: bool
    clear_summary_chunk_data: bool
    update_ingestion_source_data: bool
    update_first_recursive_chunk_data: bool
    update_second_recursive_chunk_data: bool
    update_sparse_second_recursive_chunk_data: bool
    update_summary_chunk_data: bool


@dataclass(frozen=True)
class IndexingEmbeddingCacheSection:
    enabled: bool
    compact_after_publish: bool
    force_reembed_on_full_rebuild: bool


@dataclass(frozen=True)
class IndexingSection:
    chunking: IndexingChunkingSection
    stages: IndexingStagesSection
    refresh: IndexingRefreshSection
    embedding_cache: IndexingEmbeddingCacheSection


@dataclass(frozen=True)
class OpsRagasMetricsSection:
    answer_relevancy_enabled: bool
    faithfulness_enabled: bool
    context_precision_enabled: bool
    context_recall_enabled: bool


@dataclass(frozen=True)
class OpsSection:
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
class EvaluationSection:
    eval_sets_dir: Path
    eval_results_dir: Path
    default_suite: str
    smoke_targets: list[str]
    full_targets: list[str]
    safety_targets: list[str]
    acl_targets: list[str]
    thresholds: dict[str, dict[str, object]]
    safety_zero_tolerance: bool
    fixture_mode: str
    fake_executor: bool
    llm_enabled: bool
    suite_min_cases: dict[str, int]
    missing_eval_set_policy: dict[str, str]


@dataclass(frozen=True)
class SummarizationSection:
    target_characters: int


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
class IntegrationHatenablogSection:
    blog_url: str


@dataclass(frozen=True)
class IntegrationNotionSection:
    api_token: str
    database_ids: list[str]


@dataclass(frozen=True)
class MinecraftWikiCategorySampleSection:
    per_category_limit: int
    categories: dict[str, str]


@dataclass(frozen=True)
class MinecraftWikiQualityGateSection:
    enabled: bool
    min_article_characters: int
    max_redirect_ratio: float
    min_indexable_pages: int
    min_chunk_count: int
    required_canonical_host: str
    policy: str


@dataclass(frozen=True)
class IntegrationMinecraftWikiSection:
    page_titles: list[str]
    api_url: str
    page_url_base: str
    max_pages: int
    rate_limit_per_minute: int
    request_interval_seconds: float
    namespaces: list[int]
    full_backfill_enabled: bool
    acquisition_mode: str
    category_sample: MinecraftWikiCategorySampleSection
    quality_gate: MinecraftWikiQualityGateSection


@dataclass(frozen=True)
class IntegrationSection:
    discord: IntegrationDiscordSection
    openclaw: IntegrationOpenClawSection
    drive: IntegrationDriveSection
    hatenablog: IntegrationHatenablogSection
    crafters_colony: IntegrationCraftersColonySection
    notion: IntegrationNotionSection
    minecraft_wiki: IntegrationMinecraftWikiSection
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
    minutes_edit_temperature: float
    minutes_edit_max_output_tokens: int
    minutes_edit_thinking_level: str
    final_summary_llm_provider: str
    final_summary_gemini_model: str
    final_summary_temperature: float
    final_summary_max_output_tokens: int
    final_summary_thinking_level: str


@dataclass(frozen=True)
class ServerManagementDockerPsSection:
    container_name_prefixes: list[str]


@dataclass(frozen=True)
class ServerManagementServerSection:
    name: str
    compose_dir: Path
    services: list[str]
    allow_file_search_paths: list[Path]
    critical_operations_enabled: bool


@dataclass(frozen=True)
class ServerManagementExecutionSection:
    timeout_seconds: int
    stdout_char_limit: int
    stderr_char_limit: int


@dataclass(frozen=True)
class ServerManagementBackupSection:
    backup_dir: Path
    max_backups: int


@dataclass(frozen=True)
class ServerManagementSection:
    default_server_name: str
    docker_ps: ServerManagementDockerPsSection
    servers: list[ServerManagementServerSection]
    execution: ServerManagementExecutionSection
    backup: ServerManagementBackupSection


@dataclass(frozen=True)
class RuntimeConfig:
    base_dir: Path
    app: AppSection
    providers: ProviderSection
    security: SecuritySection
    scheduler: SchedulerSection
    autonomous_agent: AutonomousAgentSection
    comprehensive_agent: ComprehensiveAgentSection
    task_management: TaskManagementSection
    event_management: EventManagementSection
    infrastructure: InfrastructureSection
    features: FeatureSection
    minecraft_wiki_rag: MinecraftWikiRagSection
    rag: RagSection
    indexing: IndexingSection
    ops: OpsSection
    evaluation: EvaluationSection
    summarization: SummarizationSection
    integrations: IntegrationSection
    model: ModelSection
    vc: VCSection
    server_management: ServerManagementSection

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
