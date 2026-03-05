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
    rerank_pool_size: int
    mmr_lambda: float


@dataclass(frozen=True)
class SourcesSection:
    drive: bool
    discord: bool
    hatenablog: bool
    crafters_colony: bool


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
class IntegrationDiscordSection:
    bot_token: str


@dataclass(frozen=True)
class IntegrationDriveSection:
    folder_id: str
    google_application_credentials: str
    max_files: int


@dataclass(frozen=True)
class IntegrationCraftersColonySection:
    author_url: str
    max_pages: int
    max_articles: int


@dataclass(frozen=True)
class IntegrationSection:
    discord: IntegrationDiscordSection
    drive: IntegrationDriveSection
    crafters_colony: IntegrationCraftersColonySection
    gemini_api_key: str


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
