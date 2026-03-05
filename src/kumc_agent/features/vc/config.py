from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.config.schema import RuntimeConfig


@dataclass(frozen=True)
class VCConfig:
    enabled: bool = False


@dataclass(frozen=True)
class VCManagerConfig:
    raw_data_dir: Path
    summery_chunk_dir: Path
    discord_guild_allow_list: tuple[int, ...]
    drive_folder_id: str
    google_application_credentials: str
    gemini_api_key: str
    llama_threads: int
    llama_gpu_layers: int
    vc_feature_enabled: bool
    vc_auto_join_enabled: bool
    vc_auto_join_weekdays: tuple[int, ...]
    vc_auto_join_start_hour: int
    vc_auto_join_start_minute: int
    vc_auto_join_duration_minutes: int
    vc_target_voice_channel_name: str
    vc_auto_join_min_participants: int
    vc_participant_check_interval_seconds: int
    vc_summary_transcribe_interval_seconds: int
    vc_transcribe_model: str
    vc_transcribe_device: str
    vc_transcribe_torch_dtype: str
    vc_transcribe_language: str
    vc_auto_quit_enabled: bool
    vc_final_summary_enabled: bool
    vc_summary_previous_max: int
    vc_summary_target_characters: int
    vc_summary_llm_provider: str
    vc_summary_gemini_model: str
    vc_summary_llama_model: str
    vc_summary_llama_model_path: str
    vc_summary_llama_ctx_size: int
    vc_summary_temperature: float
    vc_summary_max_output_tokens: int
    vc_summary_thinking_level: str
    vc_minutes_enabled: bool
    vc_minutes_drive_dir: str
    vc_minutes_fetch_max_retries: int
    vc_minutes_apply_max_retries: int
    vc_minutes_llm_max_retries: int
    vc_minutes_history_summary_max: int
    vc_minutes_image_batch_size: int
    vc_minutes_edit_llm_provider: str
    vc_minutes_edit_gemini_model: str
    vc_minutes_edit_llama_model: str
    vc_minutes_edit_llama_model_path: str
    vc_minutes_edit_llama_ctx_size: int
    vc_minutes_edit_temperature: float
    vc_minutes_edit_max_output_tokens: int
    vc_minutes_edit_thinking_level: str
    vc_final_summary_llm_provider: str
    vc_final_summary_gemini_model: str
    vc_final_summary_llama_model: str
    vc_final_summary_llama_model_path: str
    vc_final_summary_llama_ctx_size: int
    vc_final_summary_temperature: float
    vc_final_summary_max_output_tokens: int
    vc_final_summary_thinking_level: str

    @classmethod
    def from_runtime(cls, config: RuntimeConfig) -> "VCManagerConfig":
        auto_join_hour, auto_join_minute = _parse_time(config.vc.auto_join_time)
        summary_llama_path = str(config.vc.summary_llama_model_path or "").strip()
        minutes_edit_llama_path = str(config.vc.minutes_edit_llama_model_path or "").strip()
        final_summary_llama_path = str(config.vc.final_summary_llama_model_path or "").strip()

        return cls(
            raw_data_dir=config.app.raw_dir,
            summery_chunk_dir=config.app.data_dir / "chunks" / "summery_chunk",
            discord_guild_allow_list=tuple(config.security.discord_guild_allow_list),
            drive_folder_id=config.integrations.drive.folder_id,
            google_application_credentials=config.integrations.drive.google_application_credentials,
            gemini_api_key=config.integrations.gemini_api_key,
            llama_threads=config.providers.llm.threads,
            llama_gpu_layers=config.providers.llm.gpu_layers,
            vc_feature_enabled=bool(config.features.vc and config.vc.feature_enabled),
            vc_auto_join_enabled=config.vc.auto_join_enabled,
            vc_auto_join_weekdays=tuple(config.vc.auto_join_weekdays),
            vc_auto_join_start_hour=auto_join_hour,
            vc_auto_join_start_minute=auto_join_minute,
            vc_auto_join_duration_minutes=config.vc.auto_join_duration_minutes,
            vc_target_voice_channel_name=config.vc.target_voice_channel_name,
            vc_auto_join_min_participants=config.vc.auto_join_min_participants,
            vc_participant_check_interval_seconds=config.vc.participant_check_interval_seconds,
            vc_summary_transcribe_interval_seconds=config.vc.summary_transcribe_interval_seconds,
            vc_transcribe_model=config.vc.transcribe_model,
            vc_transcribe_device=config.vc.transcribe_device,
            vc_transcribe_torch_dtype=config.vc.transcribe_torch_dtype,
            vc_transcribe_language=config.vc.transcribe_language,
            vc_auto_quit_enabled=config.vc.auto_quit_enabled,
            vc_final_summary_enabled=config.vc.final_summary_enabled,
            vc_summary_previous_max=config.vc.summary_previous_max,
            vc_summary_target_characters=config.vc.summary_target_characters,
            vc_summary_llm_provider=config.vc.summary_llm_provider,
            vc_summary_gemini_model=config.vc.summary_gemini_model,
            vc_summary_llama_model=_llama_model_name(summary_llama_path),
            vc_summary_llama_model_path=summary_llama_path,
            vc_summary_llama_ctx_size=config.vc.summary_llama_ctx_size,
            vc_summary_temperature=config.vc.summary_temperature,
            vc_summary_max_output_tokens=config.vc.summary_max_output_tokens,
            vc_summary_thinking_level=config.vc.summary_thinking_level,
            vc_minutes_enabled=config.vc.minutes_enabled,
            vc_minutes_drive_dir=config.vc.minutes_drive_dir,
            vc_minutes_fetch_max_retries=config.vc.minutes_fetch_max_retries,
            vc_minutes_apply_max_retries=config.vc.minutes_apply_max_retries,
            vc_minutes_llm_max_retries=config.vc.minutes_llm_max_retries,
            vc_minutes_history_summary_max=config.vc.minutes_history_summary_max,
            vc_minutes_image_batch_size=config.vc.minutes_image_batch_size,
            vc_minutes_edit_llm_provider=config.vc.minutes_edit_llm_provider,
            vc_minutes_edit_gemini_model=config.vc.minutes_edit_gemini_model,
            vc_minutes_edit_llama_model=_llama_model_name(minutes_edit_llama_path),
            vc_minutes_edit_llama_model_path=minutes_edit_llama_path,
            vc_minutes_edit_llama_ctx_size=config.vc.minutes_edit_llama_ctx_size,
            vc_minutes_edit_temperature=config.vc.minutes_edit_temperature,
            vc_minutes_edit_max_output_tokens=config.vc.minutes_edit_max_output_tokens,
            vc_minutes_edit_thinking_level=config.vc.minutes_edit_thinking_level,
            vc_final_summary_llm_provider=config.vc.final_summary_llm_provider,
            vc_final_summary_gemini_model=config.vc.final_summary_gemini_model,
            vc_final_summary_llama_model=_llama_model_name(final_summary_llama_path),
            vc_final_summary_llama_model_path=final_summary_llama_path,
            vc_final_summary_llama_ctx_size=config.vc.final_summary_llama_ctx_size,
            vc_final_summary_temperature=config.vc.final_summary_temperature,
            vc_final_summary_max_output_tokens=config.vc.final_summary_max_output_tokens,
            vc_final_summary_thinking_level=config.vc.final_summary_thinking_level,
        )


def _parse_time(value: str) -> tuple[int, int]:
    raw = (value or "").strip()
    if ":" not in raw:
        return 20, 0
    hour_str, minute_str = raw.split(":", 1)
    try:
        hour = int(hour_str)
        minute = int(minute_str)
    except ValueError:
        return 20, 0
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return 20, 0
    return hour, minute


def _llama_model_name(path: str) -> str:
    cleaned = (path or "").strip()
    if not cleaned:
        return ""
    return Path(cleaned).name or cleaned
