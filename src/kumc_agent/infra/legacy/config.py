from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from langchain_core.embeddings import Embeddings
from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit

## コンフィグ ##
# Embedding Model Settings
DEFAULT_EMBEDDING_MODEL: str = "embeddinggemma-300M-Q8_0.gguf"
DEFAULT_CROSS_ENCODER_MODEL: str = ""
DEFAULT_EMBEDDING_MODEL_DIR: str = "app/model/embedding"
DEFAULT_CROSS_ENCODER_MODEL_DIR: str = "app/model/cross-encoder"
DEFAULT_WHISPER_MODEL_DIR: str = "app/model/whisper"
DEFAULT_PDF_OCR_MODEL: str = "app/model/ocr/tencent/HunyuanOCR"

# Answering LLM Settings
DEFAULT_LLM_PROVIDER: str = "gemini"
DEFAULT_GENAI_MODEL: str = "gemini-3-flash-preview" # gemini
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_THINKING_LEVEL: str = "minimal"
DEFAULT_GEMINI_REQUESTS_PER_MINUTE: int = 60
DEFAULT_GEMINI_SUMMARY_REQUESTS_PER_MINUTE: int = DEFAULT_GEMINI_REQUESTS_PER_MINUTE
DEFAULT_MAX_OUTPUT_TOKENS: int = 512
DEFAULT_CHAT_HISTORY_ENABLED: bool = False
DEFAULT_CHAT_HISTORY_MAX_TURNS: int = 5
DEFAULT_PROMPT_HISTORY_DEFAULT_TURNS: int = 3
DEFAULT_PROMPT_HISTORY_ADDITIONAL_TURNS: int = 10
DEFAULT_CHATBOT_CAPABILITIES_INFO: str = ""
DEFAULT_CIRCLE_BASIC_INFO: str = ""

_REQUIRED_PROMPT_ENV_NAMES: tuple[str, ...] = (
    "PROMPT_CHATBOT_CAPABILITIES_INFO",
    "PROMPT_CIRCLE_BASIC_INFO",
    "PROMPT_SYSTEM_RULES",
    "PROMPT_LLM_CHUNK_SYSTEM_PROMPT",
    "PROMPT_SUMMERY_CHUNK_MESSAGES_TEMPLATE",
    "PROMPT_SUMMERY_CHUNK_SHEETS_TEMPLATE",
    "PROMPT_SUMMERY_CHUNK_DEFAULT_TEMPLATE",
    "PROMPT_OUTPUT_INSTRUCTIONS_COMMON",
    "PROMPT_OUTPUT_INSTRUCTIONS_RAG",
    "PROMPT_OUTPUT_INSTRUCTIONS_RAG_IDEA",
    "PROMPT_OUTPUT_INSTRUCTIONS_NO_RAG",
    "PROMPT_OUTPUT_INSTRUCTIONS_REFUSAL",
    "PROMPT_MODE_INSTRUCTIONS_RAG",
    "PROMPT_MODE_INSTRUCTIONS_RAG_IDEA",
    "PROMPT_MODE_INSTRUCTIONS_NO_RAG",
    "PROMPT_MODE_INSTRUCTIONS_REFUSAL",
    "PROMPT_EMPTY_CONTEXT",
    "PROMPT_EMPTY_HISTORY",
    "PROMPT_HISTORY_USER_PREFIX",
    "PROMPT_HISTORY_ASSISTANT_PREFIX",
    "PROMPT_HISTORY_SOURCES_LABEL",
    "PROMPT_GEMINI_HEADER_CHAT_HISTORY",
    "PROMPT_GEMINI_HEADER_RETRY_HISTORY",
    "PROMPT_GEMINI_HEADER_CIRCLE_INFO",
    "PROMPT_GEMINI_HEADER_CAPABILITIES",
    "PROMPT_GEMINI_HEADER_CONTEXT",
    "PROMPT_GEMINI_HEADER_OUTPUT_FORMAT",
    "PROMPT_GEMINI_HEADER_INSTRUCTIONS",
    "PROMPT_GEMINI_HEADER_QUESTION",
    "PROMPT_MATERIAL_SEARCH_SELECTOR_SYSTEM",
    "PROMPT_MATERIAL_SEARCH_SELECTOR_USER_TEMPLATE",
)


def _decode_prompt_env_value(value: str) -> str:
    return value.replace("\\n", "\n")


@lru_cache(maxsize=None)
def get_required_prompt_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required prompt environment variable: {name}")
    return _decode_prompt_env_value(value.strip())


def ensure_required_prompt_envs() -> None:
    for env_name in _REQUIRED_PROMPT_ENV_NAMES:
        get_required_prompt_env(env_name)


def _parse_system_rule_templates(raw: str) -> tuple[str, ...]:
    parts = [part.strip() for part in raw.split("||") if part.strip()]
    if not parts:
        parts = [raw.strip()]
    templates = tuple(part for part in parts if part)
    if not templates:
        raise RuntimeError("PROMPT_SYSTEM_RULES must contain at least one rule.")
    return templates


def _jst_today_label() -> str:
    today = datetime.now(ZoneInfo("Asia/Tokyo"))
    weekday = ["月", "火", "水", "木", "金", "土", "日"][today.weekday()]
    return today.strftime("%Y年%m月%d日") + f"（{weekday}）"


def _build_default_system_rules(today_label: str) -> tuple[str, ...]:
    templates = _parse_system_rule_templates(
        get_required_prompt_env("PROMPT_SYSTEM_RULES")
    )
    rules: list[str] = []
    for template in templates:
        try:
            rules.append(template.format(today_label=today_label))
        except KeyError as exc:
            placeholder = exc.args[0]
            raise RuntimeError(
                "PROMPT_SYSTEM_RULES contains unsupported placeholder: "
                f"{placeholder}"
            ) from exc
    return tuple(rules)


class _DailySystemRules(Sequence[str]):
    def __init__(self) -> None:
        self._cached_label: str | None = None
        self._cached_rules: tuple[str, ...] = tuple()

    def _current_rules(self) -> tuple[str, ...]:
        today_label = _jst_today_label()
        if today_label != self._cached_label:
            self._cached_label = today_label
            self._cached_rules = _build_default_system_rules(today_label)
        return self._cached_rules

    def __iter__(self):
        return iter(self._current_rules())

    def __len__(self) -> int:
        return len(self._current_rules())

    def __getitem__(self, index):
        return self._current_rules()[index]


DEFAULT_SYSTEM_RULES: Sequence[str] = _DailySystemRules()

# No-RAG Answer LLM Settings
DEFAULT_NO_RAG_LLM_PROVIDER: str = DEFAULT_LLM_PROVIDER
DEFAULT_NO_RAG_GENAI_MODEL: str = DEFAULT_GENAI_MODEL
DEFAULT_NO_RAG_TEMPERATURE: float = DEFAULT_TEMPERATURE
DEFAULT_NO_RAG_MAX_OUTPUT_TOKENS: int = DEFAULT_MAX_OUTPUT_TOKENS
DEFAULT_NO_RAG_THINKING_LEVEL: str = DEFAULT_THINKING_LEVEL

# Refusal Answer LLM Settings
DEFAULT_REFUSAL_LLM_PROVIDER: str = DEFAULT_NO_RAG_LLM_PROVIDER
DEFAULT_REFUSAL_GENAI_MODEL: str = DEFAULT_NO_RAG_GENAI_MODEL
DEFAULT_REFUSAL_TEMPERATURE: float = DEFAULT_NO_RAG_TEMPERATURE
DEFAULT_REFUSAL_MAX_OUTPUT_TOKENS: int = DEFAULT_NO_RAG_MAX_OUTPUT_TOKENS
DEFAULT_REFUSAL_THINKING_LEVEL: str = DEFAULT_NO_RAG_THINKING_LEVEL

# Function Calling (RAG routing) Settings
DEFAULT_FUNCTION_CALL_PROVIDER: str = "gemini"
DEFAULT_FUNCTION_CALL_GEMINI_MODEL: str = DEFAULT_GENAI_MODEL
DEFAULT_FUNCTION_CALL_TEMPERATURE: float = 0.0
DEFAULT_FUNCTION_CALL_MAX_NEW_TOKENS: int = 64
DEFAULT_FUNCTION_CALL_MAX_RETRIES: int = 2
DEFAULT_FUNCTION_CALL_ENABLED: bool = True
DEFAULT_FUNCTION_CALL_LOG_ENABLED: bool = False
DEFAULT_RAG_IDEA_TEMPERATURE: float = 0.8
DEFAULT_MATERIAL_SEARCH_MAX_NAMES: int = 3
DEFAULT_MATERIAL_SEARCH_PARTIAL_MATCH_SEMANTIC_TOP_K: int = 1
DEFAULT_MATERIAL_SEARCH_CHAR_LIMIT: int = 3000
DEFAULT_MATERIAL_SEARCH_MAX_SELECTED_SUMMARY_CHUNKS: int = 3
DEFAULT_MATERIAL_SEARCH_SUMMARY_MISSING_FIRST_REC_TOP_K: int = 3
DEFAULT_MATERIAL_SEARCH_SELECTOR_MAX_RETRIES: int = 2
DEFAULT_MATERIAL_SEARCH_SELECTOR_LLM_PROVIDER: str = DEFAULT_LLM_PROVIDER
DEFAULT_MATERIAL_SEARCH_SELECTOR_GEMINI_MODEL: str = DEFAULT_GENAI_MODEL
DEFAULT_MATERIAL_SEARCH_SELECTOR_TEMPERATURE: float = 0.0
DEFAULT_MATERIAL_SEARCH_SELECTOR_MAX_OUTPUT_TOKENS: int = 128
DEFAULT_MATERIAL_SEARCH_SELECTOR_THINKING_LEVEL: str = DEFAULT_THINKING_LEVEL

# First Recursive Chunking Settings
DEFAULT_FIRST_REC_CHUNK_SIZE: int = 1024
DEFAULT_FIRST_REC_CHUNK_OVERLAP: int = 128

# Second Recursive Chunking Settings
DEFAULT_SECOND_REC_ENABLED: bool = True
DEFAULT_SECOND_REC_CHUNK_SIZE: int = 128
DEFAULT_SECOND_REC_CHUNK_OVERLAP: int = 32

# Summery Chunking Settings
DEFAULT_SUMMERY_ENABLED: bool = True
DEFAULT_SUMMERY_CHARACTERS: int = 200
DEFAULT_SUMMERY_PROVIDER: str = "gemini"
DEFAULT_SUMMERY_GEMINI_MODEL: str = "gemini-3-flash-preview"
DEFAULT_SUMMERY_MAX_OUTPUT_TOKENS: int = 1024
DEFAULT_SUMMERY_TEMPERATURE: float = 0.2
DEFAULT_SUMMERY_MAX_RETRIES: int = 2
DEFAULT_SUMMERY_BATCH_SIZE: int = 1

def get_llm_chunk_system_prompt() -> str:
    return get_required_prompt_env("PROMPT_LLM_CHUNK_SYSTEM_PROMPT")

# Clear Data Settings
DEFAULT_CLEAR_RAW_DATA: bool = False
DEFAULT_CLEAR_FIRST_REC_CHUNK_DATA: bool = False
DEFAULT_CLEAR_SECOND_REC_CHUNK_DATA: bool = False
DEFAULT_CLEAR_SUMMERY_CHUNK_DATA: bool = False

# Incremental Update Settings
DEFAULT_UPDATE_RAW_DATA: bool = True
DEFAULT_UPDATE_FIRST_REC_CHUNK_DATA: bool = True
DEFAULT_UPDATE_SECOND_REC_CHUNK_DATA: bool = True
DEFAULT_UPDATE_SPARSE_SECOND_REC_CHUNK_DATA: bool = True
DEFAULT_UPDATE_SUMMERY_CHUNK_DATA: bool = True

# Retrieval Settings
DEFAULT_TOP_K: int = 5
DEFAULT_DENSE_SEARCH_TOP_K: int = 20
DEFAULT_SPARSE_SEARCH_TOP_K: int = 20
DEFAULT_SPARSE_SEARCH_ORIGINAL_TOP_K: int = DEFAULT_SPARSE_SEARCH_TOP_K
DEFAULT_SPARSE_SEARCH_TRANSFORM_TOP_K: int = DEFAULT_SPARSE_SEARCH_TOP_K
DEFAULT_SPARSE_SEARCH_INITIAL_SPARSE_TOP_K: int = DEFAULT_SPARSE_SEARCH_TOP_K
DEFAULT_SPARSE_SEARCH_ORIGINAL_SPARSE_TOP_K: int = (
    DEFAULT_SPARSE_SEARCH_ORIGINAL_TOP_K
)
DEFAULT_PARENT_DOC_ENABLED: bool = True
DEFAULT_PARENT_CHUNK_CAP: int = 2
DEFAULT_RERANK_ENABLED: bool = True
DEFAULT_RERANK_POOL_SIZE: int = 20
DEFAULT_RECENCY_WEIGHT_SOFT: float = 0.3
DEFAULT_RECENCY_WEIGHT_HARD: float = 0.8
DEFAULT_RECENCY_HALF_LIFE_DAYS: float = 30.0
DEFAULT_MMR_LAMBDA: float = 0.5
DEFAULT_SUDACHI_MODE: str = "B"
DEFAULT_SPARSE_BM25_K1: float = 1.5
DEFAULT_SPARSE_BM25_B: float = 0.75
DEFAULT_SPARSE_USE_NORMALIZED_FORM: bool = True
DEFAULT_SPARSE_REMOVE_SYMBOLS: bool = True
DEFAULT_SOURCE_MAX_COUNT: int = 3
DEFAULT_ANSWER_JSON_MAX_RETRIES: int = 2
DEFAULT_ANSWER_RESEARCH_MAX_RETRIES: int = 3
DEFAULT_EVAL_ANSWER_RELEVANCY_ENABLED: bool = True
DEFAULT_EVAL_FAITHFULNESS_ENABLED: bool = True
DEFAULT_EVAL_CONTEXT_PRECISION_ENABLED: bool = True
DEFAULT_EVAL_CONTEXT_RECALL_ENABLED: bool = True

# Google Drive Settings
DEFAULT_DRIVE_MAX_FILES: int = 0
DEFAULT_CRAFTERS_COLONY_AUTHOR_URL: str = (
    "https://minecraft-mcworld.com/author/2937761467834624754e30c1ed9db1390dc5f974/"
)
DEFAULT_CRAFTERS_COLONY_MAX_PAGES: int = 100
DEFAULT_CRAFTERS_COLONY_MAX_ARTICLES: int = 0

# Command Prefix
DEFAULT_COMMAND_PREFIX: str = "/ai "
DEFAULT_INDEX_COMMAND_PREFIX: str = "/ai build-index"
DEFAULT_MAINTENANCE_COMMAND_AUTHOR_IDS: str = ""
DEFAULT_AUTO_INDEX_ENABLED: bool = False
DEFAULT_AUTO_INDEX_TIME: str = "03:00"
DEFAULT_AUTO_INDEX_WEEKDAYS: str = "mon,tue,wed,thu,fri"
DEFAULT_WARMUP_INTERVAL_MINUTES: int = 60
DEFAULT_INDEX_UPDATE_ESTIMATE_MIN_MINUTES: int = 30
DEFAULT_INDEX_UPDATE_ESTIMATE_MAX_MINUTES: int = 60
DEFAULT_DISCORD_GUILD_ALLOW_LIST: str = ""
DEFAULT_MAX_INPUT_CHARACTERS: int = 0
DEFAULT_PROMPT_FULL_LOG_ENABLED: bool = True
DEFAULT_SPECIAL_CHANNEL_HISTORY_LIMIT: int = 30
DEFAULT_SPECIAL_CHANNEL_CUSTOM_INSTRUCTION: str = ""
DEFAULT_ANSWER_RECORD_LOG_ENABLED: bool = True
DEFAULT_ANSWER_RECORD_LOG_PATH: str = "logs/answer_records.jsonl"

# VC Meeting Settings
DEFAULT_VC_FEATURE_ENABLED: bool = False
DEFAULT_VC_AUTO_JOIN_ENABLED: bool = False
DEFAULT_VC_AUTO_JOIN_WEEKDAYS: str = "sat"
DEFAULT_VC_AUTO_JOIN_TIME: str = "20:00"
DEFAULT_VC_AUTO_JOIN_DURATION_MINUTES: int = 30
DEFAULT_VC_TARGET_VOICE_CHANNEL_NAME: str = "例会"
DEFAULT_VC_AUTO_JOIN_MIN_PARTICIPANTS: int = 3
DEFAULT_VC_PARTICIPANT_CHECK_INTERVAL_SECONDS: int = 10
DEFAULT_VC_SUMMARY_TRANSCRIBE_INTERVAL_SECONDS: int = 300
DEFAULT_VC_TRANSCRIBE_MODEL: str = "kotoba-tech/kotoba-whisper-v2.2"
DEFAULT_VC_TRANSCRIBE_DEVICE: str = "auto"
DEFAULT_VC_TRANSCRIBE_TORCH_DTYPE: str = "auto"
DEFAULT_VC_TRANSCRIBE_LANGUAGE: str = "ja"
DEFAULT_VC_AUTO_QUIT_ENABLED: bool = True
DEFAULT_VC_FINAL_SUMMARY_ENABLED: bool = True
DEFAULT_VC_SUMMARY_PREVIOUS_MAX: int = 2
DEFAULT_VC_SUMMARY_TARGET_CHARACTERS: int = 100
DEFAULT_VC_SUMMARY_LLM_PROVIDER: str = DEFAULT_LLM_PROVIDER
DEFAULT_VC_SUMMARY_GEMINI_MODEL: str = DEFAULT_GENAI_MODEL
DEFAULT_VC_SUMMARY_TEMPERATURE: float = 0.2
DEFAULT_VC_SUMMARY_MAX_OUTPUT_TOKENS: int = 256
DEFAULT_VC_SUMMARY_THINKING_LEVEL: str = DEFAULT_THINKING_LEVEL
DEFAULT_VC_MINUTES_ENABLED: bool = True
DEFAULT_VC_MINUTES_DRIVE_DIR: str = "議事録"
DEFAULT_VC_MINUTES_FETCH_MAX_RETRIES: int = 2
DEFAULT_VC_MINUTES_APPLY_MAX_RETRIES: int = 2
DEFAULT_VC_MINUTES_LLM_MAX_RETRIES: int = 2
DEFAULT_VC_MINUTES_HISTORY_SUMMARY_MAX: int = 2
DEFAULT_VC_MINUTES_IMAGE_BATCH_SIZE: int = 10
DEFAULT_VC_MINUTES_EDIT_LLM_PROVIDER: str = DEFAULT_LLM_PROVIDER
DEFAULT_VC_MINUTES_EDIT_GEMINI_MODEL: str = DEFAULT_GENAI_MODEL
DEFAULT_VC_MINUTES_EDIT_TEMPERATURE: float = 0.2
DEFAULT_VC_MINUTES_EDIT_MAX_OUTPUT_TOKENS: int = 512
DEFAULT_VC_MINUTES_EDIT_THINKING_LEVEL: str = DEFAULT_THINKING_LEVEL
DEFAULT_VC_FINAL_SUMMARY_LLM_PROVIDER: str = DEFAULT_LLM_PROVIDER
DEFAULT_VC_FINAL_SUMMARY_GEMINI_MODEL: str = DEFAULT_GENAI_MODEL
DEFAULT_VC_FINAL_SUMMARY_TEMPERATURE: float = 0.2
DEFAULT_VC_FINAL_SUMMARY_MAX_OUTPUT_TOKENS: int = 512
DEFAULT_VC_FINAL_SUMMARY_THINKING_LEVEL: str = DEFAULT_THINKING_LEVEL



def _env_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_time(value: str | None, *, default: str) -> tuple[int, int]:
    raw = (value if value is not None else default).strip()
    try:
        hour_str, minute_str = raw.split(":", maxsplit=1)
        hour = int(hour_str)
        minute = int(minute_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid AUTO_INDEX_TIME '{raw}'. Expected HH:MM in 24h format."
        ) from exc

    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(
            f"Invalid AUTO_INDEX_TIME '{raw}'. Expected HH:MM in 24h format."
        )
    return hour, minute


def _parse_weekdays(value: str | None, *, default: str) -> tuple[int, ...]:
    raw = (value if value is not None else default).strip()
    if not raw:
        return tuple()

    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if any(token in {"*", "all", "every"} for token in tokens):
        return (0, 1, 2, 3, 4, 5, 6)

    weekday_map = {
        "mon": 0,
        "tue": 1,
        "wed": 2,
        "thu": 3,
        "fri": 4,
        "sat": 5,
        "sun": 6,
    }
    weekdays: list[int] = []
    for token in tokens:
        if token.isdigit():
            value_int = int(token)
            if value_int < 0 or value_int > 6:
                raise ValueError(
                    f"Invalid AUTO_INDEX_WEEKDAYS entry '{token}'. "
                    "Use 0-6 or mon-sun."
                )
            weekdays.append(value_int)
            continue
        key = token[:3]
        if key not in weekday_map:
            raise ValueError(
                f"Invalid AUTO_INDEX_WEEKDAYS entry '{token}'. "
                "Use 0-6 or mon-sun."
            )
        weekdays.append(weekday_map[key])

    deduped: list[int] = []
    seen = set()
    for day in weekdays:
        if day in seen:
            continue
        seen.add(day)
        deduped.append(day)
    return tuple(deduped)


def _parse_id_list(
    value: str | None,
    *,
    default: str,
    env_name: str = "DISCORD_GUILD_ALLOW_LIST",
) -> tuple[int, ...]:
    raw = (value if value is not None else default).strip()
    if not raw:
        return tuple()
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    ids: list[int] = []
    for token in tokens:
        if not token.isdigit():
            raise ValueError(
                f"Invalid {env_name} entry '{token}'. "
                "Use comma-separated numeric IDs."
            )
        ids.append(int(token))
    deduped: list[int] = []
    seen = set()
    for value_int in ids:
        if value_int in seen:
            continue
        seen.add(value_int)
        deduped.append(value_int)
    return tuple(deduped)


def _resolve_dir(path_value: str, *, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        return base_dir / path
    return path


def _resolve_model_path(
    *,
    model_name: str,
    model_dir: Path,
    base_dir: Path,
) -> str:
    if not model_name:
        return ""
    normalized = model_name.strip()
    lowered = normalized.lower()
    if lowered.startswith("gemini:") or lowered.startswith("gemini/"):
        return normalized
    path = Path(model_name)
    if path.is_absolute():
        return str(path)
    if "/" in model_name or "\\" in model_name:
        if model_name.startswith((".", "~", "app/", "app\\")):
            return str(base_dir / path)
        candidate = model_dir / path
        if candidate.exists():
            return str(candidate)
        base_candidate = base_dir / path
        if base_candidate.exists():
            return str(base_candidate)
        return model_name
    if path.parent != Path("."):
        return str(base_dir / path)
    return str(model_dir / path)


def _resolve_local_model_path(
    *,
    model_name: str,
    model_dir: Path,
    base_dir: Path,
) -> str:
    if not model_name:
        return ""
    path = Path(model_name).expanduser()
    if path.is_absolute():
        return str(path)
    if model_name.startswith(("app/", "app\\", ".", "..")):
        return str(base_dir / path)
    return str(model_dir / path)


def render_prompt_template(template_env_name: str, **kwargs: object) -> str:
    template = get_required_prompt_env(template_env_name)
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        placeholder = exc.args[0]
        raise RuntimeError(
            f"{template_env_name} is missing placeholder value: {placeholder}"
        ) from exc


def build_summery_chunk_prompt(
    *,
    text: str,
    target_characters: int,
    source_type: str | None = None,
    drive_file_path: str | None = None,
) -> str:
    normalized_type = (source_type or "").strip().lower()
    drive_path = (drive_file_path or "").strip()
    drive_path_display = drive_path if drive_path else "不明"

    if normalized_type in {"messages", "discord_message", "x_posts"}:
        return render_prompt_template(
            "PROMPT_SUMMERY_CHUNK_MESSAGES_TEMPLATE",
            target_characters=target_characters,
            text=text,
        )

    if normalized_type == "sheets":
        return render_prompt_template(
            "PROMPT_SUMMERY_CHUNK_SHEETS_TEMPLATE",
            target_characters=target_characters,
            drive_path_display=drive_path_display,
            text=text,
        )

    return render_prompt_template(
        "PROMPT_SUMMERY_CHUNK_DEFAULT_TEMPLATE",
        target_characters=target_characters,
        drive_path_display=drive_path_display,
        text=text,
    )


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    raw_data_dir: Path
    first_rec_chunk_dir: Path
    second_rec_chunk_dir: Path
    sparse_second_rec_chunk_dir: Path
    summery_chunk_dir: Path
    index_dir: Path
    discord_bot_token: str = ""
    discord_guild_allow_list: tuple[int, ...] = ()
    maintenance_command_author_ids: tuple[int, ...] = ()
    gemini_api_key: str = ""
    gemini_requests_per_minute: int = DEFAULT_GEMINI_REQUESTS_PER_MINUTE
    gemini_summary_requests_per_minute: int = (
        DEFAULT_GEMINI_SUMMARY_REQUESTS_PER_MINUTE
    )
    drive_folder_id: str = ""
    google_application_credentials: str = ""
    drive_max_files: int = DEFAULT_DRIVE_MAX_FILES
    crafters_colony_author_url: str = DEFAULT_CRAFTERS_COLONY_AUTHOR_URL
    crafters_colony_max_pages: int = DEFAULT_CRAFTERS_COLONY_MAX_PAGES
    crafters_colony_max_articles: int = DEFAULT_CRAFTERS_COLONY_MAX_ARTICLES
    pdf_ocr_model_path: str = DEFAULT_PDF_OCR_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    cross_encoder_model_path: str = DEFAULT_CROSS_ENCODER_MODEL
    first_rec_chunk_size: int = DEFAULT_FIRST_REC_CHUNK_SIZE
    first_rec_chunk_overlap: int = DEFAULT_FIRST_REC_CHUNK_OVERLAP
    second_rec_enabled: bool = DEFAULT_SECOND_REC_ENABLED
    second_rec_chunk_size: int = DEFAULT_SECOND_REC_CHUNK_SIZE
    second_rec_chunk_overlap: int = DEFAULT_SECOND_REC_CHUNK_OVERLAP
    summery_enabled: bool = DEFAULT_SUMMERY_ENABLED
    summery_characters: int = DEFAULT_SUMMERY_CHARACTERS
    summery_provider: str = DEFAULT_SUMMERY_PROVIDER
    summery_gemini_model: str = DEFAULT_SUMMERY_GEMINI_MODEL
    summery_temperature: float = DEFAULT_SUMMERY_TEMPERATURE
    summery_max_output_tokens: int = DEFAULT_SUMMERY_MAX_OUTPUT_TOKENS
    summery_max_retries: int = DEFAULT_SUMMERY_MAX_RETRIES
    summery_batch_size: int = DEFAULT_SUMMERY_BATCH_SIZE
    llm_provider: str = DEFAULT_LLM_PROVIDER
    genai_model: str = DEFAULT_GENAI_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    thinking_level: str = DEFAULT_THINKING_LEVEL
    no_rag_llm_provider: str = DEFAULT_NO_RAG_LLM_PROVIDER
    no_rag_genai_model: str = DEFAULT_NO_RAG_GENAI_MODEL
    no_rag_temperature: float = DEFAULT_NO_RAG_TEMPERATURE
    no_rag_max_output_tokens: int = DEFAULT_NO_RAG_MAX_OUTPUT_TOKENS
    no_rag_thinking_level: str = DEFAULT_NO_RAG_THINKING_LEVEL
    refusal_llm_provider: str = DEFAULT_REFUSAL_LLM_PROVIDER
    refusal_genai_model: str = DEFAULT_REFUSAL_GENAI_MODEL
    refusal_temperature: float = DEFAULT_REFUSAL_TEMPERATURE
    refusal_max_output_tokens: int = DEFAULT_REFUSAL_MAX_OUTPUT_TOKENS
    refusal_thinking_level: str = DEFAULT_REFUSAL_THINKING_LEVEL
    function_call_provider: str = DEFAULT_FUNCTION_CALL_PROVIDER
    function_call_gemini_model: str = DEFAULT_FUNCTION_CALL_GEMINI_MODEL
    function_call_temperature: float = DEFAULT_FUNCTION_CALL_TEMPERATURE
    function_call_max_new_tokens: int = DEFAULT_FUNCTION_CALL_MAX_NEW_TOKENS
    function_call_max_retries: int = DEFAULT_FUNCTION_CALL_MAX_RETRIES
    function_call_enabled: bool = DEFAULT_FUNCTION_CALL_ENABLED
    function_call_log_enabled: bool = DEFAULT_FUNCTION_CALL_LOG_ENABLED
    material_search_max_names: int = DEFAULT_MATERIAL_SEARCH_MAX_NAMES
    material_search_partial_match_semantic_top_k: int = (
        DEFAULT_MATERIAL_SEARCH_PARTIAL_MATCH_SEMANTIC_TOP_K
    )
    material_search_char_limit: int = DEFAULT_MATERIAL_SEARCH_CHAR_LIMIT
    material_search_max_selected_summary_chunks: int = (
        DEFAULT_MATERIAL_SEARCH_MAX_SELECTED_SUMMARY_CHUNKS
    )
    material_search_summary_missing_first_rec_top_k: int = (
        DEFAULT_MATERIAL_SEARCH_SUMMARY_MISSING_FIRST_REC_TOP_K
    )
    material_search_selector_max_retries: int = (
        DEFAULT_MATERIAL_SEARCH_SELECTOR_MAX_RETRIES
    )
    material_search_selector_llm_provider: str = (
        DEFAULT_MATERIAL_SEARCH_SELECTOR_LLM_PROVIDER
    )
    material_search_selector_gemini_model: str = (
        DEFAULT_MATERIAL_SEARCH_SELECTOR_GEMINI_MODEL
    )
    material_search_selector_temperature: float = (
        DEFAULT_MATERIAL_SEARCH_SELECTOR_TEMPERATURE
    )
    material_search_selector_max_output_tokens: int = (
        DEFAULT_MATERIAL_SEARCH_SELECTOR_MAX_OUTPUT_TOKENS
    )
    material_search_selector_thinking_level: str = (
        DEFAULT_MATERIAL_SEARCH_SELECTOR_THINKING_LEVEL
    )
    rag_idea_temperature: float = DEFAULT_RAG_IDEA_TEMPERATURE
    chat_history_enabled: bool = DEFAULT_CHAT_HISTORY_ENABLED
    chat_history_max_turns: int = DEFAULT_CHAT_HISTORY_MAX_TURNS
    prompt_history_default_turns: int = DEFAULT_PROMPT_HISTORY_DEFAULT_TURNS
    prompt_history_additional_turns: int = (
        DEFAULT_PROMPT_HISTORY_ADDITIONAL_TURNS
    )
    chatbot_capabilities_info: str = DEFAULT_CHATBOT_CAPABILITIES_INFO
    circle_basic_info: str = DEFAULT_CIRCLE_BASIC_INFO
    top_k: int = DEFAULT_TOP_K
    dense_search_top_k: int = DEFAULT_DENSE_SEARCH_TOP_K
    sparse_search_top_k: int = DEFAULT_SPARSE_SEARCH_TOP_K
    sparse_search_original_top_k: int = DEFAULT_SPARSE_SEARCH_ORIGINAL_TOP_K
    sparse_search_transform_top_k: int = DEFAULT_SPARSE_SEARCH_TRANSFORM_TOP_K
    sparse_search_initial_sparse_top_k: int = (
        DEFAULT_SPARSE_SEARCH_INITIAL_SPARSE_TOP_K
    )
    sparse_search_original_sparse_top_k: int = (
        DEFAULT_SPARSE_SEARCH_ORIGINAL_SPARSE_TOP_K
    )
    parent_doc_enabled: bool = DEFAULT_PARENT_DOC_ENABLED
    parent_chunk_cap: int = DEFAULT_PARENT_CHUNK_CAP
    rerank_enabled: bool = DEFAULT_RERANK_ENABLED
    rerank_pool_size: int = DEFAULT_RERANK_POOL_SIZE
    recency_weight_soft: float = DEFAULT_RECENCY_WEIGHT_SOFT
    recency_weight_hard: float = DEFAULT_RECENCY_WEIGHT_HARD
    recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    sudachi_mode: str = DEFAULT_SUDACHI_MODE
    sparse_bm25_k1: float = DEFAULT_SPARSE_BM25_K1
    sparse_bm25_b: float = DEFAULT_SPARSE_BM25_B
    sparse_use_normalized_form: bool = DEFAULT_SPARSE_USE_NORMALIZED_FORM
    sparse_remove_symbols: bool = DEFAULT_SPARSE_REMOVE_SYMBOLS
    source_max_count: int = DEFAULT_SOURCE_MAX_COUNT
    answer_json_max_retries: int = DEFAULT_ANSWER_JSON_MAX_RETRIES
    answer_research_max_retries: int = DEFAULT_ANSWER_RESEARCH_MAX_RETRIES
    eval_answer_relevancy_enabled: bool = (
        DEFAULT_EVAL_ANSWER_RELEVANCY_ENABLED
    )
    eval_faithfulness_enabled: bool = DEFAULT_EVAL_FAITHFULNESS_ENABLED
    eval_context_precision_enabled: bool = (
        DEFAULT_EVAL_CONTEXT_PRECISION_ENABLED
    )
    eval_context_recall_enabled: bool = DEFAULT_EVAL_CONTEXT_RECALL_ENABLED
    max_input_characters: int = DEFAULT_MAX_INPUT_CHARACTERS
    prompt_full_log_enabled: bool = DEFAULT_PROMPT_FULL_LOG_ENABLED
    special_channel_history_limit: int = DEFAULT_SPECIAL_CHANNEL_HISTORY_LIMIT
    special_channel_custom_instruction: str = (
        DEFAULT_SPECIAL_CHANNEL_CUSTOM_INSTRUCTION
    )
    answer_record_log_enabled: bool = DEFAULT_ANSWER_RECORD_LOG_ENABLED
    answer_record_log_path: str = DEFAULT_ANSWER_RECORD_LOG_PATH
    command_prefix: str = DEFAULT_COMMAND_PREFIX
    index_command_prefix: str = DEFAULT_INDEX_COMMAND_PREFIX
    system_rules: Sequence[str] = DEFAULT_SYSTEM_RULES
    auto_index_enabled: bool = DEFAULT_AUTO_INDEX_ENABLED
    auto_index_weekdays: tuple[int, ...] = ()
    auto_index_hour: int = 0
    auto_index_minute: int = 0
    warmup_interval_minutes: int = DEFAULT_WARMUP_INTERVAL_MINUTES
    index_update_estimate_min_minutes: int = (
        DEFAULT_INDEX_UPDATE_ESTIMATE_MIN_MINUTES
    )
    index_update_estimate_max_minutes: int = (
        DEFAULT_INDEX_UPDATE_ESTIMATE_MAX_MINUTES
    )
    vc_feature_enabled: bool = DEFAULT_VC_FEATURE_ENABLED
    vc_auto_join_enabled: bool = DEFAULT_VC_AUTO_JOIN_ENABLED
    vc_auto_join_weekdays: tuple[int, ...] = ()
    vc_auto_join_start_hour: int = 20
    vc_auto_join_start_minute: int = 0
    vc_auto_join_duration_minutes: int = (
        DEFAULT_VC_AUTO_JOIN_DURATION_MINUTES
    )
    vc_target_voice_channel_name: str = DEFAULT_VC_TARGET_VOICE_CHANNEL_NAME
    vc_auto_join_min_participants: int = (
        DEFAULT_VC_AUTO_JOIN_MIN_PARTICIPANTS
    )
    vc_participant_check_interval_seconds: int = (
        DEFAULT_VC_PARTICIPANT_CHECK_INTERVAL_SECONDS
    )
    vc_summary_transcribe_interval_seconds: int = (
        DEFAULT_VC_SUMMARY_TRANSCRIBE_INTERVAL_SECONDS
    )
    vc_transcribe_model: str = DEFAULT_VC_TRANSCRIBE_MODEL
    vc_transcribe_device: str = DEFAULT_VC_TRANSCRIBE_DEVICE
    vc_transcribe_torch_dtype: str = DEFAULT_VC_TRANSCRIBE_TORCH_DTYPE
    vc_transcribe_language: str = DEFAULT_VC_TRANSCRIBE_LANGUAGE
    vc_auto_quit_enabled: bool = DEFAULT_VC_AUTO_QUIT_ENABLED
    vc_final_summary_enabled: bool = DEFAULT_VC_FINAL_SUMMARY_ENABLED
    vc_summary_previous_max: int = DEFAULT_VC_SUMMARY_PREVIOUS_MAX
    vc_summary_target_characters: int = (
        DEFAULT_VC_SUMMARY_TARGET_CHARACTERS
    )
    vc_summary_llm_provider: str = DEFAULT_VC_SUMMARY_LLM_PROVIDER
    vc_summary_gemini_model: str = DEFAULT_VC_SUMMARY_GEMINI_MODEL
    vc_summary_temperature: float = DEFAULT_VC_SUMMARY_TEMPERATURE
    vc_summary_max_output_tokens: int = (
        DEFAULT_VC_SUMMARY_MAX_OUTPUT_TOKENS
    )
    vc_summary_thinking_level: str = DEFAULT_VC_SUMMARY_THINKING_LEVEL
    vc_minutes_enabled: bool = DEFAULT_VC_MINUTES_ENABLED
    vc_minutes_drive_dir: str = DEFAULT_VC_MINUTES_DRIVE_DIR
    vc_minutes_fetch_max_retries: int = DEFAULT_VC_MINUTES_FETCH_MAX_RETRIES
    vc_minutes_apply_max_retries: int = DEFAULT_VC_MINUTES_APPLY_MAX_RETRIES
    vc_minutes_llm_max_retries: int = DEFAULT_VC_MINUTES_LLM_MAX_RETRIES
    vc_minutes_history_summary_max: int = DEFAULT_VC_MINUTES_HISTORY_SUMMARY_MAX
    vc_minutes_image_batch_size: int = DEFAULT_VC_MINUTES_IMAGE_BATCH_SIZE
    vc_minutes_edit_llm_provider: str = DEFAULT_VC_MINUTES_EDIT_LLM_PROVIDER
    vc_minutes_edit_gemini_model: str = DEFAULT_VC_MINUTES_EDIT_GEMINI_MODEL
    vc_minutes_edit_temperature: float = DEFAULT_VC_MINUTES_EDIT_TEMPERATURE
    vc_minutes_edit_max_output_tokens: int = (
        DEFAULT_VC_MINUTES_EDIT_MAX_OUTPUT_TOKENS
    )
    vc_minutes_edit_thinking_level: str = DEFAULT_VC_MINUTES_EDIT_THINKING_LEVEL
    vc_final_summary_llm_provider: str = (
        DEFAULT_VC_FINAL_SUMMARY_LLM_PROVIDER
    )
    vc_final_summary_gemini_model: str = DEFAULT_VC_FINAL_SUMMARY_GEMINI_MODEL
    vc_final_summary_temperature: float = (
        DEFAULT_VC_FINAL_SUMMARY_TEMPERATURE
    )
    vc_final_summary_max_output_tokens: int = (
        DEFAULT_VC_FINAL_SUMMARY_MAX_OUTPUT_TOKENS
    )
    vc_final_summary_thinking_level: str = (
        DEFAULT_VC_FINAL_SUMMARY_THINKING_LEVEL
    )
    clear_raw_data: bool = DEFAULT_CLEAR_RAW_DATA
    clear_first_rec_chunk_data: bool = DEFAULT_CLEAR_FIRST_REC_CHUNK_DATA
    clear_second_rec_chunk_data: bool = DEFAULT_CLEAR_SECOND_REC_CHUNK_DATA
    clear_summery_chunk_data: bool = DEFAULT_CLEAR_SUMMERY_CHUNK_DATA
    update_raw_data: bool = DEFAULT_UPDATE_RAW_DATA
    update_first_rec_chunk_data: bool = DEFAULT_UPDATE_FIRST_REC_CHUNK_DATA
    update_second_rec_chunk_data: bool = DEFAULT_UPDATE_SECOND_REC_CHUNK_DATA
    update_sparse_second_rec_chunk_data: bool = (
        DEFAULT_UPDATE_SPARSE_SECOND_REC_CHUNK_DATA
    )
    update_summery_chunk_data: bool = DEFAULT_UPDATE_SUMMERY_CHUNK_DATA

    @classmethod
    def from_here(
        cls,
        *,
        embedding_model: str | None = None,
        cross_encoder_model_path: str | None = None,
        embedding_model_dir: str | None = None,
        whisper_model_dir: str | None = None,
        cross_encoder_model_dir: str | None = None,
        first_rec_chunk_size: int | None = None,
        first_rec_chunk_overlap: int | None = None,
        second_rec_enabled: bool | None = None,
        second_rec_chunk_size: int | None = None,
        second_rec_chunk_overlap: int | None = None,
        summery_enabled: bool | None = None,
        summery_characters: int | None = None,
        summery_provider: str | None = None,
        summery_gemini_model: str | None = None,
        summery_temperature: float | None = None,
        summery_max_output_tokens: int | None = None,
        summery_max_retries: int | None = None,
        summery_batch_size: int | None = None,
        llm_provider: str | None = None,
        genai_model: str | None = None,
        discord_bot_token: str | None = None,
        discord_guild_allow_list: str | None = None,
        maintenance_command_author_ids: str | None = None,
        gemini_api_key: str | None = None,
        gemini_requests_per_minute: int | None = None,
        gemini_summary_requests_per_minute: int | None = None,
        drive_folder_id: str | None = None,
        google_application_credentials: str | None = None,
        drive_max_files: int | None = None,
        crafters_colony_author_url: str | None = None,
        crafters_colony_max_pages: int | None = None,
        crafters_colony_max_articles: int | None = None,
        pdf_ocr_model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        thinking_level: str | None = None,
        no_rag_llm_provider: str | None = None,
        no_rag_genai_model: str | None = None,
        no_rag_temperature: float | None = None,
        no_rag_max_output_tokens: int | None = None,
        no_rag_thinking_level: str | None = None,
        refusal_llm_provider: str | None = None,
        refusal_genai_model: str | None = None,
        refusal_temperature: float | None = None,
        refusal_max_output_tokens: int | None = None,
        refusal_thinking_level: str | None = None,
        function_call_provider: str | None = None,
        function_call_gemini_model: str | None = None,
        function_call_temperature: float | None = None,
        function_call_max_new_tokens: int | None = None,
        function_call_max_retries: int | None = None,
        function_call_enabled: bool | None = None,
        function_call_log_enabled: bool | None = None,
        rag_idea_temperature: float | None = None,
        chat_history_enabled: bool | None = None,
        chat_history_max_turns: int | None = None,
        prompt_history_default_turns: int | None = None,
        prompt_history_additional_turns: int | None = None,
        chatbot_capabilities_info: str | None = None,
        circle_basic_info: str | None = None,
        top_k: int | None = None,
        dense_search_top_k: int | None = None,
        sparse_search_top_k: int | None = None,
        sparse_search_original_top_k: int | None = None,
        sparse_search_transform_top_k: int | None = None,
        sparse_search_initial_sparse_top_k: int | None = None,
        sparse_search_original_sparse_top_k: int | None = None,
        parent_doc_enabled: bool | None = None,
        parent_chunk_cap: int | None = None,
        rerank_enabled: bool | None = None,
        rerank_pool_size: int | None = None,
        recency_weight_soft: float | None = None,
        recency_weight_hard: float | None = None,
        recency_half_life_days: float | None = None,
        mmr_lambda: float | None = None,
        sudachi_mode: str | None = None,
        sparse_bm25_k1: float | None = None,
        sparse_bm25_b: float | None = None,
        sparse_use_normalized_form: bool | None = None,
        sparse_remove_symbols: bool | None = None,
        source_max_count: int | None = None,
        answer_json_max_retries: int | None = None,
        answer_research_max_retries: int | None = None,
        eval_answer_relevancy_enabled: bool | None = None,
        eval_faithfulness_enabled: bool | None = None,
        eval_context_precision_enabled: bool | None = None,
        eval_context_recall_enabled: bool | None = None,
        max_input_characters: int | None = None,
        prompt_full_log_enabled: bool | None = None,
        special_channel_history_limit: int | None = None,
        special_channel_custom_instruction: str | None = None,
        answer_record_log_enabled: bool | None = None,
        answer_record_log_path: str | None = None,
        auto_index_enabled: bool | None = None,
        auto_index_weekdays: str | None = None,
        auto_index_time: str | None = None,
        warmup_interval_minutes: int | None = None,
        index_update_estimate_min_minutes: int | None = None,
        index_update_estimate_max_minutes: int | None = None,
        clear_raw_data: bool | None = None,
        clear_first_rec_chunk_data: bool | None = None,
        clear_second_rec_chunk_data: bool | None = None,
        clear_summery_chunk_data: bool | None = None,
        update_raw_data: bool | None = None,
        update_first_rec_chunk_data: bool | None = None,
        update_second_rec_chunk_data: bool | None = None,
        update_sparse_second_rec_chunk_data: bool | None = None,
        update_summery_chunk_data: bool | None = None,
        command_prefix: str | None = None,
        system_rules: Sequence[str] | None = None,
        base_dir: Path | None = None,
    ) -> "AppConfig":
        resolved_base = base_dir or Path(__file__).resolve().parents[2]
        ensure_required_prompt_envs()
        embedding_model_dir_value = embedding_model_dir or os.getenv(
            "EMBEDDING_MODEL_DIR", DEFAULT_EMBEDDING_MODEL_DIR
        )
        cross_encoder_model_dir_value = cross_encoder_model_dir or os.getenv(
            "CROSS_ENCODER_MODEL_DIR", DEFAULT_CROSS_ENCODER_MODEL_DIR
        )
        whisper_model_dir_value = whisper_model_dir or os.getenv(
            "WHISPER_MODEL_DIR", DEFAULT_WHISPER_MODEL_DIR
        )

        embedding_model_dir_path = _resolve_dir(
            embedding_model_dir_value, base_dir=resolved_base
        )
        cross_encoder_model_dir_path = _resolve_dir(
            cross_encoder_model_dir_value, base_dir=resolved_base
        )
        whisper_model_dir_path = _resolve_dir(
            whisper_model_dir_value, base_dir=resolved_base
        )
        ocr_model_dir_path = resolved_base / "app" / "model" / "ocr"

        raw_embedding_model_name = (
            embedding_model
            if embedding_model is not None
            else os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )
        resolved_embedding_model = _resolve_model_path(
            model_name=raw_embedding_model_name,
            model_dir=embedding_model_dir_path,
            base_dir=resolved_base,
        )

        raw_pdf_ocr_model_name = (
            pdf_ocr_model
            if pdf_ocr_model is not None
            else os.getenv("PDF_OCR_MODEL", DEFAULT_PDF_OCR_MODEL)
        )
        resolved_pdf_ocr_model_path = _resolve_local_model_path(
            model_name=raw_pdf_ocr_model_name,
            model_dir=ocr_model_dir_path,
            base_dir=resolved_base,
        )

        function_call_provider_value = function_call_provider or os.getenv(
            "FUNCTION_CALL_PROVIDER", DEFAULT_FUNCTION_CALL_PROVIDER
        )
        function_call_gemini_model_value = (
            function_call_gemini_model
            if function_call_gemini_model is not None
            else os.getenv(
                "FUNCTION_CALL_GEMINI_MODEL",
                DEFAULT_FUNCTION_CALL_GEMINI_MODEL,
            )
        )

        raw_cross_encoder_model_name = (
            cross_encoder_model_path
            if cross_encoder_model_path is not None
            else os.getenv("CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)
        )
        resolved_cross_encoder_model_path = _resolve_model_path(
            model_name=raw_cross_encoder_model_name,
            model_dir=cross_encoder_model_dir_path,
            base_dir=resolved_base,
        )

        raw_vc_transcribe_model_name = os.getenv(
            "VC_TRANSCRIBE_MODEL",
            DEFAULT_VC_TRANSCRIBE_MODEL,
        )
        resolved_vc_transcribe_model_path = _resolve_local_model_path(
            model_name=raw_vc_transcribe_model_name,
            model_dir=whisper_model_dir_path,
            base_dir=resolved_base,
        )

        summery_provider_value = summery_provider or os.getenv(
            "SUMMERY_PROVIDER", DEFAULT_SUMMERY_PROVIDER
        )
        no_rag_provider_value = no_rag_llm_provider or os.getenv(
            "NO_RAG_LLM_PROVIDER", DEFAULT_NO_RAG_LLM_PROVIDER
        )
        refusal_provider_value = refusal_llm_provider or os.getenv(
            "REFUSAL_LLM_PROVIDER", DEFAULT_REFUSAL_LLM_PROVIDER
        )
        material_search_selector_provider_value = os.getenv(
            "MATERIAL_SEARCH_SELECTOR_LLM_PROVIDER",
            DEFAULT_MATERIAL_SEARCH_SELECTOR_LLM_PROVIDER,
        )
        vc_summary_provider_value = os.getenv(
            "VC_SUMMARY_LLM_PROVIDER",
            DEFAULT_VC_SUMMARY_LLM_PROVIDER,
        )
        vc_minutes_edit_provider_value = os.getenv(
            "VC_MINUTES_EDIT_LLM_PROVIDER",
            DEFAULT_VC_MINUTES_EDIT_LLM_PROVIDER,
        )
        vc_final_summary_provider_value = os.getenv(
            "VC_FINAL_SUMMARY_LLM_PROVIDER",
            DEFAULT_VC_FINAL_SUMMARY_LLM_PROVIDER,
        )
        summery_gemini_model_value = (
            summery_gemini_model
            if summery_gemini_model is not None
            else os.getenv("SUMMERY_GEMINI_MODEL", DEFAULT_SUMMERY_GEMINI_MODEL)
        )
        no_rag_gemini_model_value = (
            no_rag_genai_model
            if no_rag_genai_model is not None
            else os.getenv("NO_RAG_GEMINI_MODEL", DEFAULT_NO_RAG_GENAI_MODEL)
        )
        refusal_gemini_model_value = (
            refusal_genai_model
            if refusal_genai_model is not None
            else os.getenv("REFUSAL_GEMINI_MODEL", DEFAULT_REFUSAL_GENAI_MODEL)
        )
        material_search_selector_gemini_model_value = os.getenv(
            "MATERIAL_SEARCH_SELECTOR_GEMINI_MODEL",
            DEFAULT_MATERIAL_SEARCH_SELECTOR_GEMINI_MODEL,
        )
        vc_summary_gemini_model_value = os.getenv(
            "VC_SUMMARY_GEMINI_MODEL",
            DEFAULT_VC_SUMMARY_GEMINI_MODEL,
        )
        vc_minutes_edit_gemini_model_value = os.getenv(
            "VC_MINUTES_EDIT_GEMINI_MODEL",
            DEFAULT_VC_MINUTES_EDIT_GEMINI_MODEL,
        )
        vc_final_summary_gemini_model_value = os.getenv(
            "VC_FINAL_SUMMARY_GEMINI_MODEL",
            DEFAULT_VC_FINAL_SUMMARY_GEMINI_MODEL,
        )
        auto_index_time_value = (
            auto_index_time
            if auto_index_time is not None
            else os.getenv("AUTO_INDEX_TIME", DEFAULT_AUTO_INDEX_TIME)
        )
        auto_index_weekdays_value = (
            auto_index_weekdays
            if auto_index_weekdays is not None
            else os.getenv("AUTO_INDEX_WEEKDAYS", DEFAULT_AUTO_INDEX_WEEKDAYS)
        )
        auto_index_hour, auto_index_minute = _parse_time(
            auto_index_time_value, default=DEFAULT_AUTO_INDEX_TIME
        )
        auto_index_weekdays_parsed = _parse_weekdays(
            auto_index_weekdays_value, default=DEFAULT_AUTO_INDEX_WEEKDAYS
        )
        warmup_interval_minutes_value = max(
            0,
            warmup_interval_minutes
            if warmup_interval_minutes is not None
            else int(
                os.getenv(
                    "WARMUP_INTERVAL_MINUTES",
                    str(DEFAULT_WARMUP_INTERVAL_MINUTES),
                )
            ),
        )
        index_update_estimate_min_minutes_value = max(
            0,
            index_update_estimate_min_minutes
            if index_update_estimate_min_minutes is not None
            else int(
                os.getenv(
                    "INDEX_UPDATE_ESTIMATE_MIN_MINUTES",
                    str(DEFAULT_INDEX_UPDATE_ESTIMATE_MIN_MINUTES),
                )
            ),
        )
        index_update_estimate_max_minutes_value = max(
            index_update_estimate_min_minutes_value,
            index_update_estimate_max_minutes
            if index_update_estimate_max_minutes is not None
            else int(
                os.getenv(
                    "INDEX_UPDATE_ESTIMATE_MAX_MINUTES",
                    str(DEFAULT_INDEX_UPDATE_ESTIMATE_MAX_MINUTES),
                )
            ),
        )
        vc_auto_join_time_value = os.getenv(
            "VC_AUTO_JOIN_TIME",
            DEFAULT_VC_AUTO_JOIN_TIME,
        )
        vc_auto_join_weekdays_value = os.getenv(
            "VC_AUTO_JOIN_WEEKDAYS",
            DEFAULT_VC_AUTO_JOIN_WEEKDAYS,
        )
        vc_auto_join_hour, vc_auto_join_minute = _parse_time(
            vc_auto_join_time_value, default=DEFAULT_VC_AUTO_JOIN_TIME
        )
        vc_auto_join_weekdays_parsed = _parse_weekdays(
            vc_auto_join_weekdays_value,
            default=DEFAULT_VC_AUTO_JOIN_WEEKDAYS,
        )
        legacy_vc_transcribe_interval_seconds = os.getenv(
            "VC_TRANSCRIBE_INTERVAL_SECONDS"
        )
        vc_summary_transcribe_interval_default = (
            legacy_vc_transcribe_interval_seconds
            if legacy_vc_transcribe_interval_seconds is not None
            else str(DEFAULT_VC_SUMMARY_TRANSCRIBE_INTERVAL_SECONDS)
        )
        discord_guild_allow_list_value = (
            discord_guild_allow_list
            if discord_guild_allow_list is not None
            else os.getenv(
                "DISCORD_GUILD_ALLOW_LIST",
                DEFAULT_DISCORD_GUILD_ALLOW_LIST,
            )
        )
        discord_guild_allow_list_parsed = _parse_id_list(
            discord_guild_allow_list_value,
            default=DEFAULT_DISCORD_GUILD_ALLOW_LIST,
        )
        maintenance_command_author_ids_value = (
            maintenance_command_author_ids
            if maintenance_command_author_ids is not None
            else os.getenv(
                "MAINTENANCE_COMMAND_AUTHOR_IDS",
                DEFAULT_MAINTENANCE_COMMAND_AUTHOR_IDS,
            )
        )
        maintenance_command_author_ids_parsed = _parse_id_list(
            maintenance_command_author_ids_value,
            default=DEFAULT_MAINTENANCE_COMMAND_AUTHOR_IDS,
            env_name="MAINTENANCE_COMMAND_AUTHOR_IDS",
        )
        base_sparse_search_top_k = (
            sparse_search_top_k
            if sparse_search_top_k is not None
            else int(
                os.getenv(
                    "SPARSE_SEARCH_TOP_K",
                    str(DEFAULT_SPARSE_SEARCH_TOP_K),
                )
            )
        )
        base_sparse_search_original_top_k = (
            sparse_search_original_top_k
            if sparse_search_original_top_k is not None
            else int(
                os.getenv(
                    "SPARSE_SEARCH_ORIGINAL_TOP_K",
                    str(base_sparse_search_top_k),
                )
            )
        )
        return cls(
            base_dir=resolved_base,
            raw_data_dir=resolved_base / "app" / "data" / "raw",
            first_rec_chunk_dir=resolved_base / "app" / "data" / "first_rec_chunk",
            second_rec_chunk_dir=resolved_base / "app" / "data" / "second_rec_chunk",
            sparse_second_rec_chunk_dir=resolved_base
            / "app"
            / "data"
            / "sparse_second_rec_chunk",
            summery_chunk_dir=resolved_base / "app" / "data" / "summery_chunk",
            index_dir=resolved_base / "app" / "data" / "index",
            discord_bot_token=discord_bot_token
            if discord_bot_token is not None
            else os.getenv("DISCORD_BOT_TOKEN", ""),
            discord_guild_allow_list=discord_guild_allow_list_parsed,
            maintenance_command_author_ids=maintenance_command_author_ids_parsed,
            gemini_api_key=gemini_api_key
            if gemini_api_key is not None
            else os.getenv("GEMINI_API_KEY", ""),
            gemini_requests_per_minute=max(
                0,
                gemini_requests_per_minute
                if gemini_requests_per_minute is not None
                else int(
                    os.getenv(
                        "KUMC_GEMINI_REQUESTS_PER_MINUTE",
                        os.getenv(
                            "GEMINI_REQUESTS_PER_MINUTE",
                            str(DEFAULT_GEMINI_REQUESTS_PER_MINUTE),
                        ),
                    )
                ),
            ),
            gemini_summary_requests_per_minute=max(
                0,
                gemini_summary_requests_per_minute
                if gemini_summary_requests_per_minute is not None
                else (
                    gemini_requests_per_minute
                    if gemini_requests_per_minute is not None
                    else int(
                        os.getenv(
                            "KUMC_GEMINI_SUMMARY_REQUESTS_PER_MINUTE",
                            os.getenv(
                                "GEMINI_SUMMARY_REQUESTS_PER_MINUTE",
                                os.getenv(
                                    "KUMC_GEMINI_REQUESTS_PER_MINUTE",
                                    os.getenv(
                                        "GEMINI_REQUESTS_PER_MINUTE",
                                        str(
                                            DEFAULT_GEMINI_SUMMARY_REQUESTS_PER_MINUTE
                                        ),
                                    ),
                                ),
                            ),
                        )
                    )
                ),
            ),
            drive_folder_id=drive_folder_id
            if drive_folder_id is not None
            else os.getenv("FOLDER_ID", ""),
            google_application_credentials=google_application_credentials
            if google_application_credentials is not None
            else os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
            drive_max_files=drive_max_files
            if drive_max_files is not None
            else int(os.getenv("DRIVE_MAX_FILES", str(DEFAULT_DRIVE_MAX_FILES))),
            crafters_colony_author_url=crafters_colony_author_url
            if crafters_colony_author_url is not None
            else os.getenv(
                "CRAFTERS_COLONY_AUTHOR_URL",
                DEFAULT_CRAFTERS_COLONY_AUTHOR_URL,
            ),
            crafters_colony_max_pages=max(
                1,
                crafters_colony_max_pages
                if crafters_colony_max_pages is not None
                else int(
                    os.getenv(
                        "CRAFTERS_COLONY_MAX_PAGES",
                        str(DEFAULT_CRAFTERS_COLONY_MAX_PAGES),
                    )
                ),
            ),
            crafters_colony_max_articles=max(
                0,
                crafters_colony_max_articles
                if crafters_colony_max_articles is not None
                else int(
                    os.getenv(
                        "CRAFTERS_COLONY_MAX_ARTICLES",
                        str(DEFAULT_CRAFTERS_COLONY_MAX_ARTICLES),
                    )
                ),
            ),
            pdf_ocr_model_path=resolved_pdf_ocr_model_path,
            embedding_model=resolved_embedding_model,
            cross_encoder_model_path=resolved_cross_encoder_model_path,
            first_rec_chunk_size=first_rec_chunk_size
            if first_rec_chunk_size is not None
            else int(
                os.getenv(
                    "FIRST_REC_CHUNK_SIZE",
                    str(DEFAULT_FIRST_REC_CHUNK_SIZE),
                )
            ),
            first_rec_chunk_overlap=first_rec_chunk_overlap
            if first_rec_chunk_overlap is not None
            else int(
                os.getenv(
                    "FIRST_REC_CHUNK_OVERLAP",
                    str(DEFAULT_FIRST_REC_CHUNK_OVERLAP),
                )
            ),
            second_rec_enabled=second_rec_enabled
            if second_rec_enabled is not None
            else _env_bool(
                os.getenv("SECOND_REC_ENABLED"),
                DEFAULT_SECOND_REC_ENABLED,
            ),
            second_rec_chunk_size=second_rec_chunk_size
            if second_rec_chunk_size is not None
            else int(
                os.getenv(
                    "SECOND_REC_CHUNK_SIZE",
                    str(DEFAULT_SECOND_REC_CHUNK_SIZE),
                )
            ),
            second_rec_chunk_overlap=second_rec_chunk_overlap
            if second_rec_chunk_overlap is not None
            else int(
                os.getenv(
                    "SECOND_REC_CHUNK_OVERLAP",
                    str(DEFAULT_SECOND_REC_CHUNK_OVERLAP),
                )
            ),
            summery_enabled=summery_enabled
            if summery_enabled is not None
            else _env_bool(
                os.getenv("SUMMERY_ENABLED"),
                DEFAULT_SUMMERY_ENABLED,
            ),
            summery_characters=summery_characters
            if summery_characters is not None
            else int(
                os.getenv(
                    "SUMMERY_CHARACTERS", str(DEFAULT_SUMMERY_CHARACTERS)
                )
            ),
            summery_provider=summery_provider_value,
            summery_gemini_model=summery_gemini_model_value,
            summery_temperature=summery_temperature
            if summery_temperature is not None
            else float(
                os.getenv(
                    "SUMMERY_TEMPERATURE",
                    str(DEFAULT_SUMMERY_TEMPERATURE),
                )
            ),
            summery_max_output_tokens=summery_max_output_tokens
            if summery_max_output_tokens is not None
            else int(
                os.getenv(
                    "SUMMERY_MAX_OUTPUT_TOKENS",
                    str(DEFAULT_SUMMERY_MAX_OUTPUT_TOKENS),
                )
            ),
            summery_max_retries=max(
                1,
                summery_max_retries
                if summery_max_retries is not None
                else int(
                    os.getenv(
                        "SUMMERY_MAX_RETRIES",
                        str(DEFAULT_SUMMERY_MAX_RETRIES),
                    )
                ),
            ),
            summery_batch_size=max(
                1,
                summery_batch_size
                if summery_batch_size is not None
                else int(
                    os.getenv(
                        "KUMC_INDEXING_SUMMARY_BATCH_SIZE",
                        os.getenv(
                            "KUMC_SUMMERY_BATCH_SIZE",
                            os.getenv(
                                "SUMMERY_BATCH_SIZE",
                                str(DEFAULT_SUMMERY_BATCH_SIZE),
                            ),
                        ),
                    )
                ),
            ),
            llm_provider=llm_provider
            or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
            genai_model=genai_model or os.getenv("GEMINI_MODEL", DEFAULT_GENAI_MODEL),
            temperature=temperature
            if temperature is not None
            else float(os.getenv("TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            max_output_tokens=max_output_tokens
            if max_output_tokens is not None
            else int(os.getenv("MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))),
            thinking_level=thinking_level
            if thinking_level is not None
            else os.getenv("THINKING_LEVEL", DEFAULT_THINKING_LEVEL),
            no_rag_llm_provider=no_rag_provider_value,
            no_rag_genai_model=no_rag_gemini_model_value,
            no_rag_temperature=no_rag_temperature
            if no_rag_temperature is not None
            else float(
                os.getenv(
                    "NO_RAG_TEMPERATURE", str(DEFAULT_NO_RAG_TEMPERATURE)
                )
            ),
            no_rag_max_output_tokens=no_rag_max_output_tokens
            if no_rag_max_output_tokens is not None
            else int(
                os.getenv(
                    "NO_RAG_MAX_OUTPUT_TOKENS",
                    str(DEFAULT_NO_RAG_MAX_OUTPUT_TOKENS),
                )
            ),
            no_rag_thinking_level=no_rag_thinking_level
            if no_rag_thinking_level is not None
            else os.getenv(
                "NO_RAG_THINKING_LEVEL", DEFAULT_NO_RAG_THINKING_LEVEL
            ),
            refusal_llm_provider=refusal_provider_value,
            refusal_genai_model=refusal_gemini_model_value,
            refusal_temperature=refusal_temperature
            if refusal_temperature is not None
            else float(
                os.getenv(
                    "REFUSAL_TEMPERATURE", str(DEFAULT_REFUSAL_TEMPERATURE)
                )
            ),
            refusal_max_output_tokens=refusal_max_output_tokens
            if refusal_max_output_tokens is not None
            else int(
                os.getenv(
                    "REFUSAL_MAX_OUTPUT_TOKENS",
                    str(DEFAULT_REFUSAL_MAX_OUTPUT_TOKENS),
                )
            ),
            refusal_thinking_level=refusal_thinking_level
            if refusal_thinking_level is not None
            else os.getenv(
                "REFUSAL_THINKING_LEVEL", DEFAULT_REFUSAL_THINKING_LEVEL
            ),
            function_call_provider=function_call_provider_value,
            function_call_gemini_model=function_call_gemini_model_value,
            function_call_temperature=function_call_temperature
            if function_call_temperature is not None
            else float(
                os.getenv(
                    "FUNCTION_CALL_TEMPERATURE",
                    str(DEFAULT_FUNCTION_CALL_TEMPERATURE),
                )
            ),
            function_call_max_new_tokens=function_call_max_new_tokens
            if function_call_max_new_tokens is not None
            else int(
                os.getenv(
                    "FUNCTION_CALL_MAX_NEW_TOKENS",
                    str(DEFAULT_FUNCTION_CALL_MAX_NEW_TOKENS),
                )
            ),
            function_call_max_retries=max(
                0,
                function_call_max_retries
                if function_call_max_retries is not None
                else int(
                    os.getenv(
                        "FUNCTION_CALL_MAX_RETRIES",
                        str(DEFAULT_FUNCTION_CALL_MAX_RETRIES),
                    )
                ),
            ),
            function_call_enabled=function_call_enabled
            if function_call_enabled is not None
            else _env_bool(
                os.getenv("FUNCTION_CALL_ENABLED"),
                DEFAULT_FUNCTION_CALL_ENABLED,
            ),
            function_call_log_enabled=function_call_log_enabled
            if function_call_log_enabled is not None
            else _env_bool(
                os.getenv("FUNCTION_CALL_LOG_ENABLED"),
                DEFAULT_FUNCTION_CALL_LOG_ENABLED,
            ),
            material_search_max_names=max(
                1,
                int(
                    os.getenv(
                        "MATERIAL_SEARCH_MAX_NAMES",
                        str(DEFAULT_MATERIAL_SEARCH_MAX_NAMES),
                    )
                ),
            ),
            material_search_partial_match_semantic_top_k=max(
                1,
                int(
                    os.getenv(
                        "MATERIAL_SEARCH_PARTIAL_MATCH_SEMANTIC_TOP_K",
                        str(DEFAULT_MATERIAL_SEARCH_PARTIAL_MATCH_SEMANTIC_TOP_K),
                    )
                ),
            ),
            material_search_char_limit=max(
                1,
                int(
                    os.getenv(
                        "MATERIAL_SEARCH_CHAR_LIMIT",
                        str(DEFAULT_MATERIAL_SEARCH_CHAR_LIMIT),
                    )
                ),
            ),
            material_search_max_selected_summary_chunks=max(
                1,
                int(
                    os.getenv(
                        "MATERIAL_SEARCH_MAX_SELECTED_SUMMARY_CHUNKS",
                        str(
                            DEFAULT_MATERIAL_SEARCH_MAX_SELECTED_SUMMARY_CHUNKS
                        ),
                    )
                ),
            ),
            material_search_summary_missing_first_rec_top_k=max(
                1,
                int(
                    os.getenv(
                        "MATERIAL_SEARCH_SUMMARY_MISSING_FIRST_REC_TOP_K",
                        str(
                            DEFAULT_MATERIAL_SEARCH_SUMMARY_MISSING_FIRST_REC_TOP_K
                        ),
                    )
                ),
            ),
            material_search_selector_max_retries=max(
                0,
                int(
                    os.getenv(
                        "MATERIAL_SEARCH_SELECTOR_MAX_RETRIES",
                        str(DEFAULT_MATERIAL_SEARCH_SELECTOR_MAX_RETRIES),
                    )
                ),
            ),
            material_search_selector_llm_provider=material_search_selector_provider_value,
            material_search_selector_gemini_model=material_search_selector_gemini_model_value,
            material_search_selector_temperature=float(
                os.getenv(
                    "MATERIAL_SEARCH_SELECTOR_TEMPERATURE",
                    str(DEFAULT_MATERIAL_SEARCH_SELECTOR_TEMPERATURE),
                )
            ),
            material_search_selector_max_output_tokens=max(
                1,
                int(
                    os.getenv(
                        "MATERIAL_SEARCH_SELECTOR_MAX_OUTPUT_TOKENS",
                        str(DEFAULT_MATERIAL_SEARCH_SELECTOR_MAX_OUTPUT_TOKENS),
                    )
                ),
            ),
            material_search_selector_thinking_level=os.getenv(
                "MATERIAL_SEARCH_SELECTOR_THINKING_LEVEL",
                DEFAULT_MATERIAL_SEARCH_SELECTOR_THINKING_LEVEL,
            ),
            rag_idea_temperature=rag_idea_temperature
            if rag_idea_temperature is not None
            else float(
                os.getenv(
                    "RAG_IDEA_TEMPERATURE", str(DEFAULT_RAG_IDEA_TEMPERATURE)
                )
            ),
            chat_history_enabled=chat_history_enabled
            if chat_history_enabled is not None
            else _env_bool(
                os.getenv("CHAT_HISTORY_ENABLED"),
                DEFAULT_CHAT_HISTORY_ENABLED,
            ),
            chat_history_max_turns=max(
                0,
                chat_history_max_turns
                if chat_history_max_turns is not None
                else int(
                    os.getenv(
                        "CHAT_HISTORY_MAX_TURNS",
                        str(DEFAULT_CHAT_HISTORY_MAX_TURNS),
                    )
                ),
            ),
            prompt_history_default_turns=max(
                0,
                prompt_history_default_turns
                if prompt_history_default_turns is not None
                else int(
                    os.getenv(
                        "PROMPT_HISTORY_DEFAULT_TURNS",
                        str(DEFAULT_PROMPT_HISTORY_DEFAULT_TURNS),
                    )
                ),
            ),
            prompt_history_additional_turns=max(
                0,
                prompt_history_additional_turns
                if prompt_history_additional_turns is not None
                else int(
                    os.getenv(
                        "PROMPT_HISTORY_ADDITIONAL_TURNS",
                        str(DEFAULT_PROMPT_HISTORY_ADDITIONAL_TURNS),
                    )
                ),
            ),
            chatbot_capabilities_info=(
                chatbot_capabilities_info
                if chatbot_capabilities_info is not None
                else get_required_prompt_env("PROMPT_CHATBOT_CAPABILITIES_INFO")
            ),
            circle_basic_info=(
                circle_basic_info
                if circle_basic_info is not None
                else get_required_prompt_env("PROMPT_CIRCLE_BASIC_INFO")
            ),
            top_k=top_k
            if top_k is not None
            else int(os.getenv("TOP_K", str(DEFAULT_TOP_K))),
            dense_search_top_k=dense_search_top_k
            if dense_search_top_k is not None
            else int(
                os.getenv(
                    "DENSE_SEARCH_TOP_K",
                    str(DEFAULT_DENSE_SEARCH_TOP_K),
                )
            ),
            sparse_search_top_k=base_sparse_search_top_k,
            sparse_search_original_top_k=base_sparse_search_original_top_k,
            sparse_search_transform_top_k=sparse_search_transform_top_k
            if sparse_search_transform_top_k is not None
            else int(
                os.getenv(
                    "SPARSE_SEARCH_TRANSFORM_TOP_K",
                    str(base_sparse_search_top_k),
                )
            ),
            sparse_search_initial_sparse_top_k=sparse_search_initial_sparse_top_k
            if sparse_search_initial_sparse_top_k is not None
            else int(
                os.getenv(
                    "SPARSE_SEARCH_INITIAL_SPARSE_TOP_K",
                    str(base_sparse_search_top_k),
                )
            ),
            sparse_search_original_sparse_top_k=sparse_search_original_sparse_top_k
            if sparse_search_original_sparse_top_k is not None
            else int(
                os.getenv(
                    "SPARSE_SEARCH_ORIGINAL_SPARSE_TOP_K",
                    str(base_sparse_search_original_top_k),
                )
            ),
            parent_doc_enabled=parent_doc_enabled
            if parent_doc_enabled is not None
            else _env_bool(
                os.getenv("PARENT_DOC_ENABLED"),
                DEFAULT_PARENT_DOC_ENABLED,
            ),
            parent_chunk_cap=parent_chunk_cap
            if parent_chunk_cap is not None
            else int(
                os.getenv("PARENT_CHUNK_CAP", str(DEFAULT_PARENT_CHUNK_CAP))
            ),
            rerank_enabled=rerank_enabled
            if rerank_enabled is not None
            else _env_bool(
                os.getenv("RERANK_ENABLED"),
                DEFAULT_RERANK_ENABLED,
            ),
            rerank_pool_size=rerank_pool_size
            if rerank_pool_size is not None
            else int(
                os.getenv("RERANK_POOL_SIZE", str(DEFAULT_RERANK_POOL_SIZE))
            ),
            recency_weight_soft=recency_weight_soft
            if recency_weight_soft is not None
            else float(
                os.getenv(
                    "RECENCY_WEIGHT_SOFT",
                    str(DEFAULT_RECENCY_WEIGHT_SOFT),
                )
            ),
            recency_weight_hard=recency_weight_hard
            if recency_weight_hard is not None
            else float(
                os.getenv(
                    "RECENCY_WEIGHT_HARD",
                    str(DEFAULT_RECENCY_WEIGHT_HARD),
                )
            ),
            recency_half_life_days=max(
                0.0001,
                recency_half_life_days
                if recency_half_life_days is not None
                else float(
                    os.getenv(
                        "RECENCY_HALF_LIFE_DAYS",
                        str(DEFAULT_RECENCY_HALF_LIFE_DAYS),
                    )
                ),
            ),
            mmr_lambda=mmr_lambda
            if mmr_lambda is not None
            else float(os.getenv("MMR_LAMBDA", str(DEFAULT_MMR_LAMBDA))),
            sudachi_mode=sudachi_mode
            if sudachi_mode is not None
            else os.getenv("SUDACHI_MODE", DEFAULT_SUDACHI_MODE),
            sparse_bm25_k1=sparse_bm25_k1
            if sparse_bm25_k1 is not None
            else float(
                os.getenv(
                    "SPARSE_BM25_K1", str(DEFAULT_SPARSE_BM25_K1)
                )
            ),
            sparse_bm25_b=sparse_bm25_b
            if sparse_bm25_b is not None
            else float(
                os.getenv("SPARSE_BM25_B", str(DEFAULT_SPARSE_BM25_B))
            ),
            sparse_use_normalized_form=sparse_use_normalized_form
            if sparse_use_normalized_form is not None
            else _env_bool(
                os.getenv("SPARSE_USE_NORMALIZED_FORM"),
                DEFAULT_SPARSE_USE_NORMALIZED_FORM,
            ),
            sparse_remove_symbols=sparse_remove_symbols
            if sparse_remove_symbols is not None
            else _env_bool(
                os.getenv("SPARSE_REMOVE_SYMBOLS"),
                DEFAULT_SPARSE_REMOVE_SYMBOLS,
            ),
            source_max_count=source_max_count
            if source_max_count is not None
            else int(
                os.getenv("SOURCE_MAX_COUNT", str(DEFAULT_SOURCE_MAX_COUNT))
            ),
            answer_json_max_retries=answer_json_max_retries
            if answer_json_max_retries is not None
            else int(
                os.getenv(
                    "ANSWER_JSON_MAX_RETRIES",
                    str(DEFAULT_ANSWER_JSON_MAX_RETRIES),
                )
            ),
            answer_research_max_retries=answer_research_max_retries
            if answer_research_max_retries is not None
            else int(
                os.getenv(
                    "ANSWER_RESEARCH_MAX_RETRIES",
                    str(DEFAULT_ANSWER_RESEARCH_MAX_RETRIES),
                )
            ),
            eval_answer_relevancy_enabled=eval_answer_relevancy_enabled
            if eval_answer_relevancy_enabled is not None
            else _env_bool(
                os.getenv("EVAL_ANSWER_RELEVANCY_ENABLED"),
                DEFAULT_EVAL_ANSWER_RELEVANCY_ENABLED,
            ),
            eval_faithfulness_enabled=eval_faithfulness_enabled
            if eval_faithfulness_enabled is not None
            else _env_bool(
                os.getenv("EVAL_FAITHFULNESS_ENABLED"),
                DEFAULT_EVAL_FAITHFULNESS_ENABLED,
            ),
            eval_context_precision_enabled=eval_context_precision_enabled
            if eval_context_precision_enabled is not None
            else _env_bool(
                os.getenv("EVAL_CONTEXT_PRECISION_ENABLED"),
                DEFAULT_EVAL_CONTEXT_PRECISION_ENABLED,
            ),
            eval_context_recall_enabled=eval_context_recall_enabled
            if eval_context_recall_enabled is not None
            else _env_bool(
                os.getenv("EVAL_CONTEXT_RECALL_ENABLED"),
                DEFAULT_EVAL_CONTEXT_RECALL_ENABLED,
            ),
            max_input_characters=max(
                0,
                max_input_characters
                if max_input_characters is not None
                else int(
                    os.getenv(
                        "MAX_INPUT_CHARACTERS",
                        str(DEFAULT_MAX_INPUT_CHARACTERS),
                    )
                ),
            ),
            prompt_full_log_enabled=prompt_full_log_enabled
            if prompt_full_log_enabled is not None
            else _env_bool(
                os.getenv("PROMPT_FULL_LOG_ENABLED"),
                DEFAULT_PROMPT_FULL_LOG_ENABLED,
            ),
            special_channel_history_limit=max(
                0,
                special_channel_history_limit
                if special_channel_history_limit is not None
                else int(
                    os.getenv(
                        "SPECIAL_CHANNEL_HISTORY_LIMIT",
                        str(DEFAULT_SPECIAL_CHANNEL_HISTORY_LIMIT),
                    )
                ),
            ),
            special_channel_custom_instruction=special_channel_custom_instruction
            if special_channel_custom_instruction is not None
            else os.getenv(
                "SPECIAL_CHANNEL_CUSTOM_INSTRUCTION",
                DEFAULT_SPECIAL_CHANNEL_CUSTOM_INSTRUCTION,
            ),
            answer_record_log_enabled=answer_record_log_enabled
            if answer_record_log_enabled is not None
            else _env_bool(
                os.getenv("ANSWER_RECORD_LOG_ENABLED"),
                DEFAULT_ANSWER_RECORD_LOG_ENABLED,
            ),
            answer_record_log_path=(
                answer_record_log_path.strip()
                if answer_record_log_path is not None
                and answer_record_log_path.strip()
                else (
                    (os.getenv("ANSWER_RECORD_LOG_PATH") or "").strip()
                    or DEFAULT_ANSWER_RECORD_LOG_PATH
                )
            ),
            command_prefix=command_prefix
            if command_prefix is not None
            else os.getenv("COMMAND_PREFIX", DEFAULT_COMMAND_PREFIX),
            system_rules=system_rules if system_rules is not None else DEFAULT_SYSTEM_RULES,
            auto_index_enabled=auto_index_enabled
            if auto_index_enabled is not None
            else _env_bool(
                os.getenv("AUTO_INDEX_ENABLED"), DEFAULT_AUTO_INDEX_ENABLED
            ),
            auto_index_weekdays=auto_index_weekdays_parsed,
            auto_index_hour=auto_index_hour,
            auto_index_minute=auto_index_minute,
            warmup_interval_minutes=warmup_interval_minutes_value,
            index_update_estimate_min_minutes=index_update_estimate_min_minutes_value,
            index_update_estimate_max_minutes=index_update_estimate_max_minutes_value,
            vc_feature_enabled=_env_bool(
                os.getenv("VC_FEATURE_ENABLED"), DEFAULT_VC_FEATURE_ENABLED
            ),
            vc_auto_join_enabled=_env_bool(
                os.getenv("VC_AUTO_JOIN_ENABLED"),
                DEFAULT_VC_AUTO_JOIN_ENABLED,
            ),
            vc_auto_join_weekdays=vc_auto_join_weekdays_parsed,
            vc_auto_join_start_hour=vc_auto_join_hour,
            vc_auto_join_start_minute=vc_auto_join_minute,
            vc_auto_join_duration_minutes=max(
                1,
                int(
                    os.getenv(
                        "VC_AUTO_JOIN_DURATION_MINUTES",
                        str(DEFAULT_VC_AUTO_JOIN_DURATION_MINUTES),
                    )
                ),
            ),
            vc_target_voice_channel_name=os.getenv(
                "VC_TARGET_VOICE_CHANNEL_NAME",
                DEFAULT_VC_TARGET_VOICE_CHANNEL_NAME,
            ),
            vc_auto_join_min_participants=max(
                1,
                int(
                    os.getenv(
                        "VC_AUTO_JOIN_MIN_PARTICIPANTS",
                        str(DEFAULT_VC_AUTO_JOIN_MIN_PARTICIPANTS),
                    )
                ),
            ),
            vc_participant_check_interval_seconds=max(
                2,
                int(
                    os.getenv(
                        "VC_PARTICIPANT_CHECK_INTERVAL_SECONDS",
                        str(DEFAULT_VC_PARTICIPANT_CHECK_INTERVAL_SECONDS),
                    )
                ),
            ),
            vc_summary_transcribe_interval_seconds=max(
                30,
                int(
                    os.getenv(
                        "VC_SUMMARY_TRANSCRIBE_INTERVAL_SECONDS",
                        vc_summary_transcribe_interval_default,
                    )
                ),
            ),
            vc_transcribe_model=resolved_vc_transcribe_model_path,
            vc_transcribe_device=os.getenv(
                "VC_TRANSCRIBE_DEVICE",
                DEFAULT_VC_TRANSCRIBE_DEVICE,
            ),
            vc_transcribe_torch_dtype=os.getenv(
                "VC_TRANSCRIBE_TORCH_DTYPE",
                DEFAULT_VC_TRANSCRIBE_TORCH_DTYPE,
            ),
            vc_transcribe_language=os.getenv(
                "VC_TRANSCRIBE_LANGUAGE",
                DEFAULT_VC_TRANSCRIBE_LANGUAGE,
            ),
            vc_auto_quit_enabled=_env_bool(
                os.getenv("VC_AUTO_QUIT_ENABLED"), DEFAULT_VC_AUTO_QUIT_ENABLED
            ),
            vc_final_summary_enabled=_env_bool(
                os.getenv("VC_FINAL_SUMMARY_ENABLED"),
                DEFAULT_VC_FINAL_SUMMARY_ENABLED,
            ),
            vc_summary_previous_max=max(
                0,
                int(
                    os.getenv(
                        "VC_SUMMARY_PREVIOUS_MAX",
                        str(DEFAULT_VC_SUMMARY_PREVIOUS_MAX),
                    )
                ),
            ),
            vc_summary_target_characters=max(
                1,
                int(
                    os.getenv(
                        "VC_SUMMARY_TARGET_CHARACTERS",
                        str(DEFAULT_VC_SUMMARY_TARGET_CHARACTERS),
                    )
                ),
            ),
            vc_summary_llm_provider=vc_summary_provider_value,
            vc_summary_gemini_model=vc_summary_gemini_model_value,
            vc_summary_temperature=float(
                os.getenv(
                    "VC_SUMMARY_TEMPERATURE",
                    str(DEFAULT_VC_SUMMARY_TEMPERATURE),
                )
            ),
            vc_summary_max_output_tokens=max(
                1,
                int(
                    os.getenv(
                        "VC_SUMMARY_MAX_OUTPUT_TOKENS",
                        str(DEFAULT_VC_SUMMARY_MAX_OUTPUT_TOKENS),
                    )
                ),
            ),
            vc_summary_thinking_level=os.getenv(
                "VC_SUMMARY_THINKING_LEVEL",
                DEFAULT_VC_SUMMARY_THINKING_LEVEL,
            ),
            vc_minutes_enabled=_env_bool(
                os.getenv("VC_MINUTES_ENABLED"), DEFAULT_VC_MINUTES_ENABLED
            ),
            vc_minutes_drive_dir=os.getenv(
                "VC_MINUTES_DRIVE_DIR",
                DEFAULT_VC_MINUTES_DRIVE_DIR,
            ),
            vc_minutes_fetch_max_retries=max(
                0,
                int(
                    os.getenv(
                        "VC_MINUTES_FETCH_MAX_RETRIES",
                        str(DEFAULT_VC_MINUTES_FETCH_MAX_RETRIES),
                    )
                ),
            ),
            vc_minutes_apply_max_retries=max(
                0,
                int(
                    os.getenv(
                        "VC_MINUTES_APPLY_MAX_RETRIES",
                        str(DEFAULT_VC_MINUTES_APPLY_MAX_RETRIES),
                    )
                ),
            ),
            vc_minutes_llm_max_retries=max(
                0,
                int(
                    os.getenv(
                        "VC_MINUTES_LLM_MAX_RETRIES",
                        str(DEFAULT_VC_MINUTES_LLM_MAX_RETRIES),
                    )
                ),
            ),
            vc_minutes_history_summary_max=max(
                0,
                int(
                    os.getenv(
                        "VC_MINUTES_HISTORY_SUMMARY_MAX",
                        str(DEFAULT_VC_MINUTES_HISTORY_SUMMARY_MAX),
                    )
                ),
            ),
            vc_minutes_image_batch_size=max(
                1,
                int(
                    os.getenv(
                        "VC_MINUTES_IMAGE_BATCH_SIZE",
                        str(DEFAULT_VC_MINUTES_IMAGE_BATCH_SIZE),
                    )
                ),
            ),
            vc_minutes_edit_llm_provider=vc_minutes_edit_provider_value,
            vc_minutes_edit_gemini_model=vc_minutes_edit_gemini_model_value,
            vc_minutes_edit_temperature=float(
                os.getenv(
                    "VC_MINUTES_EDIT_TEMPERATURE",
                    str(DEFAULT_VC_MINUTES_EDIT_TEMPERATURE),
                )
            ),
            vc_minutes_edit_max_output_tokens=max(
                1,
                int(
                    os.getenv(
                        "VC_MINUTES_EDIT_MAX_OUTPUT_TOKENS",
                        str(DEFAULT_VC_MINUTES_EDIT_MAX_OUTPUT_TOKENS),
                    )
                ),
            ),
            vc_minutes_edit_thinking_level=os.getenv(
                "VC_MINUTES_EDIT_THINKING_LEVEL",
                DEFAULT_VC_MINUTES_EDIT_THINKING_LEVEL,
            ),
            vc_final_summary_llm_provider=vc_final_summary_provider_value,
            vc_final_summary_gemini_model=vc_final_summary_gemini_model_value,
            vc_final_summary_temperature=float(
                os.getenv(
                    "VC_FINAL_SUMMARY_TEMPERATURE",
                    str(DEFAULT_VC_FINAL_SUMMARY_TEMPERATURE),
                )
            ),
            vc_final_summary_max_output_tokens=max(
                1,
                int(
                    os.getenv(
                        "VC_FINAL_SUMMARY_MAX_OUTPUT_TOKENS",
                        str(DEFAULT_VC_FINAL_SUMMARY_MAX_OUTPUT_TOKENS),
                    )
                ),
            ),
            vc_final_summary_thinking_level=os.getenv(
                "VC_FINAL_SUMMARY_THINKING_LEVEL",
                DEFAULT_VC_FINAL_SUMMARY_THINKING_LEVEL,
            ),
            clear_raw_data=clear_raw_data
            if clear_raw_data is not None
            else _env_bool(os.getenv("CLEAR_RAW_DATA"), DEFAULT_CLEAR_RAW_DATA),
            clear_first_rec_chunk_data=clear_first_rec_chunk_data
            if clear_first_rec_chunk_data is not None
            else _env_bool(
                os.getenv("CLEAR_FIRST_REC_CHUNK_DATA"),
                DEFAULT_CLEAR_FIRST_REC_CHUNK_DATA,
            ),
            clear_second_rec_chunk_data=clear_second_rec_chunk_data
            if clear_second_rec_chunk_data is not None
            else _env_bool(
                os.getenv("CLEAR_SECOND_REC_CHUNK_DATA"),
                DEFAULT_CLEAR_SECOND_REC_CHUNK_DATA,
            ),
            clear_summery_chunk_data=clear_summery_chunk_data
            if clear_summery_chunk_data is not None
            else _env_bool(
                os.getenv("CLEAR_SUMMERY_CHUNK_DATA"),
                DEFAULT_CLEAR_SUMMERY_CHUNK_DATA,
            ),
            update_raw_data=update_raw_data
            if update_raw_data is not None
            else _env_bool(os.getenv("UPDATE_RAW_DATA"), DEFAULT_UPDATE_RAW_DATA),
            update_first_rec_chunk_data=update_first_rec_chunk_data
            if update_first_rec_chunk_data is not None
            else _env_bool(
                os.getenv("UPDATE_FIRST_REC_CHUNK_DATA"),
                DEFAULT_UPDATE_FIRST_REC_CHUNK_DATA,
            ),
            update_second_rec_chunk_data=update_second_rec_chunk_data
            if update_second_rec_chunk_data is not None
            else _env_bool(
                os.getenv("UPDATE_SECOND_REC_CHUNK_DATA"),
                DEFAULT_UPDATE_SECOND_REC_CHUNK_DATA,
            ),
            update_sparse_second_rec_chunk_data=update_sparse_second_rec_chunk_data
            if update_sparse_second_rec_chunk_data is not None
            else _env_bool(
                os.getenv("UPDATE_SPARSE_SECOND_REC_CHUNK_DATA"),
                DEFAULT_UPDATE_SPARSE_SECOND_REC_CHUNK_DATA,
            ),
            update_summery_chunk_data=update_summery_chunk_data
            if update_summery_chunk_data is not None
            else _env_bool(
                os.getenv("UPDATE_SUMMERY_CHUNK_DATA"),
                DEFAULT_UPDATE_SUMMERY_CHUNK_DATA,
            ),
        )


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, *, model_path: str) -> None:
        if not model_path:
            raise RuntimeError("Embedding model path is required.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embedding access."
            ) from exc

        self._model_path = model_path
        self._model = SentenceTransformer(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._use_e5_prefix = _is_multilingual_e5(model_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._use_e5_prefix:
            texts = [self._apply_e5_prefix(text, prefix="document:") for text in texts]
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return _vectors_to_list(vectors)

    def embed_query(self, text: str) -> list[float]:
        query = text if text else " "
        if self._use_e5_prefix:
            query = self._apply_e5_prefix(query, prefix="query:")
        vectors = self._model.encode([query], normalize_embeddings=True)
        return _vectors_to_list(vectors)[0] if vectors is not None else []

    @staticmethod
    def _apply_e5_prefix(text: str, *, prefix: str) -> str:
        stripped = (text or "").lstrip()
        lower = stripped.lower()
        if lower.startswith("query:") or lower.startswith("document:"):
            return stripped
        if not stripped:
            return f"{prefix} "
        return f"{prefix} {stripped}"


class GeminiEmbeddings(Embeddings):
    _BATCH_SIZE = 96

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
        requests_per_minute: int | None = None,
    ) -> None:
        self._model_name = (model_name or "").strip()
        if not self._model_name:
            raise RuntimeError("Gemini embedding model name is required.")

        resolved_api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not resolved_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Please set it in .env")

        self._api_key = resolved_api_key
        self._requests_per_minute = max(
            0,
            requests_per_minute
            if requests_per_minute is not None
            else int(
                os.getenv(
                    "KUMC_GEMINI_REQUESTS_PER_MINUTE",
                    os.getenv(
                        "GEMINI_REQUESTS_PER_MINUTE",
                        str(DEFAULT_GEMINI_REQUESTS_PER_MINUTE),
                    ),
                )
            ),
        )
        self._client = _gemini_embedding_client(self._api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        normalized_texts = [_normalize_embedding_text(text) for text in texts]
        vectors: list[list[float]] = []
        for i in range(0, len(normalized_texts), self._BATCH_SIZE):
            batch = normalized_texts[i : i + self._BATCH_SIZE]
            wait_for_gemini_rate_limit(
                max_requests_per_minute=self._requests_per_minute
            )
            response = self._client.models.embed_content(
                model=self._model_name,
                contents=batch,
                config=_gemini_embed_config(task_type="RETRIEVAL_DOCUMENT"),
            )
            vectors.extend(_extract_gemini_embedding_vectors(response))
        if len(vectors) != len(normalized_texts):
            raise RuntimeError(
                "Gemini embedding response count mismatch for documents: "
                f"requested={len(normalized_texts)} got={len(vectors)}"
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        wait_for_gemini_rate_limit(
            max_requests_per_minute=self._requests_per_minute
        )
        response = self._client.models.embed_content(
            model=self._model_name,
            contents=[_normalize_embedding_text(text)],
            config=_gemini_embed_config(task_type="RETRIEVAL_QUERY"),
        )
        vectors = _extract_gemini_embedding_vectors(response)
        if not vectors or not vectors[0]:
            raise RuntimeError(
                "Gemini embedding response did not contain a query vector."
            )
        return vectors[0]


class EmbeddingFactory:
    def __init__(self, model_name: str, *, api_key: str | None = None) -> None:
        self._model_name = model_name
        self._api_key = api_key

    @property
    def model_name(self) -> str:
        return self._model_name

    @lru_cache(maxsize=1)
    def get_embeddings(self) -> Embeddings:
        provider, model_name = _parse_embedding_model_spec(self._model_name)
        if provider == "gemini":
            return GeminiEmbeddings(model_name=model_name, api_key=self._api_key)
        return SentenceTransformerEmbeddings(model_path=model_name)


def _vectors_to_list(vectors) -> list[list[float]]:
    tolist = getattr(vectors, "tolist", None)
    if callable(tolist):
        return tolist()
    return [list(vector) for vector in vectors]


def _is_multilingual_e5(model_path: str) -> bool:
    normalized = (model_path or "").lower()
    return "multilingual-e5" in normalized or "multilingual_e5" in normalized


def _normalize_embedding_text(text: str | None) -> str:
    normalized = text if text else " "
    return normalized if normalized.strip() else " "


def _parse_embedding_model_spec(model_name: str) -> tuple[str, str]:
    raw = (model_name or "").strip()
    lowered = raw.lower()
    if lowered.startswith("gemini:"):
        parsed = raw.split(":", maxsplit=1)[1].strip()
        if not parsed:
            raise RuntimeError(
                "Gemini embedding model is missing. "
                "Use EMBEDDING_MODEL=gemini:<model-name>."
            )
        return "gemini", parsed
    if lowered.startswith("gemini/"):
        parsed = raw.split("/", maxsplit=1)[1].strip()
        if not parsed:
            raise RuntimeError(
                "Gemini embedding model is missing. "
                "Use EMBEDDING_MODEL=gemini/<model-name>."
            )
        return "gemini", parsed
    return "local", raw


def _extract_gemini_embedding_vectors(response) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None) or []
    if not embeddings:
        single = getattr(response, "embedding", None)
        if single is not None:
            embeddings = [single]
    vectors: list[list[float]] = []
    for embedding in embeddings:
        values = getattr(embedding, "values", None) or []
        vectors.append([float(value) for value in values])
    return vectors


def _gemini_embed_config(*, task_type: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for Gemini embedding access."
        ) from exc
    return genai.types.EmbedContentConfig(task_type=task_type)


@lru_cache(maxsize=1)
def _gemini_embedding_client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for Gemini embedding access."
        ) from exc
    return genai.Client(api_key=api_key)
