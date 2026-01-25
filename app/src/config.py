from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from langchain_core.embeddings import Embeddings

## コンフィグ ##
# Embedding Model Settings
DEFAULT_EMBEDDING_MODEL: str = "embeddinggemma-300M-Q8_0.gguf"
DEFAULT_RAPTOR_EMBEDDING_MODEL: str = ""
DEFAULT_CROSS_ENCODER_MODEL: str = ""

# Answering LLM Settings
DEFAULT_LLM_PROVIDER: str = "llama" # gemini or llama
DEFAULT_GENAI_MODEL: str = "gemini-3-flash-preview" # gemini
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_THINKING_LEVEL: str = "minimal"
DEFAULT_LLAMA_CTX_SIZE: int = 1024 # llama
DEFAULT_MAX_OUTPUT_TOKENS: int = 512
DEFAULT_SYSTEM_RULES: Sequence[str] = (
    "あなたはcontextに基づいて回答するアシスタントで、敬語のみで解答してください。",
    "400字以内で回答しますが、出力には字数を表示することは避けてください。",
    "コンテキストに書かれていないことは推測せず、『分かりません』と答えてください。",
    "質問に関連しない情報を解答に含めることは必ず避けてください。",
)

# Recursive Chunking Settings
DEFAULT_REC_CHUNK_SIZE: int = 1200
DEFAULT_REC_CHUNK_OVERLAP: int = 40
DEFAULT_REC_MIN_CHUNK_TOKENS: int = 0

LLM_CHUNK_SYSTEM_PROMPT: str = (
    "You are a text chunking assistant. Output JSON only."
)

# Proposition Chunking Settings
DEFAULT_PROP_CHUNK_ENABLED: bool = False
DEFAULT_PROP_CHUNK_PROVIDER: str = DEFAULT_LLM_PROVIDER
DEFAULT_PROP_CHUNK_LLAMA_MODEL: str = "gemma-3n-E4B-it-IQ4_XS.gguf"
DEFAULT_PROP_CHUNK_TEMPERATURE: float = 0.0
DEFAULT_PROP_CHUNK_SIZE: int = 100
DEFAULT_PROP_CHUNK_LLAMA_CTX_SIZE: int = 2048
DEFAULT_PROP_CHUNK_MAX_OUTPUT_TOKENS: int = 2048
DEFAULT_PROP_CHUNK_MAX_RETRIES: int = 2

# RAPTOR Settings
DEFAULT_RAPTOR_ENABLED: bool = False
DEFAULT_RAPTOR_SUMMARY_PROVIDER: str = DEFAULT_LLM_PROVIDER
DEFAULT_RAPTOR_SUMMARY_LLAMA_MODEL: str = "gemma-3n-E4B-it-IQ4_XS.gguf"
DEFAULT_RAPTOR_SUMMARY_TEMPERATURE: float = 0.0
DEFAULT_RAPTOR_SUMMARY_LLAMA_CTX_SIZE: int = DEFAULT_LLAMA_CTX_SIZE
DEFAULT_RAPTOR_CLUSTER_MAX_TOKENS: int = 1024
DEFAULT_RAPTOR_SUMMARY_MAX_TOKENS: int = 256
DEFAULT_RAPTOR_STOP_CHUNK_COUNT: int = 20
DEFAULT_RAPTOR_K_MAX: int = 8
DEFAULT_RAPTOR_K_SELECTION: str = "elbow"
DEFAULT_RAPTOR_SUMMARY_MAX_RETRIES: int = 3
RAPTOR_SUMMARY_SYSTEM_PROMPT: str = (
    "You are a summarization assistant."
)

# CPU/GPU Settings
DEFAULT_LLAMA_GPU_LAYERS: int = 0
DEFAULT_LLAMA_THREADS: int = 4

# Clear Data Settings
DEFAULT_CLEAR_RAW_DATA: bool = False
DEFAULT_CLEAR_REC_CHUNK_DATA: bool = False
DEFAULT_CLEAR_PROP_CHUNK_DATA: bool = False
DEFAULT_CLEAR_RAPTOR_CHUNK_DATA: bool = False

# Retrieval Settings
DEFAULT_TOP_K: int = 5
DEFAULT_RAPTOR_SEARCH_TOP_K: int = 20
DEFAULT_KEYWORD_SEARCH_TOP_K: int = 20
DEFAULT_PARENT_DOC_ENABLED: bool = True

# Google Drive Settings
DEFAULT_DRIVE_MAX_FILES: int = 0

# Command Prefix
DEFAULT_COMMAND_PREFIX: str = "/ai "
DEFAULT_INDEX_COMMAND_PREFIX: str = "/build_index"



def _env_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_proposition_chunk_prompt(
    *,
    text: str,
    chunk_size: int,
) -> str:
    return (
    "Contentを明確でシンプルな命題に分解し、文脈に関係なく解釈できるようにしてください。\n"
    "1. 複文を単純な文に分割する。可能な限り、入力の元の言い回しを維持する。\n"
    "2. 代名詞（例：その、彼は）を、それらが参照するエンティティのフルネームで置き換えることで、命題を非文脈化する。\n"
    "3. 1つの命題に周辺のとても詳細な文脈を可能な限り含める（各命題の情報は重複しても良い）。\n"
    "4. 1つの命題に周辺のとても詳細な文脈を可能な限り含める（各命題の情報は重複しても良い）。\n\n"
    "## Content\n"
    "2025/02/08 例会議事録\n参加者：社不、prince、マグナム、orange、ブノシ\n議題：\n①新しい新刊企画\n②他企画の方針\n①\n考慮すべき事項\n・参加者はVCが可能か\n・プレイ媒体→統合版が便利\n・参加人数\n・対象層→幅広い内容を用意して選んでもらう？\n案\n・アスレ（バージョン問わずやりやすい）\n・ビルドバトル（VCの有無と経験・技量でバランス調整）\n②\n・RPG→R7年度NFでの公開を目指す、夏休みまでに建築を完成させる\n・ブログ→そろそろ書き始める\n"
    "## Output\n"
    "[\n"
    "  \"「2025/02/08 例会議事録」という文書である。\",\n"
    "  \"参加者は社不, prince, マグナム, orange, ブノシである。\",\n"
    "  \"議題は「新しい新刊企画」と「他企画の方針」である。\",\n"
    "  \"「新しい新刊企画」で考慮すべき事項は「参加者はVCが可能か」「プレイ媒体」「参加人数」「対象層」である。\",\n"
    "  \"「新しい新刊企画」では「プレイ媒体」について「統合版が便利」という記述がある。\",\n"
    "  \"「新しい新刊企画」では「対象層」について「幅広い内容を用意して選んでもらう？」という案が示されている。\",\n"
    "  \"「新しい新刊企画」の案の1つは「アスレ」である。\",\n"
    "  \"「新しい新刊企画」では「アスレ」について「バージョン問わずやりやすい」という記述がある。\",\n"
    "  \"「新しい新刊企画」の案の1つは「ビルドバトル」である。\",\n"
    "  \"「新しい新刊企画」では「ビルドバトル」について「VCの有無と経験・技量でバランス調整」という記述がある。\",\n"
    "  \"「他企画の方針」では「RPG」について「R7年度NFでの公開を目指す」という方針が記載されている。\",\n"
    "  \"「他企画の方針」では「RPG」について「夏休みまでに建築を完成させる」という方針が記載されている。\",\n"
    "  \"「他企画の方針」では「ブログ」について「そろそろ書き始める」という方針が記載されている。\"\n"
    "]\n\n"
    "## Content\n"
    f"{text}\n\n"
    "## Output\n"
    )


def build_raptor_summary_prompt(*, text: str, target_tokens: int) -> str:
    return (
        "Documentを、すべての重要な事実およびエンティティを保持したまま要約してください。\n"
        f"要約はおおよそ {target_tokens} トークン以内にしてください。\n"
        "新しい情報は追加しないでください。要約文のみを出力してください。\n\n"
        "Document:\n"
        "<<<\n"
        f"{text}\n"
        ">>>"
    )


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    raw_data_dir: Path
    rec_chunk_dir: Path
    prop_chunk_dir: Path
    raptor_chunk_dir: Path
    index_dir: Path
    discord_bot_token: str = ""
    gemini_api_key: str = ""
    drive_folder_id: str = ""
    google_application_credentials: str = ""
    drive_max_files: int = DEFAULT_DRIVE_MAX_FILES
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    raptor_embedding_model: str = DEFAULT_RAPTOR_EMBEDDING_MODEL
    cross_encoder_model_path: str = DEFAULT_CROSS_ENCODER_MODEL
    rec_chunk_size: int = DEFAULT_REC_CHUNK_SIZE
    rec_chunk_overlap: int = DEFAULT_REC_CHUNK_OVERLAP
    rec_min_chunk_tokens: int = DEFAULT_REC_MIN_CHUNK_TOKENS
    llm_provider: str = DEFAULT_LLM_PROVIDER
    genai_model: str = DEFAULT_GENAI_MODEL
    llama_model_path: str = ""
    llama_ctx_size: int = DEFAULT_LLAMA_CTX_SIZE
    llama_gpu_layers: int = DEFAULT_LLAMA_GPU_LAYERS
    llama_threads: int = DEFAULT_LLAMA_THREADS
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    thinking_level: str = DEFAULT_THINKING_LEVEL
    top_k: int = DEFAULT_TOP_K
    raptor_search_top_k: int = DEFAULT_RAPTOR_SEARCH_TOP_K
    keyword_search_top_k: int = DEFAULT_KEYWORD_SEARCH_TOP_K
    parent_doc_enabled: bool = DEFAULT_PARENT_DOC_ENABLED
    command_prefix: str = DEFAULT_COMMAND_PREFIX
    index_command_prefix: str = DEFAULT_INDEX_COMMAND_PREFIX
    system_rules: Sequence[str] = DEFAULT_SYSTEM_RULES
    prop_chunk_enabled: bool = DEFAULT_PROP_CHUNK_ENABLED
    prop_chunk_provider: str = DEFAULT_PROP_CHUNK_PROVIDER
    prop_chunk_model: str = DEFAULT_PROP_CHUNK_LLAMA_MODEL
    prop_chunk_llama_model_path: str = ""
    prop_chunk_llama_ctx_size: int = DEFAULT_PROP_CHUNK_LLAMA_CTX_SIZE
    prop_chunk_temperature: float = DEFAULT_PROP_CHUNK_TEMPERATURE
    prop_chunk_max_output_tokens: int = DEFAULT_PROP_CHUNK_MAX_OUTPUT_TOKENS
    prop_chunk_size: int = DEFAULT_PROP_CHUNK_SIZE
    prop_chunk_max_retries: int = DEFAULT_PROP_CHUNK_MAX_RETRIES
    raptor_enabled: bool = DEFAULT_RAPTOR_ENABLED
    raptor_cluster_max_tokens: int = DEFAULT_RAPTOR_CLUSTER_MAX_TOKENS
    raptor_summary_max_tokens: int = DEFAULT_RAPTOR_SUMMARY_MAX_TOKENS
    raptor_stop_chunk_count: int = DEFAULT_RAPTOR_STOP_CHUNK_COUNT
    raptor_k_max: int = DEFAULT_RAPTOR_K_MAX
    raptor_k_selection: str = DEFAULT_RAPTOR_K_SELECTION
    raptor_summary_provider: str = DEFAULT_RAPTOR_SUMMARY_PROVIDER
    raptor_summary_model: str = DEFAULT_RAPTOR_SUMMARY_LLAMA_MODEL
    raptor_summary_llama_model_path: str = ""
    raptor_summary_llama_ctx_size: int = DEFAULT_RAPTOR_SUMMARY_LLAMA_CTX_SIZE
    raptor_summary_temperature: float = DEFAULT_RAPTOR_SUMMARY_TEMPERATURE
    raptor_summary_max_retries: int = DEFAULT_RAPTOR_SUMMARY_MAX_RETRIES
    clear_raw_data: bool = DEFAULT_CLEAR_RAW_DATA
    clear_rec_chunk_data: bool = DEFAULT_CLEAR_REC_CHUNK_DATA
    clear_prop_chunk_data: bool = DEFAULT_CLEAR_PROP_CHUNK_DATA
    clear_raptor_chunk_data: bool = DEFAULT_CLEAR_RAPTOR_CHUNK_DATA

    @classmethod
    def from_here(
        cls,
        *,
        embedding_model: str | None = None,
        raptor_embedding_model: str | None = None,
        cross_encoder_model_path: str | None = None,
        rec_chunk_size: int | None = None,
        rec_chunk_overlap: int | None = None,
        rec_min_chunk_tokens: int | None = None,
        llm_provider: str | None = None,
        genai_model: str | None = None,
        discord_bot_token: str | None = None,
        gemini_api_key: str | None = None,
        drive_folder_id: str | None = None,
        google_application_credentials: str | None = None,
        drive_max_files: int | None = None,
        llama_model_path: str | None = None,
        llama_ctx_size: int | None = None,
        llama_gpu_layers: int | None = None,
        llama_threads: int | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        thinking_level: str | None = None,
        top_k: int | None = None,
        raptor_search_top_k: int | None = None,
        keyword_search_top_k: int | None = None,
        parent_doc_enabled: bool | None = None,
        prop_chunk_enabled: bool | None = None,
        prop_chunk_provider: str | None = None,
        prop_chunk_model: str | None = None,
        prop_chunk_llama_model_path: str | None = None,
        prop_chunk_llama_ctx_size: int | None = None,
        prop_chunk_temperature: float | None = None,
        prop_chunk_max_output_tokens: int | None = None,
        prop_chunk_size: int | None = None,
        prop_chunk_max_retries: int | None = None,
        raptor_enabled: bool | None = None,
        raptor_cluster_max_tokens: int | None = None,
        raptor_summary_max_tokens: int | None = None,
        raptor_stop_chunk_count: int | None = None,
        raptor_k_max: int | None = None,
        raptor_k_selection: str | None = None,
        raptor_summary_provider: str | None = None,
        raptor_summary_model: str | None = None,
        raptor_summary_llama_model_path: str | None = None,
        raptor_summary_llama_ctx_size: int | None = None,
        raptor_summary_temperature: float | None = None,
        raptor_summary_max_retries: int | None = None,
        clear_raw_data: bool | None = None,
        clear_rec_chunk_data: bool | None = None,
        clear_prop_chunk_data: bool | None = None,
        clear_raptor_chunk_data: bool | None = None,
        command_prefix: str | None = None,
        system_rules: Sequence[str] | None = None,
        base_dir: Path | None = None,
    ) -> "AppConfig":
        resolved_base = base_dir or Path(__file__).resolve().parents[2]
        raw_embedding_model = (
            embedding_model
            if embedding_model is not None
            else os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )
        if raw_embedding_model:
            embedding_path = Path(raw_embedding_model)
            if not embedding_path.is_absolute():
                raw_embedding_model = str(
                    resolved_base / "app" / "model" / "embedding" / embedding_path
                )
        raw_raptor_embedding_model = (
            raptor_embedding_model
            if raptor_embedding_model is not None
            else os.getenv("RAPTOR_EMBEDDING_MODEL", "")
        )
        if raw_raptor_embedding_model:
            raptor_embedding_path = Path(raw_raptor_embedding_model)
            if not raptor_embedding_path.is_absolute():
                raw_raptor_embedding_model = str(
                    resolved_base
                    / "app"
                    / "model"
                    / "embedding"
                    / raptor_embedding_path
                )
        else:
            raw_raptor_embedding_model = raw_embedding_model

        raw_llama_model_path = llama_model_path or os.getenv("LLAMA_MODEL_PATH", "")
        if raw_llama_model_path:
            llama_path = Path(raw_llama_model_path)
            if not llama_path.is_absolute():
                raw_llama_model_path = str(resolved_base / llama_path)

        raw_cross_encoder_model_path = (
            cross_encoder_model_path
            if cross_encoder_model_path is not None
            else os.getenv("CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)
        )
        if raw_cross_encoder_model_path:
            cross_encoder_path = Path(raw_cross_encoder_model_path)
            if not cross_encoder_path.is_absolute():
                raw_cross_encoder_model_path = str(
                    resolved_base
                    / "app"
                    / "model"
                    / "cross-encoder"
                    / cross_encoder_path
                )

        raw_prop_chunk_llama_model_path = (
            prop_chunk_llama_model_path
            or os.getenv("PROP_CHUNK_LLAMA_MODEL_PATH", "")
            or raw_llama_model_path
        )
        if raw_prop_chunk_llama_model_path:
            llama_path = Path(raw_prop_chunk_llama_model_path)
            if not llama_path.is_absolute():
                raw_prop_chunk_llama_model_path = str(resolved_base / llama_path)

        raw_raptor_summary_llama_model_path = (
            raptor_summary_llama_model_path
            or os.getenv("RAPTOR_SUMMARY_LLAMA_MODEL_PATH", "")
            or raw_llama_model_path
        )
        if raw_raptor_summary_llama_model_path:
            llama_path = Path(raw_raptor_summary_llama_model_path)
            if not llama_path.is_absolute():
                raw_raptor_summary_llama_model_path = str(resolved_base / llama_path)

        prop_chunk_provider_value = prop_chunk_provider or os.getenv(
            "PROP_CHUNK_PROVIDER", DEFAULT_PROP_CHUNK_PROVIDER
        )
        raw_prop_chunk_model = (
            prop_chunk_model
            if prop_chunk_model is not None
            else os.getenv("PROP_CHUNK_MODEL", DEFAULT_PROP_CHUNK_LLAMA_MODEL)
        )
        if (prop_chunk_provider_value or "").lower() == "llama":
            prop_chunk_model_value = raw_prop_chunk_llama_model_path or raw_prop_chunk_model
        else:
            prop_chunk_model_value = raw_prop_chunk_model

        raptor_summary_provider_value = raptor_summary_provider or os.getenv(
            "RAPTOR_SUMMARY_PROVIDER", DEFAULT_RAPTOR_SUMMARY_PROVIDER
        )
        raw_raptor_summary_model = (
            raptor_summary_model
            if raptor_summary_model is not None
            else os.getenv(
                "RAPTOR_SUMMARY_MODEL", DEFAULT_RAPTOR_SUMMARY_LLAMA_MODEL
            )
        )
        if (raptor_summary_provider_value or "").lower() == "llama":
            raptor_summary_model_value = (
                raw_raptor_summary_llama_model_path or raw_raptor_summary_model
            )
        else:
            raptor_summary_model_value = raw_raptor_summary_model

        return cls(
            base_dir=resolved_base,
            raw_data_dir=resolved_base / "app" / "data" / "raw",
            rec_chunk_dir=resolved_base / "app" / "data" / "rec_chunk",
            prop_chunk_dir=resolved_base / "app" / "data" / "prop_chunk",
            raptor_chunk_dir=resolved_base / "app" / "data" / "raptor_chunk",
            index_dir=resolved_base / "app" / "data" / "index",
            discord_bot_token=discord_bot_token
            if discord_bot_token is not None
            else os.getenv("DISCORD_BOT_TOKEN", ""),
            gemini_api_key=gemini_api_key
            if gemini_api_key is not None
            else os.getenv("GEMINI_API_KEY", ""),
            drive_folder_id=drive_folder_id
            if drive_folder_id is not None
            else os.getenv("FOLDER_ID", ""),
            google_application_credentials=google_application_credentials
            if google_application_credentials is not None
            else os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
            drive_max_files=drive_max_files
            if drive_max_files is not None
            else int(os.getenv("DRIVE_MAX_FILES", str(DEFAULT_DRIVE_MAX_FILES))),
            embedding_model=raw_embedding_model,
            raptor_embedding_model=raw_raptor_embedding_model,
            cross_encoder_model_path=raw_cross_encoder_model_path,
            rec_chunk_size=rec_chunk_size
            if rec_chunk_size is not None
            else int(os.getenv("REC_CHUNK_SIZE", str(DEFAULT_REC_CHUNK_SIZE))),
            rec_chunk_overlap=rec_chunk_overlap
            if rec_chunk_overlap is not None
            else int(
                os.getenv("REC_CHUNK_OVERLAP", str(DEFAULT_REC_CHUNK_OVERLAP))
            ),
            rec_min_chunk_tokens=rec_min_chunk_tokens
            if rec_min_chunk_tokens is not None
            else int(
                os.getenv(
                    "REC_MIN_CHUNK_TOKENS", str(DEFAULT_REC_MIN_CHUNK_TOKENS)
                )
            ),
            llm_provider=llm_provider
            or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
            genai_model=genai_model or os.getenv("GEMINI_MODEL", DEFAULT_GENAI_MODEL),
            llama_model_path=raw_llama_model_path,
            llama_ctx_size=llama_ctx_size
            if llama_ctx_size is not None
            else int(os.getenv("LLAMA_CTX_SIZE", str(DEFAULT_LLAMA_CTX_SIZE))),
            llama_gpu_layers=llama_gpu_layers
            if llama_gpu_layers is not None
            else int(os.getenv("LLAMA_GPU_LAYERS", str(DEFAULT_LLAMA_GPU_LAYERS))),
            llama_threads=llama_threads
            if llama_threads is not None
            else int(os.getenv("LLAMA_THREADS", str(DEFAULT_LLAMA_THREADS))),
            temperature=temperature
            if temperature is not None
            else float(os.getenv("TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            max_output_tokens=max_output_tokens
            if max_output_tokens is not None
            else int(os.getenv("MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))),
            thinking_level=thinking_level
            if thinking_level is not None
            else os.getenv("THINKING_LEVEL", DEFAULT_THINKING_LEVEL),
            top_k=top_k
            if top_k is not None
            else int(os.getenv("TOP_K", str(DEFAULT_TOP_K))),
            raptor_search_top_k=raptor_search_top_k
            if raptor_search_top_k is not None
            else int(
                os.getenv(
                    "RAPTOR_SEARCH_TOP_K", str(DEFAULT_RAPTOR_SEARCH_TOP_K)
                )
            ),
            keyword_search_top_k=keyword_search_top_k
            if keyword_search_top_k is not None
            else int(
                os.getenv(
                    "KEYWORD_SEARCH_TOP_K", str(DEFAULT_KEYWORD_SEARCH_TOP_K)
                )
            ),
            parent_doc_enabled=parent_doc_enabled
            if parent_doc_enabled is not None
            else _env_bool(
                os.getenv("PARENT_DOC_ENABLED"),
                DEFAULT_PARENT_DOC_ENABLED,
            ),
            command_prefix=command_prefix
            if command_prefix is not None
            else os.getenv("COMMAND_PREFIX", DEFAULT_COMMAND_PREFIX),
            system_rules=system_rules if system_rules is not None else DEFAULT_SYSTEM_RULES,
            prop_chunk_enabled=prop_chunk_enabled
            if prop_chunk_enabled is not None
            else _env_bool(os.getenv("PROP_CHUNK_ENABLED"), DEFAULT_PROP_CHUNK_ENABLED),
            prop_chunk_provider=prop_chunk_provider_value,
            prop_chunk_model=prop_chunk_model_value,
            prop_chunk_llama_model_path=raw_prop_chunk_llama_model_path,
            prop_chunk_llama_ctx_size=prop_chunk_llama_ctx_size
            if prop_chunk_llama_ctx_size is not None
            else int(
                os.getenv(
                    "PROP_CHUNK_LLAMA_CTX_SIZE",
                    str(DEFAULT_PROP_CHUNK_LLAMA_CTX_SIZE),
                )
            ),
            prop_chunk_temperature=prop_chunk_temperature
            if prop_chunk_temperature is not None
            else float(
                os.getenv(
                    "PROP_CHUNK_TEMPERATURE", str(DEFAULT_PROP_CHUNK_TEMPERATURE)
                )
            ),
            prop_chunk_max_output_tokens=prop_chunk_max_output_tokens
            if prop_chunk_max_output_tokens is not None
            else int(
                os.getenv(
                    "PROP_CHUNK_MAX_OUTPUT_TOKENS",
                    str(DEFAULT_PROP_CHUNK_MAX_OUTPUT_TOKENS),
                )
            ),
            prop_chunk_size=prop_chunk_size
            if prop_chunk_size is not None
            else int(os.getenv("PROP_CHUNK_SIZE", str(DEFAULT_PROP_CHUNK_SIZE))),
            prop_chunk_max_retries=max(
                1,
                prop_chunk_max_retries
                if prop_chunk_max_retries is not None
                else int(
                    os.getenv(
                        "PROP_CHUNK_MAX_RETRIES",
                        str(DEFAULT_PROP_CHUNK_MAX_RETRIES),
                    )
                ),
            ),
            raptor_enabled=raptor_enabled
            if raptor_enabled is not None
            else _env_bool(os.getenv("RAPTOR_ENABLED"), DEFAULT_RAPTOR_ENABLED),
            raptor_cluster_max_tokens=raptor_cluster_max_tokens
            if raptor_cluster_max_tokens is not None
            else int(
                os.getenv(
                    "RAPTOR_CLUSTER_MAX_TOKENS",
                    str(DEFAULT_RAPTOR_CLUSTER_MAX_TOKENS),
                )
            ),
            raptor_summary_max_tokens=raptor_summary_max_tokens
            if raptor_summary_max_tokens is not None
            else int(
                os.getenv(
                    "RAPTOR_SUMMARY_MAX_TOKENS",
                    str(DEFAULT_RAPTOR_SUMMARY_MAX_TOKENS),
                )
            ),
            raptor_stop_chunk_count=raptor_stop_chunk_count
            if raptor_stop_chunk_count is not None
            else int(
                os.getenv(
                    "RAPTOR_STOP_CHUNK_COUNT",
                    str(DEFAULT_RAPTOR_STOP_CHUNK_COUNT),
                )
            ),
            raptor_k_max=raptor_k_max
            if raptor_k_max is not None
            else int(os.getenv("RAPTOR_K_MAX", str(DEFAULT_RAPTOR_K_MAX))),
            raptor_k_selection=raptor_k_selection
            if raptor_k_selection is not None
            else os.getenv("RAPTOR_K_SELECTION", DEFAULT_RAPTOR_K_SELECTION),
            raptor_summary_provider=raptor_summary_provider_value,
            raptor_summary_model=raptor_summary_model_value,
            raptor_summary_llama_model_path=raw_raptor_summary_llama_model_path,
            raptor_summary_llama_ctx_size=raptor_summary_llama_ctx_size
            if raptor_summary_llama_ctx_size is not None
            else int(
                os.getenv(
                    "RAPTOR_SUMMARY_LLAMA_CTX_SIZE",
                    str(DEFAULT_RAPTOR_SUMMARY_LLAMA_CTX_SIZE),
                )
            ),
            raptor_summary_temperature=raptor_summary_temperature
            if raptor_summary_temperature is not None
            else float(
                os.getenv(
                    "RAPTOR_SUMMARY_TEMPERATURE",
                    str(DEFAULT_RAPTOR_SUMMARY_TEMPERATURE),
                )
            ),
            raptor_summary_max_retries=max(
                1,
                raptor_summary_max_retries
                if raptor_summary_max_retries is not None
                else int(
                    os.getenv(
                        "RAPTOR_SUMMARY_MAX_RETRIES",
                        str(DEFAULT_RAPTOR_SUMMARY_MAX_RETRIES),
                    )
                ),
            ),
            clear_raw_data=clear_raw_data
            if clear_raw_data is not None
            else _env_bool(os.getenv("CLEAR_RAW_DATA"), DEFAULT_CLEAR_RAW_DATA),
            clear_rec_chunk_data=clear_rec_chunk_data
            if clear_rec_chunk_data is not None
            else _env_bool(
                os.getenv("CLEAR_REC_CHUNK_DATA"), DEFAULT_CLEAR_REC_CHUNK_DATA
            ),
            clear_prop_chunk_data=clear_prop_chunk_data
            if clear_prop_chunk_data is not None
            else _env_bool(
                os.getenv("CLEAR_PROP_CHUNK_DATA"), DEFAULT_CLEAR_PROP_CHUNK_DATA
            ),
            clear_raptor_chunk_data=clear_raptor_chunk_data
            if clear_raptor_chunk_data is not None
            else _env_bool(
                os.getenv("CLEAR_RAPTOR_CHUNK_DATA"),
                DEFAULT_CLEAR_RAPTOR_CHUNK_DATA,
            ),
        )


class LlamaCppEmbeddings(Embeddings):
    def __init__(self, *, model_path: str) -> None:
        if not model_path:
            raise RuntimeError("Embedding model path is required.")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required for embedding access."
            ) from exc

        self._model_path = model_path
        self._llama = Llama(model_path=model_path, embedding=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        payload = text if text else " "
        response = self._llama.create_embedding(payload)
        data = response.get("data") or []
        if not data:
            return []
        embedding = data[0].get("embedding") or []
        return self._normalize(embedding)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        if not vector:
            return vector
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class EmbeddingFactory:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    @lru_cache(maxsize=1)
    def get_embeddings(self) -> Embeddings:
        return LlamaCppEmbeddings(model_path=self._model_name)
