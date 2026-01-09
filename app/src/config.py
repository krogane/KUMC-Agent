from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
DEFAULT_CHUNK_SIZE: int = 400
DEFAULT_CHUNK_OVERLAP: int = 80
DEFAULT_LLM_PROVIDER: str = "gemini"
DEFAULT_GENAI_MODEL: str = "gemini-3-flash-preview"
DEFAULT_MAX_OUTPUT_TOKENS: int = 512
DEFAULT_LLM_CHUNKING_ENABLED: bool = True
DEFAULT_LLM_CHUNK_PROVIDER: str = "gemini"
DEFAULT_LLM_CHUNK_MODEL: str = DEFAULT_GENAI_MODEL
DEFAULT_LLM_CHUNK_TEMPERATURE: float = 0.0
DEFAULT_LLM_CHUNK_SIZE: int = DEFAULT_CHUNK_SIZE
DEFAULT_LLM_CHUNK_MAX_OUTPUT_TOKENS: int = DEFAULT_MAX_OUTPUT_TOKENS
DEFAULT_LLM_CHUNK_MAX_RETRIES: int = 3
DEFAULT_LLAMA_CTX_SIZE: int = 1024
DEFAULT_LLAMA_GPU_LAYERS: int = 0
DEFAULT_LLAMA_THREADS: int = 4
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_THINKING_LEVEL: str = "minimal"
DEFAULT_TOP_K: int = 5
DEFAULT_COMMAND_PREFIX: str = "/ai "
DEFAULT_SYSTEM_RULES: Sequence[str] = (
    "あなたはcontextに基づいて回答するアシスタントで、敬語のみで解答してください。",
    "400字以内で回答しますが、出力には字数を表示することは避けてください。",
    "コンテキストに書かれていないことは推測せず、『分かりません』と答えてください。",
    "質問に関連しない情報を解答に含めることは必ず避けてください。",
)
LLM_CHUNK_SYSTEM_PROMPT: str = (
    "You are a text chunking assistant. Output JSON only."
)


def _env_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_llm_chunk_prompt(
    *,
    text: str,
    chunk_size: int,
) -> str:
    return (
        "Split the document into propositions and generate one sentence per proposition.\n"
        f"The target chunk size is approximately {chunk_size} characters.\n"
        "Preserve the original order, do not add or remove any content, and maintain the original wording as much as possible.\n"
        "If a noun phrase contains supplementary or explanatory information, separate that information into an independent proposition.\n"
        "Decontextualize each proposition so that it is self-contained, for example by replacing pronouns with the nouns they refer to.\n"
        "Return a JSON array consisting of strings only. Do not output any extra text.\n\n"
        "Document:\n"
        "<<<\n"
        f"{text}\n"
        ">>>"
    )


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    raw_data_dir: Path
    chunk_dir: Path
    llm_chunk_dir: Path
    index_dir: Path
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
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
    command_prefix: str = DEFAULT_COMMAND_PREFIX
    system_rules: Sequence[str] = DEFAULT_SYSTEM_RULES
    llm_chunking_enabled: bool = DEFAULT_LLM_CHUNKING_ENABLED
    llm_chunk_provider: str = DEFAULT_LLM_CHUNK_PROVIDER
    llm_chunk_model: str = DEFAULT_LLM_CHUNK_MODEL
    llm_chunk_llama_model_path: str = ""
    llm_chunk_temperature: float = DEFAULT_LLM_CHUNK_TEMPERATURE
    llm_chunk_max_output_tokens: int = DEFAULT_LLM_CHUNK_MAX_OUTPUT_TOKENS
    llm_chunk_size: int = DEFAULT_LLM_CHUNK_SIZE
    llm_chunk_max_retries: int = DEFAULT_LLM_CHUNK_MAX_RETRIES

    @classmethod
    def from_here(
        cls,
        *,
        embedding_model_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        llm_provider: str | None = None,
        genai_model: str | None = None,
        llama_model_path: str | None = None,
        llama_ctx_size: int | None = None,
        llama_gpu_layers: int | None = None,
        llama_threads: int | None = None,
        llm_chunking_enabled: bool | None = None,
        llm_chunk_provider: str | None = None,
        llm_chunk_model: str | None = None,
        llm_chunk_llama_model_path: str | None = None,
        llm_chunk_temperature: float | None = None,
        llm_chunk_max_output_tokens: int | None = None,
        llm_chunk_size: int | None = None,
        llm_chunk_max_retries: int | None = None,
        command_prefix: str | None = None,
        system_rules: Sequence[str] | None = None,
        base_dir: Path | None = None,
    ) -> "AppConfig":
        resolved_base = base_dir or Path(__file__).resolve().parents[2]
        raw_llama_model_path = llama_model_path or os.getenv("LLAMA_MODEL_PATH", "")
        if raw_llama_model_path:
            llama_path = Path(raw_llama_model_path)
            if not llama_path.is_absolute():
                raw_llama_model_path = str(resolved_base / llama_path)
        raw_llm_chunk_llama_model_path = (
            llm_chunk_llama_model_path
            or os.getenv("LLM_CHUNK_LLAMA_MODEL_PATH", "")
            or raw_llama_model_path
        )
        if raw_llm_chunk_llama_model_path:
            llama_path = Path(raw_llm_chunk_llama_model_path)
            if not llama_path.is_absolute():
                raw_llm_chunk_llama_model_path = str(resolved_base / llama_path)
        return cls(
            base_dir=resolved_base,
            raw_data_dir=resolved_base / "app" / "data" / "raw",
            chunk_dir=resolved_base / "app" / "data" / "chunk",
            llm_chunk_dir=resolved_base / "app" / "data" / "llm_chunk",
            index_dir=resolved_base / "app" / "data" / "index",
            embedding_model_name=embedding_model_name
            if embedding_model_name is not None
            else DEFAULT_EMBEDDING_MODEL_NAME,
            chunk_size=chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP,
            llm_provider=llm_provider or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
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
            command_prefix=command_prefix
            if command_prefix is not None
            else os.getenv("COMMAND_PREFIX", DEFAULT_COMMAND_PREFIX),
            system_rules=system_rules if system_rules is not None else DEFAULT_SYSTEM_RULES,
            llm_chunking_enabled=llm_chunking_enabled
            if llm_chunking_enabled is not None
            else _env_bool(
                os.getenv("LLM_CHUNKING_ENABLED"), DEFAULT_LLM_CHUNKING_ENABLED
            ),
            llm_chunk_provider=llm_chunk_provider
            or os.getenv("LLM_CHUNK_PROVIDER", DEFAULT_LLM_CHUNK_PROVIDER),
            llm_chunk_model=llm_chunk_model
            or os.getenv("LLM_CHUNK_MODEL", DEFAULT_LLM_CHUNK_MODEL),
            llm_chunk_llama_model_path=raw_llm_chunk_llama_model_path,
            llm_chunk_temperature=llm_chunk_temperature
            if llm_chunk_temperature is not None
            else float(
                os.getenv(
                    "LLM_CHUNK_TEMPERATURE", str(DEFAULT_LLM_CHUNK_TEMPERATURE)
                )
            ),
            llm_chunk_max_output_tokens=llm_chunk_max_output_tokens
            if llm_chunk_max_output_tokens is not None
            else int(
                os.getenv(
                    "LLM_CHUNK_MAX_OUTPUT_TOKENS",
                    str(DEFAULT_LLM_CHUNK_MAX_OUTPUT_TOKENS),
                )
            ),
            llm_chunk_size=llm_chunk_size
            if llm_chunk_size is not None
            else int(os.getenv("LLM_CHUNK_SIZE", str(DEFAULT_LLM_CHUNK_SIZE))),
            llm_chunk_max_retries=max(
                1,
                llm_chunk_max_retries
                if llm_chunk_max_retries is not None
                else int(
                    os.getenv(
                        "LLM_CHUNK_MAX_RETRIES",
                        str(DEFAULT_LLM_CHUNK_MAX_RETRIES),
                    )
                ),
            ),
        )


class EmbeddingFactory:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    @lru_cache(maxsize=1)
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self._model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
