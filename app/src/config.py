from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
DEFAULT_CHUNK_SIZE: int = 200
DEFAULT_CHUNK_OVERLAP: int = 40
DEFAULT_LLM_PROVIDER: str = "gemini"
DEFAULT_GENAI_MODEL: str = "gemini-3-flash-preview"
DEFAULT_LLAMA_CTX_SIZE: int = 1024
DEFAULT_LLAMA_GPU_LAYERS: int = 0
DEFAULT_LLAMA_THREADS: int = 4
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_OUTPUT_TOKENS: int = 512
DEFAULT_THINKING_LEVEL: str = "minimal"
DEFAULT_TOP_K: int = 5
DEFAULT_SYSTEM_RULES: Sequence[str] = (
    "あなたはcontextに基づいて回答するアシスタントで、敬語のみで解答してください。",
    "400字以内で回答しますが、出力には字数を表示することは避けてください。",
    "コンテキストに書かれていないことは推測せず、『分かりません』と答えてください。",
    "質問に関連しない情報を解答に含めることは必ず避けてください。",
)


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    raw_data_dir: Path
    chunk_dir: Path
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
    system_rules: Sequence[str] = DEFAULT_SYSTEM_RULES

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
        system_rules: Sequence[str] | None = None,
        base_dir: Path | None = None,
    ) -> "AppConfig":
        resolved_base = base_dir or Path(__file__).resolve().parents[2]
        raw_llama_model_path = llama_model_path or os.getenv("LLAMA_MODEL_PATH", "")
        if raw_llama_model_path:
            llama_path = Path(raw_llama_model_path)
            if not llama_path.is_absolute():
                raw_llama_model_path = str(resolved_base / llama_path)
        return cls(
            base_dir=resolved_base,
            raw_data_dir=resolved_base / "app" / "data" / "raw",
            chunk_dir=resolved_base / "app" / "data" / "chunk",
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
            system_rules=system_rules if system_rules is not None else DEFAULT_SYSTEM_RULES,
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
