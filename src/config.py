from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_SYSTEM_RULES: Sequence[str] = (
    "あなたは外部資料に基づいて回答するアシスタントです。",
    "- 敬語の日本語で解答してください。",
    "- 200字以内で回答してください。",
    "- 解答が明らかに200字以上になる場合は、回答できない旨を伝えてください。",
    "- コンテキストに書かれていないことは推測せず、『分かりません』と答えてください。",
)


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    raw_data_dir: Path
    chunk_dir: Path
    index_dir: Path
    embedding_model_name: str = "intfloat/multilingual-e5-small"
    chunk_size: int = 500
    chunk_overlap: int = 100
    genai_model: str = "gemini-3-flash-preview"
    temperature: float = 1.0
    max_output_tokens: int = 256
    thinking_level: str = "minimal"
    top_k: int = 10
    system_rules: Sequence[str] = DEFAULT_SYSTEM_RULES

    @classmethod
    def from_here(
        cls,
        *,
        embedding_model_name: str = "intfloat/multilingual-e5-small",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        system_rules: Sequence[str] = DEFAULT_SYSTEM_RULES,
        base_dir: Path | None = None,
    ) -> "AppConfig":
        resolved_base = base_dir or Path(__file__).resolve().parents[1]
        return cls(
            base_dir=resolved_base,
            raw_data_dir=resolved_base / "data" / "raw",
            chunk_dir=resolved_base / "data" / "chunk",
            index_dir=resolved_base / "data" / "index",
            embedding_model_name=embedding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            system_rules=system_rules,
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
