from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    top_k: int
    dense_top_k: int
    sparse_top_k: int
    rerank_pool_size: int
    source_max_count: int
    recency_mode: str
    llm_temperature: float
    llm_max_output_tokens: int
    llm_thinking_level: str
