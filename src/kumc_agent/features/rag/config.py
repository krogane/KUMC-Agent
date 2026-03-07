from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    top_k: int
    dense_top_k: int
    sparse_top_k: int
    rerank_pool_size: int
    mmr_lambda: float
    source_max_count: int
    recency_mode: str
    llm_temperature: float
    llm_max_output_tokens: int
    llm_thinking_level: str
    history_enabled: bool = False
    history_max_turns: int = 5
    prompt_default_turns: int = 3
    prompt_additional_turns: int = 10
    fast_model_notice: str = "※負荷軽減のために軽量モードを使用しました。"
