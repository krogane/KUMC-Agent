from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagGenerationSettings:
    provider: str
    temperature: float
    max_output_tokens: int
    thinking_level: str
    prompt_name: str


@dataclass(frozen=True)
class RagIdeaGenerationSettings:
    prompt_name: str
    temperature: float


@dataclass(frozen=True)
class RagPromptTextSettings:
    empty_context: str = "(コンテキストなし)"
    empty_history: str = "(履歴なし)"
    history_user_prefix: str = "ユーザー: "
    history_assistant_prefix: str = "アシスタント: "
    history_sources_label: str = "参照ソース:"
    gemini_header_chat_history: str = "# チャット履歴"
    gemini_header_retry_history: str = "# 再検索前の質問と回答"
    gemini_header_circle_info: str = "# サークルの基本情報"
    gemini_header_capabilities: str = "# チャットボット自身の機能情報"
    gemini_header_context: str = "# コンテキスト"
    gemini_header_output_format: str = "# 出力形式"
    gemini_header_instructions: str = "## 指示"
    gemini_header_question: str = "# ユーザーの質問"
    llama_header_question: str = "### Question"
    llama_header_previous_attempt: str = "### Previous attempt (Question/Answer)"
    llama_header_circle_info: str = "### サークルの基本情報"
    llama_header_capabilities: str = "### チャットボット自身の機能情報"
    llama_header_context: str = "### Context"
    llama_header_output_format: str = "### Output format"
    llama_header_instructions: str = "## 指示"


@dataclass(frozen=True)
class RagConfig:
    top_k: int
    dense_top_k: int
    sparse_top_k: int
    rerank_pool_size: int
    mmr_lambda: float
    recency_weight_soft: float
    recency_weight_hard: float
    recency_half_life_days: float
    source_max_count: int
    recency_mode: str
    rag_generation: RagGenerationSettings
    no_rag_generation: RagGenerationSettings
    refusal_generation: RagGenerationSettings
    idea_generation: RagIdeaGenerationSettings
    history_enabled: bool = False
    history_max_turns: int = 5
    prompt_default_turns: int = 3
    prompt_additional_turns: int = 10
    fast_model_notice: str = "※負荷軽減のために軽量モードを使用しました。"
