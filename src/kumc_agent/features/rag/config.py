from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagGenerationSettings:
    provider: str
    temperature: float
    max_output_tokens: int
    prompt_name: str


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


@dataclass(frozen=True)
class RagConfig:
    top_k: int
    dense_top_k: int
    sparse_top_k: int
    sparse_initial_sparse_top_k: int
    rerank_pool_size: int
    mmr_lambda: float
    recency_weight_soft: float
    recency_weight_hard: float
    recency_half_life_days: float
    source_max_count: int
    recency_mode: str
    rag_generation: RagGenerationSettings
    no_rag_generation: RagGenerationSettings
    material_search_max_names: int = 3
    parent_doc_enabled: bool = True
    parent_chunk_cap: int = 2
    answer_json_max_retries: int = 2
    sparse_normalized_ratio: float | None = None
    history_enabled: bool = False
    history_max_turns: int = 5
    prompt_default_turns: int = 3
    prompt_additional_turns: int = 10
    material_full_text_char_limit: int = 3000
    fast_model_notice: str = "※負荷軽減のために軽量モードを使用しました。"
    rrf_k: int = 60
    allowed_guild_ids: tuple[str, ...] = tuple()
    admin_user_ids: tuple[str, ...] = tuple()
    minecraft_wiki_top_k: int | None = None
    minecraft_wiki_dense_top_k: int | None = None
    minecraft_wiki_sparse_top_k: int | None = None
    minecraft_wiki_sparse_initial_sparse_top_k: int | None = None
    minecraft_wiki_sparse_normalized_ratio: float | None = None
    minecraft_wiki_rerank_pool_size: int | None = None
    minecraft_wiki_rrf_k: int | None = None
    minecraft_wiki_mmr_lambda: float | None = None
    minecraft_wiki_parent_doc_enabled: bool | None = None
    minecraft_wiki_parent_chunk_cap: int | None = None
    minecraft_wiki_sparse_sudachi_mode: str | None = None
    minecraft_wiki_sparse_use_normalized_form: bool | None = None
    minecraft_wiki_sparse_remove_symbols: bool | None = None
