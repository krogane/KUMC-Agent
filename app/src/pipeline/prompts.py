from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from config import AppConfig


def format_context(parts: Sequence[str]) -> str:
    if not parts:
        return "(コンテキストなし)"
    return "\n\n---\n\n".join(parts)


def build_gemini_prompt(*, query: str, docs: list[Document]) -> str:
    context = format_context([d.page_content for d in docs])
    return f"# コンテキスト\n{context}\n\n# 質問\n{query}"


def build_llama_messages(
    *,
    query: str,
    docs: list[Document],
    config: AppConfig,
) -> list[dict[str, str]]:
    context = format_context([d.page_content for d in docs])
    system = "\n".join(config.system_rules)
    user = (
        "### Question\n"
        f"{query}"
        "### Context\n"
        f"{context}\n\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
