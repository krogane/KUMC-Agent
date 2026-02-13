from __future__ import annotations

import re
from typing import Sequence

from langchain_core.documents import Document

from config import AppConfig


def _doc_to_context(doc: Document) -> str:
    metadata = doc.metadata or {}
    source_type = str(metadata.get("source_type") or "").strip().lower()
    if source_type == "vc_transcript":
        meeting_label = str(metadata.get("meeting_label") or "").strip()
        if meeting_label:
            return f"meeting: {meeting_label}\n{doc.page_content}"
        meeting_date = str(metadata.get("meeting_date") or "").strip()
        if meeting_date:
            return f"meeting_date: {meeting_date}\n{doc.page_content}"
        return doc.page_content
    if source_type == "hatenablog":
        lines: list[str] = []
        title = str(metadata.get("hatenablog_title") or "").strip()
        if title:
            lines.append(f"hatenablog_title: {title}")
        created_at = str(metadata.get("hatenablog_created_at") or "").strip()
        if created_at:
            lines.append(f"hatenablog_created_at: {created_at}")
        url = str(metadata.get("hatenablog_url") or "").strip()
        if url:
            lines.append(f"hatenablog_url: {url}")
        if lines:
            header = "\n".join(lines)
            return f"{header}\n{doc.page_content}"
        return doc.page_content
    first_message_date = str(metadata.get("first_message_date") or "").strip()
    category_name = str(metadata.get("category_name") or "").strip()
    channel_name = str(metadata.get("channel_name") or "").strip()
    if channel_name:
        channel_display = (
            f"{category_name} / {channel_name}" if category_name else channel_name
        )
        if first_message_date:
            return (
                f"channel_name: {channel_display}\n"
                f"first_message_date: {first_message_date}\n"
                f"{doc.page_content}"
            )
        return f"channel_name: {channel_display}\n{doc.page_content}"
    drive_path = str(metadata.get("drive_file_path") or "").strip()
    drive_path_display = drive_path if drive_path else "不明"
    if first_message_date:
        return (
            f"drive_file_path: {drive_path_display}\n"
            f"first_message_date: {first_message_date}\n"
            f"{doc.page_content}"
        )
    return f"drive_file_path: {drive_path_display}\n{doc.page_content}"


def doc_to_context(doc: Document) -> str:
    return _doc_to_context(doc)


def format_doc_context(docs: Sequence[Document]) -> str:
    if not docs:
        return "(コンテキストなし)"
    parts: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        parts.append(f"[{idx}]\n{doc_to_context(doc)}")
    return "\n\n---\n\n".join(parts)


def format_output_instructions() -> str:
    return (
        "JSONのみで出力してください。説明文やコードフェンスは不要です。answerには必要に応じて改行などを含めて可読性を高めてください。\n"
        "answer には `<@123...>` / `<@!123...>` / `<@&123...>` のようなメンション記法を絶対に含めないでください。\n"
        "answer にはコンテキスト番号（[1]など）を含めないでください。"
        "コンテキストに必要なサークル関連情報が含まれていない場合や、回答に追加のコンテキストがあると望ましい場合は、 follow_up_queries に具体的な追加検索クエリを複数入れてください。十分な場合は [] を入れてください。\n"
        "回答に追加のチャット履歴があると望ましい場合（例: 質問に指示語が含まれている・質問の文脈が曖昧・質問が過去のチャットに関連する）は needs_additional_memory を true にしてください。不要なら false を入れてください。\n"
        "氏名と思われる単語は回答に含めないでください（ユーザーネームは回答に含めてよい）。\n"
        "質問と同じ言語で回答してください。たとえば、英語で質問されたら英語で回答します。",
        "出力形式:\n"
        "{\"answer\": \"...\", \"sources\": [2], \"follow_up_queries\": [\"...\"], \"needs_additional_memory\": false}\n"
        "- answer: 質問への回答。\n"
        "- sources: 回答に直接参照したコンテキストの番号（[1]のような番号）の配列。根拠がない場合は []。\n"
        "- follow_up_queries: 追加検索が必要な場合の具体的な検索クエリの配列。不要なら []。\n"
        "- needs_additional_memory: 追加のチャット履歴が必要かどうか。\n"
    )


ChatHistoryEntry = tuple[str, str, Sequence[str]]


_HISTORY_SOURCE_INDEX_PATTERN = re.compile(r"^\[\d+\]\s*\n?")


def _clean_history_sources(sources: Sequence[str]) -> list[str]:
    cleaned_sources: list[str] = []
    for source in sources:
        cleaned = (source or "").strip()
        if not cleaned:
            continue
        cleaned = _HISTORY_SOURCE_INDEX_PATTERN.sub("", cleaned, count=1).strip()
        if cleaned:
            cleaned_sources.append(cleaned)
    return cleaned_sources


def format_chat_history(
    history: Sequence[ChatHistoryEntry],
    *,
    include_sources: bool = True,
) -> str:
    if not history:
        return "(履歴なし)"
    parts: list[str] = []
    for user_text, assistant_text, sources in history:
        user_value = (user_text or "").strip()
        assistant_value = (assistant_text or "").strip()
        turn_parts: list[str] = []
        if user_value:
            turn_parts.append(f"ユーザー: {user_value}")
        if assistant_value:
            turn_parts.append(f"アシスタント: {assistant_value}")
        if include_sources and sources:
            cleaned_sources = _clean_history_sources(sources)
            if cleaned_sources:
                turn_parts.append("参照ソース:")
                turn_parts.append("\n\n---\n\n".join(cleaned_sources))
        if turn_parts:
            parts.append("\n".join(turn_parts))
    return "\n\n".join(parts) if parts else "(履歴なし)"


def format_retry_history(history: Sequence[tuple[str, str]]) -> str:
    if not history:
        return "(履歴なし)"
    parts: list[str] = []
    for user_text, assistant_text in history:
        user_value = (user_text or "").strip()
        assistant_value = (assistant_text or "").strip()
        if user_value:
            parts.append(f"ユーザー: {user_value}")
        if assistant_value:
            parts.append(f"アシスタント: {assistant_value}")
    return "\n".join(parts) if parts else "(履歴なし)"


QUERY_TRANSFORM_SYSTEM_PROMPT = (
    "You are a query keyword conversion assistant."
)


def history_to_messages(
    history: Sequence[ChatHistoryEntry],
    *,
    include_sources: bool = True,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_text, assistant_text, sources in history:
        user_value = (user_text or "").strip()
        assistant_value = (assistant_text or "").strip()
        if user_value:
            messages.append({"role": "user", "content": user_value})
        if assistant_value or (include_sources and sources):
            assistant_parts: list[str] = []
            if assistant_value:
                assistant_parts.append(assistant_value)
            if include_sources and sources:
                cleaned_sources = _clean_history_sources(sources)
                if cleaned_sources:
                    assistant_parts.append(
                        "参照ソース:\n" + "\n\n---\n\n".join(cleaned_sources)
                    )
            messages.append(
                {"role": "assistant", "content": "\n\n".join(assistant_parts)}
            )
    return messages


def build_query_transform_prompt(*, query: str) -> str:
    return (
        "あなたは検索クエリ生成器です。質問に答えず、クエリに関連する追加キーワードのみを出力してください。\n"
        "- 出力は半角スペース区切りのキーワードのみ。\n"
        "- 固有名詞は改変せず、推測で新しい固有名詞を作らない。\n"
        "- キーワードは最大で10個。\n"
        "- 余計な説明、記号、引用符、JSON、箇条書きは禁止。\n\n"
        "## クエリ\n"
        "2024/11/30例会で『openにするかclosedにするか』はどう結論づいた？理由も簡潔に。\n"
        "## 出力\n"
        "例会 議事録 運営方針 公開範囲 決定 理由 背景\n\n"
        "## クエリ\n"
        "北田さんプロジェクトの双方のタスクについてまとめて\n"
        "## 出力\n"
        "プロジェクト 役割 一覧 進捗 担当 作業\n\n"
        "## クエリ\n"
        "NFの企画登録会はいつ開催で、参加できる時間帯は？\n"
        "## 出力\n"
        "企画登録会 日程 時間帯 スケジュール 告知\n\n"
        "## クエリ\n"
        "京大RPGについて\n"
        "## 出力\n"
        "企画 ゲーム 制作 イベント 発表\n\n"
        "## クエリ\n"
        "団体広報原稿によると、主な活動場所はどこで、入会希望者はどこから連絡すればよい？\n"
        "## 出力\n"
        "団体 広報 原稿 活動場所 拠点 連絡先 問い合わせ 案内\n\n"
        f"## クエリ\n{query}\n"
        "## 出力\n"
    )


def build_gemini_prompt(
    *,
    query: str,
    docs: list[Document],
    history: Sequence[ChatHistoryEntry] | None = None,
    retry_history: Sequence[tuple[str, str]] | None = None,
    circle_basic_info: str = "",
    chatbot_capabilities_info: str = "",
    include_history_sources: bool = True,
) -> str:
    context = format_doc_context(docs)
    sections: list[str] = []
    if history is not None:
        sections.append(
            "# チャット履歴\n"
            f"{format_chat_history(history, include_sources=include_history_sources)}"
        )
    if retry_history:
        sections.append(
            f"# 再検索前の質問と回答\n{format_retry_history(retry_history)}"
        )
    basic_info = (circle_basic_info or "").strip()
    if basic_info:
        sections.append(f"# サークルの基本情報\n{basic_info}")
    capabilities_info = (chatbot_capabilities_info or "").strip()
    if capabilities_info:
        sections.append(
            f"# チャットボット自身の機能情報\n{capabilities_info}"
        )
    sections.append(f"# コンテキスト\n{context}")
    sections.append(f"# 出力形式\n{format_output_instructions()}")
    sections.append(f"# 質問\n{query}")
    return "\n\n".join(sections)


def build_llama_messages(
    *,
    query: str,
    docs: list[Document],
    config: AppConfig,
    history: Sequence[ChatHistoryEntry] | None = None,
    retry_history: Sequence[tuple[str, str]] | None = None,
    include_history_sources: bool = True,
) -> list[dict[str, str]]:
    context = format_doc_context(docs)
    system = "\n".join(config.system_rules)
    user_sections = ["### Question", f"{query}"]
    if retry_history:
        user_sections.extend(
            [
                "### Previous attempt (Question/Answer)",
                format_retry_history(retry_history),
            ]
        )
    basic_info = (config.circle_basic_info or "").strip()
    if basic_info:
        user_sections.extend(["### サークルの基本情報", basic_info, ""])
    capabilities_info = (config.chatbot_capabilities_info or "").strip()
    if capabilities_info:
        user_sections.extend(
            ["### チャットボット自身の機能情報", capabilities_info, ""]
        )
    user_sections.extend(
        [
            "### Context",
            f"{context}",
            "",
            "### Output format",
            f"{format_output_instructions()}",
            "",
        ]
    )
    user = "\n".join(user_sections)
    messages = [{"role": "system", "content": system}]
    if history is not None:
        messages.extend(
            history_to_messages(
                history,
                include_sources=include_history_sources,
            )
        )
    messages.append({"role": "user", "content": user})
    return messages
