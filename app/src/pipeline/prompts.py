from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from config import AppConfig


def _doc_to_context(doc: Document) -> str:
    metadata = doc.metadata or {}
    first_message_date = str(metadata.get("first_message_date") or "").strip()
    channel_name = str(metadata.get("channel_name") or "").strip()
    if channel_name:
        if first_message_date:
            return (
                f"channel_name: {channel_name}\n"
                f"first_message_date: {first_message_date}\n"
                f"{doc.page_content}"
            )
        return f"channel_name: {channel_name}\n{doc.page_content}"
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
        "コンテキストに必要なサークル関連情報が含まれていない場合や、回答に追加のコンテキストがあると望ましい場合は、 follow_up_queries に具体的な追加検索クエリを複数入れてください。十分な場合は [] を入れてください。\n"
        "追加のチャット履歴が必要だと判断した場合は needs_additional_memory を true にしてください。不要なら false を入れてください。\n"
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


def format_chat_history(history: Sequence[ChatHistoryEntry]) -> str:
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
        if sources:
            turn_parts.append("参照ソース:")
            for source in sources:
                cleaned = (source or "").strip()
                if cleaned:
                    turn_parts.append(cleaned)
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
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_text, assistant_text, sources in history:
        user_value = (user_text or "").strip()
        assistant_value = (assistant_text or "").strip()
        if user_value:
            messages.append({"role": "user", "content": user_value})
        if assistant_value or sources:
            assistant_parts: list[str] = []
            if assistant_value:
                assistant_parts.append(assistant_value)
            if sources:
                cleaned_sources = [s.strip() for s in sources if s and s.strip()]
                if cleaned_sources:
                    assistant_parts.append(
                        "参照ソース:\n" + "\n\n".join(cleaned_sources)
                    )
            messages.append(
                {"role": "assistant", "content": "\n\n".join(assistant_parts)}
            )
    return messages


def build_query_transform_prompt(*, query: str) -> str:
    return (
        "あなたは検索クエリ生成器です。質問に答えず、検索に有利なキーワードのみを出力してください。\n"
        "- 出力は半角スペース区切りのキーワードのみ。\n"
        "- 固有名詞は改変せず、推測で新しい固有名詞を作らない。\n"
        "- キーワードは最大で10個。\n"
        "- 余計な説明、記号、引用符、JSON、箇条書きは禁止。\n\n"
        "## クエリ\n"
        "2024/11/30例会で『openにするかclosedにするか』はどう結論づいた？理由も簡潔に。\n"
        "## 出力\n"
        "2024/11/30 例会 open closed 結論 決定 理由 方針 判断\n\n"
        "## クエリ\n"
        "北田さんプロジェクトの双方のタスクについてまとめて\n"
        "## 出力\n"
        "北田 プロジェクト 双方 タスク 担当 役割 作業 内容 分担\n\n"
        "## クエリ\n"
        "NFの企画登録会はいつ開催で、参加できる時間帯は？\n"
        "## 出力\n"
        "NF 企画 登録会 開催日 日程 開催時間 時間帯 参加可能\n\n"
        "## クエリ\n"
        "京大RPGについて\n"
        "## 出力\n"
        "京大RPG 団体 活動 内容 概要 企画\n\n"
        "## クエリ\n"
        "団体広報原稿によると、主な活動場所はどこで、入会希望者はどこから連絡すればよい？\n"
        "## 出力\n"
        "団体 広報 原稿 活動場所 主な活動 連絡先 入会 希望 問い合わせ\n\n"
        f"## クエリ\n{query}\n"
        "## 出力\n"
    )


def build_gemini_prompt(
    *,
    query: str,
    docs: list[Document],
    history: Sequence[ChatHistoryEntry] | None = None,
    retry_history: Sequence[tuple[str, str]] | None = None,
) -> str:
    context = format_doc_context(docs)
    sections: list[str] = []
    if history is not None:
        sections.append(f"# チャット履歴\n{format_chat_history(history)}")
    if retry_history:
        sections.append(
            f"# 再検索前の質問と回答\n{format_retry_history(retry_history)}"
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
        messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": user})
    return messages
