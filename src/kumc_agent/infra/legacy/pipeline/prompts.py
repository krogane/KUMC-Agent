from __future__ import annotations

import re
from typing import Literal, Sequence

from langchain_core.documents import Document

from kumc_agent.infra.legacy.config import AppConfig, get_required_prompt_env


PromptMode = Literal["rag", "rag_idea", "no_rag", "refusal"]


_OUTPUT_INSTRUCTIONS_ENV_BY_MODE: dict[PromptMode, str] = {
    "rag": "PROMPT_OUTPUT_INSTRUCTIONS_RAG",
    "rag_idea": "PROMPT_OUTPUT_INSTRUCTIONS_RAG_IDEA",
    "no_rag": "PROMPT_OUTPUT_INSTRUCTIONS_NO_RAG",
    "refusal": "PROMPT_OUTPUT_INSTRUCTIONS_REFUSAL",
}

_MODE_INSTRUCTIONS_ENV_BY_MODE: dict[PromptMode, str] = {
    "rag": "PROMPT_MODE_INSTRUCTIONS_RAG",
    "rag_idea": "PROMPT_MODE_INSTRUCTIONS_RAG_IDEA",
    "no_rag": "PROMPT_MODE_INSTRUCTIONS_NO_RAG",
    "refusal": "PROMPT_MODE_INSTRUCTIONS_REFUSAL",
}

_DISCORD_DATE_LINE_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")
_RAG_DISCORD_SUBSOURCE_INSTRUCTION = (
    "When citing Discord context, write each item in sources as "
    "\"source_index-sub_index\"."
)


def _is_discord_message_line(line: str) -> bool:
    value = (line or "").strip()
    if not value:
        return False
    return _DISCORD_DATE_LINE_RE.fullmatch(value) is None


def _annotate_discord_subsources(*, text: str, source_index: int) -> str:
    if not text:
        return ""
    annotated: list[str] = []
    sub_index = 1
    for raw_line in text.splitlines():
        if _is_discord_message_line(raw_line):
            annotated.append(f"[{source_index}-{sub_index}] {raw_line}")
            sub_index += 1
        else:
            annotated.append(raw_line)
    return "\n".join(annotated)


def _doc_to_context(doc: Document, *, source_index: int | None = None) -> str:
    metadata = doc.metadata or {}
    source_type = str(metadata.get("source_type") or "").strip().lower()
    annotated_content = doc.page_content
    if (
        source_index is not None
        and source_type in {"messages", "discord_message"}
    ):
        annotated_content = _annotate_discord_subsources(
            text=doc.page_content,
            source_index=source_index,
        )
    if source_type == "vc_transcript":
        meeting_label = str(metadata.get("meeting_label") or "").strip()
        if meeting_label:
            return f"meeting: {meeting_label}\n{annotated_content}"
        meeting_date = str(metadata.get("meeting_date") or "").strip()
        if meeting_date:
            return f"meeting_date: {meeting_date}\n{annotated_content}"
        return annotated_content
    if source_type == "hatenablog":
        lines: list[str] = []
        title = str(metadata.get("hatenablog_title") or "").strip()
        if title:
            lines.append(f"hatenablog_title: {title}")
        created_at = str(metadata.get("hatenablog_created_at") or "").strip()
        if created_at:
            lines.append(f"hatenablog_created_at: {created_at}")
        source_date = str(metadata.get("source_date") or "").strip()
        if source_date:
            lines.append(f"source_date: {source_date}")
        url = str(metadata.get("hatenablog_url") or "").strip()
        if url:
            lines.append(f"hatenablog_url: {url}")
        if lines:
            header = "\n".join(lines)
            return f"{header}\n{annotated_content}"
        return annotated_content
    if source_type == "crafters_colony":
        lines: list[str] = []
        title = str(metadata.get("crafters_colony_title") or "").strip()
        if title:
            lines.append(f"crafters_colony_title: {title}")
        published_at = str(metadata.get("crafters_colony_published_at") or "").strip()
        if published_at:
            lines.append(f"crafters_colony_published_at: {published_at}")
        source_date = str(metadata.get("source_date") or "").strip()
        if source_date:
            lines.append(f"source_date: {source_date}")
        url = str(metadata.get("crafters_colony_article_url") or "").strip()
        if url:
            lines.append(f"crafters_colony_article_url: {url}")
        if lines:
            header = "\n".join(lines)
            return f"{header}\n{annotated_content}"
        return annotated_content
    first_message_date = str(metadata.get("first_message_date") or "").strip()
    guild_name = str(metadata.get("guild_name") or "").strip()
    category_name = str(metadata.get("category_name") or "").strip()
    channel_name = str(metadata.get("channel_name") or "").strip()
    if channel_name:
        channel_parts: list[str] = []
        if guild_name:
            channel_parts.append(guild_name)
        if category_name:
            channel_parts.append(category_name)
        channel_parts.append(channel_name)
        channel_display = " / ".join(channel_parts)
        if first_message_date:
            return (
                f"channel_name: {channel_display}\n"
                f"first_message_date: {first_message_date}\n"
                f"{annotated_content}"
            )
        return f"channel_name: {channel_display}\n{annotated_content}"
    drive_path = str(metadata.get("drive_file_path") or "").strip()
    drive_path_display = drive_path if drive_path else "不明"
    source_date = str(metadata.get("source_date") or "").strip()
    source_date_display = source_date if source_date else "不明"
    if first_message_date:
        return (
            f"drive_file_path: {drive_path_display}\n"
            f"source_date: {source_date_display}\n"
            f"first_message_date: {first_message_date}\n"
            f"{annotated_content}"
        )
    return (
        f"drive_file_path: {drive_path_display}\n"
        f"source_date: {source_date_display}\n"
        f"{annotated_content}"
    )


def doc_to_context(doc: Document) -> str:
    return _doc_to_context(doc)


def format_doc_context(docs: Sequence[Document]) -> str:
    if not docs:
        return get_required_prompt_env("PROMPT_EMPTY_CONTEXT")
    parts: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        parts.append(f"[{idx}]\n{_doc_to_context(doc, source_index=idx)}")
    return "\n\n---\n\n".join(parts)


def format_output_instructions(*, mode: PromptMode) -> str:
    common = get_required_prompt_env("PROMPT_OUTPUT_INSTRUCTIONS_COMMON")
    mode_specific = get_required_prompt_env(_OUTPUT_INSTRUCTIONS_ENV_BY_MODE[mode])
    if mode == "rag":
        mode_specific = (
            f"{mode_specific}\n{_RAG_DISCORD_SUBSOURCE_INSTRUCTION}"
            if mode_specific
            else _RAG_DISCORD_SUBSOURCE_INSTRUCTION
        )
    return f"{common}\n{mode_specific}"


def format_mode_instructions(
    *,
    mode: PromptMode,
    extra_mode_instruction: str | None = None,
) -> str:
    base = get_required_prompt_env(_MODE_INSTRUCTIONS_ENV_BY_MODE[mode])
    extra = (extra_mode_instruction or "").strip()
    if not extra:
        return base
    return f"{base}\n{extra}" if base else extra


ChatHistoryEntry = tuple[str, str, Sequence[str]]


def format_chat_history(
    history: Sequence[ChatHistoryEntry],
) -> str:
    if not history:
        return get_required_prompt_env("PROMPT_EMPTY_HISTORY")
    parts: list[str] = []
    for user_text, assistant_text, _ in history:
        user_value = (user_text or "").strip()
        assistant_value = (assistant_text or "").strip()
        turn_parts: list[str] = []
        if user_value:
            turn_parts.append(
                f"{get_required_prompt_env('PROMPT_HISTORY_USER_PREFIX')}{user_value}"
            )
        if assistant_value:
            turn_parts.append(
                f"{get_required_prompt_env('PROMPT_HISTORY_ASSISTANT_PREFIX')}{assistant_value}"
            )
        if turn_parts:
            parts.append("\n".join(turn_parts))
    return "\n\n".join(parts) if parts else get_required_prompt_env("PROMPT_EMPTY_HISTORY")


def format_retry_history(history: Sequence[tuple[str, str]]) -> str:
    if not history:
        return get_required_prompt_env("PROMPT_EMPTY_HISTORY")
    parts: list[str] = []
    for user_text, assistant_text in history:
        user_value = (user_text or "").strip()
        assistant_value = (assistant_text or "").strip()
        if user_value:
            parts.append(
                f"{get_required_prompt_env('PROMPT_HISTORY_USER_PREFIX')}{user_value}"
            )
        if assistant_value:
            parts.append(
                f"{get_required_prompt_env('PROMPT_HISTORY_ASSISTANT_PREFIX')}{assistant_value}"
            )
    return "\n".join(parts) if parts else get_required_prompt_env("PROMPT_EMPTY_HISTORY")


def history_to_messages(
    history: Sequence[ChatHistoryEntry],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_text, assistant_text, _ in history:
        user_value = (user_text or "").strip()
        assistant_value = (assistant_text or "").strip()
        if user_value:
            messages.append({"role": "user", "content": user_value})
        if assistant_value:
            messages.append({"role": "assistant", "content": assistant_value})
    return messages


def _format_question_block(*, query: str, question_author: str | None = None) -> str:
    query_value = (query or "").strip()
    author_value = " ".join(
        segment.strip()
        for segment in str(question_author or "").splitlines()
        if segment.strip()
    )
    if author_value:
        return f"author: {author_value}\n{query_value}"
    return query_value


def build_gemini_prompt(
    *,
    query: str,
    question_author: str | None = None,
    prompt_mode: PromptMode,
    docs: list[Document],
    history: Sequence[ChatHistoryEntry] | None = None,
    retry_history: Sequence[tuple[str, str]] | None = None,
    circle_basic_info: str = "",
    chatbot_capabilities_info: str = "",
    include_capabilities_info: bool = True,
    extra_mode_instruction: str | None = None,
) -> str:
    context = format_doc_context(docs)
    sections: list[str] = []
    if history is not None:
        sections.append(
            f"{get_required_prompt_env('PROMPT_GEMINI_HEADER_CHAT_HISTORY')}\n"
            f"{format_chat_history(history)}"
        )
    if retry_history:
        sections.append(
            f"{get_required_prompt_env('PROMPT_GEMINI_HEADER_RETRY_HISTORY')}\n"
            f"{format_retry_history(retry_history)}"
        )
    basic_info = (circle_basic_info or "").strip()
    if basic_info:
        sections.append(
            f"{get_required_prompt_env('PROMPT_GEMINI_HEADER_CIRCLE_INFO')}\n{basic_info}"
        )
    capabilities_info = (
        (chatbot_capabilities_info or "").strip()
        if include_capabilities_info
        else ""
    )
    if capabilities_info:
        sections.append(
            f"{get_required_prompt_env('PROMPT_GEMINI_HEADER_CAPABILITIES')}\n"
            f"{capabilities_info}"
        )
    sections.append(f"{get_required_prompt_env('PROMPT_GEMINI_HEADER_CONTEXT')}\n{context}")
    sections.append(
        f"{get_required_prompt_env('PROMPT_GEMINI_HEADER_OUTPUT_FORMAT')}\n"
        f"{format_output_instructions(mode=prompt_mode)}"
    )
    sections.append(
        f"{get_required_prompt_env('PROMPT_GEMINI_HEADER_INSTRUCTIONS')}\n"
        f"{format_mode_instructions(mode=prompt_mode, extra_mode_instruction=extra_mode_instruction)}"
    )
    sections.append(
        f"{get_required_prompt_env('PROMPT_GEMINI_HEADER_QUESTION')}\n"
        f"{_format_question_block(query=query, question_author=question_author)}"
    )
    return "\n\n".join(sections)
