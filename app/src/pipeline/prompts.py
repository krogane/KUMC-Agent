from __future__ import annotations

import re
from typing import Literal, Sequence

from langchain_core.documents import Document

from config import AppConfig, get_required_prompt_env


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
        return get_required_prompt_env("PROMPT_EMPTY_CONTEXT")
    parts: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        parts.append(f"[{idx}]\n{doc_to_context(doc)}")
    return "\n\n---\n\n".join(parts)


def format_output_instructions(*, mode: PromptMode) -> str:
    common = get_required_prompt_env("PROMPT_OUTPUT_INSTRUCTIONS_COMMON")
    mode_specific = get_required_prompt_env(_OUTPUT_INSTRUCTIONS_ENV_BY_MODE[mode])
    return f"{common}\n{mode_specific}"


def format_mode_instructions(*, mode: PromptMode) -> str:
    return get_required_prompt_env(_MODE_INSTRUCTIONS_ENV_BY_MODE[mode])


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
        return get_required_prompt_env("PROMPT_EMPTY_HISTORY")
    parts: list[str] = []
    for user_text, assistant_text, sources in history:
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
        if include_sources and sources:
            cleaned_sources = _clean_history_sources(sources)
            if cleaned_sources:
                turn_parts.append(get_required_prompt_env("PROMPT_HISTORY_SOURCES_LABEL"))
                turn_parts.append("\n\n---\n\n".join(cleaned_sources))
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
                        f"{get_required_prompt_env('PROMPT_HISTORY_SOURCES_LABEL')}\n"
                        + "\n\n---\n\n".join(cleaned_sources)
                    )
            messages.append(
                {"role": "assistant", "content": "\n\n".join(assistant_parts)}
            )
    return messages

def build_gemini_prompt(
    *,
    query: str,
    prompt_mode: PromptMode,
    docs: list[Document],
    history: Sequence[ChatHistoryEntry] | None = None,
    retry_history: Sequence[tuple[str, str]] | None = None,
    circle_basic_info: str = "",
    chatbot_capabilities_info: str = "",
    include_capabilities_info: bool = True,
    include_history_sources: bool = True,
) -> str:
    context = format_doc_context(docs)
    sections: list[str] = []
    if history is not None:
        sections.append(
            f"{get_required_prompt_env('PROMPT_GEMINI_HEADER_CHAT_HISTORY')}\n"
            f"{format_chat_history(history, include_sources=include_history_sources)}"
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
        f"{format_mode_instructions(mode=prompt_mode)}"
    )
    sections.append(f"{get_required_prompt_env('PROMPT_GEMINI_HEADER_QUESTION')}\n{query}")
    return "\n\n".join(sections)


def build_llama_messages(
    *,
    query: str,
    prompt_mode: PromptMode,
    docs: list[Document],
    config: AppConfig,
    history: Sequence[ChatHistoryEntry] | None = None,
    retry_history: Sequence[tuple[str, str]] | None = None,
    include_capabilities_info: bool = True,
    include_history_sources: bool = True,
    circle_basic_info: str | None = None,
) -> list[dict[str, str]]:
    context = format_doc_context(docs)
    system = "\n".join(config.system_rules)
    user_sections = [
        get_required_prompt_env("PROMPT_LLAMA_HEADER_QUESTION"),
        f"{query}",
    ]
    if retry_history:
        user_sections.extend(
            [
                get_required_prompt_env("PROMPT_LLAMA_HEADER_PREVIOUS_ATTEMPT"),
                format_retry_history(retry_history),
            ]
        )
    basic_info_raw = (
        config.circle_basic_info
        if circle_basic_info is None
        else circle_basic_info
    )
    basic_info = (basic_info_raw or "").strip()
    if basic_info:
        user_sections.extend(
            [
                get_required_prompt_env("PROMPT_LLAMA_HEADER_CIRCLE_INFO"),
                basic_info,
                "",
            ]
        )
    capabilities_info = (
        (config.chatbot_capabilities_info or "").strip()
        if include_capabilities_info
        else ""
    )
    if capabilities_info:
        user_sections.extend(
            [
                get_required_prompt_env("PROMPT_LLAMA_HEADER_CAPABILITIES"),
                capabilities_info,
                "",
            ]
        )
    user_sections.extend(
        [
            get_required_prompt_env("PROMPT_LLAMA_HEADER_CONTEXT"),
            f"{context}",
            "",
            get_required_prompt_env("PROMPT_LLAMA_HEADER_OUTPUT_FORMAT"),
            f"{format_output_instructions(mode=prompt_mode)}",
            "",
            get_required_prompt_env("PROMPT_LLAMA_HEADER_INSTRUCTIONS"),
            f"{format_mode_instructions(mode=prompt_mode)}",
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
