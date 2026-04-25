from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Sequence
from zoneinfo import ZoneInfo

from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.source import Source
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.domain.ports.prompts import PromptRepositoryPort
from kumc_agent.domain.policies.source_format import (
    SOURCE_DISCLAIMER_TEXT,
    format_sources,
)
from kumc_agent.features.rag.config import RagPromptTextSettings

_MASKED_MENTION = "（メンション非表示）"
_USER_MENTION_RE = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION_RE = re.compile(r"<@&\d+>")
_DISCORD_DATE_LINE_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")
_DISCORD_SOURCE_SELECTION_RE = re.compile(r"^(\d+)(?:-(\d+))?$")
_ANSWER_FIELD_START_RE = re.compile(r'"answer"\s*:\s*"')


@dataclass(frozen=True)
class _SourceSelection:
    doc_index: int
    sub_index: int | None = None


@dataclass(frozen=True)
class _DiscordChunkLine:
    text: str
    message_id: str | None


class GenerationComponent:
    def __init__(
        self,
        *,
        llm: LLMPort,
        no_rag_llm: LLMPort | None = None,
        prompts: PromptRepositoryPort,
        source_max_count: int,
        raw_dir: Path | None = None,
        prompt_texts: RagPromptTextSettings | None = None,
    ) -> None:
        self._rag_llm = llm
        self._no_rag_llm = no_rag_llm or llm
        self._prompts = prompts
        self._source_max_count = max(0, int(source_max_count))
        self._raw_dir = raw_dir
        self._prompt_texts = prompt_texts or RagPromptTextSettings()
        self._discord_raw_cache: dict[tuple[str, str], tuple[_DiscordChunkLine, ...]] = {}

    def generate_rag_answer(
        self,
        *,
        query: str,
        chunks: list[Chunk],
        history: Sequence[tuple[str, str, Sequence[str]]] | None,
        provider: str = "gemini",
        include_capabilities_info: bool = False,
        temperature: float,
        max_output_tokens: int,
        answer_prompt_name: str = "answer_rag",
        append_sources_to_response: bool = True,
        extra_mode_instruction: str | None = None,
        json_max_retries: int = 2,
        force_all_sources: bool = False,
    ) -> Answer:
        context = self._format_context(chunks)
        prompt = self._prompt_first_available(
            answer_prompt_name,
            "answer_rag",
            default="",
        )
        history_text = self._format_history(history)
        circle_basic_info_text = self._circle_basic_info_text()
        capabilities_text = self._capabilities_text(
            include_capabilities_info=include_capabilities_info
        )
        instruction_text = (extra_mode_instruction or "").strip()
        sections = [
            self._section(
                header=self._question_header(provider=provider),
                body=(query or "").strip(),
            ),
            self._section(
                header=self._history_header(
                    provider=provider,
                    retry_mode=False,
                ),
                body=history_text,
            ),
        ]
        if circle_basic_info_text:
            sections.append(
                self._section(
                    header=self._circle_info_header(provider=provider),
                    body=circle_basic_info_text,
                )
            )
        sections.append(
            self._section(
                header=self._context_header(provider=provider),
                body=context,
            )
        )
        if capabilities_text:
            sections.append(
                self._section(
                    header=self._capabilities_header(provider=provider),
                    body=capabilities_text,
                )
            )
        if instruction_text:
            sections.append(
                self._section(
                    header=self._instructions_header(provider=provider),
                    body=instruction_text,
                )
            )
        sections.append(
            self._section(
                header=self._output_format_header(provider=provider),
                body=prompt,
            )
        )
        user_prompt = "\n\n".join(sections)
        system_prompt = self._system_prompt()

        retries = max(0, int(json_max_retries))
        last_raw = ""
        answer_text = ""
        source_selections: list[_SourceSelection] = []
        is_json = False
        has_answer = False
        best_effort_answer = ""
        for attempt in range(retries + 1):
            raw = self._rag_llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            last_raw = raw
            answer_text, source_selections, is_json, has_answer = self._parse_answer_payload(
                raw,
                max_source_index=len(chunks),
            )
            if is_json and has_answer:
                break
            if has_answer:
                best_effort_answer = answer_text
            if attempt < retries:
                continue
            answer_text = best_effort_answer or (last_raw or "").strip()
            source_selections = []

        answer_text = _mask_discord_mentions(answer_text)
        if not answer_text:
            answer_text = "回答生成中に不具合が発生しました。もう一度お試しください。"

        selected_sources = self._sources_from_chunks(
            chunks=chunks,
            selections=source_selections,
            force_all_sources=force_all_sources,
        )
        include_disclaimer = SOURCE_DISCLAIMER_TEXT not in answer_text
        final_text = (
            answer_text
            + format_sources(
                selected_sources,
                include_disclaimer=include_disclaimer,
            )
            if append_sources_to_response
            else answer_text
        )
        return Answer(
            text=final_text,
            route="rag",
            sources=selected_sources,
            metadata={
                "raw": last_raw,
                "source_selections": [
                    (
                        str(selection.doc_index)
                        if selection.sub_index is None
                        else f"{selection.doc_index}-{selection.sub_index}"
                    )
                    for selection in source_selections
                ],
                "contexts": [chunk.text for chunk in chunks],
                "answer_payload_is_json": is_json,
                "llm_prompt": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
            },
        )

    def generate_no_rag(
        self,
        *,
        query: str,
        history: Sequence[tuple[str, str, Sequence[str]]] | None,
        provider: str = "gemini",
        include_capabilities_info: bool = False,
        temperature: float,
        max_output_tokens: int,
        answer_prompt_name: str = "answer_no_rag",
        extra_mode_instruction: str | None = None,
        json_max_retries: int = 2,
    ) -> Answer:
        prompt = self._prompt_first_available(
            answer_prompt_name,
            "answer_no_rag",
            "answer_rag",
            default="",
        )
        history_text = self._format_history(history)
        capabilities_text = self._capabilities_text(
            include_capabilities_info=include_capabilities_info
        )
        instruction_text = (extra_mode_instruction or "").strip()
        sections = [
            self._section(
                header=self._question_header(provider=provider),
                body=(query or "").strip(),
            ),
            self._section(
                header=self._history_header(provider=provider, retry_mode=True),
                body=history_text,
            ),
        ]
        if capabilities_text:
            sections.append(
                self._section(
                    header=self._capabilities_header(provider=provider),
                    body=capabilities_text,
                )
            )
        if instruction_text:
            sections.append(
                self._section(
                    header=self._instructions_header(provider=provider),
                    body=instruction_text,
                )
            )
        sections.append(
            self._section(
                header=self._output_format_header(provider=provider),
                body=prompt,
            )
        )
        user_prompt = "\n\n".join(sections)
        system_prompt = self._system_prompt()

        retries = max(0, int(json_max_retries))
        last_raw = ""
        answer_text = ""
        is_json = False
        has_answer = False
        best_effort_answer = ""
        for attempt in range(retries + 1):
            raw = self._no_rag_llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            last_raw = raw
            answer_text, _, is_json, has_answer = self._parse_answer_payload(
                raw,
                max_source_index=0,
            )
            if is_json and has_answer:
                break
            if has_answer:
                best_effort_answer = answer_text
            if attempt < retries:
                continue
            answer_text = best_effort_answer or (last_raw or "").strip()

        answer_text = _mask_discord_mentions(answer_text)
        if not answer_text:
            answer_text = "回答生成中に不具合が発生しました。もう一度お試しください。"
        return Answer(
            text=answer_text,
            route="no_rag",
            sources=[],
            metadata={
                "raw": last_raw,
                "answer_payload_is_json": is_json,
                "llm_prompt": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
            },
        )

    def _format_context(self, chunks: list[Chunk]) -> str:
        if not chunks:
            return self._prompt_texts.empty_context
        parts: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            parts.append(f"[{idx}]\n{self._chunk_to_context(chunk, source_index=idx)}")
        return "\n\n---\n\n".join(parts)

    def _chunk_to_context(self, chunk: Chunk, *, source_index: int | None = None) -> str:
        metadata = chunk.metadata or {}
        source_type = str(metadata.get("source_type") or "").strip().lower()
        annotated_content = chunk.text
        if source_index is not None and source_type in {"messages", "discord_message"}:
            annotated_content = self._annotate_discord_subsources(
                text=chunk.text,
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
            lines = []
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
        drive_path = str(
            metadata.get("drive_file_path")
            or metadata.get("path")
            or metadata.get("source_name")
            or ""
        ).strip()
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

    @staticmethod
    def _annotate_discord_subsources(*, text: str, source_index: int) -> str:
        if not text:
            return ""
        annotated: list[str] = []
        sub_index = 1
        for raw_line in text.splitlines():
            value = (raw_line or "").strip()
            if value and _DISCORD_DATE_LINE_RE.fullmatch(value) is None:
                annotated.append(f"[{source_index}-{sub_index}] {raw_line}")
                sub_index += 1
            else:
                annotated.append(raw_line)
        return "\n".join(annotated)

    @staticmethod
    def _parse_answer_payload(
        text: str,
        *,
        max_source_index: int,
    ) -> tuple[str, list[_SourceSelection], bool, bool]:
        raw = (text or "").strip()
        if not raw:
            return "", [], False, False

        payload = GenerationComponent._load_json_payload(raw)
        if payload is None:
            recovered_answer = GenerationComponent._extract_answer_from_malformed_payload(
                raw
            )
            if recovered_answer is not None:
                return recovered_answer, [], False, bool(recovered_answer)
            return raw, [], False, False

        answer = str(payload.get("answer") or "").strip()
        sources_raw = payload.get("sources")
        source_selections: list[_SourceSelection] = []
        seen: set[tuple[int, int | None]] = set()
        if isinstance(sources_raw, list):
            for item in sources_raw:
                selection = GenerationComponent._parse_source_selection_item(
                    item=item,
                    max_source_index=max_source_index,
                )
                if selection is None:
                    continue
                key = (selection.doc_index, selection.sub_index)
                if key in seen:
                    continue
                seen.add(key)
                source_selections.append(selection)

        has_answer = bool(answer)
        return answer, source_selections, True, has_answer

    @staticmethod
    def _extract_answer_from_malformed_payload(text: str) -> str | None:
        cleaned = GenerationComponent._strip_code_fence(text).strip()
        if not cleaned:
            return None
        key_match = _ANSWER_FIELD_START_RE.search(cleaned)
        if key_match is None:
            return None
        value_start = key_match.end()
        escaped = False
        chars: list[str] = []
        closed = False
        for ch in cleaned[value_start:]:
            if escaped:
                chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                chars.append(ch)
                escaped = True
                continue
            if ch == '"':
                closed = True
                break
            chars.append(ch)

        if not chars and not closed:
            return None
        decoded = GenerationComponent._decode_json_string_fragment(
            value="".join(chars),
            closed=closed,
        )
        if decoded is None:
            return None
        result = decoded.strip()
        return result or None

    @staticmethod
    def _decode_json_string_fragment(*, value: str, closed: bool) -> str | None:
        candidate = value
        if not closed and candidate:
            while candidate.endswith("\\"):
                candidate = candidate[:-1]
        try:
            return str(json.loads(f'"{candidate}"'))
        except json.JSONDecodeError:
            normalized = candidate
            normalized = normalized.replace("\\\\", "\\")
            normalized = normalized.replace('\\"', '"')
            normalized = normalized.replace("\\n", "\n")
            normalized = normalized.replace("\\r", "\r")
            normalized = normalized.replace("\\t", "\t")
            return normalized

    @staticmethod
    def _parse_source_selection_item(
        *,
        item: object,
        max_source_index: int,
    ) -> _SourceSelection | None:
        doc_index: int | None = None
        sub_index: int | None = None
        if isinstance(item, int):
            doc_index = item
        elif isinstance(item, float) and item.is_integer():
            doc_index = int(item)
        elif isinstance(item, str):
            value = item.strip()
            if not value:
                return None
            match = _DISCORD_SOURCE_SELECTION_RE.fullmatch(value)
            if not match:
                return None
            doc_index = int(match.group(1))
            sub_text = match.group(2)
            if sub_text is not None:
                sub_index = int(sub_text)
        else:
            return None

        if doc_index is None:
            return None
        if doc_index < 1 or doc_index > max_source_index:
            return None
        if sub_index is not None and sub_index < 1:
            return None
        return _SourceSelection(doc_index=doc_index, sub_index=sub_index)

    @staticmethod
    def _load_json_payload(text: str) -> dict[str, object] | None:
        cleaned = GenerationComponent._strip_code_fence(text).strip()
        if not cleaned:
            return None
        parsed = GenerationComponent._load_json_object(cleaned)
        if parsed is not None:
            return parsed
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        return GenerationComponent._load_json_object(cleaned[start : end + 1])

    @staticmethod
    def _load_json_object(text: str) -> dict[str, object] | None:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        if not isinstance(parsed, str):
            return None
        try:
            nested = json.loads(parsed)
        except json.JSONDecodeError:
            return None
        if isinstance(nested, dict):
            return nested
        return None

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = (text or "").strip()
        if not stripped.startswith("```"):
            return text
        lines = stripped.splitlines()
        if len(lines) < 2:
            return text
        if not lines[-1].strip().startswith("```"):
            return text
        return "\n".join(lines[1:-1]).strip()

    def _sources_from_chunks(
        self,
        *,
        chunks: list[Chunk],
        selections: list[_SourceSelection],
        force_all_sources: bool,
    ) -> list[Source]:
        if not chunks:
            return []
        effective_selections = list(selections)
        if force_all_sources:
            effective_selections = [
                _SourceSelection(doc_index=idx)
                for idx in range(1, len(chunks) + 1)
            ]
        if not effective_selections:
            return []

        limit = self._source_max_count
        selected: list[Source] = []
        seen: set[str] = set()
        for selection in effective_selections:
            pos = selection.doc_index - 1
            if pos < 0 or pos >= len(chunks):
                continue
            chunk = chunks[pos]
            ref = self._source_ref_for_selection(
                chunk=chunk,
                sub_index=selection.sub_index,
            )
            if not ref or ref in seen:
                continue
            seen.add(ref)
            source_id = (
                f"{chunk.id}:{selection.doc_index}"
                if selection.sub_index is None
                else f"{chunk.id}:{selection.doc_index}-{selection.sub_index}"
            )
            selected.append(Source(id=source_id, label=ref, uri=ref))
            if limit is not None and len(selected) >= limit:
                break
        return selected

    def _source_ref_for_selection(
        self,
        *,
        chunk: Chunk,
        sub_index: int | None,
    ) -> str | None:
        metadata = chunk.metadata or {}
        source_type = str(metadata.get("source_type") or "").strip().lower()
        if source_type in {"messages", "discord_message"}:
            ref = self._discord_url_for_selection(chunk=chunk, sub_index=sub_index)
            if ref:
                return ref
        ref = _x_url_from_metadata(metadata)
        if ref:
            return ref
        ref = _hatenablog_url_from_metadata(metadata)
        if ref:
            return ref
        ref = _crafters_colony_url_from_metadata(metadata)
        if ref:
            return ref
        ref = _notion_url_from_metadata(metadata)
        if ref:
            return ref
        ref = _drive_url_from_metadata(metadata)
        if ref:
            return ref
        return _vc_source_label_from_metadata(metadata)

    def _discord_url_for_selection(
        self,
        *,
        chunk: Chunk,
        sub_index: int | None,
    ) -> str | None:
        metadata = chunk.metadata or {}
        guild_id = str(metadata.get("guild_id") or "").strip()
        channel_id = str(metadata.get("channel_id") or "").strip()
        if not guild_id or not channel_id:
            return _discord_url_from_metadata(metadata)

        message_id = self._resolve_discord_message_id(
            chunk=chunk,
            sub_index=sub_index,
        )
        if not message_id:
            return _discord_url_from_metadata(metadata)
        return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"

    def _resolve_discord_message_id(
        self,
        *,
        chunk: Chunk,
        sub_index: int | None,
    ) -> str | None:
        metadata = chunk.metadata or {}
        first_message_id = str(metadata.get("first_message_id") or "").strip()
        if not first_message_id:
            first_message_id = str(metadata.get("message_id") or "").strip()
        if not first_message_id and metadata.get("chunk_stage") == "discord_message":
            first_message_id = str(metadata.get("chunk_id") or "").strip()
        if not first_message_id:
            return None
        if sub_index is None or sub_index <= 1:
            return first_message_id

        guild_id = str(metadata.get("guild_id") or "").strip()
        channel_id = str(metadata.get("channel_id") or "").strip()
        if not guild_id or not channel_id:
            return first_message_id

        chunk_lines = self._discord_chunk_lines(chunk.text)
        message_line_indices = self._discord_message_line_indices(chunk_lines)
        if not message_line_indices:
            return first_message_id

        target_position = sub_index - 1
        if target_position >= len(message_line_indices):
            return first_message_id
        target_line_index = message_line_indices[target_position]

        raw_lines = self._discord_raw_chunk_lines(
            guild_id=guild_id,
            channel_id=channel_id,
        )
        if not raw_lines:
            return first_message_id

        start_index = self._resolve_discord_chunk_start_index(
            raw_lines=raw_lines,
            chunk_lines=chunk_lines,
            first_message_id=first_message_id,
        )
        if start_index is None:
            return first_message_id

        raw_target_index = start_index + target_line_index
        if raw_target_index < 0 or raw_target_index >= len(raw_lines):
            return first_message_id
        resolved = raw_lines[raw_target_index].message_id
        return resolved or first_message_id

    @staticmethod
    def _discord_chunk_lines(text: str) -> list[str]:
        return (text or "").splitlines()

    @staticmethod
    def _discord_message_line_indices(lines: Sequence[str]) -> list[int]:
        indices: list[int] = []
        for idx, line in enumerate(lines):
            value = (line or "").strip()
            if not value:
                continue
            if _DISCORD_DATE_LINE_RE.fullmatch(value):
                continue
            indices.append(idx)
        return indices

    def _discord_raw_chunk_lines(
        self,
        *,
        guild_id: str,
        channel_id: str,
    ) -> tuple[_DiscordChunkLine, ...]:
        key = (guild_id, channel_id)
        cached = self._discord_raw_cache.get(key)
        if cached is not None:
            return cached
        if self._raw_dir is None:
            self._discord_raw_cache[key] = tuple()
            return tuple()

        path = self._raw_dir / "messages" / guild_id / f"{channel_id}.jsonl"
        if not path.exists():
            self._discord_raw_cache[key] = tuple()
            return tuple()

        lines: list[_DiscordChunkLine] = []
        last_date: str | None = None
        try:
            with path.open("r", encoding="utf-8") as fr:
                for raw_line in fr:
                    value = raw_line.strip()
                    if not value:
                        continue
                    try:
                        payload = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    text = payload.get("text")
                    metadata = payload.get("metadata")
                    if not isinstance(text, str) or not isinstance(metadata, dict):
                        continue

                    message_id: str | None = None
                    raw_message_id = metadata.get("message_id")
                    if raw_message_id is None:
                        raw_message_id = metadata.get("chunk_id")
                    if raw_message_id is not None:
                        message_id = str(raw_message_id).strip() or None

                    message_date = self._parse_discord_message_date(
                        str(metadata.get("message_timestamp") or "")
                    )
                    if last_date and message_date and message_date != last_date:
                        lines.append(_DiscordChunkLine(text=message_date, message_id=None))
                    if message_date:
                        last_date = message_date

                    author_name = str(metadata.get("author_name") or "unknown").strip()
                    for part in text.splitlines():
                        cleaned_part = part.strip()
                        if not cleaned_part:
                            continue
                        lines.append(
                            _DiscordChunkLine(
                                text=f"{author_name}: {cleaned_part}",
                                message_id=message_id,
                            )
                        )
        except Exception:
            self._discord_raw_cache[key] = tuple()
            return tuple()

        resolved = tuple(lines)
        self._discord_raw_cache[key] = resolved
        return resolved

    @staticmethod
    def _parse_discord_message_date(value: str) -> str | None:
        raw = (value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y/%m/%d")

    def _resolve_discord_chunk_start_index(
        self,
        *,
        raw_lines: Sequence[_DiscordChunkLine],
        chunk_lines: Sequence[str],
        first_message_id: str,
    ) -> int | None:
        candidates = [
            idx
            for idx, raw_line in enumerate(raw_lines)
            if raw_line.message_id == first_message_id
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        best_index = candidates[0]
        best_matches = -1
        best_compared = -1
        for candidate in candidates:
            matches, compared = self._discord_alignment_score(
                raw_lines=raw_lines,
                chunk_lines=chunk_lines,
                start_index=candidate,
            )
            if matches > best_matches or (
                matches == best_matches and compared > best_compared
            ):
                best_index = candidate
                best_matches = matches
                best_compared = compared
        return best_index

    @staticmethod
    def _discord_alignment_score(
        *,
        raw_lines: Sequence[_DiscordChunkLine],
        chunk_lines: Sequence[str],
        start_index: int,
    ) -> tuple[int, int]:
        max_length = min(len(chunk_lines), len(raw_lines) - start_index)
        if max_length <= 0:
            return 0, 0
        matches = 0
        compared = 0
        for offset in range(max_length):
            chunk_value = GenerationComponent._normalize_discord_line(chunk_lines[offset])
            raw_value = GenerationComponent._normalize_discord_line(
                raw_lines[start_index + offset].text
            )
            if not chunk_value or not raw_value:
                continue
            compared += 1
            if (
                chunk_value == raw_value
                or chunk_value in raw_value
                or raw_value in chunk_value
            ):
                matches += 1
        return matches, compared

    @staticmethod
    def _normalize_discord_line(value: str) -> str:
        return " ".join(str(value or "").split())

    def _format_history(
        self,
        history: Sequence[tuple[str, str, Sequence[str]]] | None,
    ) -> str:
        if not history:
            return self._prompt_texts.empty_history
        turns: list[str] = []
        for user_text, assistant_text, sources in history:
            user_value = str(user_text or "").strip()
            assistant_value = str(assistant_text or "").strip()
            turn_lines: list[str] = []
            if user_value:
                turn_lines.append(f"{self._prompt_texts.history_user_prefix}{user_value}")
            if assistant_value:
                turn_lines.append(
                    f"{self._prompt_texts.history_assistant_prefix}{assistant_value}"
                )
            source_values = [str(source or "").strip() for source in (sources or [])]
            source_values = [source for source in source_values if source]
            if source_values:
                label = self._prompt_texts.history_sources_label.strip() or "Sources:"
                turn_lines.append(f"{label} {', '.join(source_values)}")
            if turn_lines:
                turns.append("\n".join(turn_lines))
        if not turns:
            return self._prompt_texts.empty_history
        return "\n\n".join(turns)

    @staticmethod
    def _section(*, header: str, body: str) -> str:
        return f"{header}\n{body}"

    def _question_header(self, *, provider: str) -> str:
        _ = provider
        return self._prompt_texts.gemini_header_question

    def _history_header(self, *, provider: str, retry_mode: bool) -> str:
        _ = provider
        if retry_mode:
            return self._prompt_texts.gemini_header_retry_history
        return self._prompt_texts.gemini_header_chat_history

    def _circle_info_header(self, *, provider: str) -> str:
        _ = provider
        return self._prompt_texts.gemini_header_circle_info

    def _capabilities_header(self, *, provider: str) -> str:
        _ = provider
        return self._prompt_texts.gemini_header_capabilities

    def _context_header(self, *, provider: str) -> str:
        _ = provider
        return self._prompt_texts.gemini_header_context

    def _output_format_header(self, *, provider: str) -> str:
        _ = provider
        return self._prompt_texts.gemini_header_output_format

    def _instructions_header(self, *, provider: str) -> str:
        _ = provider
        return self._prompt_texts.gemini_header_instructions

    def _system_prompt(self) -> str:
        default_system_prompt = "あなたはKUMC Agentです。"
        template = self._prompt_or_default("system_rules", default="")
        if not template:
            return default_system_prompt
        today = datetime.now(ZoneInfo("Asia/Tokyo"))
        weekday = ["月", "火", "水", "木", "金", "土", "日"][today.weekday()]
        today_label = today.strftime("%Y年%m月%d日") + f"（{weekday}）"
        return template.replace("{today_label}", today_label)

    def _capabilities_text(self, *, include_capabilities_info: bool) -> str:
        if not include_capabilities_info:
            return ""
        return self._prompt_or_default("chatbot_capabilities", default="")

    def _circle_basic_info_text(self) -> str:
        return self._prompt_or_default("circle_basic_info", default="")

    def _prompt_or_default(self, name: str, *, default: str) -> str:
        try:
            return self._prompts.get(name).strip()
        except FileNotFoundError:
            return default

    def _prompt_first_available(self, *names: str, default: str = "") -> str:
        for name in names:
            value = self._prompt_or_default(name, default="")
            if value:
                return value
        return default


def _drive_url_from_metadata(metadata: dict[str, object] | None) -> str | None:
    if not metadata:
        return None
    file_id = metadata.get("drive_file_id")
    if not file_id:
        return None

    source_type = str(metadata.get("source_type") or "").strip().lower()
    mime_type = str(metadata.get("drive_mime_type") or "").strip().lower()
    if source_type == "sheets" or "spreadsheet" in mime_type:
        base = "https://docs.google.com/spreadsheets/d/"
    else:
        base = "https://docs.google.com/document/d/"
    return f"{base}{file_id}/"


def _hatenablog_url_from_metadata(metadata: dict[str, object] | None) -> str | None:
    if not metadata:
        return None
    url = str(metadata.get("hatenablog_url") or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return None
    return url


def _crafters_colony_url_from_metadata(
    metadata: dict[str, object] | None,
) -> str | None:
    if not metadata:
        return None
    url = str(metadata.get("crafters_colony_article_url") or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return None
    return url


def _notion_url_from_metadata(metadata: dict[str, object] | None) -> str | None:
    if not metadata:
        return None
    url = str(metadata.get("notion_url") or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return None
    return url


def _discord_url_from_metadata(metadata: dict[str, object] | None) -> str | None:
    if not metadata:
        return None
    source_type = str(metadata.get("source_type") or "").strip().lower()
    if source_type not in {"messages", "discord_message"}:
        return None
    guild_id = str(metadata.get("guild_id") or "").strip()
    channel_id = str(metadata.get("channel_id") or "").strip()
    message_id = str(metadata.get("first_message_id") or "").strip()
    if not message_id:
        message_id = str(metadata.get("message_id") or "").strip()
    if not message_id and metadata.get("chunk_stage") == "discord_message":
        message_id = str(metadata.get("chunk_id") or "").strip()
    if not guild_id or not channel_id or not message_id:
        return None
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def _x_url_from_metadata(metadata: dict[str, object] | None) -> str | None:
    if not metadata:
        return None
    source_type = str(metadata.get("source_type") or "").strip().lower()
    if source_type != "x_posts":
        return None
    direct_url = str(metadata.get("x_post_url") or "").strip()
    if direct_url.lower().startswith(("http://", "https://")):
        return direct_url

    post_id = str(
        metadata.get("x_post_id")
        or metadata.get("tweet_id")
        or metadata.get("first_message_id")
        or metadata.get("message_id")
        or ""
    ).strip()
    if not post_id.isdigit():
        return None
    handle = str(metadata.get("x_author_handle") or "").strip().lstrip("@")
    if handle:
        return f"https://x.com/{handle}/status/{post_id}"
    return f"https://x.com/i/web/status/{post_id}"


def _vc_source_label_from_metadata(metadata: dict[str, object] | None) -> str | None:
    if not metadata:
        return None
    source_type = str(metadata.get("source_type") or "").strip().lower()
    if source_type != "vc_transcript":
        return None

    meeting_date = str(metadata.get("meeting_date") or "").strip()
    if not meeting_date:
        meeting_label = str(metadata.get("meeting_label") or "").strip()
        if meeting_label:
            meeting_date = meeting_label.split(" ", maxsplit=1)[0].strip()
    if not meeting_date:
        return None
    return f"{meeting_date}例会 文字起こし"


def _mask_discord_mentions(text: str) -> str:
    if not text:
        return ""
    masked = _USER_MENTION_RE.sub(_MASKED_MENTION, text)
    return _ROLE_MENTION_RE.sub(_MASKED_MENTION, masked)
