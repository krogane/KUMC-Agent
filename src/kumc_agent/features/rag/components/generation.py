from __future__ import annotations

from datetime import datetime
import json
from typing import Sequence
from zoneinfo import ZoneInfo

from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.source import Source
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.domain.ports.prompts import PromptRepositoryPort
from kumc_agent.domain.policies.source_format import format_sources
from kumc_agent.features.rag.config import RagPromptTextSettings


class GenerationComponent:
    def __init__(
        self,
        *,
        llm: LLMPort,
        no_rag_llm: LLMPort | None = None,
        refusal_llm: LLMPort | None = None,
        prompts: PromptRepositoryPort,
        source_max_count: int,
        prompt_texts: RagPromptTextSettings | None = None,
    ) -> None:
        self._rag_llm = llm
        self._no_rag_llm = no_rag_llm or llm
        self._refusal_llm = refusal_llm or self._no_rag_llm
        self._prompts = prompts
        self._source_max_count = source_max_count
        self._prompt_texts = prompt_texts or RagPromptTextSettings()

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
        thinking_level: str,
        answer_prompt_name: str = "answer_json",
        append_sources_to_response: bool = True,
        extra_mode_instruction: str | None = None,
    ) -> Answer:
        context = "\n\n".join(
            f"[{i}] {chunk.text}"
            for i, chunk in enumerate(chunks, start=1)
        ) or self._prompt_texts.empty_context
        prompt = self._prompt_or_default(
            answer_prompt_name,
            default=self._prompts.get("answer_json"),
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
        raw = self._rag_llm.generate(
            system_prompt=self._system_prompt(),
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
        )
        answer_text, source_indexes = self._parse_answer(raw)
        selected_sources = self._sources_from_chunks(chunks, source_indexes)
        if not answer_text:
            answer_text = raw.strip()
        final_text = (
            answer_text + format_sources(selected_sources)
            if append_sources_to_response
            else answer_text
        )
        return Answer(
            text=final_text,
            route="rag",
            sources=selected_sources,
            metadata={
                "raw": raw,
                "source_indexes": source_indexes,
                "contexts": [chunk.text for chunk in chunks],
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
        thinking_level: str,
        answer_prompt_name: str = "answer_json",
        extra_mode_instruction: str | None = None,
    ) -> Answer:
        prompt = self._prompt_or_default(
            answer_prompt_name,
            default=self._prompts.get("answer_json"),
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
        raw = self._no_rag_llm.generate(
            system_prompt=self._system_prompt(),
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
        )
        answer_text, _ = self._parse_answer(raw)
        if not answer_text:
            answer_text = raw.strip()
        return Answer(
            text=answer_text or "回答生成中に不具合が発生しました。もう一度お試しください。",
            route="no_rag",
            sources=[],
            metadata={"raw": raw},
        )

    def generate_refusal(
        self,
        *,
        query: str,
        history: Sequence[tuple[str, str, Sequence[str]]] | None,
        provider: str = "gemini",
        temperature: float,
        max_output_tokens: int,
        thinking_level: str,
        refusal_prompt_name: str = "refusal",
        extra_mode_instruction: str | None = None,
    ) -> Answer:
        history_text = self._format_history(history)
        refusal = self._prompt_or_default(
            refusal_prompt_name,
            default=self._prompt_or_default("refusal", default=""),
        )
        fixed_prefix = "安全上の理由により、この質問には回答できません。"
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
        instruction_parts: list[str] = []
        if instruction_text:
            instruction_parts.append(instruction_text)
        if refusal:
            instruction_parts.append(refusal)
        if instruction_parts:
            sections.append(
                self._section(
                    header=self._instructions_header(provider=provider),
                    body="\n\n".join(instruction_parts),
                )
            )
        sections.append(
            self._section(
                header=self._output_format_header(provider=provider),
                body=(
                    "- 安全上の理由で回答できないことを簡潔に伝えてください。\n"
                    "- 機密情報の推測・言い換え・部分開示はしないでください。"
                ),
            )
        )
        user_prompt = "\n\n".join(sections)
        raw = self._refusal_llm.generate(
            system_prompt=self._system_prompt(),
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
        )
        text = raw.strip()
        if not text:
            text = fixed_prefix if not refusal else f"{fixed_prefix}\n\n{refusal}"
        return Answer(
            text=text,
            route="refusal",
            sources=[],
            metadata={"query": query, "raw": raw},
        )

    @staticmethod
    def _parse_answer(raw: str) -> tuple[str, list[int]]:
        try:
            payload = json.loads(raw)
            answer = str(payload.get("answer") or "").strip()
            source_indexes: list[int] = []
            raw_sources = payload.get("sources")
            if isinstance(raw_sources, list):
                for item in raw_sources:
                    try:
                        source_indexes.append(int(str(item)))
                    except ValueError:
                        continue
            return answer, source_indexes
        except json.JSONDecodeError:
            return "", []

    def _sources_from_chunks(self, chunks: list[Chunk], indexes: list[int]) -> list[Source]:
        if not chunks:
            return []
        selected: list[Source] = []
        if indexes:
            for idx in indexes:
                pos = idx - 1
                if pos < 0 or pos >= len(chunks):
                    continue
                chunk = chunks[pos]
                selected.append(
                    Source(
                        id=chunk.id,
                        label=str(chunk.metadata.get("source_name") or chunk.document_id),
                        uri=str(chunk.metadata.get("source_uri") or ""),
                    )
                )
        if not selected:
            for chunk in chunks[: self._source_max_count]:
                selected.append(
                    Source(
                        id=chunk.id,
                        label=str(chunk.metadata.get("source_name") or chunk.document_id),
                        uri=str(chunk.metadata.get("source_uri") or ""),
                    )
                )
        return selected[: self._source_max_count]

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

    def _is_llama_provider(self, provider: str) -> bool:
        normalized = (provider or "").strip().lower().replace(".", "_")
        return normalized in {"llama", "llama_cpp"}

    def _question_header(self, *, provider: str) -> str:
        if self._is_llama_provider(provider):
            return self._prompt_texts.llama_header_question
        return self._prompt_texts.gemini_header_question

    def _history_header(self, *, provider: str, retry_mode: bool) -> str:
        if self._is_llama_provider(provider):
            return self._prompt_texts.llama_header_previous_attempt
        if retry_mode:
            return self._prompt_texts.gemini_header_retry_history
        return self._prompt_texts.gemini_header_chat_history

    def _circle_info_header(self, *, provider: str) -> str:
        if self._is_llama_provider(provider):
            return self._prompt_texts.llama_header_circle_info
        return self._prompt_texts.gemini_header_circle_info

    def _capabilities_header(self, *, provider: str) -> str:
        if self._is_llama_provider(provider):
            return self._prompt_texts.llama_header_capabilities
        return self._prompt_texts.gemini_header_capabilities

    def _context_header(self, *, provider: str) -> str:
        if self._is_llama_provider(provider):
            return self._prompt_texts.llama_header_context
        return self._prompt_texts.gemini_header_context

    def _output_format_header(self, *, provider: str) -> str:
        if self._is_llama_provider(provider):
            return self._prompt_texts.llama_header_output_format
        return self._prompt_texts.gemini_header_output_format

    def _instructions_header(self, *, provider: str) -> str:
        if self._is_llama_provider(provider):
            return self._prompt_texts.llama_header_instructions
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
