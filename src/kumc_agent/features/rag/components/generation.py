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


class GenerationComponent:
    def __init__(
        self,
        *,
        llm: LLMPort,
        prompts: PromptRepositoryPort,
        source_max_count: int,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._source_max_count = source_max_count

    def generate_rag_answer(
        self,
        *,
        query: str,
        chunks: list[Chunk],
        history: Sequence[tuple[str, str, Sequence[str]]] | None,
        include_capabilities_info: bool = False,
        temperature: float,
        max_output_tokens: int,
        thinking_level: str,
        append_sources_to_response: bool = True,
        extra_mode_instruction: str | None = None,
    ) -> Answer:
        context = "\n\n".join(
            f"[{i}] {chunk.text}"
            for i, chunk in enumerate(chunks, start=1)
        )
        prompt = self._prompts.get("answer_json")
        history_text = self._format_history(history)
        capabilities_text = self._capabilities_text(
            include_capabilities_info=include_capabilities_info
        )
        instruction_text = (extra_mode_instruction or "").strip()
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"History:\n{history_text}\n\n"
            f"Context:\n{context}\n\n"
        )
        if capabilities_text:
            user_prompt += f"Capabilities:\n{capabilities_text}\n\n"
        if instruction_text:
            user_prompt += f"Mode instruction:\n{instruction_text}\n\n"
        user_prompt += f"Output instruction:\n{prompt}"
        raw = self._llm.generate(
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
            metadata={"raw": raw, "source_indexes": source_indexes},
        )

    def generate_no_rag(
        self,
        *,
        query: str,
        history: Sequence[tuple[str, str, Sequence[str]]] | None,
        include_capabilities_info: bool = False,
        temperature: float,
        max_output_tokens: int,
        thinking_level: str,
        extra_mode_instruction: str | None = None,
    ) -> Answer:
        prompt = self._prompts.get("answer_json")
        history_text = self._format_history(history)
        capabilities_text = self._capabilities_text(
            include_capabilities_info=include_capabilities_info
        )
        instruction_text = (extra_mode_instruction or "").strip()
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"History:\n{history_text}\n\n"
        )
        if capabilities_text:
            user_prompt += f"Capabilities:\n{capabilities_text}\n\n"
        if instruction_text:
            user_prompt += f"Mode instruction:\n{instruction_text}\n\n"
        user_prompt += f"Output instruction:\n{prompt}"
        raw = self._llm.generate(
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
        temperature: float,
        max_output_tokens: int,
        thinking_level: str,
        extra_mode_instruction: str | None = None,
    ) -> Answer:
        _ = (query, history, temperature, max_output_tokens, thinking_level)
        refusal = self._prompts.get("refusal").strip()
        fixed_prefix = "安全上の理由により、この質問には回答できません。"
        extra = (extra_mode_instruction or "").strip()
        if extra:
            refusal = f"{refusal}\n\n{extra}" if refusal else extra
        text = fixed_prefix if not refusal else f"{fixed_prefix}\n\n{refusal}"
        return Answer(
            text=text,
            route="refusal",
            sources=[],
            metadata={"query": query},
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

    @staticmethod
    def _format_history(
        history: Sequence[tuple[str, str, Sequence[str]]] | None,
    ) -> str:
        if not history:
            return "なし"
        lines: list[str] = []
        for user_text, assistant_text, _ in history:
            user_value = str(user_text or "").strip()
            assistant_value = str(assistant_text or "").strip()
            if user_value:
                lines.append(f"User: {user_value}")
            if assistant_value:
                lines.append(f"Assistant: {assistant_value}")
        return "\n".join(lines) if lines else "なし"

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

    def _prompt_or_default(self, name: str, *, default: str) -> str:
        try:
            return self._prompts.get(name).strip()
        except FileNotFoundError:
            return default
