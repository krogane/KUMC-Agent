from __future__ import annotations

import json

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
        temperature: float,
        max_output_tokens: int,
        thinking_level: str,
    ) -> Answer:
        context = "\n\n".join(
            f"[{i}] {chunk.text}"
            for i, chunk in enumerate(chunks, start=1)
        )
        prompt = self._prompts.get("answer_json")
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            f"Output instruction:\n{prompt}"
        )
        raw = self._llm.generate(
            system_prompt="あなたはKUMC Agentです。",
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
        )
        answer_text, source_indexes = self._parse_answer(raw)
        selected_sources = self._sources_from_chunks(chunks, source_indexes)
        if not answer_text:
            answer_text = raw.strip()
        answer_text = answer_text + format_sources(selected_sources)
        return Answer(
            text=answer_text,
            route="rag",
            sources=selected_sources,
            metadata={"raw": raw},
        )

    def generate_refusal(
        self,
        *,
        query: str,
        temperature: float,
        max_output_tokens: int,
        thinking_level: str,
    ) -> Answer:
        refusal = self._prompts.get("refusal")
        return Answer(
            text=refusal,
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
