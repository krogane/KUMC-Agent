from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Sequence

from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.domain.ports.prompts import PromptRepositoryPort

logger = logging.getLogger(__name__)

ChatHistoryEntry = tuple[str, str, Sequence[str]]


@dataclass(frozen=True)
class QuerySynthesisResult:
    synthetic_query: str
    used: bool
    fallback: bool
    raw: str = ""


class QuerySynthesizer:
    def __init__(
        self,
        *,
        llm: LLMPort,
        prompts: PromptRepositoryPort,
        prompt_name: str = "query_synthesis",
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        max_retries: int = 1,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._prompt_name = str(prompt_name or "query_synthesis").strip() or "query_synthesis"
        self._temperature = float(temperature)
        self._max_output_tokens = max(1, int(max_output_tokens))
        self._max_retries = max(0, int(max_retries))

    def synthesize(
        self,
        *,
        query: str,
        history: Sequence[ChatHistoryEntry] | None,
        additional_queries: Sequence[str],
        use_additional_memory: bool,
    ) -> QuerySynthesisResult:
        cleaned_query = str(query or "").strip()
        if not cleaned_query:
            return QuerySynthesisResult("", used=False, fallback=True)

        normalized_additional = [
            str(value or "").strip()
            for value in additional_queries
            if str(value or "").strip()
        ]
        should_synthesize = bool(use_additional_memory or normalized_additional)
        if not should_synthesize:
            return QuerySynthesisResult(cleaned_query, used=False, fallback=False)

        system_prompt = self._prompt()
        user_prompt = self._user_prompt(
            query=cleaned_query,
            history=history,
            additional_queries=normalized_additional,
        )
        last_raw = ""
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=self._temperature,
                    max_output_tokens=self._max_output_tokens,
                )
            except Exception:
                logger.exception("Query synthesis generation failed.")
                break
            last_raw = raw
            parsed = self._parse(raw)
            if parsed:
                return QuerySynthesisResult(
                    synthetic_query=parsed,
                    used=True,
                    fallback=False,
                    raw=last_raw,
                )
            if attempt < self._max_retries:
                continue
        return QuerySynthesisResult(
            synthetic_query=cleaned_query,
            used=True,
            fallback=True,
            raw=last_raw,
        )

    def _prompt(self) -> str:
        try:
            prompt = self._prompts.get(self._prompt_name)
        except Exception:
            prompt = ""
        if prompt:
            return prompt
        return (
            "あなたはKUMCサークル情報RAGの検索クエリ合成エンジンです。"
            "入力質問、同一チャンネル履歴、追加観点を使い、検索に使う単一の日本語クエリだけをJSONで返してください。"
            "回答や説明は書かないでください。"
        )

    @staticmethod
    def _user_prompt(
        *,
        query: str,
        history: Sequence[ChatHistoryEntry] | None,
        additional_queries: Sequence[str],
    ) -> str:
        history_lines: list[str] = []
        for user, assistant, _sources in history or []:
            user_text = str(user or "").strip()
            assistant_text = str(assistant or "").strip()
            if user_text:
                history_lines.append(f"ユーザー: {user_text}")
            if assistant_text:
                history_lines.append(f"アシスタント: {assistant_text}")
        payload = {
            "query": query,
            "history": "\n".join(history_lines),
            "additional_queries": list(additional_queries),
            "output_format": {"synthetic_query": "string"},
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _parse(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        cleaned = raw
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2 and lines[-1].strip().startswith("```"):
                cleaned = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return raw if len(raw) <= 512 and "\n" not in raw else ""
        if isinstance(payload, dict):
            value = str(payload.get("synthetic_query") or "").strip()
            return value[:512]
        if isinstance(payload, str):
            return payload.strip()[:512]
        return ""
