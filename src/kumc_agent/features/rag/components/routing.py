from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Sequence

from kumc_agent.domain.models.routing import RoutingDecision

logger = logging.getLogger(__name__)

ChatHistoryEntry = tuple[str, str, Sequence[str]]


class QueryRouter:
    def __init__(
        self,
        *,
        refusal_keywords: list[str],
        routing_enabled: bool,
        provider: str,
        gemini_model: str,
        llama_model_path: str,
        temperature: float,
        max_new_tokens: int,
        max_retries: int,
        log_enabled: bool,
        material_search_max_names: int,
        llm_thinking_level: str,
        llm_threads: int,
        llm_gpu_layers: int,
        llm_ctx_size: int,
        gemini_api_key: str,
    ) -> None:
        self._refusal_keywords = list(refusal_keywords)
        self._routing_enabled = routing_enabled
        self._provider = provider
        self._gemini_model = gemini_model
        self._llama_model_path = llama_model_path
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._max_retries = max_retries
        self._log_enabled = log_enabled
        self._material_search_max_names = max(1, material_search_max_names)
        self._llm_thinking_level = llm_thinking_level
        self._llm_threads = llm_threads
        self._llm_gpu_layers = llm_gpu_layers
        self._llm_ctx_size = llm_ctx_size
        self._gemini_api_key = gemini_api_key

    def route(
        self,
        query: str,
        *,
        question_author: str | None = None,
        history: Sequence[ChatHistoryEntry] | None = None,
    ) -> RoutingDecision:
        if self._routing_enabled:
            routed = self._route_with_legacy_function_calling(
                query=query,
                question_author=question_author,
                history=history,
            )
            if routed is not None:
                return routed
        return self._heuristic_route(query)

    def _route_with_legacy_function_calling(
        self,
        *,
        query: str,
        question_author: str | None,
        history: Sequence[ChatHistoryEntry] | None,
    ) -> RoutingDecision | None:
        try:
            from kumc_agent.infra.legacy.pipeline.function_calling import decide_tools
        except Exception:
            return None

        config = SimpleNamespace(
            function_call_max_retries=self._max_retries,
            function_call_provider=self._provider,
            function_call_gemini_model=self._gemini_model,
            function_call_llama_model_path=self._llama_model_path,
            function_call_temperature=self._temperature,
            function_call_max_new_tokens=self._max_new_tokens,
            function_call_log_enabled=self._log_enabled,
            material_search_max_names=self._material_search_max_names,
            thinking_level=self._llm_thinking_level,
            gemini_api_key=self._gemini_api_key,
            llama_ctx_size=self._llm_ctx_size,
            llama_threads=self._llm_threads,
            llama_gpu_layers=self._llm_gpu_layers,
        )
        try:
            decision = decide_tools(
                query=query,
                question_author=question_author,
                config=config,
                history=history,
            )
        except Exception:
            logger.exception("Legacy function-call routing failed. Fallback to heuristic.")
            return None

        return RoutingDecision(
            target_model=str(decision.target_model or "rag"),
            recency_mode=str(decision.recency_mode or "off"),
            material_names=list(decision.material_names or []),
            idea_generation=bool(decision.idea_generation),
            include_capabilities_info=bool(decision.include_capabilities_info),
            use_additional_memory=bool(decision.use_additional_memory),
            needs_additional_query=bool(decision.needs_additional_query),
            additional_queries=list(decision.additional_queries or []),
        )

    def _heuristic_route(self, query: str) -> RoutingDecision:
        text = (query or "").strip()
        lowered = text.lower()
        recency_mode = self._heuristic_recency_mode(lowered)
        material_names = self._extract_material_names(text)
        use_additional_memory = any(
            token in text
            for token in ("それ", "これ", "前回", "さっき", "先ほど", "この件", "その件")
        )
        if material_names and "資料" in text:
            return RoutingDecision(
                target_model="material_search",
                recency_mode=recency_mode,
                material_names=material_names[: self._material_search_max_names],
                include_capabilities_info=False,
                use_additional_memory=use_additional_memory,
            )
        is_general = not any(
            token in lowered
            for token in (
                "kumc",
                "京大",
                "サークル",
                "例会",
                "minecraft",
                "マイクラ",
                "同好会",
            )
        )
        if is_general:
            return RoutingDecision(
                target_model="no_rag",
                recency_mode="off",
                use_additional_memory=use_additional_memory,
            )
        return RoutingDecision(
            target_model="rag",
            recency_mode=recency_mode,
            use_additional_memory=use_additional_memory,
            needs_additional_query=False,
            additional_queries=[],
        )

    @staticmethod
    def _heuristic_recency_mode(lowered: str) -> str:
        if any(token in lowered for token in ("最新", "今日", "きょう", "直近", "今週")):
            return "hard"
        if any(token in lowered for token in ("最近", "今月", "近況")):
            return "soft"
        return "off"

    @staticmethod
    def _extract_material_names(text: str) -> list[str]:
        names: list[str] = []
        for pattern in (r"「([^」]+)」", r"\"([^\"]+)\""):
            for match in re.findall(pattern, text):
                value = str(match).strip()
                if value:
                    names.append(value)
        if not names and "資料" in text:
            tail = text.split("資料", 1)[-1].strip(" :：")
            if tail:
                names.append(tail)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in names:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped
