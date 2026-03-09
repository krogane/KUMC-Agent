from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
import logging
from typing import Sequence

from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.routing import RoutingDecision
from kumc_agent.features.rag.config import RagConfig, RagGenerationSettings
from kumc_agent.features.rag.components.generation import GenerationComponent
from kumc_agent.features.rag.components.retrieval import RetrievalComponent
from kumc_agent.features.rag.components.routing import QueryRouter
from kumc_agent.infra.retrieval.cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)

ChatHistoryEntry = tuple[str, str, Sequence[str]]


class RagService:
    def __init__(
        self,
        *,
        config: RagConfig,
        router: QueryRouter,
        retrieval: RetrievalComponent,
        generation: GenerationComponent,
        reranker: CrossEncoderReranker | None,
    ) -> None:
        self._config = config
        self._router = router
        self._retrieval = retrieval
        self._generation = generation
        self._reranker = reranker
        self._chat_histories: dict[str, deque[ChatHistoryEntry]] = {}

    def answer(
        self,
        *,
        query: str,
        question_author: str | None = None,
        history_scope: str | int | None = None,
        force_fast_mode: bool = False,
        force_disable_additional_memory: bool = False,
        routing_history_override: Sequence[ChatHistoryEntry] | None = None,
        generation_history_override: Sequence[ChatHistoryEntry] | None = None,
        append_sources_to_response: bool = True,
        extra_mode_instruction: str | None = None,
    ) -> Answer:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return Answer(text="", route="none", metadata={"reason": "empty_query"})

        routing_history: Sequence[ChatHistoryEntry] | None
        if routing_history_override is not None:
            routing_history = list(routing_history_override)
        else:
            routing_history = self._history_for_prompt(
                limit=self._config.prompt_default_turns,
                include_sources=False,
                history_scope=history_scope,
            )
        decision = self._router.route(
            cleaned_query,
            question_author=question_author,
            history=routing_history,
        )
        if force_disable_additional_memory and decision.use_additional_memory:
            decision = replace(decision, use_additional_memory=False)
        if force_fast_mode:
            decision = replace(
                decision,
                target_model="rag",
                material_names=[],
                needs_additional_query=False,
                additional_queries=[],
            )

        if generation_history_override is not None:
            generation_history = list(generation_history_override)
        else:
            generation_history = self._history_for_prompt(
                limit=(
                    self._config.prompt_additional_turns
                    if decision.use_additional_memory
                    else self._config.prompt_default_turns
                ),
                include_sources=False,
                history_scope=history_scope,
            )

        if decision.target_model == "refusal":
            refusal_generation = self._resolve_generation_settings(
                target_model="refusal",
                idea_generation=False,
            )
            answer = self._generation.generate_refusal(
                query=cleaned_query,
                history=generation_history,
                provider=refusal_generation.provider,
                temperature=refusal_generation.temperature,
                max_output_tokens=refusal_generation.max_output_tokens,
                thinking_level=refusal_generation.thinking_level,
                refusal_prompt_name=refusal_generation.prompt_name,
                extra_mode_instruction=extra_mode_instruction,
            )
            return self._finalize_answer(
                query=cleaned_query,
                answer=answer,
                routing_decision=decision,
                force_fast_mode=force_fast_mode,
                history_scope=history_scope,
            )

        if decision.target_model == "no_rag":
            no_rag_generation = self._resolve_generation_settings(
                target_model="no_rag",
                idea_generation=decision.idea_generation,
            )
            answer = self._generation.generate_no_rag(
                query=cleaned_query,
                history=generation_history,
                provider=no_rag_generation.provider,
                include_capabilities_info=decision.include_capabilities_info,
                temperature=no_rag_generation.temperature,
                max_output_tokens=no_rag_generation.max_output_tokens,
                thinking_level=no_rag_generation.thinking_level,
                answer_prompt_name=no_rag_generation.prompt_name,
                extra_mode_instruction=extra_mode_instruction,
            )
            return self._finalize_answer(
                query=cleaned_query,
                answer=answer,
                routing_decision=decision,
                force_fast_mode=force_fast_mode,
                history_scope=history_scope,
            )

        effective_recency_mode = self._resolve_recency_mode(decision.recency_mode)
        chunks = self._retrieve_chunks(
            query=cleaned_query,
            decision=decision,
            force_fast_mode=force_fast_mode,
            recency_mode=effective_recency_mode,
        )
        if decision.target_model == "material_search":
            material_filtered = self._filter_material_chunks(
                chunks=chunks,
                material_names=decision.material_names,
            )
            if material_filtered:
                chunks = material_filtered
            else:
                logger.info(
                    "Material search route had no matched chunks. Falling back to standard RAG chunks."
                )

        if self._reranker is not None and chunks and not force_fast_mode:
            chunks = self._reranker.rerank(
                cleaned_query,
                chunks,
                top_k=min(self._config.rerank_pool_size, len(chunks)),
            )
        if chunks and not force_fast_mode:
            chunks = self._retrieval.reorder_with_mmr(
                query=cleaned_query,
                chunks=chunks,
                mmr_lambda=self._config.mmr_lambda,
            )
        chunks = self._apply_recency_order(
            chunks=chunks,
            recency_mode=effective_recency_mode,
            recency_weight_soft=self._config.recency_weight_soft,
            recency_weight_hard=self._config.recency_weight_hard,
            recency_half_life_days=self._config.recency_half_life_days,
        )

        chunks = chunks[: self._config.top_k]
        if not chunks:
            no_rag_generation = self._resolve_generation_settings(
                target_model="no_rag",
                idea_generation=decision.idea_generation,
            )
            answer = self._generation.generate_no_rag(
                query=cleaned_query,
                history=generation_history,
                provider=no_rag_generation.provider,
                include_capabilities_info=decision.include_capabilities_info,
                temperature=no_rag_generation.temperature,
                max_output_tokens=no_rag_generation.max_output_tokens,
                thinking_level=no_rag_generation.thinking_level,
                answer_prompt_name=no_rag_generation.prompt_name,
                extra_mode_instruction=extra_mode_instruction,
            )
            return self._finalize_answer(
                query=cleaned_query,
                answer=answer,
                routing_decision=decision,
                force_fast_mode=force_fast_mode,
                history_scope=history_scope,
            )

        rag_generation = self._resolve_generation_settings(
            target_model="rag",
            idea_generation=decision.idea_generation,
        )
        answer = self._generation.generate_rag_answer(
            query=cleaned_query,
            chunks=chunks,
            history=generation_history,
            provider=rag_generation.provider,
            include_capabilities_info=decision.include_capabilities_info,
            temperature=rag_generation.temperature,
            max_output_tokens=rag_generation.max_output_tokens,
            thinking_level=rag_generation.thinking_level,
            answer_prompt_name=rag_generation.prompt_name,
            append_sources_to_response=append_sources_to_response,
            extra_mode_instruction=extra_mode_instruction,
        )
        if decision.target_model == "material_search":
            answer = replace(answer, route="material_search")
        return self._finalize_answer(
            query=cleaned_query,
            answer=answer,
            routing_decision=decision,
            force_fast_mode=force_fast_mode,
            history_scope=history_scope,
        )

    def _resolve_generation_settings(
        self,
        *,
        target_model: str,
        idea_generation: bool,
    ) -> RagGenerationSettings:
        normalized = (target_model or "").strip().lower()
        if normalized == "no_rag":
            base = self._config.no_rag_generation
        elif normalized == "refusal":
            base = self._config.refusal_generation
        else:
            base = self._config.rag_generation
        if normalized != "refusal" and idea_generation:
            prompt_name = (self._config.idea_generation.prompt_name or "").strip()
            if not prompt_name:
                prompt_name = base.prompt_name
            return replace(
                base,
                prompt_name=prompt_name,
                temperature=self._config.idea_generation.temperature,
            )
        return base

    def _retrieve_chunks(
        self,
        *,
        query: str,
        decision: RoutingDecision,
        force_fast_mode: bool,
        recency_mode: str,
    ):
        chunks = self._retrieval.retrieve(
            query,
            dense_top_k=self._config.dense_top_k,
            sparse_top_k=self._config.sparse_top_k,
            recency_mode=recency_mode,
            recency_weight_soft=self._config.recency_weight_soft,
            recency_weight_hard=self._config.recency_weight_hard,
            recency_half_life_days=self._config.recency_half_life_days,
            mmr_lambda=self._config.mmr_lambda,
        )
        if force_fast_mode:
            return chunks
        if decision.needs_additional_query and decision.additional_queries:
            merged = {chunk.id: chunk for chunk in chunks}
            for transformed_query in decision.additional_queries:
                extra = self._retrieval.retrieve(
                    transformed_query,
                    dense_top_k=self._config.dense_top_k,
                    sparse_top_k=self._config.sparse_top_k,
                    recency_mode=recency_mode,
                    recency_weight_soft=self._config.recency_weight_soft,
                    recency_weight_hard=self._config.recency_weight_hard,
                    recency_half_life_days=self._config.recency_half_life_days,
                    mmr_lambda=self._config.mmr_lambda,
                )
                for chunk in extra:
                    merged[chunk.id] = chunk
            return list(merged.values())
        return chunks

    def _resolve_recency_mode(self, recency_mode: str) -> str:
        normalized = str(recency_mode or "").strip().lower()
        if normalized in {"off", "soft", "hard"}:
            return normalized
        fallback = str(self._config.recency_mode or "off").strip().lower()
        if fallback in {"off", "soft", "hard"}:
            return fallback
        return "off"

    @staticmethod
    def _apply_recency_order(
        *,
        chunks,
        recency_mode: str,
        recency_weight_soft: float,
        recency_weight_hard: float,
        recency_half_life_days: float,
    ):
        mode = (recency_mode or "off").strip().lower()
        if mode not in {"soft", "hard"}:
            return chunks
        if len(chunks) <= 1:
            return chunks

        soft_weight = max(0.0, min(1.0, float(recency_weight_soft)))
        hard_weight = max(0.0, min(1.0, float(recency_weight_hard)))
        weight = soft_weight if mode == "soft" else hard_weight
        half_life_days = max(0.1, float(recency_half_life_days))
        now = datetime.now(timezone.utc)
        scored = []
        count = max(1, len(chunks))
        for idx, chunk in enumerate(chunks):
            base_score = 1.0 - (idx / count)
            updated_at = RagService._chunk_updated_at(chunk)
            if updated_at is None:
                recency_score = 0.5
            else:
                age_days = max(0.0, (now - updated_at).total_seconds() / 86400.0)
                recency_score = 0.5 ** (age_days / half_life_days)
            final_score = ((1.0 - weight) * base_score) + (weight * recency_score)
            scored.append((final_score, idx, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk for _, _, chunk in scored]

    @staticmethod
    def _chunk_updated_at(chunk) -> datetime | None:
        metadata = chunk.metadata or {}
        for key in (
            "updated_at",
            "message_timestamp",
            "drive_modified_time",
            "hatenablog_updated_at",
            "hatenablog_created_at",
            "crafters_colony_published_at",
            "created_at",
            "source_date",
            "first_message_date",
        ):
            raw = str(metadata.get(key) or "").strip()
            if not raw:
                continue
            parsed = RagService._parse_datetime(raw)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        raw = (value or "").strip()
        if not raw or raw == "不明":
            return None
        iso_value = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(iso_value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def _filter_material_chunks(
        *,
        chunks,
        material_names: Sequence[str],
    ):
        if not chunks or not material_names:
            return []
        lowered_names = [name.casefold() for name in material_names if name.strip()]
        out = []
        for chunk in chunks:
            metadata = chunk.metadata or {}
            candidates = [
                str(metadata.get("source_name") or ""),
                str(metadata.get("source_file_name") or ""),
                str(metadata.get("drive_file_id") or ""),
                str(metadata.get("path") or ""),
            ]
            composite = " ".join(candidates).casefold()
            if any(name in composite for name in lowered_names):
                out.append(chunk)
        return out

    def _finalize_answer(
        self,
        *,
        query: str,
        answer: Answer,
        routing_decision: RoutingDecision,
        force_fast_mode: bool,
        history_scope: str | int | None,
    ) -> Answer:
        text = answer.text
        if force_fast_mode and text:
            text = f"{self._config.fast_model_notice}\n\n{text}"
        metadata = dict(answer.metadata)
        metadata["routing_decision"] = {
            "target_model": routing_decision.target_model,
            "recency_mode": routing_decision.recency_mode,
            "material_names": list(routing_decision.material_names),
            "idea_generation": routing_decision.idea_generation,
            "include_capabilities_info": routing_decision.include_capabilities_info,
            "use_additional_memory": routing_decision.use_additional_memory,
            "needs_additional_query": routing_decision.needs_additional_query,
            "additional_queries": list(routing_decision.additional_queries),
        }
        metadata["fast_mode"] = force_fast_mode
        finalized = replace(answer, text=text, metadata=metadata)
        self._record_history(
            query=query,
            answer=finalized.text,
            sources=[source.label for source in finalized.sources],
            history_scope=history_scope,
        )
        return finalized

    def _history_for_prompt(
        self,
        *,
        limit: int,
        include_sources: bool,
        history_scope: str | int | None,
    ) -> list[ChatHistoryEntry] | None:
        if not self._config.history_enabled:
            return None
        history_bucket = self._history_bucket(history_scope=history_scope)
        if limit <= 0 or history_bucket.maxlen == 0:
            return []
        selected = list(history_bucket)
        if len(selected) > limit:
            selected = selected[-limit:]
        if include_sources:
            return selected
        return [(user, assistant, []) for user, assistant, _ in selected]

    def _record_history(
        self,
        *,
        query: str,
        answer: str,
        sources: Sequence[str],
        history_scope: str | int | None,
    ) -> None:
        if not self._config.history_enabled:
            return
        history_bucket = self._history_bucket(history_scope=history_scope)
        user_text = (query or "").strip()
        assistant_text = (answer or "").strip()
        if not user_text or not assistant_text:
            return
        history_bucket.append((user_text, assistant_text, list(sources)))

    def _history_bucket(
        self,
        *,
        history_scope: str | int | None,
    ) -> deque[ChatHistoryEntry]:
        key = self._normalize_history_scope(history_scope)
        bucket = self._chat_histories.get(key)
        if bucket is not None:
            return bucket
        bucket = deque(maxlen=max(0, self._config.history_max_turns))
        self._chat_histories[key] = bucket
        return bucket

    @staticmethod
    def _normalize_history_scope(history_scope: str | int | None) -> str:
        if history_scope is None:
            return "__default__"
        value = str(history_scope).strip()
        return value or "__default__"
