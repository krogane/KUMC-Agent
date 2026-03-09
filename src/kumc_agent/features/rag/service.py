from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Sequence

from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.routing import RoutingDecision
from kumc_agent.features.rag.config import RagConfig, RagGenerationSettings
from kumc_agent.features.rag.components.generation import GenerationComponent
from kumc_agent.features.rag.components.retrieval import RetrievalComponent
from kumc_agent.features.rag.components.routing import QueryRouter
from kumc_agent.infra.retrieval.cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)

ChatHistoryEntry = tuple[str, str, Sequence[str]]
_MATERIAL_SEARCH_EXCLUDED_SOURCE_TYPES = frozenset(
    {"messages", "discord_message", "x_posts"}
)


@dataclass(frozen=True)
class _MaterialCatalogEntry:
    material_id: str
    source_type: str
    source_key: str
    canonical_name: str
    aliases: tuple[str, ...]
    raw_path: str


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
                json_max_retries=self._config.answer_json_max_retries,
            )
            return self._finalize_answer(
                query=cleaned_query,
                answer=answer,
                routing_decision=decision,
                force_fast_mode=force_fast_mode,
                history_scope=history_scope,
            )

        effective_recency_mode = self._resolve_recency_mode(decision.recency_mode)
        if decision.target_model == "material_search":
            chunks = self._retrieve_material_route_chunks(
                query=cleaned_query,
                decision=decision,
                recency_mode=effective_recency_mode,
                force_fast_mode=force_fast_mode,
            )
            material_route = True
        else:
            chunks = self._retrieve_chunks(
                query=cleaned_query,
                decision=decision,
                force_fast_mode=force_fast_mode,
                recency_mode=effective_recency_mode,
                excluded_source_types=None,
            )
            material_route = False

        chunks = self._rank_and_select_chunks(
            query=cleaned_query,
            chunks=chunks,
            recency_mode=effective_recency_mode,
            force_fast_mode=force_fast_mode,
        )

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
                json_max_retries=self._config.answer_json_max_retries,
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
            json_max_retries=self._config.answer_json_max_retries,
            force_all_sources=material_route,
        )
        if material_route:
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
        excluded_source_types: set[str] | None,
    ) -> list[Chunk]:
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
        chunks = self._filter_chunks_by_source_type(
            chunks,
            excluded_source_types=excluded_source_types,
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
                extra = self._filter_chunks_by_source_type(
                    extra,
                    excluded_source_types=excluded_source_types,
                )
                for chunk in extra:
                    merged[chunk.id] = chunk
            return list(merged.values())
        return chunks

    def _retrieve_material_route_chunks(
        self,
        *,
        query: str,
        decision: RoutingDecision,
        recency_mode: str,
        force_fast_mode: bool,
    ) -> list[Chunk]:
        excluded_source_types = set(_MATERIAL_SEARCH_EXCLUDED_SOURCE_TYPES)
        matched_entries = self._match_material_entries(
            material_names=decision.material_names,
            query=query,
            excluded_source_types=excluded_source_types,
        )
        if not matched_entries:
            return self._retrieve_chunks(
                query=query,
                decision=replace(
                    decision,
                    target_model="rag",
                    needs_additional_query=False,
                    additional_queries=[],
                ),
                force_fast_mode=force_fast_mode,
                recency_mode=recency_mode,
                excluded_source_types=excluded_source_types,
            )

        material_keys = {
            self._normalize_material_key(entry.source_type, entry.source_key)
            for entry in matched_entries
        }
        searched = self._retrieve_material_limited_chunks(
            query=query,
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
            recency_mode=recency_mode,
        )
        if not searched:
            return self._retrieve_chunks(
                query=query,
                decision=replace(
                    decision,
                    target_model="rag",
                    needs_additional_query=False,
                    additional_queries=[],
                ),
                force_fast_mode=force_fast_mode,
                recency_mode=recency_mode,
                excluded_source_types=excluded_source_types,
            )

        contexts = self._build_material_context_chunks(
            query=query,
            entries=matched_entries,
            searched_chunks=searched,
            force_fast_mode=force_fast_mode,
        )
        if contexts:
            return contexts
        return self._retrieve_chunks(
            query=query,
            decision=replace(
                decision,
                target_model="rag",
                needs_additional_query=False,
                additional_queries=[],
            ),
            force_fast_mode=force_fast_mode,
            recency_mode=recency_mode,
            excluded_source_types=excluded_source_types,
        )

    def _rank_and_select_chunks(
        self,
        *,
        query: str,
        chunks: list[Chunk],
        recency_mode: str,
        force_fast_mode: bool,
    ) -> list[Chunk]:
        if not chunks:
            return []

        ranked = list(chunks)
        if self._reranker is not None and ranked and not force_fast_mode:
            scored = self._reranker.score_documents(query=query, chunks=ranked)
            scored = self._apply_recency_scores(
                scored=scored,
                recency_mode=recency_mode,
            )
            scored.sort(key=lambda item: (-item[0], item[1]))
            ranked = [chunk for _, _, chunk in scored]

        ranked = self._apply_parent_chunk_cap(ranked)

        if self._reranker is not None and ranked and not force_fast_mode:
            pool_size = max(0, int(self._config.rerank_pool_size))
            if pool_size > 0:
                ranked = ranked[:pool_size]

        if force_fast_mode:
            selected = ranked[: max(0, int(self._config.top_k))]
        else:
            selected = self._retrieval.reorder_with_mmr(
                query=query,
                chunks=ranked,
                mmr_lambda=self._config.mmr_lambda,
            )
            selected = selected[: max(0, int(self._config.top_k))]

        return self._append_parent_chunks(selected)

    def _resolve_recency_mode(self, recency_mode: str) -> str:
        normalized = str(recency_mode or "").strip().lower()
        if normalized in {"off", "soft", "hard"}:
            return normalized
        fallback = str(self._config.recency_mode or "off").strip().lower()
        if fallback in {"off", "soft", "hard"}:
            return fallback
        return "off"

    def _recency_weight_for_mode(self, recency_mode: str) -> float:
        mode = self._resolve_recency_mode(recency_mode)
        if mode == "soft":
            value = float(self._config.recency_weight_soft)
        elif mode == "hard":
            value = float(self._config.recency_weight_hard)
        else:
            return 0.0
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _apply_recency_scores(
        self,
        *,
        scored: Sequence[tuple[float, int, Chunk]],
        recency_mode: str,
    ) -> list[tuple[float, int, Chunk]]:
        if not scored:
            return []

        weight = self._recency_weight_for_mode(recency_mode)
        if weight <= 0.0:
            return list(scored)

        half_life_days = max(0.0001, float(self._config.recency_half_life_days))
        now = datetime.now(timezone.utc)
        adjusted: list[tuple[float, int, Chunk]] = []
        for base_score, original_index, chunk in scored:
            updated_at = self._chunk_updated_at(chunk)
            if updated_at is None:
                recency_score = 0.5
            else:
                age_days = max(0.0, (now - updated_at).total_seconds() / 86400.0)
                recency_score = 0.5 ** (age_days / half_life_days)
            final_score = ((1.0 - weight) * float(base_score)) + (weight * recency_score)
            adjusted.append((final_score, original_index, chunk))
        return adjusted

    @staticmethod
    def _chunk_updated_at(chunk: Chunk) -> datetime | None:
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

    def _apply_parent_chunk_cap(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        cap = int(self._config.parent_chunk_cap)
        if cap <= 0:
            return chunks

        counts: dict[tuple[object, ...], int] = {}
        capped: list[Chunk] = []
        for chunk in chunks:
            key = self._parent_cap_key(chunk)
            count = counts.get(key, 0)
            if count >= cap:
                continue
            counts[key] = count + 1
            capped.append(chunk)
        return capped

    def _parent_cap_key(self, chunk: Chunk) -> tuple[object, ...]:
        metadata = chunk.metadata or {}
        stage = metadata.get("chunk_stage")
        parent_id = self._normalize_chunk_id(metadata.get("parent_chunk_id"))
        if stage == "proposition":
            resolved = self._resolve_first_parent_id(metadata, parent_id)
            if resolved is not None:
                parent_id = resolved
        if parent_id is None:
            return ("self", self._chunk_doc_key(chunk))
        key = self._chunk_lookup_key(metadata, parent_id)
        if key is None:
            return ("self", self._chunk_doc_key(chunk))
        return ("parent",) + key

    def _resolve_first_parent_id(
        self,
        metadata: dict[str, object],
        parent_id: int | None,
    ) -> int | None:
        if parent_id is None:
            return None
        key = self._chunk_lookup_key(metadata, parent_id)
        if key is None:
            return None
        second_chunk = self._second_rec_chunk_map().get(key)
        if second_chunk is None:
            return None
        return self._normalize_chunk_id((second_chunk.metadata or {}).get("parent_chunk_id"))

    def _append_parent_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        if not chunks or not self._config.parent_doc_enabled:
            return chunks

        ordered: list[Chunk] = []
        seen: set[tuple[object, ...]] = set()
        for chunk in chunks:
            key = self._chunk_doc_key(chunk)
            if key not in seen:
                seen.add(key)
                ordered.append(chunk)
            for candidate in self._parent_candidates_for_chunk(chunk):
                candidate_key = self._chunk_doc_key(candidate)
                if candidate_key in seen:
                    continue
                seen.add(candidate_key)
                ordered.append(candidate)
        return ordered

    def _parent_candidates_for_chunk(self, chunk: Chunk) -> list[Chunk]:
        metadata = chunk.metadata or {}
        if self._metadata_flag_enabled(metadata.get("skip_parent_context")):
            return []
        stage = metadata.get("chunk_stage")
        if stage == "proposition":
            second_parent_id = self._normalize_chunk_id(metadata.get("parent_chunk_id"))
            first_parent_id = self._resolve_first_parent_id(metadata, second_parent_id)
            if first_parent_id is None:
                return []
            return self._first_or_summary_candidates(
                metadata=metadata,
                first_parent_id=first_parent_id,
            )

        if stage == "second_recursive":
            first_parent_id = self._normalize_chunk_id(metadata.get("parent_chunk_id"))
            if first_parent_id is None:
                return []
            return self._first_or_summary_candidates(
                metadata=metadata,
                first_parent_id=first_parent_id,
            )
        return []

    def _first_or_summary_candidates(
        self,
        *,
        metadata: dict[str, object],
        first_parent_id: int,
    ) -> list[Chunk]:
        key = self._chunk_lookup_key(metadata, first_parent_id)
        if key is None:
            return []
        summary_chunks = self._summary_chunk_map().get(key)
        if summary_chunks:
            return list(summary_chunks)
        first_chunk = self._first_rec_chunk_map().get(key)
        if first_chunk is None:
            return []
        return [first_chunk]

    @lru_cache(maxsize=1)
    def _second_rec_chunk_map(self) -> dict[tuple[object, ...], Chunk]:
        return self._chunk_map_for_stage(self._chunks_root_dir() / "second_rec_chunk")

    @lru_cache(maxsize=1)
    def _first_rec_chunk_map(self) -> dict[tuple[object, ...], Chunk]:
        return self._chunk_map_for_stage(self._chunks_root_dir() / "first_rec_chunk")

    @lru_cache(maxsize=1)
    def _summary_chunk_map(self) -> dict[tuple[object, ...], list[Chunk]]:
        stage_dir = self._chunks_root_dir() / "summary_chunk"
        chunks = self._load_stage_chunks(stage_dir)
        mapping: dict[tuple[object, ...], list[Chunk]] = {}
        for chunk in chunks:
            metadata = chunk.metadata or {}
            parent_id = self._normalize_chunk_id(metadata.get("parent_chunk_id"))
            if parent_id is None:
                continue
            key = self._chunk_lookup_key(metadata, parent_id)
            if key is None:
                continue
            mapping.setdefault(key, []).append(chunk)
        return mapping

    def _chunk_map_for_stage(self, stage_dir: Path) -> dict[tuple[object, ...], Chunk]:
        chunks = self._load_stage_chunks(stage_dir)
        mapping: dict[tuple[object, ...], Chunk] = {}
        for chunk in chunks:
            metadata = chunk.metadata or {}
            chunk_id = self._normalize_chunk_id(metadata.get("chunk_id"))
            if chunk_id is None:
                continue
            key = self._chunk_lookup_key(metadata, chunk_id)
            if key is None or key in mapping:
                continue
            mapping[key] = chunk
        return mapping

    @staticmethod
    def _load_stage_chunks(stage_dir: Path) -> list[Chunk]:
        if not stage_dir.exists():
            return []
        chunks: list[Chunk] = []
        for path in sorted(stage_dir.rglob("*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as fr:
                    for line in fr:
                        raw = line.strip()
                        if not raw:
                            continue
                        payload = json.loads(raw)
                        if not isinstance(payload, dict):
                            continue
                        metadata = payload.get("metadata")
                        if not isinstance(metadata, dict):
                            metadata = {}
                        chunks.append(
                            Chunk(
                                id=str(payload.get("id") or ""),
                                document_id=str(payload.get("document_id") or ""),
                                text=str(payload.get("text") or ""),
                                index=int(payload.get("index") or 0),
                                metadata={str(key): value for key, value in metadata.items()},
                            )
                        )
            except Exception:
                logger.exception("Failed to load stage chunks from %s", path)
        return chunks

    def _chunks_root_dir(self) -> Path:
        return self._retrieval.index_dir.parent / "chunks"

    @staticmethod
    def _chunk_lookup_key(
        metadata: dict[str, object],
        chunk_id: int,
    ) -> tuple[object, ...] | None:
        source = metadata.get("drive_file_id") or metadata.get("source_file_name")
        if not source:
            return None
        source_type = metadata.get("source_type") or ""
        return (source_type, source, chunk_id)

    @staticmethod
    def _normalize_chunk_id(value: object) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _metadata_flag_enabled(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    @staticmethod
    def _chunk_doc_key(chunk: Chunk) -> tuple[object, ...]:
        metadata = chunk.metadata or {}
        stage = metadata.get("chunk_stage")
        source = metadata.get("drive_file_id") or metadata.get("source_file_name")
        chunk_id = metadata.get("chunk_id")
        raptor_level = metadata.get("raptor_level")
        raptor_cluster = metadata.get("raptor_cluster_id")
        if stage or source or chunk_id or raptor_level or raptor_cluster:
            return (stage, source, chunk_id, raptor_level, raptor_cluster)
        return ("content", chunk.text)

    @staticmethod
    def _normalize_source_type_filters(
        source_types: set[str] | None,
    ) -> set[str]:
        if not source_types:
            return set()
        normalized: set[str] = set()
        for value in source_types:
            source_type = str(value or "").strip().lower()
            if source_type:
                normalized.add(source_type)
        return normalized

    def _filter_chunks_by_source_type(
        self,
        chunks: Sequence[Chunk],
        *,
        excluded_source_types: set[str] | None,
    ) -> list[Chunk]:
        excluded = self._normalize_source_type_filters(excluded_source_types)
        if not excluded:
            return list(chunks)
        filtered: list[Chunk] = []
        for chunk in chunks:
            source_type = str((chunk.metadata or {}).get("source_type") or "").strip().lower()
            if source_type in excluded:
                continue
            filtered.append(chunk)
        return filtered

    def _normalize_material_name(self, value: str) -> str:
        normalized = str(value or "").replace("\\", "/").strip().casefold()
        return " ".join(normalized.split())

    @staticmethod
    def _normalize_material_key(
        source_type: str,
        source_key: str,
    ) -> tuple[str, str]:
        return (
            str(source_type or "").strip().lower(),
            str(source_key or "").strip().casefold(),
        )

    def _metadata_material_key(
        self,
        metadata: dict[str, object] | None,
    ) -> tuple[str, str] | None:
        if not metadata:
            return None
        source_type = str(metadata.get("source_type") or "").strip().lower()
        if not source_type:
            return None
        source_key = str(
            metadata.get("drive_file_id") or metadata.get("source_file_name") or ""
        ).strip()
        if not source_key:
            return None
        return self._normalize_material_key(source_type, source_key)

    def _chunk_matches_material_keys(
        self,
        *,
        chunk: Chunk,
        material_keys: set[tuple[str, str]] | None,
        excluded_source_types: set[str] | None,
    ) -> bool:
        excluded = self._normalize_source_type_filters(excluded_source_types)
        metadata = chunk.metadata or {}
        if excluded:
            source_type = str(metadata.get("source_type") or "").strip().lower()
            if source_type in excluded:
                return False
        if not material_keys:
            return True
        key = self._metadata_material_key(metadata)
        return key in material_keys if key is not None else False

    @lru_cache(maxsize=1)
    def _material_catalog_entries(self) -> tuple[_MaterialCatalogEntry, ...]:
        path = self._retrieval.index_dir / "material_catalog.json"
        if not path.exists():
            return tuple()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load material catalog: %s", path)
            return tuple()
        if not isinstance(payload, dict):
            return tuple()
        materials = payload.get("materials")
        if not isinstance(materials, list):
            return tuple()

        entries: list[_MaterialCatalogEntry] = []
        for row in materials:
            if not isinstance(row, dict):
                continue
            aliases_raw = row.get("aliases")
            aliases: list[str] = []
            if isinstance(aliases_raw, list):
                aliases = [str(value) for value in aliases_raw if str(value).strip()]
            entry = _MaterialCatalogEntry(
                material_id=str(row.get("material_id") or ""),
                source_type=str(row.get("source_type") or ""),
                source_key=str(row.get("source_key") or ""),
                canonical_name=str(row.get("canonical_name") or ""),
                aliases=tuple(aliases),
                raw_path=str(row.get("raw_path") or ""),
            )
            if not entry.material_id or not entry.source_type or not entry.source_key:
                continue
            entries.append(entry)
        return tuple(entries)

    def _match_material_entries(
        self,
        *,
        material_names: Sequence[str],
        query: str,
        excluded_source_types: set[str],
    ) -> list[_MaterialCatalogEntry]:
        normalized_names: list[str] = []
        seen_names: set[str] = set()
        for raw in material_names:
            normalized = self._normalize_material_name(raw)
            if not normalized or normalized in seen_names:
                continue
            seen_names.add(normalized)
            normalized_names.append(normalized)
            if len(normalized_names) >= 3:
                break
        if not normalized_names:
            return []

        entries = self._material_catalog_entries()
        if not entries:
            return []

        def aliases(entry: _MaterialCatalogEntry) -> list[str]:
            values = [entry.canonical_name, *entry.aliases]
            deduped: list[str] = []
            seen_aliases: set[str] = set()
            for value in values:
                normalized = self._normalize_material_name(value)
                if not normalized or normalized in seen_aliases:
                    continue
                seen_aliases.add(normalized)
                deduped.append(normalized)
            return deduped

        strict_matches: list[_MaterialCatalogEntry] = []
        for name in normalized_names:
            for entry in entries:
                if name in aliases(entry):
                    strict_matches.append(entry)
        if strict_matches:
            return self._dedupe_material_entries(strict_matches, limit=3)

        partial_matches: list[_MaterialCatalogEntry] = []
        for name in normalized_names:
            for entry in entries:
                alias_values = aliases(entry)
                if any((name in alias) or (alias in name) for alias in alias_values):
                    partial_matches.append(entry)
        deduped_partial_matches = self._dedupe_material_entries(partial_matches, limit=None)
        if len(deduped_partial_matches) <= 1:
            return deduped_partial_matches

        selected = self._select_partial_material_match_by_semantic(
            query=query,
            entries=deduped_partial_matches,
            excluded_source_types=excluded_source_types,
        )
        if selected is not None:
            return [selected]
        return deduped_partial_matches[:3]

    def _dedupe_material_entries(
        self,
        entries: Sequence[_MaterialCatalogEntry],
        *,
        limit: int | None,
    ) -> list[_MaterialCatalogEntry]:
        deduped: list[_MaterialCatalogEntry] = []
        seen_ids: set[str] = set()
        resolved_limit = max(1, int(limit)) if limit is not None else None
        for entry in entries:
            if entry.material_id in seen_ids:
                continue
            seen_ids.add(entry.material_id)
            deduped.append(entry)
            if resolved_limit is not None and len(deduped) >= resolved_limit:
                break
        return deduped

    def _select_partial_material_match_by_semantic(
        self,
        *,
        query: str,
        entries: Sequence[_MaterialCatalogEntry],
        excluded_source_types: set[str],
    ) -> _MaterialCatalogEntry | None:
        cleaned_query = (query or "").strip()
        if not cleaned_query or len(entries) <= 1:
            return None

        material_keys = {
            self._normalize_material_key(entry.source_type, entry.source_key)
            for entry in entries
        }
        ranked = self._retrieve_filtered_candidates(
            query=cleaned_query,
            dense_top_k=max(1, self._config.dense_top_k),
            sparse_top_k=max(1, self._config.sparse_top_k),
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
        )
        if not ranked:
            return None
        top_key = self._metadata_material_key(ranked[0].metadata or {})
        if top_key is None:
            return None
        for entry in entries:
            entry_key = self._normalize_material_key(entry.source_type, entry.source_key)
            if entry_key == top_key:
                return entry
        return None

    def _retrieve_material_limited_chunks(
        self,
        *,
        query: str,
        material_keys: set[tuple[str, str]],
        excluded_source_types: set[str],
        recency_mode: str,
    ) -> list[Chunk]:
        _ = recency_mode
        chunks = self._retrieve_filtered_candidates(
            query=query,
            dense_top_k=self._config.dense_top_k,
            sparse_top_k=self._config.sparse_top_k,
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
        )
        if chunks:
            return chunks
        return self._retrieve_filtered_candidates(
            query=query,
            dense_top_k=self._config.dense_top_k,
            sparse_top_k=0,
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
        )

    def _retrieve_filtered_candidates(
        self,
        *,
        query: str,
        dense_top_k: int,
        sparse_top_k: int,
        material_keys: set[tuple[str, str]] | None,
        excluded_source_types: set[str] | None,
    ) -> list[Chunk]:
        results: list[Chunk] = []
        seen: set[tuple[object, ...]] = set()
        target = max(1, dense_top_k)
        for mult in (1, 2, 4, 8):
            dense_k = max(0, int(dense_top_k) * mult)
            sparse_k = max(0, int(sparse_top_k) * mult)
            candidates = self._retrieval.retrieve(
                query,
                dense_top_k=dense_k,
                sparse_top_k=sparse_k,
                recency_mode="off",
                recency_weight_soft=self._config.recency_weight_soft,
                recency_weight_hard=self._config.recency_weight_hard,
                recency_half_life_days=self._config.recency_half_life_days,
                mmr_lambda=self._config.mmr_lambda,
            )
            for chunk in candidates:
                if not self._chunk_matches_material_keys(
                    chunk=chunk,
                    material_keys=material_keys,
                    excluded_source_types=excluded_source_types,
                ):
                    continue
                key = self._chunk_doc_key(chunk)
                if key in seen:
                    continue
                seen.add(key)
                results.append(chunk)
                if len(results) >= target:
                    return results
        return results

    def _build_material_context_chunks(
        self,
        *,
        query: str,
        entries: Sequence[_MaterialCatalogEntry],
        searched_chunks: Sequence[Chunk],
        force_fast_mode: bool,
    ) -> list[Chunk]:
        _ = query
        if not entries:
            return []

        entry_by_key = {
            self._normalize_material_key(entry.source_type, entry.source_key): entry
            for entry in entries
        }
        docs_by_key: dict[tuple[str, str], list[Chunk]] = {}
        ordered_keys: list[tuple[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()
        for chunk in searched_chunks:
            key = self._metadata_material_key(chunk.metadata or {})
            if key is None or key not in entry_by_key:
                continue
            docs_by_key.setdefault(key, []).append(chunk)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ordered_keys.append(key)
        if not ordered_keys:
            ordered_keys = list(entry_by_key.keys())

        contexts: list[Chunk] = []
        char_limit = 3000
        for key in ordered_keys:
            entry = entry_by_key.get(key)
            if entry is None:
                continue
            representative_metadata = self._representative_metadata_for_material(
                material_key=key,
                searched_chunks=docs_by_key.get(key) or [],
                entry=entry,
            )
            raw_text = self._read_material_raw_text(entry)
            if raw_text and len(raw_text) <= char_limit:
                contexts.append(
                    Chunk(
                        id=f"material:{entry.material_id}:raw",
                        document_id=f"material:{entry.material_id}",
                        text=raw_text,
                        index=0,
                        metadata=representative_metadata,
                    )
                )
                continue

            fallback = self._first_rec_chunks_for_material_key(key)
            if not fallback:
                fallback = docs_by_key.get(key) or []
            if force_fast_mode or self._reranker is None or len(fallback) <= 1:
                contexts.extend(list(fallback[:3]))
            else:
                reranked = self._reranker.rerank(
                    query=query,
                    chunks=list(fallback),
                    top_k=min(3, len(fallback)),
                )
                contexts.extend(reranked)

        return self._merge_unique_chunks(contexts)

    def _first_rec_chunks_for_material_key(
        self,
        material_key: tuple[str, str],
    ) -> list[Chunk]:
        matched: list[tuple[int, Chunk]] = []
        for key, chunk in self._first_rec_chunk_map().items():
            if len(key) < 3:
                continue
            source_type = str(key[0] or "")
            source_key = str(key[1] or "")
            normalized = self._normalize_material_key(source_type, source_key)
            if normalized != material_key:
                continue
            chunk_id_raw = key[2]
            try:
                chunk_id = int(chunk_id_raw)
            except (TypeError, ValueError):
                chunk_id = len(matched)
            matched.append((chunk_id, chunk))
        matched.sort(key=lambda item: item[0])
        return [chunk for _, chunk in matched]

    def _representative_metadata_for_material(
        self,
        *,
        material_key: tuple[str, str],
        searched_chunks: Sequence[Chunk],
        entry: _MaterialCatalogEntry,
    ) -> dict[str, object]:
        first_chunks = self._first_rec_chunks_for_material_key(material_key)
        if first_chunks:
            metadata = dict(first_chunks[0].metadata or {})
        elif searched_chunks:
            metadata = dict(searched_chunks[0].metadata or {})
        else:
            metadata = {}
        metadata.setdefault("source_type", entry.source_type)
        metadata.setdefault("source_file_name", entry.source_key)
        return metadata

    def _read_material_raw_text(self, entry: _MaterialCatalogEntry) -> str:
        if not entry.raw_path:
            return ""
        raw_path = Path(entry.raw_path)
        candidates: list[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            base_dir = self._base_dir()
            data_dir = self._retrieval.index_dir.parent
            raw_dir = self._raw_data_dir()
            candidates.extend(
                [
                    base_dir / raw_path,  # Legacy catalog stores path relative to base dir.
                    data_dir / raw_path,
                    raw_dir / raw_path,
                ]
            )

        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if not resolved.exists() or not resolved.is_file():
                continue
            try:
                return resolved.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                return resolved.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                logger.warning(
                    "Failed to read material raw file: %s",
                    resolved,
                    exc_info=True,
                )
                return ""
        return ""

    def _base_dir(self) -> Path:
        return self._retrieval.index_dir.parent.parent

    def _raw_data_dir(self) -> Path:
        return self._retrieval.index_dir.parent / "raw"

    @staticmethod
    def _merge_unique_chunks(chunks: Sequence[Chunk]) -> list[Chunk]:
        merged: list[Chunk] = []
        seen: set[tuple[object, ...]] = set()
        for chunk in chunks:
            key = RagService._chunk_doc_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
        return merged

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
