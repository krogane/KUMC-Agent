from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
from typing import Sequence
import unicodedata

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
_MATERIAL_NAME_SEPARATORS_RE = re.compile(r"[\s/\\_.\-:：,，、]+")
_MATERIAL_NAME_NOISE_RE = re.compile(r"[^0-9a-zぁ-んァ-ン一-龠々ー]+")
_MATERIAL_DATE_RE = re.compile(
    r"(?P<y>\d{4})\D{0,3}(?P<m>\d{1,2})\D{0,3}(?P<d>\d{1,2})"
)
_MATERIAL_VARIANT_FILLERS = ("例会", "定例会", "meeting", "mtg", "ミーティング")
_MATERIAL_LABEL_PREFIXES = ("議事録",)


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
        disable_history: bool = False,
    ) -> Answer:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return Answer(text="", route="none", metadata={"reason": "empty_query"})

        routing_history: Sequence[ChatHistoryEntry] | None
        if routing_history_override is not None:
            routing_history = list(routing_history_override)
        elif disable_history:
            routing_history = []
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
                material_names=[],
                additional_queries=[],
            )

        if generation_history_override is not None:
            generation_history = list(generation_history_override)
        elif disable_history:
            generation_history = []
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

        self._prepare_reranker_runtime(force_fast_mode=force_fast_mode)
        effective_recency_mode = self._resolve_recency_mode(decision.recency_mode)
        material_route = bool(decision.material_names)
        if material_route:
            chunks = self._retrieve_material_route_chunks(
                query=cleaned_query,
                decision=decision,
                recency_mode=effective_recency_mode,
                force_fast_mode=force_fast_mode,
            )
        else:
            chunks = self._retrieve_chunks(
                query=cleaned_query,
                decision=decision,
                force_fast_mode=force_fast_mode,
                recency_mode=effective_recency_mode,
                excluded_source_types=None,
            )

        chunks = self._rank_and_select_chunks(
            query=cleaned_query,
            chunks=chunks,
            recency_mode=effective_recency_mode,
            force_fast_mode=force_fast_mode,
        )

        if not chunks:
            no_rag_generation = self._resolve_generation_settings(
                target_model="no_rag",
            )
            answer = self._generation.generate_no_rag(
                query=cleaned_query,
                history=generation_history,
                provider=no_rag_generation.provider,
                include_capabilities_info=decision.include_capabilities_info,
                temperature=no_rag_generation.temperature,
                max_output_tokens=no_rag_generation.max_output_tokens,
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
                disable_history=disable_history,
            )

        rag_generation = self._resolve_generation_settings(
            target_model="rag",
        )
        answer = self._generation.generate_rag_answer(
            query=cleaned_query,
            chunks=chunks,
            history=generation_history,
            provider=rag_generation.provider,
            include_capabilities_info=decision.include_capabilities_info,
            temperature=rag_generation.temperature,
            max_output_tokens=rag_generation.max_output_tokens,
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
            disable_history=disable_history,
        )

    def _resolve_generation_settings(
        self,
        *,
        target_model: str,
    ) -> RagGenerationSettings:
        normalized = (target_model or "").strip().lower()
        if normalized == "no_rag":
            base = self._config.no_rag_generation
        else:
            base = self._config.rag_generation
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
        if force_fast_mode or not decision.additional_queries:
            return self._retrieve_single_query_chunks(
                query=query,
                recency_mode=recency_mode,
                excluded_source_types=excluded_source_types,
            )

        query_candidates = [query, *decision.additional_queries]
        normalized_queries = [str(value or "").strip() for value in query_candidates]
        normalized_queries = [value for value in normalized_queries if value]
        if not normalized_queries:
            return []
        if len(normalized_queries) == 1:
            return self._retrieve_single_query_chunks(
                query=normalized_queries[0],
                recency_mode=recency_mode,
                excluded_source_types=excluded_source_types,
            )

        max_workers = max(1, min(len(normalized_queries), 4))
        results_by_index: dict[int, list[Chunk]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._retrieve_single_query_chunks,
                    query=retrieval_query,
                    recency_mode=recency_mode,
                    excluded_source_types=excluded_source_types,
                ): index
                for index, retrieval_query in enumerate(normalized_queries)
            }
            for future in as_completed(futures):
                index = futures[future]
                retrieval_query = normalized_queries[index]
                try:
                    results_by_index[index] = future.result()
                except Exception:
                    logger.exception(
                        "Query retrieval failed for transformed query: %s",
                        retrieval_query,
                    )
                    results_by_index[index] = []

        merged: dict[str, Chunk] = {}
        for index in range(len(normalized_queries)):
            for chunk in results_by_index.get(index, []):
                merged[chunk.id] = chunk
        return list(merged.values())

    def _retrieve_single_query_chunks(
        self,
        *,
        query: str,
        recency_mode: str,
        excluded_source_types: set[str] | None,
    ) -> list[Chunk]:
        chunks = self._retrieval.retrieve(
            query,
            dense_top_k=self._config.dense_top_k,
            sparse_top_k=self._config.sparse_top_k,
            sparse_initial_sparse_top_k=self._config.sparse_initial_sparse_top_k,
            recency_mode=recency_mode,
            recency_weight_soft=self._config.recency_weight_soft,
            recency_weight_hard=self._config.recency_weight_hard,
            recency_half_life_days=self._config.recency_half_life_days,
            mmr_lambda=self._config.mmr_lambda,
            rrf_k=self._config.rrf_k,
        )
        return self._filter_chunks_by_source_type(
            chunks,
            excluded_source_types=excluded_source_types,
        )

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
            dense_name_contexts = self._material_name_dense_fallback_contexts(
                material_names=decision.material_names,
                query=query,
                excluded_source_types=excluded_source_types,
                force_fast_mode=force_fast_mode,
            )
            if dense_name_contexts:
                return dense_name_contexts
            return self._retrieve_chunks(
                query=query,
                decision=decision,
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
            direct_contexts = self._build_material_context_chunks(
                query=query,
                entries=matched_entries,
                searched_chunks=[],
                force_fast_mode=force_fast_mode,
            )
            if direct_contexts:
                return direct_contexts
            dense_name_contexts = self._material_name_dense_fallback_contexts(
                material_names=decision.material_names,
                query=query,
                excluded_source_types=excluded_source_types,
                force_fast_mode=force_fast_mode,
            )
            if dense_name_contexts:
                return dense_name_contexts
            return self._retrieve_chunks(
                query=query,
                decision=decision,
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
        dense_name_contexts = self._material_name_dense_fallback_contexts(
            material_names=decision.material_names,
            query=query,
            excluded_source_types=excluded_source_types,
            force_fast_mode=force_fast_mode,
        )
        if dense_name_contexts:
            return dense_name_contexts
        return self._retrieve_chunks(
            query=query,
            decision=decision,
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

    def _prepare_reranker_runtime(self, *, force_fast_mode: bool) -> None:
        if force_fast_mode or self._reranker is None:
            return
        prepare_runtime = getattr(self._reranker, "prepare_runtime", None)
        if not callable(prepare_runtime):
            return
        try:
            prepare_runtime()
        except Exception:
            logger.debug("Failed to preload reranker runtime.", exc_info=True)

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
        parent_id = self._normalize_chunk_id(metadata.get("parent_chunk_id"))
        if parent_id is None:
            return ("self", self._chunk_doc_key(chunk))
        key = self._chunk_lookup_key(metadata, parent_id)
        if key is None:
            return ("self", self._chunk_doc_key(chunk))
        return ("parent",) + key

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
        if stage or source or chunk_id:
            return (stage, source, chunk_id)
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
        variants = self._material_name_variants(value)
        return variants[0] if variants else ""

    def _material_name_variants(self, value: str) -> tuple[str, ...]:
        base = self._material_name_base(value)
        if not base:
            return tuple()

        candidates: list[str] = [base]
        for filler in _MATERIAL_VARIANT_FILLERS:
            if filler in base:
                candidates.append(base.replace(filler, ""))
        for label_prefix in _MATERIAL_LABEL_PREFIXES:
            if base.startswith(label_prefix):
                candidates.append(base[len(label_prefix) :])
            if base.endswith(label_prefix):
                candidates.append(base[: -len(label_prefix)])

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = self._material_name_base(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return tuple(deduped)

    @staticmethod
    def _material_name_base(value: str) -> str:
        text = unicodedata.normalize("NFKC", str(value or "")).casefold()
        text = text.replace("\\", "/")
        text = _MATERIAL_DATE_RE.sub(
            lambda match: (
                f"{int(match.group('y')):04d}"
                f"{int(match.group('m')):02d}"
                f"{int(match.group('d')):02d}"
            ),
            text,
        )
        text = _MATERIAL_NAME_SEPARATORS_RE.sub(" ", text).strip()
        text = _MATERIAL_NAME_NOISE_RE.sub("", text)
        return text

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
        max_names = max(1, int(self._config.material_search_max_names))
        excluded = self._normalize_source_type_filters(excluded_source_types)
        normalized_names: list[tuple[str, ...]] = []
        seen_names: set[str] = set()
        for raw in material_names:
            variants = self._material_name_variants(raw)
            if not variants:
                continue
            canonical = variants[0]
            if canonical in seen_names:
                continue
            seen_names.add(canonical)
            normalized_names.append(variants)
            if len(normalized_names) >= max_names:
                break
        if not normalized_names:
            return []

        entries = [
            entry
            for entry in self._material_catalog_entries()
            if str(entry.source_type or "").strip().lower() not in excluded
        ]
        if not entries:
            return []

        def aliases(entry: _MaterialCatalogEntry) -> tuple[str, ...]:
            values = [entry.canonical_name, *entry.aliases]
            deduped: list[str] = []
            seen_aliases: set[str] = set()
            for value in values:
                for normalized in self._material_name_variants(value):
                    if normalized in seen_aliases:
                        continue
                    seen_aliases.add(normalized)
                    deduped.append(normalized)
            return tuple(deduped)

        aliases_by_entry = {entry.material_id: aliases(entry) for entry in entries}

        strict_matches: list[_MaterialCatalogEntry] = []
        for name_variants in normalized_names:
            name_set = set(name_variants)
            for entry in entries:
                alias_values = aliases_by_entry.get(entry.material_id, tuple())
                if any(alias in name_set for alias in alias_values):
                    strict_matches.append(entry)
        if strict_matches:
            return self._dedupe_material_entries(strict_matches, limit=max_names)

        partial_matches: list[_MaterialCatalogEntry] = []
        for name_variants in normalized_names:
            for entry in entries:
                alias_values = aliases_by_entry.get(entry.material_id, tuple())
                if any(
                    (name in alias) or (alias in name)
                    for name in name_variants
                    for alias in alias_values
                ):
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
        return deduped_partial_matches[:max_names]

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

    def _material_name_dense_fallback_contexts(
        self,
        *,
        material_names: Sequence[str],
        query: str,
        excluded_source_types: set[str],
        force_fast_mode: bool,
    ) -> list[Chunk]:
        selected = self._select_material_entry_by_dense_name(
            material_names=material_names,
            excluded_source_types=excluded_source_types,
        )
        if selected is None:
            return []
        return self._build_material_context_chunks(
            query=query,
            entries=[selected],
            searched_chunks=[],
            force_fast_mode=force_fast_mode,
        )

    def _select_material_entry_by_dense_name(
        self,
        *,
        material_names: Sequence[str],
        excluded_source_types: set[str],
    ) -> _MaterialCatalogEntry | None:
        dense_ranker = getattr(self._retrieval, "rank_texts_by_dense", None)
        if not callable(dense_ranker):
            return None

        excluded = self._normalize_source_type_filters(excluded_source_types)
        entries = [
            entry
            for entry in self._material_catalog_entries()
            if str(entry.source_type or "").strip().lower() not in excluded
        ]
        if not entries:
            return None

        dense_queries: list[str] = []
        seen_queries: set[str] = set()
        for value in material_names:
            for candidate in self._material_name_variants(value):
                if not candidate or candidate in seen_queries:
                    continue
                seen_queries.add(candidate)
                dense_queries.append(candidate)
        if not dense_queries:
            return None

        texts = [self._material_entry_dense_text(entry) for entry in entries]
        best: tuple[float, _MaterialCatalogEntry] | None = None
        for dense_query in dense_queries:
            ranked = dense_ranker(query=dense_query, texts=texts, top_k=1)
            if not ranked:
                continue
            index, score = ranked[0]
            if index < 0 or index >= len(entries):
                continue
            candidate = entries[index]
            if best is None or score > best[0]:
                best = (score, candidate)
        return best[1] if best is not None else None

    @staticmethod
    def _material_entry_dense_text(entry: _MaterialCatalogEntry) -> str:
        parts = [entry.canonical_name, entry.source_key, *entry.aliases]
        values: list[str] = []
        seen: set[str] = set()
        for value in parts:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            values.append(text)
        return "\n".join(values)

    def _retrieve_material_limited_chunks(
        self,
        *,
        query: str,
        material_keys: set[tuple[str, str]],
        excluded_source_types: set[str],
        recency_mode: str,
    ) -> list[Chunk]:
        chunks = self._retrieve_filtered_candidates(
            query=query,
            dense_top_k=self._config.dense_top_k,
            sparse_top_k=self._config.sparse_top_k,
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
            recency_mode=recency_mode,
        )
        if chunks:
            return chunks
        return self._retrieve_filtered_candidates(
            query=query,
            dense_top_k=self._config.dense_top_k,
            sparse_top_k=0,
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
            recency_mode=recency_mode,
        )

    def _retrieve_filtered_candidates(
        self,
        *,
        query: str,
        dense_top_k: int,
        sparse_top_k: int,
        material_keys: set[tuple[str, str]] | None,
        excluded_source_types: set[str] | None,
        recency_mode: str = "off",
    ) -> list[Chunk]:
        results: list[Chunk] = []
        seen: set[tuple[object, ...]] = set()
        target = max(1, dense_top_k)
        effective_recency_mode = self._resolve_recency_mode(recency_mode)
        for mult in (1, 2, 4, 8):
            dense_k = max(0, int(dense_top_k) * mult)
            sparse_k = max(0, int(sparse_top_k) * mult)
            candidates = self._retrieval.retrieve(
                query,
                dense_top_k=dense_k,
                sparse_top_k=sparse_k,
                recency_mode=effective_recency_mode,
                recency_weight_soft=self._config.recency_weight_soft,
                recency_weight_hard=self._config.recency_weight_hard,
                recency_half_life_days=self._config.recency_half_life_days,
                mmr_lambda=self._config.mmr_lambda,
                rrf_k=self._config.rrf_k,
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
        char_limit = max(1, int(self._config.material_full_text_char_limit))
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
            if raw_text and len(raw_text) < char_limit:
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
            dense_contexts = self._select_dense_chunks_with_char_limit(
                query=query,
                chunks=fallback,
                char_limit=char_limit,
                fallback_metadata=representative_metadata,
            )
            if dense_contexts:
                contexts.extend(dense_contexts)
                continue
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

    def _select_dense_chunks_with_char_limit(
        self,
        *,
        query: str,
        chunks: Sequence[Chunk],
        char_limit: int,
        fallback_metadata: dict[str, object],
    ) -> list[Chunk]:
        limit = max(1, int(char_limit))
        candidates = [chunk for chunk in chunks if (chunk.text or "").strip()]
        if not candidates:
            return []

        ordered = list(candidates)
        dense_ranker = getattr(self._retrieval, "rank_texts_by_dense", None)
        if callable(dense_ranker):
            texts = [chunk.text for chunk in candidates]
            ranked = dense_ranker(query=query, texts=texts, top_k=len(texts))
            ranked_chunks: list[Chunk] = []
            seen_indices: set[int] = set()
            for index, _score in ranked:
                if index in seen_indices:
                    continue
                if index < 0 or index >= len(candidates):
                    continue
                seen_indices.add(index)
                ranked_chunks.append(candidates[index])
            if ranked_chunks:
                ordered = ranked_chunks

        selected: list[Chunk] = []
        used_chars = 0
        for chunk in ordered:
            remaining = limit - used_chars
            if remaining <= 0:
                break
            text = chunk.text or ""
            if not text.strip():
                continue
            if len(text) <= remaining:
                selected.append(self._with_fallback_metadata(chunk, fallback_metadata))
                used_chars += len(text)
                continue
            truncated = text[:remaining].rstrip()
            if not truncated:
                break
            selected.append(
                self._with_fallback_metadata(
                    replace(chunk, text=truncated),
                    fallback_metadata,
                )
            )
            used_chars += len(truncated)
            break
        return selected

    @staticmethod
    def _with_fallback_metadata(chunk: Chunk, fallback_metadata: dict[str, object]) -> Chunk:
        if not fallback_metadata:
            return chunk
        metadata = dict(chunk.metadata or {})
        if not metadata:
            metadata = dict(fallback_metadata)
        else:
            for key, value in fallback_metadata.items():
                metadata.setdefault(key, value)
        return replace(chunk, metadata=metadata)

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
        disable_history: bool,
    ) -> Answer:
        text = answer.text
        if force_fast_mode and text:
            text = f"{self._config.fast_model_notice}\n\n{text}"
        metadata = dict(answer.metadata)
        metadata["routing_decision"] = {
            "recency_mode": routing_decision.recency_mode,
            "material_names": list(routing_decision.material_names),
            "include_capabilities_info": routing_decision.include_capabilities_info,
            "use_additional_memory": routing_decision.use_additional_memory,
            "additional_queries": list(routing_decision.additional_queries),
        }
        metadata["fast_mode"] = force_fast_mode
        finalized = replace(answer, text=text, metadata=metadata)
        if not disable_history:
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
