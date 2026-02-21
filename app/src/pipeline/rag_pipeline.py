from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import math
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import (
    AppConfig,
    EmbeddingFactory,
    get_required_prompt_env,
    render_prompt_template,
)
from date_metadata import SOURCE_DATE_UNKNOWN, infer_source_date, source_date_to_date
from indexing.chunks import load_chunks_from_dirs
from indexing.llm_client import generate_text
from indexing.material_catalog import MaterialCatalogEntry, load_material_catalog
from indexing.keyword_inverted_index import (
    KEYWORD_CORPUS_SECOND_REC_SPARSE,
    KEYWORD_CORPUS_SPARSE,
    KEYWORD_CORPUS_SPARSE_SECOND_REC,
    KeywordInvertedIndex,
    load_keyword_index,
    tokenize_sparse_doc,
)
from indexing.sparse_sources import (
    second_rec_chunk_dirs as resolve_second_rec_chunk_dirs,
    sparse_chunk_dirs as resolve_sparse_chunk_dirs,
    sparse_second_rec_chunk_dirs as resolve_sparse_second_rec_chunk_dirs,
)
from pipeline.function_calling import FunctionRoutingDecision, decide_tools
from pipeline.llm_clients import (
    generate_with_gemini_config,
    generate_with_llama_config,
)
from pipeline.prompts import (
    ChatHistoryEntry,
    build_gemini_prompt,
    build_llama_messages,
    doc_to_context,
)
from pipeline.reranker import CrossEncoderReranker
from pipeline.vectorstore import load_faiss_index
from sparse_normalizer import SparseNormalizer, SparseNormalizerConfig

logger = logging.getLogger(__name__)

_SECOND_REC_SPARSE_STAGE = "second_recursive_sparse"
_MASKED_MENTION = "（メンション非表示）"
_USER_MENTION_RE = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION_RE = re.compile(r"<@&\d+>")
_DISCORD_DATE_LINE_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")
_DISCORD_SOURCE_SELECTION_RE = re.compile(r"^(\d+)(?:-(\d+))?$")
_SECOND_REC_SOURCE_DIRS = (
    "docs",
    "sheets",
    "messages",
    "x",
    "vc",
    "hatenablog",
    "crafters_colony",
)
_JST = ZoneInfo("Asia/Tokyo")
_MATERIAL_SELECTOR_RESPONSE_KEYS = (
    "selected_indices",
    "indices",
    "chunks",
    "selected_chunks",
)
_MATERIAL_SEARCH_EXCLUDED_SOURCE_TYPES = frozenset(
    {"messages", "discord_message", "x_posts"}
)


class GenerationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class _DenseSearchResult:
    docs: list[Document]
    query_vector: np.ndarray | None
    doc_vectors_by_key: dict[tuple[object, ...], np.ndarray]


@dataclass(frozen=True)
class _MaterialMatch:
    entry: MaterialCatalogEntry
    matched_name: str
    strict: bool


@dataclass(frozen=True)
class _SourceSelection:
    doc_index: int
    sub_index: int | None = None


@dataclass(frozen=True)
class _DiscordChunkLine:
    text: str
    message_id: str | None


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled("Generation cancelled.")


class RagPipeline:
    def __init__(
        self,
        *,
        index_dir: Path,
        embedding_factory: EmbeddingFactory,
        llm_api_key: str,
        config: AppConfig,
    ) -> None:
        self._index_dir = index_dir
        self._embedding_factory = embedding_factory
        self._llm_api_key = llm_api_key
        self._config = config
        self._reranker = CrossEncoderReranker(config)
        self._chat_histories: dict[str, deque[ChatHistoryEntry]] = {}
        self._rerank_scores_by_query: dict[str, dict[tuple[object, ...], float]] = {}
        self._rerank_score_query_order: deque[str] = deque()
        self._rerank_score_cache_lock = threading.Lock()

    def retrieve(
        self,
        query: str,
        *,
        re_search: bool = False,
        recency_mode: str = "off",
        disable_rerank: bool = False,
        disable_mmr: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        query = query.strip()
        if not query:
            return []

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                self._dense_search, query, cancel_event=cancel_event
            )
            sparse_future = executor.submit(
                self._retrieve_sparse_docs,
                query,
                re_search=re_search,
                cancel_event=cancel_event,
            )
            done, _ = wait(
                [dense_future, sparse_future], return_when=FIRST_EXCEPTION
            )
            for future in done:
                exc = future.exception()
                if exc is not None:
                    dense_future.cancel()
                    sparse_future.cancel()
                    raise exc
            dense_result = dense_future.result()
            sparse_docs = sparse_future.result()

        return self._finalize_retrieved_docs(
            query=query,
            dense_result=dense_result,
            sparse_docs=sparse_docs,
            recency_mode=recency_mode,
            disable_rerank=disable_rerank,
            disable_mmr=disable_mmr,
            cancel_event=cancel_event,
        )

    def _finalize_retrieved_docs(
        self,
        *,
        query: str,
        dense_result: _DenseSearchResult,
        sparse_docs: list[Document],
        recency_mode: str = "off",
        disable_rerank: bool = False,
        disable_mmr: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        dense_docs = dense_result.docs
        merged = self._merge_docs([dense_docs, sparse_docs])
        if merged:
            rerank_enabled = self._config.rerank_enabled and not disable_rerank
            ranked = (
                self._rerank_docs(
                    query=query,
                    docs=merged,
                    recency_mode=recency_mode,
                )
                if rerank_enabled
                else merged
            )
            capped = self._apply_parent_doc_cap(ranked)
            pooled = (
                self._limit_rerank_pool(capped)
                if rerank_enabled
                else capped
            )
            selected = (
                pooled[: max(0, self._config.top_k)]
                if disable_mmr
                else self._select_with_mmr(
                    query=query,
                    docs=pooled,
                    query_vector=dense_result.query_vector,
                    doc_vectors_by_key=dense_result.doc_vectors_by_key,
                )
            )
            return self._append_parent_docs(selected)

        _raise_if_cancelled(cancel_event)
        q = f"query: {query}"
        docs = self._vectorstore().similarity_search(q, k=self._config.top_k)
        return self._append_parent_docs(docs)

    def _retrieve_transformed_query(
        self,
        query: str,
        *,
        recency_mode: str = "off",
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        cleaned = (query or "").strip()
        if not cleaned:
            return []
        dense_result = self._dense_search(cleaned, cancel_event=cancel_event)
        transform_k = max(0, self._config.sparse_search_transform_top_k)
        sparse_docs = self._sparse_search_once(
            cleaned,
            transform_k,
            cancel_event=cancel_event,
        )
        return self._finalize_retrieved_docs(
            query=cleaned,
            dense_result=dense_result,
            sparse_docs=sparse_docs,
            recency_mode=recency_mode,
            cancel_event=cancel_event,
        )

    def _retrieve_sparse_docs(
        self,
        query: str,
        *,
        re_search: bool,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        if re_search:
            return self._sparse_search(
                original_query=query,
                transformed_query=None,
                cancel_event=cancel_event,
            )
        return self._sparse_search_initial(
            query, cancel_event=cancel_event
        )

    @lru_cache(maxsize=1)
    def _material_catalog_entries(self) -> tuple[MaterialCatalogEntry, ...]:
        entries = load_material_catalog(self._index_dir)
        return tuple(entries)

    @staticmethod
    def _normalize_material_name(value: str) -> str:
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
        self, metadata: dict[str, object] | None
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

    def _doc_matches_material_keys(
        self,
        doc: Document,
        material_keys: set[tuple[str, str]] | None,
        excluded_source_types: set[str] | None = None,
    ) -> bool:
        excluded = self._normalize_source_type_filters(excluded_source_types)
        if excluded:
            source_type = str(
                (doc.metadata or {}).get("source_type") or ""
            ).strip().lower()
            if source_type in excluded:
                return False
        if not material_keys:
            return True
        key = self._metadata_material_key(doc.metadata or {})
        return key in material_keys if key is not None else False

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

    def _filter_docs_by_source_type(
        self,
        docs: Sequence[Document],
        excluded_source_types: set[str] | None,
    ) -> list[Document]:
        excluded = self._normalize_source_type_filters(excluded_source_types)
        if not excluded:
            return list(docs)
        filtered: list[Document] = []
        for doc in docs:
            source_type = str(
                (doc.metadata or {}).get("source_type") or ""
            ).strip().lower()
            if source_type in excluded:
                continue
            filtered.append(doc)
        return filtered

    def _match_material_entries(
        self,
        material_names: Sequence[str],
        *,
        query: str | None = None,
        excluded_source_types: set[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[_MaterialMatch]:
        normalized_names: list[str] = []
        seen_names: set[str] = set()
        for raw in material_names:
            normalized = self._normalize_material_name(raw)
            if not normalized or normalized in seen_names:
                continue
            seen_names.add(normalized)
            normalized_names.append(normalized)
            if len(normalized_names) >= max(1, self._config.material_search_max_names):
                break
        if not normalized_names:
            return []

        entries = self._material_catalog_entries()
        if not entries:
            return []

        def _aliases(entry: MaterialCatalogEntry) -> list[str]:
            values = [entry.canonical_name, *entry.aliases]
            deduped: list[str] = []
            seen: set[str] = set()
            for value in values:
                normalized = self._normalize_material_name(value)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                deduped.append(normalized)
            return deduped

        strict_matches: list[_MaterialMatch] = []
        for name in normalized_names:
            for entry in entries:
                if name in _aliases(entry):
                    strict_matches.append(
                        _MaterialMatch(entry=entry, matched_name=name, strict=True)
                    )
        if strict_matches:
            return self._dedupe_material_matches(
                strict_matches,
                limit=max(1, int(self._config.material_search_max_names)),
            )

        partial_matches: list[_MaterialMatch] = []
        for name in normalized_names:
            for entry in entries:
                aliases = _aliases(entry)
                if any((name in alias) or (alias in name) for alias in aliases):
                    partial_matches.append(
                        _MaterialMatch(entry=entry, matched_name=name, strict=False)
                    )
        deduped_partial_matches = self._dedupe_material_matches(
            partial_matches,
            limit=None,
        )
        if len(deduped_partial_matches) <= 1:
            return deduped_partial_matches

        selected_match = self._select_partial_material_match_by_second_rec_semantic(
            query=query or "",
            matches=deduped_partial_matches,
            excluded_source_types=excluded_source_types,
            cancel_event=cancel_event,
        )
        if selected_match is not None:
            logger.info(
                "Material partial-match candidates resolved by semantic search: "
                "selected=%s candidates=%d",
                selected_match.entry.material_id,
                len(deduped_partial_matches),
            )
            return [selected_match]

        return deduped_partial_matches[
            : max(1, int(self._config.material_search_max_names))
        ]

    def _select_partial_material_match_by_second_rec_semantic(
        self,
        *,
        query: str,
        matches: Sequence[_MaterialMatch],
        excluded_source_types: set[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> _MaterialMatch | None:
        _raise_if_cancelled(cancel_event)
        cleaned_query = (query or "").strip()
        if not cleaned_query or len(matches) <= 1:
            return None

        material_keys = {
            self._normalize_material_key(match.entry.source_type, match.entry.source_key)
            for match in matches
        }
        if not material_keys:
            return None

        search_k = max(
            1,
            int(self._config.material_search_partial_match_semantic_top_k),
        )
        q = f"query: {cleaned_query}"
        query_vector = self._dense_query_vector(q)
        ranked_docs = self._dense_search_with_filters(
            query=q,
            query_vector=(
                query_vector.astype(float).tolist() if query_vector is not None else None
            ),
            k=search_k,
            stages={"second_recursive"},
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
            cancel_event=cancel_event,
        )
        if not ranked_docs:
            return None

        top_key = self._metadata_material_key(ranked_docs[0].metadata or {})
        if top_key is None:
            return None

        for match in matches:
            match_key = self._normalize_material_key(
                match.entry.source_type,
                match.entry.source_key,
            )
            if match_key == top_key:
                return match
        return None

    def _dedupe_material_matches(
        self,
        matches: Sequence[_MaterialMatch],
        *,
        limit: int | None = None,
    ) -> list[_MaterialMatch]:
        deduped: list[_MaterialMatch] = []
        seen_ids: set[str] = set()
        resolved_limit = (
            max(1, int(limit))
            if limit is not None
            else None
        )
        for match in matches:
            material_id = match.entry.material_id
            if material_id in seen_ids:
                continue
            seen_ids.add(material_id)
            deduped.append(match)
            if resolved_limit is not None and len(deduped) >= resolved_limit:
                break
        return deduped

    def _retrieve_material_limited_docs(
        self,
        *,
        query: str,
        material_keys: set[tuple[str, str]],
        excluded_source_types: set[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        sparse_k = max(0, self._config.sparse_search_top_k)
        sparse_sparse_k = max(0, self._config.sparse_search_initial_sparse_top_k)
        if sparse_k > 0:
            sparse_docs = self._sparse_search_mixed_sources(
                query,
                top_k=sparse_k,
                sparse_top_k=sparse_sparse_k,
                material_keys=material_keys,
                excluded_source_types=excluded_source_types,
                cancel_event=cancel_event,
            )
            if sparse_docs:
                return sparse_docs
        return self._dense_search_for_materials(
            query,
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
            cancel_event=cancel_event,
        )

    def _dense_search_for_materials(
        self,
        query: str,
        *,
        material_keys: set[tuple[str, str]],
        excluded_source_types: set[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        k = max(0, self._config.dense_search_top_k)
        if k <= 0:
            return []
        q = f"query: {query}"
        query_vector = self._dense_query_vector(q)
        stages = self._dense_stages()
        return self._dense_search_with_filters(
            query=q,
            query_vector=(
                query_vector.astype(float).tolist() if query_vector is not None else None
            ),
            k=k,
            stages=stages,
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
            cancel_event=cancel_event,
        )

    def _dense_search_with_filters(
        self,
        *,
        query: str,
        query_vector: list[float] | None,
        k: int,
        stages: set[str],
        material_keys: set[tuple[str, str]] | None,
        excluded_source_types: set[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        results: list[Document] = []
        seen: set[tuple[object, ...]] = set()
        fetch_k = max(k * 4, k + 10)
        max_fetch = max(fetch_k, k * 10)

        while True:
            _raise_if_cancelled(cancel_event)
            if query_vector is None:
                docs = self._vectorstore().similarity_search(query, k=fetch_k)
            else:
                docs_with_scores = self._vectorstore().similarity_search_with_score_by_vector(
                    query_vector,
                    k=fetch_k,
                )
                docs = [doc for doc, _ in docs_with_scores]

            for doc in docs:
                if stages and doc.metadata.get("chunk_stage") not in stages:
                    continue
                if not self._doc_matches_material_keys(
                    doc,
                    material_keys,
                    excluded_source_types=excluded_source_types,
                ):
                    continue
                key = self._doc_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                results.append(doc)
                if len(results) >= k:
                    return results

            if fetch_k >= max_fetch or len(docs) < fetch_k:
                break
            fetch_k = min(fetch_k * 2, max_fetch)
        return results

    def generate(
        self,
        *,
        query: str,
        question_author: str | None = None,
        docs: list[Document],
        retry_history: Sequence[tuple[str, str]] | None = None,
        history: Sequence[ChatHistoryEntry] | None = None,
        history_override: Sequence[ChatHistoryEntry] | None = None,
        include_capabilities_info: bool = True,
        idea_generation: bool = False,
        fast_mode: bool = False,
        extra_mode_instruction: str | None = None,
        history_scope: str | int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        _raise_if_cancelled(cancel_event)
        provider = (self._config.llm_provider or "").lower()
        gemini_model = self._config.genai_model
        llama_model_path = self._config.llama_model_path
        llama_ctx_size = self._config.llama_ctx_size
        temperature = (
            self._config.rag_idea_temperature
            if idea_generation
            else self._config.temperature
        )
        max_output_tokens = self._config.max_output_tokens
        thinking_level = self._config.thinking_level
        if history is None:
            if history_override is not None:
                history = list(history_override)
            else:
                history = self._history_for_prompt(
                    limit=self._config.prompt_history_default_turns,
                    history_scope=history_scope,
                )
        if provider == "gemini":
            prompt = build_gemini_prompt(
                query=query,
                question_author=question_author,
                prompt_mode="rag_idea" if idea_generation else "rag",
                docs=docs,
                history=history,
                retry_history=retry_history,
                circle_basic_info=self._config.circle_basic_info,
                chatbot_capabilities_info=self._config.chatbot_capabilities_info,
                include_capabilities_info=include_capabilities_info,
                extra_mode_instruction=extra_mode_instruction,
            )
            if self._config.prompt_full_log_enabled:
                logger.info("Answer LLM prompt (gemini): %s", prompt)
            text = generate_with_gemini_config(
                api_key=self._config.gemini_api_key,
                prompt=prompt,
                system_rules=self._config.system_rules,
                model=gemini_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                thinking_level=thinking_level,
            )
            if self._config.prompt_full_log_enabled:
                logger.info("Answer LLM output (gemini): %s", text)
        elif provider == "llama":
            messages = build_llama_messages(
                query=query,
                question_author=question_author,
                prompt_mode="rag_idea" if idea_generation else "rag",
                docs=docs,
                config=self._config,
                history=history,
                retry_history=retry_history,
                include_capabilities_info=include_capabilities_info,
                circle_basic_info=self._config.circle_basic_info,
                extra_mode_instruction=extra_mode_instruction,
            )
            if self._config.prompt_full_log_enabled:
                logger.info("Answer LLM prompt (llama): %s", messages)
            text = generate_with_llama_config(
                messages=messages,
                model_path=llama_model_path,
                ctx_size=llama_ctx_size,
                threads=self._config.llama_threads,
                gpu_layers=self._config.llama_gpu_layers,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stop=["\n---"],
            )
            if self._config.prompt_full_log_enabled:
                logger.info("Answer LLM output (llama): %s", text)
        else:
            raise ValueError(
                f"Unsupported llm_provider: {provider}. "
                "Use 'gemini' or 'llama'."
            )

        if not text:
            return "回答生成中に不具合が発生しました、もう一度お試しください。"

        _raise_if_cancelled(cancel_event)
        return _mask_discord_mentions(text)

    def _generate_no_rag(
        self,
        *,
        query: str,
        question_author: str | None = None,
        retry_history: Sequence[tuple[str, str]] | None = None,
        history: Sequence[ChatHistoryEntry] | None = None,
        history_override: Sequence[ChatHistoryEntry] | None = None,
        include_capabilities_info: bool = True,
        fast_mode: bool = False,
        extra_mode_instruction: str | None = None,
        history_scope: str | int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        _raise_if_cancelled(cancel_event)
        provider = (self._config.no_rag_llm_provider or "").lower()
        if history is None:
            if history_override is not None:
                history = list(history_override)
            else:
                history = self._history_for_prompt(
                    limit=self._config.prompt_history_default_turns,
                    history_scope=history_scope,
                )
        docs: list[Document] = []
        if provider == "gemini":
            prompt = build_gemini_prompt(
                query=query,
                question_author=question_author,
                prompt_mode="no_rag",
                docs=docs,
                history=history,
                retry_history=retry_history,
                circle_basic_info="",
                chatbot_capabilities_info=self._config.chatbot_capabilities_info,
                include_capabilities_info=include_capabilities_info,
                extra_mode_instruction=extra_mode_instruction,
            )
            if self._config.prompt_full_log_enabled:
                logger.info("No-RAG LLM prompt (gemini): %s", prompt)
            text = generate_with_gemini_config(
                api_key=self._config.gemini_api_key,
                prompt=prompt,
                system_rules=self._config.system_rules,
                model=self._config.no_rag_genai_model,
                temperature=self._config.no_rag_temperature,
                max_output_tokens=self._config.no_rag_max_output_tokens,
                thinking_level=self._config.no_rag_thinking_level,
            )
            if self._config.prompt_full_log_enabled:
                logger.info("No-RAG LLM output (gemini): %s", text)
        elif provider == "llama":
            messages = build_llama_messages(
                query=query,
                question_author=question_author,
                prompt_mode="no_rag",
                docs=docs,
                config=self._config,
                history=history,
                retry_history=retry_history,
                include_capabilities_info=include_capabilities_info,
                circle_basic_info="",
                extra_mode_instruction=extra_mode_instruction,
            )
            if self._config.prompt_full_log_enabled:
                logger.info("No-RAG LLM prompt (llama): %s", messages)
            text = generate_with_llama_config(
                messages=messages,
                model_path=self._config.no_rag_llama_model_path,
                ctx_size=self._config.no_rag_llama_ctx_size,
                threads=self._config.llama_threads,
                gpu_layers=self._config.llama_gpu_layers,
                temperature=self._config.no_rag_temperature,
                max_output_tokens=self._config.no_rag_max_output_tokens,
                stop=["\n---"],
            )
            if self._config.prompt_full_log_enabled:
                logger.info("No-RAG LLM output (llama): %s", text)
        else:
            raise ValueError(
                "Unsupported no_rag_llm_provider: "
                f"{self._config.no_rag_llm_provider}. Use 'gemini' or 'llama'."
            )

        if not text:
            return "回答生成中に不具合が発生しました、もう一度お試しください。"

        _raise_if_cancelled(cancel_event)
        return _mask_discord_mentions(text)

    def _generate_refusal(
        self,
        *,
        query: str,
        question_author: str | None = None,
        history: Sequence[ChatHistoryEntry] | None = None,
        include_capabilities_info: bool = False,
        fast_mode: bool = False,
        extra_mode_instruction: str | None = None,
        history_scope: str | int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        _raise_if_cancelled(cancel_event)
        provider = (
            self._config.llm_provider
            if fast_mode
            else self._config.refusal_llm_provider
        ).lower()
        if history is None:
            history = self._history_for_prompt(
                limit=self._config.prompt_history_default_turns,
                history_scope=history_scope,
            )
        docs: list[Document] = []

        if provider == "gemini":
            prompt = build_gemini_prompt(
                query=query,
                question_author=question_author,
                prompt_mode="refusal",
                docs=docs,
                history=history,
                circle_basic_info="",
                chatbot_capabilities_info=self._config.chatbot_capabilities_info,
                include_capabilities_info=include_capabilities_info,
                extra_mode_instruction=extra_mode_instruction,
            )
            text = generate_with_gemini_config(
                api_key=self._config.gemini_api_key,
                prompt=prompt,
                system_rules=self._config.system_rules,
                model=(
                    self._config.genai_model
                    if fast_mode
                    else self._config.refusal_genai_model
                ),
                temperature=(
                    self._config.temperature
                    if fast_mode
                    else self._config.refusal_temperature
                ),
                max_output_tokens=(
                    self._config.max_output_tokens
                    if fast_mode
                    else self._config.refusal_max_output_tokens
                ),
                thinking_level=(
                    self._config.thinking_level
                    if fast_mode
                    else self._config.refusal_thinking_level
                ),
            )
        elif provider == "llama":
            messages = build_llama_messages(
                query=query,
                question_author=question_author,
                prompt_mode="refusal",
                docs=docs,
                config=self._config,
                history=history,
                include_capabilities_info=include_capabilities_info,
                circle_basic_info="",
                extra_mode_instruction=extra_mode_instruction,
            )
            text = generate_with_llama_config(
                messages=messages,
                model_path=(
                    self._config.llama_model_path
                    if fast_mode
                    else self._config.refusal_llama_model_path
                ),
                ctx_size=(
                    self._config.llama_ctx_size
                    if fast_mode
                    else self._config.refusal_llama_ctx_size
                ),
                threads=self._config.llama_threads,
                gpu_layers=self._config.llama_gpu_layers,
                temperature=(
                    self._config.temperature
                    if fast_mode
                    else self._config.refusal_temperature
                ),
                max_output_tokens=(
                    self._config.max_output_tokens
                    if fast_mode
                    else self._config.refusal_max_output_tokens
                ),
                stop=["\n---"],
            )
        else:
            provider_label = (
                "llm_provider"
                if fast_mode
                else "refusal_llm_provider"
            )
            raise ValueError(
                f"Unsupported {provider_label}: {provider}. "
                "Use 'gemini' or 'llama'."
            )

        _raise_if_cancelled(cancel_event)
        parsed_answer, _, is_json, has_answer = self._parse_answer_payload(
            text,
            max_source_index=0,
        )
        if is_json and has_answer:
            return _mask_discord_mentions(parsed_answer)
        return _mask_discord_mentions(text or "")

    def answer(
        self,
        query: str,
        *,
        history_scope: str | int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        docs = self.retrieve(query, cancel_event=cancel_event)
        answer, final, source_ids, used_docs, history_sources = (
            self._answer_with_docs(
                query=query,
                docs=docs,
                history_scope=history_scope,
                cancel_event=cancel_event,
            )
        )
        self._record_history(
            query=query,
            answer=answer,
            sources=history_sources,
            history_scope=history_scope,
        )
        return final

    def answer_with_routing(
        self,
        query: str,
        *,
        question_author: str | None = None,
        on_routing_decided: Callable[[FunctionRoutingDecision], None] | None = None,
        on_research_start: Callable[[], None] | None = None,
        on_memory_start: Callable[[], None] | None = None,
        on_research_and_memory_start: Callable[[], None] | None = None,
        history_scope: str | int | None = None,
        routing_history_override: Sequence[ChatHistoryEntry] | None = None,
        generation_history_override: Sequence[ChatHistoryEntry] | None = None,
        force_disable_additional_memory: bool = False,
        append_sources_to_response: bool = True,
        extra_mode_instruction: str | None = None,
        force_fast_mode: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> str:
        _raise_if_cancelled(cancel_event)
        routing: FunctionRoutingDecision
        if not self._config.function_call_enabled:
            routing = FunctionRoutingDecision(
                target_model="rag",
                material_names=[],
                idea_generation=False,
                include_capabilities_info=False,
                recency_mode="off",
                use_additional_memory=False,
                needs_additional_query=False,
                additional_queries=[],
            )
        else:
            if routing_history_override is not None:
                routing_history = list(routing_history_override)
                logger.info(
                    "Routing history override applied. turns=%s",
                    len(routing_history),
                )
            else:
                routing_history = self._history_for_prompt(
                    limit=self._config.prompt_history_default_turns,
                    include_sources=False,
                    history_scope=history_scope,
                )
            routing = decide_tools(
                query=query,
                question_author=question_author,
                config=self._config,
                history=routing_history,
            )
        if force_disable_additional_memory and routing.use_additional_memory:
            routing = replace(routing, use_additional_memory=False)
            logger.info("Forced additional memory off after routing.")
        if force_fast_mode:
            routing = replace(
                routing,
                target_model="rag",
                material_names=[],
                needs_additional_query=False,
                additional_queries=[],
            )
            logger.info(
                "Fast mode enabled: routing target forced to rag and additional query expansion disabled."
            )
        logger.info(
            "Function-call routing decision: %s",
            routing,
        )
        if on_routing_decided is not None:
            try:
                on_routing_decided(routing)
            except Exception:
                logger.exception("Failed to run routing decision callback.")
        if not append_sources_to_response:
            logger.info("Source appending is disabled for this response.")

        if routing.needs_additional_query and routing.use_additional_memory:
            if on_research_and_memory_start is not None:
                try:
                    on_research_and_memory_start()
                except Exception:
                    logger.exception(
                        "Failed to send research+memory start notification"
                    )
        elif routing.needs_additional_query:
            if on_research_start is not None:
                try:
                    on_research_start()
                except Exception:
                    logger.exception("Failed to send research start notification")
        elif routing.use_additional_memory:
            if on_memory_start is not None:
                try:
                    on_memory_start()
                except Exception:
                    logger.exception("Failed to send memory start notification")

        if routing.target_model == "refusal":
            answer = self.answer_refusal(
                query=query,
                question_author=question_author,
                use_additional_history=routing.use_additional_memory,
                include_capabilities_info=False,
                history_override=generation_history_override,
                extra_mode_instruction=extra_mode_instruction,
                history_scope=history_scope,
                fast_mode=force_fast_mode,
                cancel_event=cancel_event,
            )
            return answer

        if routing.target_model == "material_search":
            return self.answer_material_search(
                query=query,
                material_names=routing.material_names,
                question_author=question_author,
                use_additional_history=routing.use_additional_memory,
                include_capabilities_info=routing.include_capabilities_info,
                recency_mode=routing.recency_mode,
                history_override=generation_history_override,
                append_sources_to_response=append_sources_to_response,
                extra_mode_instruction=extra_mode_instruction,
                history_scope=history_scope,
                fast_mode=force_fast_mode,
                cancel_event=cancel_event,
            )

        if routing.target_model == "rag":
            docs = self.retrieve(
                query,
                re_search=self._config.function_call_enabled,
                recency_mode=routing.recency_mode,
                disable_rerank=force_fast_mode,
                disable_mmr=force_fast_mode,
                cancel_event=cancel_event,
            )
            if routing.needs_additional_query and routing.additional_queries:
                retrieved_groups: list[list[Document]] = [docs]
                for transformed_query in routing.additional_queries:
                    retrieved_groups.append(
                        self._retrieve_transformed_query(
                            transformed_query,
                            recency_mode=routing.recency_mode,
                            cancel_event=cancel_event,
                        )
                    )
                docs = self._merge_docs(retrieved_groups)
            answer, final, source_ids, used_docs, history_sources = (
                self._answer_with_docs(
                    query=query,
                    question_author=question_author,
                    docs=docs,
                    use_additional_history=routing.use_additional_memory,
                    include_capabilities_info=routing.include_capabilities_info,
                    idea_generation=routing.idea_generation,
                    recency_mode=routing.recency_mode,
                    history_override=generation_history_override,
                    append_sources_to_response=append_sources_to_response,
                    extra_mode_instruction=extra_mode_instruction,
                    history_scope=history_scope,
                    fast_mode=force_fast_mode,
                    cancel_event=cancel_event,
                )
            )
            self._record_history(
                query=query,
                answer=answer,
                sources=history_sources,
                history_scope=history_scope,
            )
            return final

        answer = self.answer_no_rag(
            query,
            question_author=question_author,
            use_additional_history=routing.use_additional_memory,
            include_capabilities_info=routing.include_capabilities_info,
            history_override=generation_history_override,
            extra_mode_instruction=extra_mode_instruction,
            history_scope=history_scope,
            fast_mode=force_fast_mode,
            cancel_event=cancel_event,
        )
        return answer

    def answer_no_rag(
        self,
        query: str,
        *,
        question_author: str | None = None,
        use_additional_history: bool = False,
        include_capabilities_info: bool = True,
        history_override: Sequence[ChatHistoryEntry] | None = None,
        extra_mode_instruction: str | None = None,
        history_scope: str | int | None = None,
        fast_mode: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> str:
        answer, _ = self._generate_no_rag_payload(
            query=query,
            question_author=question_author,
            use_additional_history=use_additional_history,
            include_capabilities_info=include_capabilities_info,
            history_override=history_override,
            fast_mode=fast_mode,
            extra_mode_instruction=extra_mode_instruction,
            history_scope=history_scope,
            cancel_event=cancel_event,
        )
        self._record_history(
            query=query,
            answer=answer,
            sources=[],
            history_scope=history_scope,
        )
        return answer

    def answer_refusal(
        self,
        query: str,
        *,
        question_author: str | None = None,
        use_additional_history: bool = False,
        include_capabilities_info: bool = False,
        history_override: Sequence[ChatHistoryEntry] | None = None,
        extra_mode_instruction: str | None = None,
        history_scope: str | int | None = None,
        fast_mode: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> str:
        if history_override is not None:
            history = list(history_override)
        else:
            history = self._history_for_prompt(
                limit=(
                    self._config.prompt_history_additional_turns
                    if use_additional_history
                    else self._config.prompt_history_default_turns
                ),
                include_sources=False,
                history_scope=history_scope,
            )
        fixed_prefix = "安全上の理由により、この質問には回答できません。"
        supplemental = ""
        try:
            supplemental = self._generate_refusal(
                query=query,
                question_author=question_author,
                history=history,
                include_capabilities_info=include_capabilities_info,
                fast_mode=fast_mode,
                extra_mode_instruction=extra_mode_instruction,
                history_scope=history_scope,
                cancel_event=cancel_event,
            ).strip()
        except Exception:
            logger.exception("Failed to generate refusal supplemental text")

        answer = (
            fixed_prefix
            if not supplemental
            else f"{fixed_prefix}\n\n{supplemental}"
        )
        self._record_history(
            query=query,
            answer=answer,
            sources=[],
            history_scope=history_scope,
        )
        return answer

    def answer_material_search(
        self,
        *,
        query: str,
        material_names: Sequence[str],
        question_author: str | None = None,
        use_additional_history: bool = False,
        include_capabilities_info: bool = True,
        recency_mode: str = "off",
        history_override: Sequence[ChatHistoryEntry] | None = None,
        append_sources_to_response: bool = True,
        extra_mode_instruction: str | None = None,
        history_scope: str | int | None = None,
        fast_mode: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> str:
        _raise_if_cancelled(cancel_event)
        excluded_source_types = set(_MATERIAL_SEARCH_EXCLUDED_SOURCE_TYPES)
        matched = self._match_material_entries(
            material_names,
            query=query,
            excluded_source_types=excluded_source_types,
            cancel_event=cancel_event,
        )
        if not matched:
            return self._answer_with_standard_rag_route(
                query=query,
                question_author=question_author,
                use_additional_history=use_additional_history,
                include_capabilities_info=include_capabilities_info,
                recency_mode=recency_mode,
                history_override=history_override,
                append_sources_to_response=append_sources_to_response,
                extra_mode_instruction=extra_mode_instruction,
                history_scope=history_scope,
                excluded_source_types=excluded_source_types,
                fast_mode=fast_mode,
                cancel_event=cancel_event,
            )

        material_entries = [match.entry for match in matched]
        material_keys = {
            self._normalize_material_key(entry.source_type, entry.source_key)
            for entry in material_entries
        }
        searched_docs = self._retrieve_material_limited_docs(
            query=query,
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
            cancel_event=cancel_event,
        )
        if not searched_docs:
            return self._answer_with_standard_rag_route(
                query=query,
                question_author=question_author,
                use_additional_history=use_additional_history,
                include_capabilities_info=include_capabilities_info,
                recency_mode=recency_mode,
                history_override=history_override,
                append_sources_to_response=append_sources_to_response,
                extra_mode_instruction=extra_mode_instruction,
                history_scope=history_scope,
                excluded_source_types=excluded_source_types,
                fast_mode=fast_mode,
                cancel_event=cancel_event,
            )

        context_docs = self._build_material_context_docs(
            query=query,
            material_entries=material_entries,
            searched_docs=searched_docs,
            fast_mode=fast_mode,
            cancel_event=cancel_event,
        )
        if not context_docs:
            return self._answer_with_standard_rag_route(
                query=query,
                question_author=question_author,
                use_additional_history=use_additional_history,
                include_capabilities_info=include_capabilities_info,
                recency_mode=recency_mode,
                history_override=history_override,
                append_sources_to_response=append_sources_to_response,
                extra_mode_instruction=extra_mode_instruction,
                history_scope=history_scope,
                excluded_source_types=excluded_source_types,
                fast_mode=fast_mode,
                cancel_event=cancel_event,
            )

        answer, _, used_docs = self._generate_answer_payload(
            query=query,
            question_author=question_author,
            docs=context_docs,
            use_additional_history=use_additional_history,
            include_capabilities_info=include_capabilities_info,
            idea_generation=False,
            history_override=history_override,
            fast_mode=fast_mode,
            extra_mode_instruction=extra_mode_instruction,
            history_scope=history_scope,
            cancel_event=cancel_event,
        )
        if append_sources_to_response:
            all_source_selections = [
                _SourceSelection(doc_index=idx)
                for idx in range(1, len(used_docs) + 1)
            ]
            final = self._append_sources(
                answer=answer,
                docs=used_docs,
                source_selections=all_source_selections,
            )
        else:
            final = answer
        self._record_history(
            query=query,
            answer=answer,
            sources=self._build_source_refs(used_docs),
            history_scope=history_scope,
        )
        return final

    def _answer_with_standard_rag_route(
        self,
        *,
        query: str,
        question_author: str | None = None,
        use_additional_history: bool = False,
        include_capabilities_info: bool = True,
        recency_mode: str = "off",
        history_override: Sequence[ChatHistoryEntry] | None = None,
        append_sources_to_response: bool = True,
        extra_mode_instruction: str | None = None,
        history_scope: str | int | None = None,
        excluded_source_types: set[str] | None = None,
        fast_mode: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> str:
        docs = self.retrieve(
            query,
            re_search=self._config.function_call_enabled,
            recency_mode=recency_mode,
            disable_rerank=fast_mode,
            disable_mmr=fast_mode,
            cancel_event=cancel_event,
        )
        docs = self._filter_docs_by_source_type(
            docs,
            excluded_source_types,
        )
        answer, final, _, _, history_sources = self._answer_with_docs(
            query=query,
            question_author=question_author,
            docs=docs,
            use_additional_history=use_additional_history,
            include_capabilities_info=include_capabilities_info,
            idea_generation=False,
            recency_mode=recency_mode,
            history_override=history_override,
            append_sources_to_response=append_sources_to_response,
            fast_mode=fast_mode,
            extra_mode_instruction=extra_mode_instruction,
            history_scope=history_scope,
            cancel_event=cancel_event,
        )
        self._record_history(
            query=query,
            answer=answer,
            sources=history_sources,
            history_scope=history_scope,
        )
        return final

    def _answer_with_docs(
        self,
        *,
        query: str,
        question_author: str | None = None,
        docs: list[Document],
        use_additional_history: bool = False,
        include_capabilities_info: bool = True,
        idea_generation: bool = False,
        recency_mode: str = "off",
        history_override: Sequence[ChatHistoryEntry] | None = None,
        append_sources_to_response: bool = True,
        fast_mode: bool = False,
        extra_mode_instruction: str | None = None,
        history_scope: str | int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, str, list[_SourceSelection], list[Document], list[str]]:
        answer, source_selections, used_docs = self._generate_answer_payload(
            query=query,
            question_author=question_author,
            docs=docs,
            use_additional_history=use_additional_history,
            include_capabilities_info=include_capabilities_info,
            idea_generation=idea_generation,
            history_override=history_override,
            fast_mode=fast_mode,
            extra_mode_instruction=extra_mode_instruction,
            history_scope=history_scope,
            cancel_event=cancel_event,
        )
        source_selections = self._order_source_selections(
            source_selections=source_selections,
            query=query,
            docs=used_docs,
            recency_mode=recency_mode,
            disable_rerank=fast_mode,
        )
        history_sources = self._build_source_refs(
            docs=used_docs,
            source_selections=source_selections,
        )
        if append_sources_to_response:
            final = self._append_sources(
                answer=answer,
                docs=used_docs,
                source_selections=source_selections,
            )
        else:
            final = answer
        return answer, final, source_selections, used_docs, history_sources

    def answer_with_contexts(
        self,
        query: str,
        *,
        history_scope: str | int | None = None,
    ) -> tuple[str, list[str]]:
        docs = self.retrieve(query)
        answer, _, used_docs = self._generate_answer_payload(
            query=query,
            docs=docs,
            history_scope=history_scope,
        )
        self._record_history(
            query=query,
            answer=answer,
            sources=[],
            history_scope=history_scope,
        )
        return answer, [doc_to_context(d) for d in used_docs]

    def refresh_index(self) -> None:
        self._vectorstore.cache_clear()
        self._docstore_id_to_faiss_index.cache_clear()
        self._sparse_index.cache_clear()
        self._sparse_second_rec_index.cache_clear()
        self._second_rec_sparse_index.cache_clear()
        self._second_rec_chunk_map.cache_clear()
        self._first_rec_chunk_map.cache_clear()
        self._summery_chunk_map.cache_clear()
        self._discord_raw_chunk_lines.cache_clear()
        self._material_catalog_entries.cache_clear()
        self._clear_rerank_score_cache()

    @property
    def config(self) -> AppConfig:
        return self._config

    def embeddings(self) -> Embeddings:
        return self._embedding_factory.get_embeddings()

    @lru_cache(maxsize=1)
    def _vectorstore(self) -> FAISS:
        return load_faiss_index(
            index_dir=self._index_dir,
            embedding_factory=self._embedding_factory,
        )

    def _dense_search(
        self, query: str, *, cancel_event: threading.Event | None = None
    ) -> _DenseSearchResult:
        _raise_if_cancelled(cancel_event)
        k = max(0, self._config.dense_search_top_k)
        if k <= 0:
            return _DenseSearchResult(
                docs=[],
                query_vector=None,
                doc_vectors_by_key={},
            )

        q = f"query: {query}"
        query_vector = self._dense_query_vector(q)
        stages = self._dense_stages()

        if query_vector is None:
            if not stages:
                docs = self._vectorstore().similarity_search(q, k=k)
            else:
                docs = self._dense_search_filtered(
                    q, k, stages, cancel_event=cancel_event
                )
            return _DenseSearchResult(
                docs=docs,
                query_vector=None,
                doc_vectors_by_key={},
            )

        query_list = query_vector.astype(float).tolist()
        if not stages:
            docs_with_scores = self._vectorstore().similarity_search_with_score_by_vector(
                query_list,
                k=k,
            )
            docs = [doc for doc, _ in docs_with_scores]
        else:
            docs = self._dense_search_filtered_by_vector(
                query_list, k, stages, cancel_event=cancel_event
            )

        return _DenseSearchResult(
            docs=docs,
            query_vector=(
                query_vector
                if self._can_reuse_dense_query_vector_for_mmr()
                else None
            ),
            doc_vectors_by_key=self._reconstruct_dense_doc_vectors(docs),
        )

    def _dense_query_vector(self, query: str) -> np.ndarray | None:
        try:
            query_vec = self.embeddings().embed_query(query)
        except Exception:
            logger.exception("Dense query embedding failed; fallback to text search.")
            return None
        return self._as_1d_float_array(query_vec)

    def _can_reuse_dense_query_vector_for_mmr(self) -> bool:
        # For multilingual-e5, embed_query() normalizes to "query: ...", so
        # dense and MMR query embeddings are equivalent.
        return bool(getattr(self.embeddings(), "_use_e5_prefix", False))

    def _dense_stages(self) -> set[str]:
        stages: set[str] = set()
        if self._config.prop_enabled and self._config.second_rec_enabled:
            stages.add("proposition")
        else:
            if self._config.second_rec_enabled:
                stages.add("second_recursive")
            else:
                stages.add("first_recursive")
                if (self._config.second_rec_chunk_dir / "vc").exists():
                    stages.add("second_recursive")
        if self._config.raptor_enabled:
            stages.add("raptor")
        return stages

    def _dense_search_filtered(
        self,
        query: str,
        k: int,
        stages: set[str],
        *,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        if len(stages) == 1:
            stage = next(iter(stages))
            try:
                return self._vectorstore().similarity_search(
                    query, k=k, filter={"chunk_stage": stage}
                )
            except TypeError:
                pass

        results: list[Document] = []
        seen: set[tuple[object, ...]] = set()
        fetch_k = max(k * 4, k + 10)
        max_fetch = max(fetch_k, k * 10)
        while True:
            _raise_if_cancelled(cancel_event)
            docs = self._vectorstore().similarity_search(query, k=fetch_k)
            for doc in docs:
                if doc.metadata.get("chunk_stage") not in stages:
                    continue
                key = self._doc_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                results.append(doc)
                if len(results) >= k:
                    return results
            if fetch_k >= max_fetch or len(docs) < fetch_k:
                break
            fetch_k = min(fetch_k * 2, max_fetch)
        return results

    def _dense_search_filtered_by_vector(
        self,
        query_vector: list[float],
        k: int,
        stages: set[str],
        *,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        if len(stages) == 1:
            stage = next(iter(stages))
            try:
                docs_with_scores = self._vectorstore().similarity_search_with_score_by_vector(
                    query_vector,
                    k=k,
                    filter={"chunk_stage": stage},
                )
                return [doc for doc, _ in docs_with_scores]
            except TypeError:
                pass

        results: list[Document] = []
        seen: set[tuple[object, ...]] = set()
        fetch_k = max(k * 4, k + 10)
        max_fetch = max(fetch_k, k * 10)
        while True:
            _raise_if_cancelled(cancel_event)
            docs_with_scores = self._vectorstore().similarity_search_with_score_by_vector(
                query_vector,
                k=fetch_k,
            )
            docs = [doc for doc, _ in docs_with_scores]
            for doc in docs:
                if doc.metadata.get("chunk_stage") not in stages:
                    continue
                key = self._doc_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                results.append(doc)
                if len(results) >= k:
                    return results
            if fetch_k >= max_fetch or len(docs) < fetch_k:
                break
            fetch_k = min(fetch_k * 2, max_fetch)
        return results

    def _reconstruct_dense_doc_vectors(
        self, docs: list[Document]
    ) -> dict[tuple[object, ...], np.ndarray]:
        if not docs:
            return {}

        index_lookup = self._docstore_id_to_faiss_index()
        if not index_lookup:
            return {}

        vectors: dict[tuple[object, ...], np.ndarray] = {}
        index = self._vectorstore().index
        for doc in docs:
            key = self._doc_key(doc)
            if key in vectors:
                continue
            doc_id = getattr(doc, "id", None)
            if not isinstance(doc_id, str) or not doc_id:
                continue
            index_id = index_lookup.get(doc_id)
            if index_id is None:
                continue
            try:
                raw_vec = index.reconstruct(int(index_id))
            except Exception:
                continue
            vector = self._as_1d_float_array(raw_vec)
            if vector is None:
                continue
            vectors[key] = vector
        return vectors

    @lru_cache(maxsize=1)
    def _docstore_id_to_faiss_index(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for index_id, docstore_id in self._vectorstore().index_to_docstore_id.items():
            if isinstance(docstore_id, str):
                mapping[docstore_id] = int(index_id)
        return mapping

    def _sparse_search(
        self,
        *,
        original_query: str,
        transformed_query: str | None,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        results: list[list[Document]] = []
        original_k = max(0, self._config.sparse_search_original_top_k)
        transform_k = max(0, self._config.sparse_search_transform_top_k)
        original_sparse_k = max(
            0, self._config.sparse_search_original_sparse_top_k
        )

        if original_k > 0:
            original_docs = self._sparse_search_mixed_sources(
                original_query,
                top_k=original_k,
                sparse_top_k=original_sparse_k,
                cancel_event=cancel_event,
            )
            if original_docs:
                results.append(original_docs)

        transformed = (transformed_query or "").strip()
        if (
            transform_k > 0
            and transformed
            and (transformed != original_query or original_k == 0)
        ):
            transformed_docs = self._sparse_search_once(
                transformed, transform_k, cancel_event=cancel_event
            )
            if transformed_docs:
                results.append(transformed_docs)

        if not results:
            return []
        return self._merge_docs(results)

    def _sparse_search_initial(
        self, query: str, *, cancel_event: threading.Event | None = None
    ) -> list[Document]:
        k = max(0, self._config.sparse_search_top_k)
        sparse_k = max(0, self._config.sparse_search_initial_sparse_top_k)
        if k <= 0:
            return []
        return self._sparse_search_mixed_sources(
            query,
            top_k=k,
            sparse_top_k=sparse_k,
            cancel_event=cancel_event,
        )

    def _sparse_search_mixed_sources(
        self,
        query: str,
        *,
        top_k: int,
        sparse_top_k: int,
        material_keys: set[tuple[str, str]] | None = None,
        excluded_source_types: set[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        total_k = max(0, top_k)
        if total_k <= 0:
            return []

        sparse_k = min(max(0, sparse_top_k), total_k)
        second_rec_k = total_k - sparse_k
        tokens = self._sudachi_tokens(query)
        if not tokens:
            return []

        with ThreadPoolExecutor(max_workers=2) as executor:
            sparse_future = executor.submit(
                self._sparse_search_tokens_once_with_index,
                tokens,
                total_k,
                index_loader=self._sparse_second_rec_index,
                material_keys=material_keys,
                excluded_source_types=excluded_source_types,
                cancel_event=cancel_event,
            )
            second_rec_future = executor.submit(
                self._sparse_search_tokens_once_with_index,
                tokens,
                total_k,
                index_loader=self._second_rec_sparse_index,
                material_keys=material_keys,
                excluded_source_types=excluded_source_types,
                cancel_event=cancel_event,
            )
            done, _ = wait(
                [sparse_future, second_rec_future], return_when=FIRST_EXCEPTION
            )
            for future in done:
                exc = future.exception()
                if exc is not None:
                    sparse_future.cancel()
                    second_rec_future.cancel()
                    raise exc
            sparse_docs = sparse_future.result()
            second_rec_docs = second_rec_future.result()

        selected_sparse = sparse_docs[:sparse_k]
        selected_second_rec = second_rec_docs[:second_rec_k]
        merged = self._merge_docs([selected_sparse, selected_second_rec])
        if len(merged) < total_k:
            merged = self._merge_docs(
                [
                    merged,
                    sparse_docs[sparse_k:],
                    second_rec_docs[second_rec_k:],
                ]
            )
        return merged[:total_k]

    def _sparse_search_once(
        self,
        query: str,
        k: int,
        *,
        material_keys: set[tuple[str, str]] | None = None,
        excluded_source_types: set[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        return self._sparse_search_once_with_index(
            query,
            k,
            index_loader=self._sparse_index,
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
            cancel_event=cancel_event,
        )

    def _sparse_search_once_with_index(
        self,
        query: str,
        k: int,
        *,
        index_loader: Callable[[], KeywordInvertedIndex],
        material_keys: set[tuple[str, str]] | None = None,
        excluded_source_types: set[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        if k <= 0:
            return []
        tokens = self._sudachi_tokens(query)
        return self._sparse_search_tokens_once_with_index(
            tokens,
            k,
            index_loader=index_loader,
            material_keys=material_keys,
            excluded_source_types=excluded_source_types,
            cancel_event=cancel_event,
        )

    def _sparse_search_tokens_once_with_index(
        self,
        tokens: Sequence[str],
        k: int,
        *,
        index_loader: Callable[[], KeywordInvertedIndex],
        material_keys: set[tuple[str, str]] | None = None,
        excluded_source_types: set[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        if k <= 0:
            return []
        if not tokens:
            return []
        keyword_index = index_loader()
        docs = keyword_index.docs
        if not docs:
            return []
        scores = keyword_index.get_scores(tokens)
        if scores is None:
            return []

        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        results: list[Document] = []
        for idx in ranked:
            _raise_if_cancelled(cancel_event)
            if scores[idx] <= 0:
                break
            restored = self._restore_sparse_hit_doc(docs[idx])
            if not self._doc_matches_material_keys(
                restored,
                material_keys,
                excluded_source_types=excluded_source_types,
            ):
                continue
            results.append(restored)
            if len(results) >= k:
                break
        return results

    @lru_cache(maxsize=1)
    def _sparse_index(self) -> KeywordInvertedIndex:
        return self._load_or_build_sparse_index(
            corpus_name=KEYWORD_CORPUS_SPARSE,
            docs_loader=self._load_sparse_docs,
        )

    @lru_cache(maxsize=1)
    def _sparse_second_rec_index(self) -> KeywordInvertedIndex:
        return self._load_or_build_sparse_index(
            corpus_name=KEYWORD_CORPUS_SPARSE_SECOND_REC,
            docs_loader=lambda: self._load_sparse_docs_for_dirs(
                self._sparse_second_rec_chunk_dirs()
            ),
        )

    @lru_cache(maxsize=1)
    def _second_rec_sparse_index(self) -> KeywordInvertedIndex:
        return self._load_or_build_sparse_index(
            corpus_name=KEYWORD_CORPUS_SECOND_REC_SPARSE,
            docs_loader=lambda: self._load_sparse_docs_for_dirs(
                self._second_rec_chunk_dirs()
            ),
        )

    def _load_sparse_docs(self) -> list[Document]:
        chunk_dirs = self._sparse_chunk_dirs()
        return self._load_sparse_docs_for_dirs(chunk_dirs)

    def _load_sparse_docs_for_dirs(self, chunk_dirs: list[Path]) -> list[Document]:
        if not chunk_dirs:
            return []

        chunks = load_chunks_from_dirs(chunk_dirs)
        return [
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in chunks
        ]

    def _sparse_second_rec_chunk_dirs(self) -> list[Path]:
        return resolve_sparse_second_rec_chunk_dirs(self._config)

    def _second_rec_chunk_dirs(self) -> list[Path]:
        return resolve_second_rec_chunk_dirs(self._config)

    def _sparse_chunk_dirs(self) -> list[Path]:
        return resolve_sparse_chunk_dirs(self._config)

    def _sudachi_tokens(self, text: str) -> list[str]:
        return self._query_normalizer().normalize_tokens(text)

    @lru_cache(maxsize=1)
    def _query_normalizer(self) -> SparseNormalizer:
        return SparseNormalizer(
            config=SparseNormalizerConfig(
                sudachi_mode=self._config.sudachi_mode,
                use_normalized_form=self._config.sparse_use_normalized_form,
                remove_symbols=self._config.sparse_remove_symbols,
                remove_stopwords=False,
            )
        )

    def _sparse_doc_tokens(self, doc: Document) -> list[str]:
        return tokenize_sparse_doc(
            doc,
            sparse_stage=_SECOND_REC_SPARSE_STAGE,
            sudachi_tokenize=self._sudachi_tokens,
        )

    def _load_or_build_sparse_index(
        self,
        *,
        corpus_name: str,
        docs_loader: Callable[[], list[Document]],
    ) -> KeywordInvertedIndex:
        prebuilt = load_keyword_index(index_dir=self._index_dir, corpus_name=corpus_name)
        if (
            prebuilt is not None
            and math.isclose(prebuilt.k1, self._config.sparse_bm25_k1)
            and math.isclose(prebuilt.b, self._config.sparse_bm25_b)
        ):
            return prebuilt
        if prebuilt is not None:
            logger.warning(
                "Keyword index parameter mismatch (corpus=%s): "
                "index(k1=%s,b=%s) current(k1=%s,b=%s). Rebuilding in-memory.",
                corpus_name,
                prebuilt.k1,
                prebuilt.b,
                self._config.sparse_bm25_k1,
                self._config.sparse_bm25_b,
            )
        else:
            logger.warning(
                "Keyword index not found for corpus=%s. Building in-memory fallback.",
                corpus_name,
            )
        docs = docs_loader()
        return KeywordInvertedIndex.build(
            docs=docs,
            tokenize_doc=self._sparse_doc_tokens,
            k1=self._config.sparse_bm25_k1,
            b=self._config.sparse_bm25_b,
        )

    def _restore_sparse_hit_doc(self, doc: Document) -> Document:
        metadata = doc.metadata or {}
        if metadata.get("chunk_stage") != _SECOND_REC_SPARSE_STAGE:
            return doc

        chunk_id = self._normalize_chunk_id(metadata.get("chunk_id"))
        if chunk_id is None:
            return doc
        key = self._chunk_lookup_key(metadata, chunk_id)
        if key is None:
            return doc

        resolved = self._second_rec_chunk_map().get(key)
        return resolved if resolved is not None else doc

    @staticmethod
    def _doc_key(doc: Document) -> tuple[object, ...]:
        metadata = doc.metadata or {}
        stage = metadata.get("chunk_stage")
        source = metadata.get("drive_file_id") or metadata.get("source_file_name")
        chunk_id = metadata.get("chunk_id")
        raptor_level = metadata.get("raptor_level")
        raptor_cluster = metadata.get("raptor_cluster_id")
        if stage or source or chunk_id or raptor_level or raptor_cluster:
            return (stage, source, chunk_id, raptor_level, raptor_cluster)
        return ("content", doc.page_content)

    def _merge_docs(self, groups: list[list[Document]]) -> list[Document]:
        merged: list[Document] = []
        seen: set[tuple[object, ...]] = set()
        for docs in groups:
            for doc in docs:
                key = self._doc_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
        return merged

    def _rerank_docs(
        self,
        *,
        query: str,
        docs: list[Document],
        recency_mode: str = "off",
    ) -> list[Document]:
        if not docs:
            return []
        if not self._config.rerank_enabled:
            return docs
        scored_base = self._reranker.score_documents(query=query, docs=docs)
        scored = self._apply_recency_scores(
            scored=scored_base,
            recency_mode=recency_mode,
        )
        self._store_rerank_scores(
            query=query,
            recency_mode=recency_mode,
            scored=scored,
        )
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _, _, doc in scored]

    def _store_rerank_scores(
        self,
        *,
        query: str,
        recency_mode: str = "off",
        scored: Sequence[tuple[float, int, Document]],
    ) -> None:
        cache_key = self._rerank_score_cache_key(
            query=query,
            recency_mode=recency_mode,
        )
        if not cache_key or not scored:
            return

        per_doc_scores: dict[tuple[object, ...], float] = {}
        for score, _, doc in scored:
            key = self._doc_key(doc)
            previous = per_doc_scores.get(key)
            if previous is None or score > previous:
                per_doc_scores[key] = score
        if not per_doc_scores:
            return

        with self._rerank_score_cache_lock:
            cached = self._rerank_scores_by_query.get(cache_key)
            if cached is None:
                cached = {}
                self._rerank_scores_by_query[cache_key] = cached
            cached.update(per_doc_scores)
            try:
                self._rerank_score_query_order.remove(cache_key)
            except ValueError:
                pass
            self._rerank_score_query_order.append(cache_key)
            while len(self._rerank_score_query_order) > 16:
                oldest = self._rerank_score_query_order.popleft()
                self._rerank_scores_by_query.pop(oldest, None)

    def _rerank_scores_for_query(
        self,
        *,
        query: str,
        recency_mode: str = "off",
    ) -> dict[tuple[object, ...], float]:
        cache_key = self._rerank_score_cache_key(
            query=query,
            recency_mode=recency_mode,
        )
        if not cache_key:
            return {}
        with self._rerank_score_cache_lock:
            cached = self._rerank_scores_by_query.get(cache_key)
            if not cached:
                return {}
            try:
                self._rerank_score_query_order.remove(cache_key)
            except ValueError:
                pass
            self._rerank_score_query_order.append(cache_key)
            return dict(cached)

    def _clear_rerank_score_cache(self) -> None:
        with self._rerank_score_cache_lock:
            self._rerank_scores_by_query.clear()
            self._rerank_score_query_order.clear()

    def _rerank_score_cache_key(self, *, query: str, recency_mode: str) -> str:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return ""
        mode = self._normalize_recency_mode(recency_mode)
        return f"{normalized_query}\t{mode}"

    @staticmethod
    def _normalize_recency_mode(recency_mode: str | None) -> str:
        mode = str(recency_mode or "").strip().lower()
        if mode in {"off", "soft", "hard"}:
            return mode
        return "off"

    def _recency_weight_for_mode(self, recency_mode: str) -> float:
        mode = self._normalize_recency_mode(recency_mode)
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
        scored: Sequence[tuple[float, int, Document]],
        recency_mode: str,
    ) -> list[tuple[float, int, Document]]:
        if not scored:
            return []

        weight = self._recency_weight_for_mode(recency_mode)
        if weight <= 0.0:
            return list(scored)

        half_life_days = max(0.0001, float(self._config.recency_half_life_days))
        today = datetime.now(_JST).date()
        adjusted: list[tuple[float, int, Document]] = []
        for base_score, original_index, doc in scored:
            doc_date = self._doc_source_date(doc)
            if doc_date is None:
                recency_score = 0.5
            else:
                age_days = max(0, (today - doc_date).days)
                recency_score = 0.5 ** (age_days / half_life_days)
            final_score = (1.0 - weight) * float(base_score) + weight * recency_score
            adjusted.append((final_score, original_index, doc))
        return adjusted

    @staticmethod
    def _doc_source_date(doc: Document):
        metadata = doc.metadata or {}
        source_date_raw = str(metadata.get("source_date") or "").strip()
        if not source_date_raw:
            source_date_raw = infer_source_date(metadata=metadata)
        if source_date_raw == SOURCE_DATE_UNKNOWN:
            return None
        return source_date_to_date(source_date_raw)

    def _apply_parent_doc_cap(self, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        cap = self._config.parent_chunk_cap
        if cap <= 0:
            return docs

        counts: dict[tuple[object, ...], int] = {}
        capped: list[Document] = []
        for doc in docs:
            key = self._parent_cap_key(doc)
            count = counts.get(key, 0)
            if count >= cap:
                continue
            counts[key] = count + 1
            capped.append(doc)
        return capped

    def _parent_cap_key(self, doc: Document) -> tuple[object, ...]:
        metadata = doc.metadata or {}
        stage = metadata.get("chunk_stage")
        parent_id = self._normalize_chunk_id(metadata.get("parent_chunk_id"))
        if stage == "proposition":
            resolved = self._resolve_first_parent_id(metadata, parent_id)
            if resolved is not None:
                parent_id = resolved
        if parent_id is None:
            return ("self", self._doc_key(doc))
        key = self._chunk_lookup_key(metadata, parent_id)
        if key is None:
            return ("self", self._doc_key(doc))
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
        second_doc = self._second_rec_chunk_map().get(key)
        if second_doc is None:
            return None
        return self._normalize_chunk_id(
            (second_doc.metadata or {}).get("parent_chunk_id")
        )

    def _limit_rerank_pool(self, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        pool_size = self._config.rerank_pool_size
        if pool_size <= 0:
            return docs
        return docs[:pool_size]

    def _select_with_mmr(
        self,
        *,
        query: str,
        docs: list[Document],
        query_vector: np.ndarray | None = None,
        doc_vectors_by_key: dict[tuple[object, ...], np.ndarray] | None = None,
    ) -> list[Document]:
        if not docs:
            return []
        top_k = max(0, self._config.top_k)
        if top_k <= 0:
            return []

        final_k = min(top_k, len(docs))
        if final_k <= 3:
            return docs[:final_k]

        fixed = min(3, final_k)
        if fixed >= final_k:
            return docs[:final_k]

        embeddings = self._mmr_embeddings(
            query=query,
            docs=docs,
            query_vector=query_vector,
            doc_vectors_by_key=doc_vectors_by_key,
        )
        if embeddings is None:
            return docs[:final_k]
        query_vec, doc_vectors = embeddings

        lambda_mult = self._config.mmr_lambda
        if lambda_mult < 0:
            lambda_mult = 0.0
        elif lambda_mult > 1:
            lambda_mult = 1.0

        sim_to_query = doc_vectors @ query_vec
        selected = list(range(fixed))
        remaining = list(range(fixed, len(docs)))

        while len(selected) < final_k and remaining:
            selected_vecs = doc_vectors[selected]
            if selected_vecs.size == 0:
                max_div = np.zeros(len(remaining))
            else:
                sims = doc_vectors[remaining] @ selected_vecs.T
                max_div = sims.max(axis=1)
            scores = lambda_mult * sim_to_query[remaining] - (
                1 - lambda_mult
            ) * max_div
            best_pos = int(np.argmax(scores))
            best_idx = remaining.pop(best_pos)
            selected.append(best_idx)

        return [docs[idx] for idx in selected]

    def _mmr_embeddings(
        self,
        *,
        query: str,
        docs: list[Document],
        query_vector: np.ndarray | None = None,
        doc_vectors_by_key: dict[tuple[object, ...], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        query_array = self._as_1d_float_array(query_vector)
        if query_array is None:
            try:
                query_vec = self.embeddings().embed_query(query)
            except Exception:
                logger.exception("MMR query embedding failed; fallback to reranked order.")
                return None
            query_array = self._as_1d_float_array(query_vec)
            if query_array is None:
                return None

        dim = query_array.shape[0]
        doc_matrix = np.zeros((len(docs), dim), dtype=float)
        lookup = doc_vectors_by_key or {}
        missing_positions: list[int] = []
        missing_texts: list[str] = []

        for idx, doc in enumerate(docs):
            cached = lookup.get(self._doc_key(doc))
            cached_array = self._as_1d_float_array(cached)
            if cached_array is not None and cached_array.shape[0] == dim:
                doc_matrix[idx] = cached_array
                continue
            missing_positions.append(idx)
            missing_texts.append(doc.page_content)

        if missing_positions:
            try:
                missing_vectors = self.embeddings().embed_documents(missing_texts)
            except Exception:
                logger.exception("MMR document embedding failed; fallback to reranked order.")
                return None
            if len(missing_vectors) != len(missing_positions):
                return None
            for idx, vector in zip(missing_positions, missing_vectors):
                vector_array = self._as_1d_float_array(vector)
                if vector_array is None or vector_array.shape[0] != dim:
                    return None
                doc_matrix[idx] = vector_array

        if doc_matrix.ndim != 2 or query_array.ndim != 1:
            return None
        if doc_matrix.shape[1] != query_array.shape[0]:
            return None

        query_norm = self._normalize_vector(query_array)
        if query_norm is None:
            return None
        doc_norm = self._normalize_matrix(doc_matrix)
        return query_norm, doc_norm

    @staticmethod
    def _as_1d_float_array(value: object) -> np.ndarray | None:
        if value is None:
            return None
        try:
            array = np.asarray(value, dtype=float)
        except Exception:
            return None
        if array.ndim != 1 or array.size == 0:
            return None
        return array

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        return vector / norm

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _append_parent_docs(
        self,
        docs: list[Document],
    ) -> list[Document]:
        if not docs or not self._config.parent_doc_enabled:
            return docs

        max_workers = min(8, len(docs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._parent_candidates_for_doc, doc)
                for doc in docs
            ]
            candidate_groups = [future.result() for future in futures]

        # Keep parent-related context adjacent to the originating chunk so
        # second_recursive / proposition style chunks and their summaries stay paired.
        ordered: list[Document] = []
        seen_doc_keys: set[tuple[object, ...]] = set()
        for doc, candidates in zip(docs, candidate_groups):
            doc_key = self._doc_key(doc)
            if doc_key not in seen_doc_keys:
                seen_doc_keys.add(doc_key)
                ordered.append(doc)
            for candidate in candidates:
                candidate_key = self._doc_key(candidate)
                if candidate_key in seen_doc_keys:
                    continue
                seen_doc_keys.add(candidate_key)
                ordered.append(candidate)
        return ordered

    def _parent_candidates_for_doc(self, doc: Document) -> list[Document]:
        metadata = doc.metadata or {}
        if self._metadata_flag_enabled(metadata.get("skip_parent_context")):
            return []
        stage = metadata.get("chunk_stage")
        if stage == "proposition":
            second_parent_id = self._normalize_chunk_id(
                metadata.get("parent_chunk_id")
            )
            first_parent_id = self._resolve_first_parent_id(
                metadata, second_parent_id
            )
            if first_parent_id is None:
                return []
            return self._first_or_summery_candidates(
                metadata=metadata,
                first_parent_id=first_parent_id,
            )

        if stage == "second_recursive":
            first_parent_id = self._normalize_chunk_id(
                metadata.get("parent_chunk_id")
            )
            if first_parent_id is None:
                return []
            return self._first_or_summery_candidates(
                metadata=metadata,
                first_parent_id=first_parent_id,
            )
        return []

    @lru_cache(maxsize=1)
    def _second_rec_chunk_map(self) -> dict[tuple[object, ...], Document]:
        if not self._config.second_rec_enabled:
            return {}
        return self._chunk_map_for_dirs(self._config.second_rec_chunk_dir)

    @lru_cache(maxsize=1)
    def _first_rec_chunk_map(self) -> dict[tuple[object, ...], Document]:
        return self._chunk_map_for_dirs(self._config.first_rec_chunk_dir)

    @lru_cache(maxsize=1)
    def _summery_chunk_map(self) -> dict[tuple[object, ...], list[Document]]:
        if not self._config.summery_enabled:
            return {}
        chunk_dirs = []
        for name in _SECOND_REC_SOURCE_DIRS:
            candidate = self._config.summery_chunk_dir / name
            if candidate.exists():
                chunk_dirs.append(candidate)
        if not chunk_dirs:
            return {}

        chunks = load_chunks_from_dirs(chunk_dirs)
        mapping: dict[tuple[object, ...], list[Document]] = {}
        for chunk in chunks:
            metadata = chunk.metadata
            parent_id = self._normalize_chunk_id(metadata.get("parent_chunk_id"))
            if parent_id is None:
                continue
            key = self._chunk_lookup_key(metadata, parent_id)
            if key is None:
                continue
            doc = Document(page_content=chunk.text, metadata=metadata)
            mapping.setdefault(key, []).append(doc)
        return mapping

    def _build_material_context_docs(
        self,
        *,
        query: str,
        material_entries: Sequence[MaterialCatalogEntry],
        searched_docs: Sequence[Document],
        fast_mode: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        if not material_entries:
            return []

        entry_by_key: dict[tuple[str, str], MaterialCatalogEntry] = {}
        for entry in material_entries:
            key = self._normalize_material_key(entry.source_type, entry.source_key)
            entry_by_key[key] = entry
        if not entry_by_key:
            return []

        docs_by_key: dict[tuple[str, str], list[Document]] = {}
        ordered_keys: list[tuple[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()
        for doc in searched_docs:
            key = self._metadata_material_key(doc.metadata or {})
            if key is None or key not in entry_by_key:
                continue
            docs_by_key.setdefault(key, []).append(doc)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ordered_keys.append(key)
        if not ordered_keys:
            ordered_keys = list(entry_by_key.keys())

        contexts: list[Document] = []
        char_limit = max(1, int(self._config.material_search_char_limit))
        for key in ordered_keys:
            _raise_if_cancelled(cancel_event)
            entry = entry_by_key.get(key)
            if entry is None:
                continue
            representative_metadata = self._representative_metadata_for_material(
                material_key=key,
                searched_docs=docs_by_key.get(key) or [],
                entry=entry,
            )
            raw_text = self._read_material_raw_text(entry)
            if raw_text and len(raw_text) <= char_limit:
                contexts.append(
                    Document(
                        page_content=raw_text,
                        metadata=representative_metadata,
                    )
                )
                continue

            summary_docs = self._summary_docs_for_material_key(key)
            if summary_docs:
                selected_summary_docs = self._select_summary_docs_for_material(
                    query=query,
                    entry=entry,
                    summary_docs=summary_docs,
                )
                first_rec_docs = self._first_rec_docs_from_summary_docs(
                    material_key=key,
                    summary_docs=selected_summary_docs,
                )
                if first_rec_docs:
                    contexts.extend(first_rec_docs)
                    continue

            fallback_docs = self._first_rec_docs_for_material_key(key)
            if not fallback_docs:
                fallback_docs = docs_by_key.get(key) or []
            contexts.extend(
                self._select_material_fallback_docs(
                    query=query,
                    docs=fallback_docs,
                    fast_mode=fast_mode,
                )
            )

        if not contexts:
            return []
        return self._merge_docs([contexts])

    def _representative_metadata_for_material(
        self,
        *,
        material_key: tuple[str, str],
        searched_docs: Sequence[Document],
        entry: MaterialCatalogEntry,
    ) -> dict[str, object]:
        first_rec_docs = self._first_rec_docs_for_material_key(material_key)
        if first_rec_docs:
            metadata = dict(first_rec_docs[0].metadata or {})
        elif searched_docs:
            metadata = dict((searched_docs[0].metadata or {}))
        else:
            metadata = {}
        metadata.setdefault("source_type", entry.source_type)
        metadata.setdefault("source_file_name", entry.source_key)
        return metadata

    def _read_material_raw_text(self, entry: MaterialCatalogEntry) -> str:
        path = self._config.base_dir / Path(entry.raw_path)
        if not path.exists() or not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            logger.warning("Failed to read material raw file: %s", path, exc_info=True)
            return ""

    def _summary_docs_for_material_key(
        self, material_key: tuple[str, str]
    ) -> list[Document]:
        docs: list[Document] = []
        for key, chunk_docs in self._summery_chunk_map().items():
            if len(key) < 2:
                continue
            source_type = str(key[0] or "")
            source_key = str(key[1] or "")
            normalized = self._normalize_material_key(source_type, source_key)
            if normalized != material_key:
                continue
            docs.extend(chunk_docs)
        return docs

    def _first_rec_docs_for_material_key(
        self, material_key: tuple[str, str]
    ) -> list[Document]:
        matched: list[tuple[int, Document]] = []
        for key, doc in self._first_rec_chunk_map().items():
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
            matched.append((chunk_id, doc))
        matched.sort(key=lambda item: item[0])
        return [doc for _, doc in matched]

    def _select_summary_docs_for_material(
        self,
        *,
        query: str,
        entry: MaterialCatalogEntry,
        summary_docs: Sequence[Document],
    ) -> list[Document]:
        max_chunks = max(
            1,
            int(self._config.material_search_max_selected_summary_chunks),
        )
        if not summary_docs:
            return []
        numbered = [
            f"[{idx}] {doc.page_content}"
            for idx, doc in enumerate(summary_docs, start=1)
        ]
        summary_chunks = "\n\n".join(numbered)
        system_prompt = get_required_prompt_env(
            "PROMPT_MATERIAL_SEARCH_SELECTOR_SYSTEM"
        )
        user_prompt = render_prompt_template(
            "PROMPT_MATERIAL_SEARCH_SELECTOR_USER_TEMPLATE",
            query=query,
            summary_chunks=summary_chunks,
            max_chunks=max_chunks,
            material_name=entry.canonical_name,
        )

        provider = (
            self._config.material_search_selector_llm_provider or ""
        ).strip().lower()
        if provider not in {"gemini", "llama"}:
            provider = "llama"
        model = (
            self._config.material_search_selector_gemini_model
            if provider == "gemini"
            else self._config.material_search_selector_llama_model
        )
        max_retries = max(0, self._config.material_search_selector_max_retries)
        for attempt in range(max_retries + 1):
            try:
                response = generate_text(
                    provider=provider,
                    api_key=self._config.gemini_api_key,
                    prompt=user_prompt,
                    model=model,
                    system_prompt=system_prompt,
                    llama_model_path=self._config.material_search_selector_llama_model_path,
                    llama_ctx_size=self._config.material_search_selector_llama_ctx_size,
                    temperature=self._config.material_search_selector_temperature,
                    max_output_tokens=self._config.material_search_selector_max_output_tokens,
                    thinking_level=self._config.material_search_selector_thinking_level,
                    llama_threads=self._config.llama_threads,
                    llama_gpu_layers=self._config.llama_gpu_layers,
                    response_mime_type="application/json",
                )
                selected_indices = self._parse_material_selector_indices(
                    response,
                    max_index=len(summary_docs),
                    max_count=max_chunks,
                )
                if selected_indices:
                    return [summary_docs[idx - 1] for idx in selected_indices]
            except Exception:
                logger.exception(
                    "Material summary selector failed (attempt %s/%s, material=%s)",
                    attempt + 1,
                    max_retries + 1,
                    entry.material_id,
                )
        return list(summary_docs[:max_chunks])

    def _parse_material_selector_indices(
        self,
        text: str,
        *,
        max_index: int,
        max_count: int,
    ) -> list[int]:
        cleaned = _strip_code_fence(text or "").strip()
        if not cleaned:
            return []

        parsed: object | None = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = _load_json_payload(cleaned)
            parsed = payload if payload is not None else None

        values: list[object] = []
        if isinstance(parsed, dict):
            for key in _MATERIAL_SELECTOR_RESPONSE_KEYS:
                candidate = parsed.get(key)
                if isinstance(candidate, list):
                    values = candidate
                    break
        elif isinstance(parsed, list):
            values = parsed

        selected: list[int] = []
        seen: set[int] = set()
        for item in values:
            value: int | None = None
            if isinstance(item, int):
                value = item
            elif isinstance(item, float) and item.is_integer():
                value = int(item)
            elif isinstance(item, str):
                stripped = item.strip()
                if stripped.isdigit():
                    value = int(stripped)
            if value is None:
                continue
            if value < 1 or value > max_index or value in seen:
                continue
            seen.add(value)
            selected.append(value)
            if len(selected) >= max_count:
                break
        return selected

    def _first_rec_docs_from_summary_docs(
        self,
        *,
        material_key: tuple[str, str],
        summary_docs: Sequence[Document],
    ) -> list[Document]:
        selected: list[Document] = []
        seen: set[tuple[object, ...]] = set()
        for summary_doc in summary_docs:
            metadata = summary_doc.metadata or {}
            parent_chunk_id = self._normalize_chunk_id(
                metadata.get("parent_chunk_id")
            )
            if parent_chunk_id is None:
                continue
            key = self._chunk_lookup_key(metadata, parent_chunk_id)
            if key is None:
                continue
            normalized = self._normalize_material_key(
                str(key[0] or ""),
                str(key[1] or ""),
            )
            if normalized != material_key:
                continue
            first_doc = self._first_rec_chunk_map().get(key)
            if first_doc is None:
                continue
            doc_key = self._doc_key(first_doc)
            if doc_key in seen:
                continue
            seen.add(doc_key)
            selected.append(first_doc)
        return selected

    def _select_material_fallback_docs(
        self,
        *,
        query: str,
        docs: Sequence[Document],
        fast_mode: bool = False,
    ) -> list[Document]:
        if not docs:
            return []
        top_k = max(
            1,
            int(self._config.material_search_summary_missing_first_rec_top_k),
        )
        if fast_mode or not self._config.rerank_enabled or len(docs) <= 1:
            return list(docs[:top_k])
        scored = self._reranker.score_documents(query=query, docs=list(docs))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _, _, doc in scored[:top_k]]

    def _chunk_map_for_dirs(self, base_dir: Path) -> dict[tuple[object, ...], Document]:
        chunk_dirs = []
        for name in _SECOND_REC_SOURCE_DIRS:
            candidate = base_dir / name
            if candidate.exists():
                chunk_dirs.append(candidate)
        if not chunk_dirs:
            return {}

        chunks = load_chunks_from_dirs(chunk_dirs)
        mapping: dict[tuple[object, ...], Document] = {}
        for chunk in chunks:
            metadata = chunk.metadata
            chunk_id = self._normalize_chunk_id(metadata.get("chunk_id"))
            if chunk_id is None:
                continue
            key = self._chunk_lookup_key(metadata, chunk_id)
            if key is None or key in mapping:
                continue
            mapping[key] = Document(page_content=chunk.text, metadata=metadata)
        return mapping

    def _first_or_summery_candidates(
        self,
        *,
        metadata: dict[str, object],
        first_parent_id: int,
    ) -> list[Document]:
        key = self._chunk_lookup_key(metadata, first_parent_id)
        if key is None:
            return []

        if self._config.summery_enabled:
            summery_docs = self._summery_chunk_map().get(key)
            if summery_docs:
                return list(summery_docs)

        parent_doc = self._first_rec_chunk_map().get(key)
        if parent_doc is None:
            return []
        return [parent_doc]

    @classmethod
    def _chunk_lookup_key(
        cls,
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

    def _append_sources(
        self,
        *,
        answer: str,
        docs: list[Document],
        source_selections: Sequence[_SourceSelection],
    ) -> str:
        if not answer or not docs or not source_selections:
            return answer

        refs = self._build_source_refs(
            docs=docs,
            source_selections=source_selections,
        )
        if not refs:
            return answer

        sources_text = "\n".join(f"- {ref}" for ref in refs)
        return (
            f"{answer}\n\n"
            "※回答は必ずしも正しいとは限りません。重要な情報は確認するようにしてください。\n"
            f"主な情報源:\n{sources_text}"
        )

    def _generate_no_rag_payload(
        self,
        *,
        query: str,
        question_author: str | None = None,
        use_additional_history: bool = False,
        include_capabilities_info: bool = True,
        history_override: Sequence[ChatHistoryEntry] | None = None,
        fast_mode: bool = False,
        extra_mode_instruction: str | None = None,
        history_scope: str | int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, list[_SourceSelection]]:
        max_json_retries = max(0, self._config.answer_json_max_retries)
        last_raw = ""
        for attempt in range(max_json_retries + 1):
            _raise_if_cancelled(cancel_event)
            if history_override is not None:
                history = list(history_override)
            else:
                history = self._history_for_prompt(
                    limit=(
                        self._config.prompt_history_additional_turns
                        if use_additional_history
                        else self._config.prompt_history_default_turns
                    ),
                    include_sources=False,
                    history_scope=history_scope,
                )
            raw = self._generate_no_rag(
                query=query,
                question_author=question_author,
                history=history,
                history_override=history_override,
                include_capabilities_info=include_capabilities_info,
                fast_mode=fast_mode,
                extra_mode_instruction=extra_mode_instruction,
                history_scope=history_scope,
                cancel_event=cancel_event,
            )
            last_raw = raw
            answer, _, is_json, has_answer = self._parse_answer_payload(
                raw,
                max_source_index=0,
            )
            _raise_if_cancelled(cancel_event)
            if is_json and has_answer:
                return answer, []
            if attempt < max_json_retries:
                logger.info(
                    "Invalid JSON from no-rag LLM. Retrying %s/%s",
                    attempt + 1,
                    max_json_retries,
                )
        return last_raw, []

    def _generate_answer_payload(
        self,
        *,
        query: str,
        question_author: str | None = None,
        docs: list[Document],
        use_additional_history: bool = False,
        include_capabilities_info: bool = True,
        idea_generation: bool = False,
        history_override: Sequence[ChatHistoryEntry] | None = None,
        fast_mode: bool = False,
        extra_mode_instruction: str | None = None,
        history_scope: str | int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, list[_SourceSelection], list[Document]]:
        max_json_retries = max(0, self._config.answer_json_max_retries)
        last_raw = ""
        for attempt in range(max_json_retries + 1):
            _raise_if_cancelled(cancel_event)
            if history_override is not None:
                history = list(history_override)
            else:
                history = self._history_for_prompt(
                    limit=(
                        self._config.prompt_history_additional_turns
                        if use_additional_history
                        else self._config.prompt_history_default_turns
                    ),
                    include_sources=False,
                    history_scope=history_scope,
                )
            raw = self.generate(
                query=query,
                question_author=question_author,
                docs=docs,
                history=history,
                history_override=history_override,
                include_capabilities_info=include_capabilities_info,
                idea_generation=idea_generation,
                fast_mode=fast_mode,
                extra_mode_instruction=extra_mode_instruction,
                history_scope=history_scope,
                cancel_event=cancel_event,
            )
            last_raw = raw
            answer, source_selections, is_json, has_answer = (
                self._parse_answer_payload(
                    raw,
                    max_source_index=len(docs),
                )
            )
            if is_json and has_answer:
                return answer, source_selections, docs
            if attempt < max_json_retries:
                logger.info(
                    "Invalid JSON from answer LLM. Retrying %s/%s",
                    attempt + 1,
                    max_json_retries,
                )
        return last_raw, [], docs

    def _order_source_ids(
        self,
        *,
        source_ids: list[int],
        query: str,
        docs: list[Document],
        recency_mode: str = "off",
        disable_rerank: bool = False,
    ) -> list[int]:
        max_count = max(0, self._config.source_max_count)
        if not source_ids or not docs or max_count == 0:
            return []

        unique_ids: list[int] = []
        seen: set[int] = set()
        for idx in source_ids:
            if idx in seen:
                continue
            if 1 <= idx <= len(docs):
                seen.add(idx)
                unique_ids.append(idx)

        if not unique_ids:
            return []
        if disable_rerank or not self._config.rerank_enabled:
            return unique_ids[:max_count]

        score_by_doc_key = self._rerank_scores_for_query(
            query=query,
            recency_mode=recency_mode,
        )
        missing_docs: list[Document] = []
        for idx in unique_ids:
            doc = docs[idx - 1]
            key = self._doc_key(doc)
            if key not in score_by_doc_key:
                missing_docs.append(doc)
        if missing_docs:
            scored_missing_base = self._reranker.score_documents(
                query=query, docs=missing_docs
            )
            scored_missing = self._apply_recency_scores(
                scored=scored_missing_base,
                recency_mode=recency_mode,
            )
            self._store_rerank_scores(
                query=query,
                recency_mode=recency_mode,
                scored=scored_missing,
            )
            for score, _, doc in scored_missing:
                key = self._doc_key(doc)
                previous = score_by_doc_key.get(key)
                if previous is None or score > previous:
                    score_by_doc_key[key] = score

        ordered = sorted(
            unique_ids,
            key=lambda idx: (
                -score_by_doc_key.get(self._doc_key(docs[idx - 1]), 0.0),
                idx - 1,
            ),
        )
        return ordered[:max_count]

    def _order_source_selections(
        self,
        *,
        source_selections: Sequence[_SourceSelection],
        query: str,
        docs: list[Document],
        recency_mode: str = "off",
        disable_rerank: bool = False,
    ) -> list[_SourceSelection]:
        if not source_selections or not docs:
            return []

        earliest_by_doc: dict[int, _SourceSelection] = {}
        for selection in source_selections:
            idx = selection.doc_index
            if idx < 1 or idx > len(docs):
                continue
            previous = earliest_by_doc.get(idx)
            if previous is None or self._is_earlier_sub_index(
                selection.sub_index,
                previous.sub_index,
            ):
                earliest_by_doc[idx] = selection

        if not earliest_by_doc:
            return []

        ordered_doc_ids = self._order_source_ids(
            source_ids=list(earliest_by_doc.keys()),
            query=query,
            docs=docs,
            recency_mode=recency_mode,
            disable_rerank=disable_rerank,
        )
        return [earliest_by_doc[idx] for idx in ordered_doc_ids]

    @staticmethod
    def _is_earlier_sub_index(
        candidate_sub_index: int | None,
        current_sub_index: int | None,
    ) -> bool:
        def _priority(value: int | None) -> int:
            if value is None:
                return 1
            return value if value >= 1 else 10**9

        return _priority(candidate_sub_index) < _priority(current_sub_index)

    def _build_source_refs(
        self,
        docs: list[Document],
        *,
        source_selections: Sequence[_SourceSelection] | None = None,
    ) -> list[str]:
        if source_selections is None:
            source_selections = [
                _SourceSelection(doc_index=idx)
                for idx in range(1, len(docs) + 1)
            ]

        refs: list[str] = []
        seen: set[str] = set()
        for selection in source_selections:
            idx = selection.doc_index
            if idx < 1 or idx > len(docs):
                continue
            doc = docs[idx - 1]
            ref = self._source_ref_for_selection(
                doc=doc,
                sub_index=selection.sub_index,
            )
            if not ref:
                continue
            if ref in seen:
                continue
            refs.append(ref)
            seen.add(ref)
        return refs

    def _source_ref_for_selection(
        self,
        *,
        doc: Document,
        sub_index: int | None,
    ) -> str | None:
        metadata = doc.metadata or {}
        source_type = str(metadata.get("source_type") or "").strip().lower()
        if source_type in {"messages", "discord_message"}:
            ref = self._discord_url_for_selection(doc=doc, sub_index=sub_index)
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
        ref = _drive_url_from_metadata(metadata)
        if ref:
            return ref
        return _vc_source_label_from_metadata(metadata)

    def _discord_url_for_selection(
        self,
        *,
        doc: Document,
        sub_index: int | None,
    ) -> str | None:
        metadata = doc.metadata or {}
        guild_id = str(metadata.get("guild_id") or "").strip()
        channel_id = str(metadata.get("channel_id") or "").strip()
        if not guild_id or not channel_id:
            return _discord_url_from_metadata(metadata)

        message_id = self._resolve_discord_message_id(
            doc=doc,
            sub_index=sub_index,
        )
        if not message_id:
            return _discord_url_from_metadata(metadata)
        return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"

    def _parse_answer_payload(
        self,
        text: str,
        *,
        max_source_index: int,
    ) -> tuple[str, list[_SourceSelection], bool, bool]:
        raw = (text or "").strip()
        if not raw:
            return "", [], False, False

        payload = _load_json_payload(raw)
        if not isinstance(payload, dict):
            return raw, [], False, False

        answer = str(payload.get("answer") or "").strip()
        sources_raw = payload.get("sources")
        source_selections: list[_SourceSelection] = []
        seen: set[tuple[int, int | None]] = set()
        if isinstance(sources_raw, list):
            for item in sources_raw:
                selection = self._parse_source_selection_item(
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

    def _resolve_discord_message_id(
        self,
        *,
        doc: Document,
        sub_index: int | None,
    ) -> str | None:
        metadata = doc.metadata or {}
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

        chunk_lines = self._discord_chunk_lines(doc.page_content)
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

    @lru_cache(maxsize=512)
    def _discord_raw_chunk_lines(
        self,
        *,
        guild_id: str,
        channel_id: str,
    ) -> tuple[_DiscordChunkLine, ...]:
        path = self._config.raw_data_dir / "messages" / guild_id / f"{channel_id}.jsonl"
        if not path.exists():
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
            logger.exception("Failed to load Discord raw messages: %s", path)
            return tuple()
        return tuple(lines)

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
        return parsed.astimezone(_JST).strftime("%Y/%m/%d")

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
            chunk_value = RagPipeline._normalize_discord_line(chunk_lines[offset])
            raw_value = RagPipeline._normalize_discord_line(
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

    def _history_for_prompt(
        self,
        *,
        limit: int,
        include_sources: bool = True,
        history_scope: str | int | None = None,
    ) -> list[ChatHistoryEntry] | None:
        if not self._config.chat_history_enabled:
            return None
        history_bucket = self._history_bucket(history_scope=history_scope)
        if limit <= 0 or history_bucket.maxlen == 0:
            return []
        history = list(history_bucket)
        selected = history if len(history) <= limit else history[-limit:]
        if include_sources:
            return selected
        return [
            (user_text, assistant_text, [])
            for user_text, assistant_text, _ in selected
        ]

    def _record_history(
        self,
        *,
        query: str,
        answer: str,
        sources: Sequence[str],
        history_scope: str | int | None = None,
    ) -> None:
        history_bucket = self._history_bucket(history_scope=history_scope)
        if history_bucket.maxlen == 0:
            return
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
        bucket = deque(maxlen=max(0, self._config.chat_history_max_turns))
        self._chat_histories[key] = bucket
        return bucket

    @staticmethod
    def _normalize_history_scope(history_scope: str | int | None) -> str:
        if history_scope is None:
            return "__default__"
        normalized = str(history_scope).strip()
        if not normalized:
            return "__default__"
        return normalized

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


def _load_json_payload(text: str) -> dict[str, object] | None:
    cleaned = _strip_code_fence(text)
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) < 2:
        return text
    if not lines[-1].strip().startswith("```"):
        return text
    return "\n".join(lines[1:-1]).strip()


def _mask_discord_mentions(text: str) -> str:
    if not text:
        return ""
    masked = _USER_MENTION_RE.sub(_MASKED_MENTION, text)
    return _ROLE_MENTION_RE.sub(_MASKED_MENTION, masked)
