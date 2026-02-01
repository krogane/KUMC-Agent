from __future__ import annotations

import json
import logging
import threading
import unicodedata
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi
from sudachipy import dictionary, tokenizer as sudachi_tokenizer

from config import AppConfig, EmbeddingFactory
from indexing.chunks import load_chunks_from_dirs
from indexing.llm_client import generate_text
from pipeline.function_calling import decide_tools
from pipeline.llm_clients import (
    generate_with_gemini,
    generate_with_gemini_config,
    generate_with_llama,
    generate_with_llama_config,
)
from pipeline.prompts import (
    QUERY_TRANSFORM_SYSTEM_PROMPT,
    ChatHistoryEntry,
    build_gemini_prompt,
    build_llama_messages,
    build_query_transform_prompt,
    doc_to_context,
)
from pipeline.reranker import CrossEncoderReranker
from pipeline.vectorstore import load_faiss_index

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_LIMIT = 3
_ADDITIONAL_HISTORY_LIMIT = 10


class GenerationCancelled(RuntimeError):
    pass


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
        self._chat_history = deque(
            maxlen=max(0, self._config.chat_history_max_turns)
        )

    def retrieve(
        self,
        query: str,
        *,
        re_search: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        query = query.strip()
        if not query:
            return []

        dense_docs = self._dense_search(query, cancel_event=cancel_event)
        _raise_if_cancelled(cancel_event)
        if re_search:
            sparse_query = self._maybe_transform_query(
                query, cancel_event=cancel_event
            )
            sparse_docs = self._sparse_search(
                original_query=query,
                transformed_query=sparse_query,
                cancel_event=cancel_event,
            )
        else:
            sparse_docs = self._sparse_search_initial(
                query, cancel_event=cancel_event
            )
        merged = self._merge_docs([dense_docs, sparse_docs])
        if merged:
            reranked = self._rerank_docs(query=query, docs=merged)
            capped = self._apply_parent_doc_cap(reranked)
            pooled = self._limit_rerank_pool(capped)
            selected = self._select_with_mmr(query=query, docs=pooled)
            return self._append_parent_docs(selected)

        _raise_if_cancelled(cancel_event)
        q = f"query: {query}"
        docs = self._vectorstore().similarity_search(q, k=self._config.top_k)
        return self._append_parent_docs(docs)

    def generate(
        self,
        *,
        query: str,
        docs: list[Document],
        retry_history: Sequence[tuple[str, str]] | None = None,
        history: Sequence[ChatHistoryEntry] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        _raise_if_cancelled(cancel_event)
        provider = (self._config.llm_provider or "").lower()
        if history is None:
            history = self._history_for_prompt(limit=_DEFAULT_HISTORY_LIMIT)
        if provider == "gemini":
            prompt = build_gemini_prompt(
                query=query,
                docs=docs,
                history=history,
                retry_history=retry_history,
            )
            logger.info("Answer LLM prompt (gemini): %s", prompt)
            text = generate_with_gemini(
                api_key=self._llm_api_key,
                prompt=prompt,
                config=self._config,
            )
            logger.info("Answer LLM output (gemini): %s", text)
        elif provider == "llama":
            messages = build_llama_messages(
                query=query,
                docs=docs,
                config=self._config,
                history=history,
                retry_history=retry_history,
            )
            logger.info("Answer LLM prompt (llama): %s", messages)
            text = generate_with_llama(messages=messages, config=self._config)
            logger.info("Answer LLM output (llama): %s", text)
        else:
            raise ValueError(
                f"Unsupported llm_provider: {self._config.llm_provider}. "
                "Use 'gemini' or 'llama'."
            )

        if not text:
            return "回答生成中に不具合が発生しました、もう一度お試しください。"

        _raise_if_cancelled(cancel_event)
        return text

    def _generate_no_rag(
        self,
        *,
        query: str,
        retry_history: Sequence[tuple[str, str]] | None = None,
        history: Sequence[ChatHistoryEntry] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        _raise_if_cancelled(cancel_event)
        provider = (self._config.no_rag_llm_provider or "").lower()
        if history is None:
            history = self._history_for_prompt(limit=_DEFAULT_HISTORY_LIMIT)
        docs: list[Document] = []
        if provider == "gemini":
            prompt = build_gemini_prompt(
                query=query,
                docs=docs,
                history=history,
                retry_history=retry_history,
            )
            logger.info("No-RAG LLM prompt (gemini): %s", prompt)
            text = generate_with_gemini_config(
                api_key=self._llm_api_key,
                prompt=prompt,
                system_rules=self._config.system_rules,
                model=self._config.no_rag_genai_model,
                temperature=self._config.no_rag_temperature,
                max_output_tokens=self._config.no_rag_max_output_tokens,
                thinking_level=self._config.no_rag_thinking_level,
            )
            logger.info("No-RAG LLM output (gemini): %s", text)
        elif provider == "llama":
            messages = build_llama_messages(
                query=query,
                docs=docs,
                config=self._config,
                history=history,
                retry_history=retry_history,
            )
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
            logger.info("No-RAG LLM output (llama): %s", text)
        else:
            raise ValueError(
                "Unsupported no_rag_llm_provider: "
                f"{self._config.no_rag_llm_provider}. Use 'gemini' or 'llama'."
            )

        if not text:
            return "回答生成中に不具合が発生しました、もう一度お試しください。"

        _raise_if_cancelled(cancel_event)
        return text

    def answer(
        self, query: str, *, cancel_event: threading.Event | None = None
    ) -> str:
        docs = self.retrieve(query, cancel_event=cancel_event)
        answer, final, source_ids, used_docs, history_sources = (
            self._answer_with_docs(
                query=query, docs=docs, cancel_event=cancel_event
            )
        )
        self._record_history(
            query=query, answer=answer, sources=history_sources
        )
        return final

    def answer_with_routing(
        self,
        query: str,
        *,
        on_research_start: Callable[[], None] | None = None,
        on_memory_start: Callable[[], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        _raise_if_cancelled(cancel_event)
        if not self._config.function_call_enabled:
            use_rag = True
        else:
            use_rag = decide_tools(query=query, config=self._config)
        logger.info(
            "Function-call routing decision: use_rag=%s",
            use_rag,
        )
        docs: list[Document] = []
        if use_rag:
            docs = self.retrieve(query, cancel_event=cancel_event)
        if use_rag:
            answer, final, source_ids, used_docs, history_sources = (
                self._answer_with_docs(
                query=query,
                docs=docs,
                on_research_start=on_research_start,
                on_memory_start=on_memory_start,
                cancel_event=cancel_event,
            )
            )
            self._record_history(
                query=query, answer=answer, sources=history_sources
            )
            return final

        answer = self.answer_no_rag(
            query, on_memory_start=on_memory_start, cancel_event=cancel_event
        )
        return answer

    def answer_no_rag(
        self,
        query: str,
        *,
        on_memory_start: Callable[[], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        answer, _ = self._generate_no_rag_payload(
            query=query,
            on_memory_start=on_memory_start,
            cancel_event=cancel_event,
        )
        self._record_history(query=query, answer=answer, sources=[])
        return answer

    def _answer_with_docs(
        self,
        *,
        query: str,
        docs: list[Document],
        on_research_start: Callable[[], None] | None = None,
        on_memory_start: Callable[[], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, str, list[int], list[Document], list[str]]:
        answer, source_ids, used_docs, _ = self._generate_answer_payload(
            query=query,
            docs=docs,
            on_research_start=on_research_start,
            on_memory_start=on_memory_start,
            cancel_event=cancel_event,
        )
        history_sources = self._sources_for_history(
            docs=used_docs, source_ids=source_ids
        )
        source_ids = self._order_source_ids(
            source_ids=source_ids,
            query=query,
            docs=used_docs,
        )
        final = self._append_sources(
            answer=answer,
            docs=used_docs,
            source_ids=source_ids,
        )
        return answer, final, source_ids, used_docs, history_sources

    def answer_with_contexts(self, query: str) -> tuple[str, list[str]]:
        docs = self.retrieve(query)
        answer, source_ids, used_docs, research_docs = self._generate_answer_payload(
            query=query, docs=docs
        )
        history_sources = self._sources_for_history(
            docs=used_docs, source_ids=source_ids
        )
        self._record_history(
            query=query, answer=answer, sources=history_sources
        )
        context_docs = research_docs if research_docs else used_docs
        return answer, [doc_to_context(d) for d in context_docs]

    def refresh_index(self) -> None:
        self._vectorstore.cache_clear()
        self._sparse_index.cache_clear()
        self._second_rec_chunk_map.cache_clear()
        self._first_rec_chunk_map.cache_clear()
        self._summery_chunk_map.cache_clear()

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
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        k = max(0, self._config.dense_search_top_k)
        if k <= 0:
            return []

        q = f"query: {query}"
        stages = self._dense_stages()
        if not stages:
            return self._vectorstore().similarity_search(q, k=k)
        return self._dense_search_filtered(
            q, k, stages, cancel_event=cancel_event
        )

    def _dense_stages(self) -> set[str]:
        stages: set[str] = set()
        if self._config.prop_enabled and self._config.second_rec_enabled:
            stages.add("proposition")
        else:
            stages.add(
                "second_recursive"
                if self._config.second_rec_enabled
                else "first_recursive"
            )
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

        if original_k > 0:
            original_docs = self._sparse_search_once(
                original_query, original_k, cancel_event=cancel_event
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
        if k <= 0:
            return []
        return self._sparse_search_once(query, k, cancel_event=cancel_event)

    def _sparse_search_once(
        self, query: str, k: int, *, cancel_event: threading.Event | None = None
    ) -> list[Document]:
        _raise_if_cancelled(cancel_event)
        if k <= 0:
            return []
        tokens = self._sudachi_tokens(query)
        if not tokens:
            return []
        docs, bm25 = self._sparse_index()
        if not docs:
            return []
        scores = bm25.get_scores(tokens)
        if scores is None:
            return []

        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        results: list[Document] = []
        for idx in ranked:
            _raise_if_cancelled(cancel_event)
            if scores[idx] <= 0:
                break
            results.append(docs[idx])
            if len(results) >= k:
                break
        return results

    def _maybe_transform_query(
        self, query: str, *, cancel_event: threading.Event | None = None
    ) -> str:
        _raise_if_cancelled(cancel_event)
        if not self._config.query_transform_enabled:
            return query
        transformed = self._transform_query_with_llm(
            query, cancel_event=cancel_event
        )
        if transformed:
            logger.info("Query keywords: %s", transformed)
            return transformed
        logger.warning("Query transform failed; fallback to original query.")
        return query

    def _transform_query_with_llm(
        self, query: str, *, cancel_event: threading.Event | None = None
    ) -> str:
        _raise_if_cancelled(cancel_event)
        provider = (self._config.query_transform_provider or "").lower()
        if provider not in {"gemini", "llama"}:
            logger.warning(
                "Unsupported query_transform_provider: %s",
                self._config.query_transform_provider,
            )
            return ""

        prompt = build_query_transform_prompt(query=query)
        model = self._select_query_transform_model(provider)
        max_retries = max(1, self._config.query_transform_max_retries)

        for attempt in range(1, max_retries + 1):
            _raise_if_cancelled(cancel_event)
            try:
                response = generate_text(
                    provider=provider,
                    api_key=self._llm_api_key,
                    prompt=prompt,
                    model=model,
                    system_prompt=QUERY_TRANSFORM_SYSTEM_PROMPT,
                    llama_model_path=self._config.query_transform_llama_model_path,
                    llama_ctx_size=self._config.query_transform_llama_ctx_size,
                    temperature=self._config.query_transform_temperature,
                    max_output_tokens=self._config.query_transform_max_output_tokens,
                    thinking_level=self._config.thinking_level,
                    llama_threads=self._config.llama_threads,
                    llama_gpu_layers=self._config.llama_gpu_layers,
                    response_mime_type="text/plain",
                )
                normalized = _normalize_query_keywords(response)
                if normalized:
                    logger.info("Query transformed: %s -> %s", query, normalized)
                    return normalized
                raise ValueError("Empty keyword output.")
            except Exception as exc:
                logger.warning(
                    "Query transform failed (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    exc,
                )
        return ""

    def _select_query_transform_model(self, provider: str) -> str:
        if provider == "llama":
            return self._config.query_transform_llama_model
        return self._config.query_transform_gemini_model

    @lru_cache(maxsize=1)
    def _sparse_index(self) -> tuple[list[Document], BM25Okapi]:
        docs = self._load_sparse_docs()
        if not docs:
            return [], BM25Okapi([[]])
        tokenized = [self._sudachi_tokens(doc.page_content) for doc in docs]
        bm25 = BM25Okapi(
            tokenized,
            k1=self._config.sparse_bm25_k1,
            b=self._config.sparse_bm25_b,
        )
        return docs, bm25

    def _load_sparse_docs(self) -> list[Document]:
        chunk_dirs = self._sparse_chunk_dirs()
        if not chunk_dirs:
            return []

        chunks = load_chunks_from_dirs(chunk_dirs)
        return [
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in chunks
        ]

    def _sparse_chunk_dirs(self) -> list[Path]:
        if self._config.prop_enabled and self._config.second_rec_enabled:
            base_dir = self._config.prop_chunk_dir
        else:
            base_dir = (
                self._config.second_rec_chunk_dir
                if self._config.second_rec_enabled
                else self._config.first_rec_chunk_dir
            )
        dirs = []
        for name in ("docs", "sheets"):
            candidate = base_dir / name
            if candidate.exists():
                dirs.append(candidate)
        if self._config.raptor_enabled:
            raptor_dir = self._config.raptor_chunk_dir
            if raptor_dir.exists():
                dirs.append(raptor_dir)
        return dirs

    def _sudachi_tokens(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        tokenizer = self._sudachi_tokenizer()
        mode = self._sudachi_mode()
        tokens: list[str] = []
        for morph in tokenizer.tokenize(text, mode):
            token = (
                morph.normalized_form()
                if self._config.sparse_use_normalized_form
                else morph.surface()
            )
            token = token.strip()
            if not token:
                continue
            if self._config.sparse_remove_symbols and _is_symbol_only(token):
                continue
            tokens.append(token)
        return tokens

    @lru_cache(maxsize=1)
    def _sudachi_tokenizer(self) -> sudachi_tokenizer.Tokenizer:
        return dictionary.Dictionary().create()

    def _sudachi_mode(self) -> sudachi_tokenizer.Tokenizer.SplitMode:
        value = (self._config.sudachi_mode or "B").upper()
        if value == "A":
            return sudachi_tokenizer.Tokenizer.SplitMode.A
        if value == "C":
            return sudachi_tokenizer.Tokenizer.SplitMode.C
        return sudachi_tokenizer.Tokenizer.SplitMode.B

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

    def _rerank_docs(self, *, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        scored = self._reranker.score_documents(query=query, docs=docs)
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _, _, doc in scored]

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

        embeddings = self._mmr_embeddings(query=query, docs=docs)
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
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            query_vec = self.embeddings().embed_query(query)
            doc_texts = [doc.page_content for doc in docs]
            doc_vecs = self.embeddings().embed_documents(doc_texts)
        except Exception:
            logger.exception("MMR embedding failed; fallback to reranked order.")
            return None

        if not query_vec or not doc_vecs or len(doc_vecs) != len(docs):
            return None

        try:
            doc_matrix = np.array(doc_vecs, dtype=float)
            query_array = np.array(query_vec, dtype=float)
        except Exception:
            return None

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

        seen_doc_keys = {self._doc_key(doc) for doc in docs}
        added: list[Document] = []

        for doc in docs:
            metadata = doc.metadata or {}
            stage = metadata.get("chunk_stage")
            if stage == "proposition":
                second_parent_id = self._normalize_chunk_id(
                    metadata.get("parent_chunk_id")
                )
                if second_parent_id is None:
                    continue
                key = self._chunk_lookup_key(metadata, second_parent_id)
                if key is None:
                    continue
                second_doc = self._second_rec_chunk_map().get(key)
                if second_doc is None:
                    continue
                first_parent_id = self._normalize_chunk_id(
                    (second_doc.metadata or {}).get("parent_chunk_id")
                )
                if first_parent_id is None:
                    continue
                self._append_first_or_summery(
                    metadata=metadata,
                    first_parent_id=first_parent_id,
                    added=added,
                    seen_doc_keys=seen_doc_keys,
                )
                continue

            if stage == "second_recursive":
                first_parent_id = self._normalize_chunk_id(
                    metadata.get("parent_chunk_id")
                )
                if first_parent_id is None:
                    continue
                self._append_first_or_summery(
                    metadata=metadata,
                    first_parent_id=first_parent_id,
                    added=added,
                    seen_doc_keys=seen_doc_keys,
                )
                continue

        if not added:
            return docs
        added.reverse()
        return docs + added

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
        for name in ("docs", "sheets", "messages"):
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

    def _chunk_map_for_dirs(self, base_dir: Path) -> dict[tuple[object, ...], Document]:
        chunk_dirs = []
        for name in ("docs", "sheets", "messages"):
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

    def _append_first_or_summery(
        self,
        *,
        metadata: dict[str, object],
        first_parent_id: int,
        added: list[Document],
        seen_doc_keys: set[tuple[object, ...]],
    ) -> None:
        key = self._chunk_lookup_key(metadata, first_parent_id)
        if key is None:
            return

        if self._config.summery_enabled:
            summery_docs = self._summery_chunk_map().get(key)
            if summery_docs:
                for doc in summery_docs:
                    doc_key = self._doc_key(doc)
                    if doc_key in seen_doc_keys:
                        continue
                    seen_doc_keys.add(doc_key)
                    added.append(doc)
                return

        parent_doc = self._first_rec_chunk_map().get(key)
        if parent_doc is None:
            return
        parent_doc_key = self._doc_key(parent_doc)
        if parent_doc_key in seen_doc_keys:
            return
        seen_doc_keys.add(parent_doc_key)
        added.append(parent_doc)

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

    def _append_sources(
        self,
        *,
        answer: str,
        docs: list[Document],
        source_ids: list[int],
    ) -> str:
        if not answer or not docs or not source_ids:
            return answer

        selected_docs: list[Document] = []
        for idx in source_ids:
            if 1 <= idx <= len(docs):
                selected_docs.append(docs[idx - 1])
        urls = self._build_source_urls(selected_docs)
        if not urls:
            return answer

        sources_text = "\n".join(f"- {url}" for url in urls)
        return f"{answer}\n\nソース:\n{sources_text}"

    def _generate_no_rag_payload(
        self,
        *,
        query: str,
        on_memory_start: Callable[[], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, list[int]]:
        max_json_retries = max(0, self._config.answer_json_max_retries)
        attempt = 0
        last_raw = ""
        answer = ""
        use_additional_history = False
        additional_memory_used = False
        memory_notified = False
        while True:
            _raise_if_cancelled(cancel_event)
            history = self._history_for_prompt(
                limit=(
                    _ADDITIONAL_HISTORY_LIMIT
                    if use_additional_history
                    else _DEFAULT_HISTORY_LIMIT
                )
            )
            raw = self._generate_no_rag(
                query=query,
                history=history,
                cancel_event=cancel_event,
            )
            last_raw = raw
            (
                answer,
                _,
                _,
                needs_additional_memory,
                is_json,
                has_answer,
            ) = self._parse_answer_payload(raw, max_source_index=0)
            _raise_if_cancelled(cancel_event)
            if is_json and has_answer:
                if needs_additional_memory and not additional_memory_used:
                    additional_memory_used = True
                    use_additional_history = True
                    if on_memory_start is not None and not memory_notified:
                        memory_notified = True
                        try:
                            on_memory_start()
                        except Exception:
                            logger.exception(
                                "Failed to send memory start notification"
                            )
                    continue
                return answer, []
            if attempt >= max_json_retries:
                break
            attempt += 1
            logger.info(
                "Invalid JSON from no-rag LLM. Retrying %s/%s",
                attempt,
                max_json_retries,
            )

        if answer:
            return answer, []
        return last_raw, []

    def _generate_answer_payload(
        self,
        *,
        query: str,
        docs: list[Document],
        on_research_start: Callable[[], None] | None = None,
        on_memory_start: Callable[[], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, list[int], list[Document], list[Document]]:
        max_json_retries = max(0, self._config.answer_json_max_retries)
        max_research_retries = max(
            0, self._config.answer_research_max_retries
        )
        research_attempt = 0
        last_raw = ""
        retry_history: list[tuple[str, str]] = []
        seen_queries = {query.strip()}
        current_docs = docs
        research_docs: list[Document] = []
        research_notified = False
        memory_notified = False
        use_additional_history = False
        additional_memory_used = False

        while True:
            _raise_if_cancelled(cancel_event)
            attempt = 0
            payload_ok = False
            answer = ""
            source_ids: list[int] = []
            follow_up_queries: list[str] = []
            needs_additional_memory = False
            is_json = False
            has_answer = False

            while True:
                _raise_if_cancelled(cancel_event)
                history = self._history_for_prompt(
                    limit=(
                        _ADDITIONAL_HISTORY_LIMIT
                        if use_additional_history
                        else _DEFAULT_HISTORY_LIMIT
                    )
                )
                raw = self.generate(
                    query=query,
                    docs=current_docs,
                    retry_history=retry_history or None,
                    history=history,
                    cancel_event=cancel_event,
                )
                last_raw = raw
                (
                    answer,
                    source_ids,
                    follow_up_queries,
                    needs_additional_memory,
                    is_json,
                    has_answer,
                ) = self._parse_answer_payload(
                    raw,
                    max_source_index=len(current_docs),
                )
                _raise_if_cancelled(cancel_event)
                payload_ok = is_json and (has_answer or follow_up_queries)
                if payload_ok:
                    break
                if attempt >= max_json_retries:
                    break
                attempt += 1
                logger.info(
                    "Invalid JSON from answer LLM. Retrying %s/%s",
                    attempt,
                    max_json_retries,
                )

            if payload_ok and needs_additional_memory and not additional_memory_used:
                additional_memory_used = True
                use_additional_history = True
                if not follow_up_queries:
                    if on_memory_start is not None and not memory_notified:
                        memory_notified = True
                        try:
                            on_memory_start()
                        except Exception:
                            logger.exception(
                                "Failed to send memory start notification"
                            )
                    continue
                if research_attempt >= max_research_retries:
                    continue

            if payload_ok and follow_up_queries:
                if research_attempt < max_research_retries:
                    if on_research_start is not None and not research_notified:
                        research_notified = True
                        try:
                            on_research_start()
                        except Exception:
                            logger.exception(
                                "Failed to send research start notification"
                            )
                    logger.info(
                        "Follow-up queries requested by answer LLM: %s",
                        follow_up_queries,
                    )
                    previous_answer = answer or "（前回の回答は空でした）"
                    retry_history.append((query, previous_answer))
                    current_docs, new_research_docs = (
                        self._extend_docs_with_queries(
                            base_docs=current_docs,
                            queries=follow_up_queries,
                            seen_queries=seen_queries,
                            cancel_event=cancel_event,
                        )
                    )
                    if new_research_docs:
                        research_docs = self._merge_docs(
                            [research_docs, new_research_docs]
                        )
                    research_attempt += 1
                    continue

            if is_json and has_answer:
                return answer, source_ids, current_docs, research_docs
            if answer:
                return answer, [], current_docs, research_docs
            return last_raw, [], current_docs, research_docs

    def _order_source_ids(
        self,
        *,
        source_ids: list[int],
        query: str,
        docs: list[Document],
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

        scored = self._reranker.score_documents(query=query, docs=docs)
        score_by_index = {idx + 1: score for score, idx, _ in scored}

        ordered = sorted(
            unique_ids,
            key=lambda idx: (-score_by_index.get(idx, 0.0), idx - 1),
        )
        return ordered[:max_count]

    @staticmethod
    def _build_source_urls(docs: list[Document]) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            url = _discord_url_from_metadata(doc.metadata)
            if not url:
                url = _drive_url_from_metadata(doc.metadata)
            if not url or url in seen:
                continue
            urls.append(url)
            seen.add(url)
        return urls

    def _parse_answer_payload(
        self,
        text: str,
        *,
        max_source_index: int,
    ) -> tuple[str, list[int], list[str], bool, bool, bool]:
        raw = (text or "").strip()
        if not raw:
            return "", [], [], False, False, False

        payload = _load_json_payload(raw)
        if not isinstance(payload, dict):
            return raw, [], [], False, False, False

        answer = str(payload.get("answer") or "").strip()
        sources_raw = payload.get("sources")
        source_ids: list[int] = []
        if isinstance(sources_raw, list):
            for item in sources_raw:
                value: int | None = None
                if isinstance(item, int):
                    value = item
                elif isinstance(item, float) and item.is_integer():
                    value = int(item)
                elif isinstance(item, str):
                    item_value = item.strip()
                    if item_value.isdigit():
                        value = int(item_value)
                if value is None:
                    continue
                if value < 1 or value > max_source_index:
                    continue
                if value in source_ids:
                    continue
                source_ids.append(value)

        follow_up_queries_raw = payload.get("follow_up_queries")
        follow_up_queries: list[str] = []
        if isinstance(follow_up_queries_raw, str):
            candidate = follow_up_queries_raw.strip()
            if candidate:
                follow_up_queries.append(candidate)
        elif isinstance(follow_up_queries_raw, list):
            for item in follow_up_queries_raw:
                if not isinstance(item, str):
                    continue
                candidate = item.strip()
                if not candidate:
                    continue
                follow_up_queries.append(candidate)

        if follow_up_queries:
            deduped: list[str] = []
            seen: set[str] = set()
            for item in follow_up_queries:
                if item in seen:
                    continue
                seen.add(item)
                deduped.append(item)
            follow_up_queries = deduped

        needs_additional_memory = _coerce_bool(
            payload.get("needs_additional_memory")
        )
        has_answer = bool(answer)
        return (
            answer,
            source_ids,
            follow_up_queries,
            needs_additional_memory,
            True,
            has_answer,
        )

    def _extend_docs_with_queries(
        self,
        *,
        base_docs: list[Document],
        queries: list[str],
        seen_queries: set[str],
        cancel_event: threading.Event | None = None,
    ) -> tuple[list[Document], list[Document]]:
        groups: list[list[Document]] = [base_docs]
        new_groups: list[list[Document]] = []
        for query in queries:
            _raise_if_cancelled(cancel_event)
            cleaned = query.strip()
            if not cleaned or cleaned in seen_queries:
                continue
            seen_queries.add(cleaned)
            logger.info("Follow-up search query: %s", cleaned)
            fetched = self.retrieve(
                cleaned, re_search=True, cancel_event=cancel_event
            )
            if fetched:
                groups.append(fetched)
                new_groups.append(fetched)
        if len(groups) == 1:
            return base_docs, []
        merged = self._merge_docs(groups)
        new_docs = self._merge_docs(new_groups) if new_groups else []
        return merged, new_docs

    def _history_for_prompt(
        self,
        *,
        limit: int,
    ) -> list[ChatHistoryEntry] | None:
        if not self._config.chat_history_enabled:
            return None
        if limit <= 0 or self._chat_history.maxlen == 0:
            return []
        history = list(self._chat_history)
        if len(history) <= limit:
            return history
        return history[-limit:]

    def _sources_for_history(
        self,
        *,
        docs: list[Document],
        source_ids: list[int],
    ) -> list[str]:
        if not docs or not source_ids:
            return []
        contexts: list[str] = []
        seen: set[int] = set()
        for idx in source_ids:
            if idx in seen:
                continue
            if 1 <= idx <= len(docs):
                contexts.append(f"[{idx}]\n{doc_to_context(docs[idx - 1])}")
                seen.add(idx)
        return contexts

    def _record_history(
        self,
        *,
        query: str,
        answer: str,
        sources: Sequence[str],
    ) -> None:
        if self._chat_history.maxlen == 0:
            return
        user_text = (query or "").strip()
        assistant_text = (answer or "").strip()
        if not user_text or not assistant_text:
            return
        self._chat_history.append((user_text, assistant_text, list(sources)))


def _is_symbol_only(text: str) -> bool:
    has_visible = False
    for ch in text:
        if ch.isspace():
            continue
        has_visible = True
        category = unicodedata.category(ch)
        if category.startswith(("L", "N", "M")):
            return False
    return has_visible


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


def _discord_url_from_metadata(metadata: dict[str, object] | None) -> str | None:
    if not metadata:
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


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return False


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


def _normalize_query_keywords(text: str) -> str:
    cleaned = _strip_code_fence(text).strip()
    if not cleaned:
        return ""

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        parts = [str(item).strip() for item in parsed if str(item).strip()]
        return " ".join(parts)
    if isinstance(parsed, str):
        cleaned = parsed.strip()

    cleaned = cleaned.replace(",", " ").replace("\n", " ").replace("\t", " ")
    cleaned = cleaned.replace("・", " ").replace("•", " ")
    tokens: list[str] = []
    for token in cleaned.split():
        if token in {"-", "・", "•"}:
            continue
        tokens.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return " ".join(deduped)
