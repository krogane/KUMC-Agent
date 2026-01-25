from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import AppConfig, EmbeddingFactory
from indexing.chunks import load_chunks_from_dirs
from pipeline.llm_clients import generate_with_gemini, generate_with_llama
from pipeline.prompts import build_gemini_prompt, build_llama_messages
from pipeline.reranker import CrossEncoderReranker
from pipeline.vectorstore import load_faiss_index

logger = logging.getLogger(__name__)


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

    def retrieve(self, query: str) -> list[Document]:
        query = query.strip()
        if not query:
            return []

        semantic_docs = self._semantic_search(query)
        keyword_docs = self._keyword_search(query)
        merged = self._merge_docs([semantic_docs, keyword_docs])
        if merged:
            reranked = self._reranker.rerank(
                query=query,
                docs=merged,
                top_k=self._config.top_k,
            )
            return self._append_parent_recursive_docs(reranked)

        q = f"query: {query}"
        docs = self._vectorstore().similarity_search(q, k=self._config.top_k)
        return self._append_parent_recursive_docs(docs)

    def generate(self, *, query: str, docs: list[Document]) -> str:
        provider = (self._config.llm_provider or "").lower()
        if provider == "gemini":
            prompt = build_gemini_prompt(query=query, docs=docs)
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

        return text

    def answer(self, query: str) -> str:
        docs = self.retrieve(query)
        return self.generate(query=query, docs=docs)

    def answer_with_contexts(self, query: str) -> tuple[str, list[str]]:
        docs = self.retrieve(query)
        answer = self.generate(query=query, docs=docs)
        return answer, [d.page_content for d in docs]

    def refresh_index(self) -> None:
        self._vectorstore.cache_clear()
        self._keyword_corpus.cache_clear()
        self._recursive_chunk_map.cache_clear()

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

    def _semantic_search(self, query: str) -> list[Document]:
        k = max(0, self._config.raptor_search_top_k)
        if k <= 0:
            return []

        q = f"query: {query}"
        stages = self._semantic_stages()
        if not stages:
            return self._vectorstore().similarity_search(q, k=k)
        return self._similarity_search_filtered(q, k, stages)

    def _semantic_stages(self) -> set[str]:
        stages: set[str] = set()
        if self._config.prop_chunk_enabled:
            stages.add("proposition")
        else:
            stages.add("recursive")
        if self._config.raptor_enabled:
            stages.add("raptor")
        return stages

    def _similarity_search_filtered(
        self,
        query: str,
        k: int,
        stages: set[str],
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

    def _keyword_search(self, query: str) -> list[Document]:
        k = max(0, self._config.keyword_search_top_k)
        if k <= 0:
            return []
        terms = self._keyword_terms(query)
        if not terms:
            return []

        scored: list[tuple[int, Document]] = []
        for doc, text in self._keyword_corpus():
            score = sum(text.count(term) for term in terms)
            if score <= 0:
                continue
            scored.append((score, doc))

        scored.sort(key=lambda item: (-item[0], self._doc_key(item[1])))
        return [doc for _, doc in scored[:k]]

    @staticmethod
    def _keyword_terms(query: str) -> list[str]:
        raw = query.strip()
        if not raw:
            return []

        pieces = [part for part in re.split(r"\s+", raw) if part]
        terms: list[str] = []
        for piece in pieces:
            subparts = [part for part in re.split(r"[^\w]+", piece) if part]
            if subparts:
                terms.extend(subparts)
            else:
                terms.append(piece)

        normalized = [term.casefold() for term in terms if term]
        full = raw.casefold()
        if full and full not in normalized:
            normalized.append(full)
        return list(dict.fromkeys(normalized))

    @lru_cache(maxsize=1)
    def _keyword_corpus(self) -> list[tuple[Document, str]]:
        docs = self._load_keyword_docs()
        corpus: list[tuple[Document, str]] = []
        for doc in docs:
            text = self._keyword_text(doc)
            if text:
                corpus.append((doc, text.casefold()))
        return corpus

    def _load_keyword_docs(self) -> list[Document]:
        chunk_dirs = self._keyword_chunk_dirs()
        if not chunk_dirs:
            return []

        chunks = load_chunks_from_dirs(chunk_dirs)
        return [
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in chunks
        ]

    def _keyword_chunk_dirs(self) -> list[Path]:
        base_dir = (
            self._config.prop_chunk_dir
            if self._config.prop_chunk_enabled
            else self._config.rec_chunk_dir
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

    @staticmethod
    def _keyword_text(doc: Document) -> str:
        metadata = doc.metadata or {}
        drive_path = str(metadata.get("drive_file_path") or "")
        drive_name = str(metadata.get("drive_file_name") or "")
        parts = [doc.page_content, drive_name, drive_path]
        return "\n".join(part for part in parts if part)

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

    def _append_parent_recursive_docs(
        self,
        docs: list[Document],
    ) -> list[Document]:
        if not docs or not self._config.parent_doc_enabled:
            return docs

        parent_map = self._recursive_chunk_map()
        if not parent_map:
            return docs

        seen_doc_keys = {self._doc_key(doc) for doc in docs}
        added: list[Document] = []
        seen_parent_keys: set[tuple[object, ...]] = set()

        for doc in docs:
            metadata = doc.metadata or {}
            if metadata.get("chunk_stage") != "proposition":
                continue
            parent_key = self._parent_lookup_key(metadata)
            if parent_key is None or parent_key in seen_parent_keys:
                continue
            seen_parent_keys.add(parent_key)
            parent_doc = parent_map.get(parent_key)
            if parent_doc is None:
                continue
            parent_doc_key = self._doc_key(parent_doc)
            if parent_doc_key in seen_doc_keys:
                continue
            seen_doc_keys.add(parent_doc_key)
            added.append(parent_doc)

        if not added:
            return docs
        return docs + added

    @lru_cache(maxsize=1)
    def _recursive_chunk_map(self) -> dict[tuple[object, ...], Document]:
        docs = self._load_recursive_docs()
        mapping: dict[tuple[object, ...], Document] = {}
        for doc in docs:
            key = self._recursive_lookup_key(doc.metadata)
            if key is None or key in mapping:
                continue
            mapping[key] = doc
        return mapping

    def _load_recursive_docs(self) -> list[Document]:
        chunk_dirs = []
        for name in ("docs", "sheets"):
            candidate = self._config.rec_chunk_dir / name
            if candidate.exists():
                chunk_dirs.append(candidate)
        if not chunk_dirs:
            return []

        chunks = load_chunks_from_dirs(chunk_dirs)
        return [
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in chunks
        ]

    @classmethod
    def _parent_lookup_key(
        cls,
        metadata: dict[str, object],
    ) -> tuple[object, ...] | None:
        parent_id = cls._normalize_chunk_id(metadata.get("parent_chunk_id"))
        if parent_id is None:
            return None
        source = metadata.get("drive_file_id") or metadata.get("source_file_name")
        if not source:
            return None
        source_type = metadata.get("source_type") or ""
        return (source_type, source, parent_id)

    @classmethod
    def _recursive_lookup_key(
        cls,
        metadata: dict[str, object],
    ) -> tuple[object, ...] | None:
        chunk_id = cls._normalize_chunk_id(metadata.get("chunk_id"))
        if chunk_id is None:
            return None
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
