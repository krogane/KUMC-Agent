from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import AppConfig, EmbeddingFactory
from pipeline.llm_clients import generate_with_gemini, generate_with_llama
from pipeline.prompts import build_gemini_prompt, build_llama_messages
from pipeline.vectorstore import load_faiss_index


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

    def retrieve(self, query: str) -> list[Document]:
        q = f"query: {query}"
        return self._vectorstore().similarity_search(q, k=self._config.top_k)

    def generate(self, *, query: str, docs: list[Document]) -> str:
        provider = (self._config.llm_provider or "").lower()
        if provider == "gemini":
            prompt = build_gemini_prompt(query=query, docs=docs)
            text = generate_with_gemini(
                api_key=self._llm_api_key,
                prompt=prompt,
                config=self._config,
            )
        elif provider == "llama":
            messages = build_llama_messages(
                query=query,
                docs=docs,
                config=self._config,
            )
            text = generate_with_llama(messages=messages, config=self._config)
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

    @property
    def config(self) -> AppConfig:
        return self._config

    def embeddings(self) -> HuggingFaceEmbeddings:
        return self._embedding_factory.get_embeddings()

    @lru_cache(maxsize=1)
    def _vectorstore(self) -> FAISS:
        return load_faiss_index(
            index_dir=self._index_dir,
            embedding_factory=self._embedding_factory,
        )
