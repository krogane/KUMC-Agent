from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

from google import genai
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import AppConfig, EmbeddingFactory


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
        prompt = self._build_prompt(query=query, docs=docs)

        response = self._genai_client().models.generate_content(
            model=self._config.genai_model,
            contents=[
                {
                    "role": "system",
                    "parts": [{"text": "\n".join(self._config.system_rules)}],
                },
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                },
            ],
            config=genai.types.GenerateContentConfig(
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
                thinking_config=genai.types.ThinkingConfig(
                    thinking_level=self._config.thinking_level
                ),
            ),
        )

        text = (response.text or "").strip()
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

    @property
    def config(self) -> AppConfig:
        return self._config

    def embeddings(self) -> HuggingFaceEmbeddings:
        return self._embedding_factory.get_embeddings()

    def _build_prompt(self, *, query: str, docs: list[Document]) -> str:
        context = _format_context([d.page_content for d in docs])
        return f"# コンテキスト\n{context}\n\n# 質問\n{query}"

    @lru_cache(maxsize=1)
    def _vectorstore(self) -> FAISS:
        if not self._index_dir.exists():
            raise FileNotFoundError(
                f"FAISS index directory not found: {self._index_dir}. "
                "Build the index first (e.g., run your build script)."
            )

        return FAISS.load_local(
            str(self._index_dir),
            self._embedding_factory.get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    @lru_cache(maxsize=1)
    def _genai_client(self) -> genai.Client:
        if not self._llm_api_key:
            raise RuntimeError("LLM_API_KEY is not set. Please set it in .env")
        return genai.Client(api_key=self._llm_api_key)


def _format_context(parts: Sequence[str]) -> str:
    if not parts:
        return "(コンテキストなし)"
    return "\n\n---\n\n".join(parts)
