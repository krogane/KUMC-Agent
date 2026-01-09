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
        provider = (self._config.llm_provider or "").lower()
        if provider == "gemini":
            prompt = self._build_prompt(query=query, docs=docs)
            text = self._generate_with_gemini(prompt)
        elif provider == "llama":
            messages = self._build_llama_messages(query=query, docs=docs)
            text = self._generate_with_llama(messages)
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

    @property
    def config(self) -> AppConfig:
        return self._config

    def embeddings(self) -> HuggingFaceEmbeddings:
        return self._embedding_factory.get_embeddings()

    def _build_prompt(self, *, query: str, docs: list[Document]) -> str:
        context = _format_context([d.page_content for d in docs])
        return f"# コンテキスト\n{context}\n\n# 質問\n{query}"

    def _build_llama_messages(self, *, query: str, docs: list[Document]) -> list[dict[str, str]]:
        context = _format_context([d.page_content for d in docs])
        system = "\n".join(self._config.system_rules)
        user = (
            "### Context\n"
            f"{context}\n\n"
            "### Question\n"
            f"{query}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

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

    def _generate_with_gemini(self, prompt: str) -> str:
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
        return (response.text or "").strip()

    def _generate_with_llama(self, messages: list[dict[str, str]]) -> str:
        llama = self._llama_client()
        result = llama.create_chat_completion(
            messages=messages,
            max_tokens=self._config.max_output_tokens,
            temperature=self._config.temperature,
            stop=["\n---"],
        )
        return (
            (result.get("choices", [{}])[0].get("message", {}) or {}).get("content")
            or ""
        ).strip()

    @lru_cache(maxsize=1)
    def _genai_client(self) -> genai.Client:
        if not self._llm_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Please set it in .env")
        return genai.Client(api_key=self._llm_api_key)

    @lru_cache(maxsize=1)
    def _llama_client(self):
        if not self._config.llama_model_path:
            raise RuntimeError("LLAMA_MODEL_PATH is not set. Please set it in .env")

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python is not installed. Please install it to use llama.cpp."
            ) from e

        return Llama(
            model_path=self._config.llama_model_path,
            n_ctx=self._config.llama_ctx_size,
            n_threads=self._config.llama_threads,
            n_gpu_layers=self._config.llama_gpu_layers,
        )

def _format_context(parts: Sequence[str]) -> str:
    if not parts:
        return "(コンテキストなし)"
    return "\n\n---\n\n".join(parts)
