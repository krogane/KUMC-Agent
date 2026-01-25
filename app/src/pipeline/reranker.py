from __future__ import annotations

import logging
import re
from functools import lru_cache

from langchain_core.documents import Document

from config import AppConfig
from pipeline.llama_lock import LLAMA_LOCK, reset_llama_cache

logger = logging.getLogger(__name__)

_RERANK_PROMPT = (
    "You are a relevance ranking assistant. "
    "Given a query and a document, output a single number between 0 and 1.\n\n"
    "Query:\n{query}\n\n"
    "Document:\n{document}\n\n"
    "Score:"
)


class CrossEncoderReranker:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def rerank(
        self,
        *,
        query: str,
        docs: list[Document],
        top_k: int,
    ) -> list[Document]:
        if top_k <= 0 or not docs:
            return []

        logger.info(
            "Rerank start: query=%s top_k=%s candidates=%s",
            query,
            top_k,
            len(docs),
        )
        for idx, doc in enumerate(docs[:top_k], start=1):
            logger.info(
                "Rerank candidate %s/%s: metadata=%s content=%s",
                idx,
                top_k,
                doc.metadata,
                doc.page_content,
            )

        model_path = (self._config.cross_encoder_model_path or "").strip()
        if not model_path:
            return docs[:top_k]

        llama = _cross_encoder_client(
            model_path,
            self._config.llama_ctx_size,
            self._config.llama_threads,
            self._config.llama_gpu_layers,
        )

        scored: list[tuple[float, int, Document]] = []
        for idx, doc in enumerate(docs):
            score = _score_pair(
                llama=llama,
                query=query,
                document=doc.page_content,
            )
            scored.append((score, idx, doc))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _, _, doc in scored[:top_k]]


def _score_pair(*, llama, query: str, document: str) -> float:
    prompt = _RERANK_PROMPT.format(query=query, document=document)
    try:
        with LLAMA_LOCK:
            reset_llama_cache(llama)
            result = llama.create_completion(
                prompt=prompt,
                max_tokens=4,
                temperature=0.0,
                stop=["\n"],
            )
    except Exception:
        logger.exception("Failed to score document with cross-encoder.")
        return 0.0

    text = (result.get("choices", [{}])[0].get("text") or "").strip()
    score = _extract_score(text)
    return max(0.0, min(1.0, score))


def _extract_score(text: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return 0.0
    try:
        return float(match.group(0))
    except ValueError:
        return 0.0


@lru_cache(maxsize=1)
def _cross_encoder_client(
    model_path: str,
    ctx_size: int,
    threads: int,
    gpu_layers: int,
):
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise RuntimeError(
            "llama-cpp-python is not installed. Please install it to use the reranker."
        ) from e

    return Llama(
        model_path=model_path,
        n_ctx=ctx_size,
        n_threads=threads,
        n_gpu_layers=gpu_layers,
    )
