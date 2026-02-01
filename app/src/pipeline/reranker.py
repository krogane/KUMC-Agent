from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document

from config import AppConfig

logger = logging.getLogger(__name__)


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

        scored = self.score_documents(query=query, docs=docs)
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _, _, doc in scored[:top_k]]

    def score_documents(
        self,
        *,
        query: str,
        docs: list[Document],
    ) -> list[tuple[float, int, Document]]:
        if not docs:
            return []

        model_path = (self._config.cross_encoder_model_path or "").strip()
        if not model_path:
            return [(0.0, idx, doc) for idx, doc in enumerate(docs)]

        cross_encoder = _cross_encoder_client(model_path)
        pairs = [(query, doc.page_content) for doc in docs]
        try:
            scores = cross_encoder.predict(pairs)
        except Exception:
            logger.exception("Failed to score documents with cross-encoder.")
            return [(0.0, idx, doc) for idx, doc in enumerate(docs)]

        scored: list[tuple[float, int, Document]] = []
        for idx, (doc, score) in enumerate(zip(docs, scores)):
            try:
                raw_value = float(score)
            except (TypeError, ValueError):
                raw_value = 0.0
            clamped = max(0.0, min(1.0, raw_value))
            scored.append((clamped, idx, doc))
        return scored


@lru_cache(maxsize=1)
def _cross_encoder_client(
    model_path: str,
):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is not installed. Please install it to use the reranker."
        ) from e

    return CrossEncoder(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
