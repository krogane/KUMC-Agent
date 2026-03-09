from __future__ import annotations

from functools import lru_cache
import logging

import numpy as np

from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit
from kumc_agent.utils.hashing import hashed_vector

logger = logging.getLogger(__name__)


class GeminiEmbedder(EmbedderPort):
    _BATCH_SIZE = 96

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        dimensions: int = 256,
        requests_per_minute: int = 60,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._dimensions = max(1, dimensions)
        self._requests_per_minute = max(0, int(requests_per_minute))
        self._client = _gemini_embedding_client(api_key)

    def embed_query(self, text: str) -> np.ndarray:
        query = _normalize_embedding_text(text)
        if self._client is None or not self._model_name:
            return hashed_vector(query, dimensions=self._dimensions)
        try:
            wait_for_gemini_rate_limit(
                max_requests_per_minute=self._requests_per_minute
            )
            response = self._client.models.embed_content(
                model=self._model_name,
                contents=[query],
                config=_gemini_embed_config(
                    task_type="RETRIEVAL_QUERY",
                    dimensions=self._dimensions,
                ),
            )
            vectors = _extract_gemini_embedding_vectors(response)
            if not vectors:
                raise RuntimeError("empty embedding response")
            resized = _resize_and_normalize(np.asarray(vectors, dtype=np.float32), dimensions=self._dimensions)
            return resized[0]
        except Exception:
            logger.exception(
                "Gemini query embedding failed. Falling back to hashed embedding."
            )
            return hashed_vector(query, dimensions=self._dimensions)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimensions), dtype=np.float32)
        if self._client is None or not self._model_name:
            return np.vstack([hashed_vector(text, dimensions=self._dimensions) for text in texts])
        normalized = [_normalize_embedding_text(text) for text in texts]
        vectors: list[list[float]] = []
        try:
            for i in range(0, len(normalized), self._BATCH_SIZE):
                batch = normalized[i : i + self._BATCH_SIZE]
                wait_for_gemini_rate_limit(
                    max_requests_per_minute=self._requests_per_minute
                )
                response = self._client.models.embed_content(
                    model=self._model_name,
                    contents=batch,
                    config=_gemini_embed_config(
                        task_type="RETRIEVAL_DOCUMENT",
                        dimensions=self._dimensions,
                    ),
                )
                vectors.extend(_extract_gemini_embedding_vectors(response))
            if len(vectors) != len(normalized):
                raise RuntimeError(
                    "Gemini embedding response count mismatch "
                    f"(requested={len(normalized)} got={len(vectors)})"
                )
            return _resize_and_normalize(np.asarray(vectors, dtype=np.float32), dimensions=self._dimensions)
        except Exception:
            logger.exception(
                "Gemini document embeddings failed. Falling back to hashed embeddings."
            )
            return np.vstack([hashed_vector(text, dimensions=self._dimensions) for text in texts])


def _normalize_embedding_text(text: str | None) -> str:
    value = text if text else " "
    return value if value.strip() else " "


def _extract_gemini_embedding_vectors(response) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None) or []
    if not embeddings:
        single = getattr(response, "embedding", None)
        if single is not None:
            embeddings = [single]
    vectors: list[list[float]] = []
    for embedding in embeddings:
        values = getattr(embedding, "values", None) or []
        vectors.append([float(value) for value in values])
    return vectors


def _gemini_embed_config(*, task_type: str, dimensions: int):
    from google import genai

    kwargs = {"task_type": task_type}
    if dimensions > 0:
        kwargs["output_dimensionality"] = int(dimensions)
    try:
        return genai.types.EmbedContentConfig(**kwargs)
    except TypeError:
        kwargs.pop("output_dimensionality", None)
        return genai.types.EmbedContentConfig(**kwargs)


@lru_cache(maxsize=1)
def _gemini_embedding_client(api_key: str):
    resolved = (api_key or "").strip()
    if not resolved:
        logger.warning(
            "Gemini API key is not configured. Falling back to hashed embeddings."
        )
        return None
    try:
        from google import genai
    except Exception:
        logger.exception(
            "google-genai import failed. Falling back to hashed embeddings."
        )
        return None
    try:
        return genai.Client(api_key=resolved)
    except Exception:
        logger.exception(
            "Failed to initialize Gemini embedding client. Falling back to hashed embeddings."
        )
        return None


def _resize_and_normalize(vectors: np.ndarray, *, dimensions: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("embedding result must be 2D")
    current_dim = int(matrix.shape[1])
    if current_dim > dimensions:
        matrix = matrix[:, :dimensions]
    elif current_dim < dimensions:
        pad = np.zeros((matrix.shape[0], dimensions - current_dim), dtype=np.float32)
        matrix = np.hstack([matrix, pad])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
