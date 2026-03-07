from __future__ import annotations

from functools import lru_cache
import logging

import numpy as np

from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.utils.hashing import hashed_vector

logger = logging.getLogger(__name__)


class LocalEmbedder(EmbedderPort):
    def __init__(self, *, model_name: str, dimensions: int = 256) -> None:
        self._model_name = model_name
        self._dimensions = max(1, dimensions)
        self._use_e5_prefix = _is_multilingual_e5(model_name)
        self._model = _load_sentence_transformer(model_name)

    def embed_query(self, text: str) -> np.ndarray:
        if self._model is None:
            return hashed_vector(text, dimensions=self._dimensions)
        query = text if text else " "
        if self._use_e5_prefix:
            query = self._apply_e5_prefix(query, prefix="query:")
        try:
            vectors = self._model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            resized = _resize_and_normalize(vectors, dimensions=self._dimensions)
            return resized[0] if resized.shape[0] > 0 else hashed_vector(query, dimensions=self._dimensions)
        except Exception:
            logger.exception(
                "Local query embedding failed. Falling back to hashed embedding."
            )
            return hashed_vector(query, dimensions=self._dimensions)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimensions), dtype=np.float32)
        if self._model is None:
            return np.vstack([hashed_vector(text, dimensions=self._dimensions) for text in texts])
        payload = texts
        if self._use_e5_prefix:
            payload = [self._apply_e5_prefix(text, prefix="document:") for text in texts]
        try:
            vectors = self._model.encode(
                payload,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return _resize_and_normalize(vectors, dimensions=self._dimensions)
        except Exception:
            logger.exception(
                "Local document embeddings failed. Falling back to hashed embeddings."
            )
            return np.vstack([hashed_vector(text, dimensions=self._dimensions) for text in texts])

    @staticmethod
    def _apply_e5_prefix(text: str, *, prefix: str) -> str:
        stripped = (text or "").lstrip()
        lowered = stripped.lower()
        if lowered.startswith("query:") or lowered.startswith("document:"):
            return stripped
        if not stripped:
            return f"{prefix} "
        return f"{prefix} {stripped}"


def _is_multilingual_e5(model_name: str) -> bool:
    lowered = (model_name or "").lower()
    return "multilingual-e5" in lowered or "multilingual_e5" in lowered


@lru_cache(maxsize=1)
def _load_sentence_transformer(model_name: str):
    if not model_name:
        logger.warning(
            "Local embedding model name is empty. Falling back to hashed embeddings."
        )
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        logger.exception(
            "sentence-transformers import failed. Falling back to hashed embeddings."
        )
        return None
    try:
        return SentenceTransformer(
            model_name,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception:
        logger.warning(
            "Local model was not found locally (%s). Trying remote fetch.",
            model_name,
        )
        try:
            return SentenceTransformer(
                model_name,
                local_files_only=False,
                trust_remote_code=False,
            )
        except Exception:
            logger.exception(
                "SentenceTransformer model load failed. Falling back to hashed embeddings."
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
