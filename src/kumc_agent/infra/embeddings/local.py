from __future__ import annotations

import numpy as np

from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.utils.hashing import hashed_vector


class LocalEmbedder(EmbedderPort):
    def __init__(self, *, model_name: str, dimensions: int = 256) -> None:
        self._model_name = model_name
        self._dimensions = dimensions

    def embed_query(self, text: str) -> np.ndarray:
        return hashed_vector(text, dimensions=self._dimensions)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimensions), dtype=np.float32)
        return np.vstack([self.embed_query(text) for text in texts])
