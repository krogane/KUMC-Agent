from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbedderPort(Protocol):
    def embed_query(self, text: str) -> np.ndarray:
        ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        ...
