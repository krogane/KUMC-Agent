from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hashed_vector(text: str, *, dimensions: int = 256) -> np.ndarray:
    if dimensions <= 0:
        raise ValueError("dimensions must be > 0")
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in (text or "").split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(0, len(digest), 4):
            chunk = digest[i : i + 4]
            if len(chunk) < 4:
                continue
            value = int.from_bytes(chunk, byteorder="big", signed=False)
            index = value % dimensions
            sign = -1.0 if (value & 1) else 1.0
            vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm > 0:
        query = query / query_norm
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    normalized = matrix / row_norms
    return normalized @ query


def merge_unique(items: Iterable[object]) -> list[object]:
    seen: set[object] = set()
    out: list[object] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
