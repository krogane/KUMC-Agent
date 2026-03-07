from __future__ import annotations

import json
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.utils.hashing import cosine_similarity_matrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class FaissLikeIndex:
    def __init__(self, *, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._faiss_path = self._index_dir / "dense_vectors.faiss"
        self._vectors_path = self._index_dir / "dense_vectors.npy"
        self._chunks_path = self._index_dir / "dense_chunks.jsonl"
        self._cached_index = None
        self._cached_index_mtime: float | None = None
        self._index_dir.mkdir(parents=True, exist_ok=True)

    def build(self, *, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != int(embeddings.shape[0]):
            raise ValueError("Chunk count and embedding row count mismatch")
        matrix = embeddings.astype(np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be 2-dimensional")
        np.save(self._vectors_path, matrix)

        try:
            import faiss

            normalized = matrix.copy()
            if normalized.shape[0] > 0:
                faiss.normalize_L2(normalized)
            index = faiss.IndexFlatIP(int(normalized.shape[1]))
            index.add(normalized)
            faiss.write_index(index, str(self._faiss_path))
            self._cached_index = index
            self._cached_index_mtime = self._faiss_path.stat().st_mtime
        except Exception:
            logger.exception(
                "FAISS index build failed. Falling back to NumPy-based dense search."
            )
            self._cached_index = None
            self._cached_index_mtime = None

        with self._chunks_path.open("w", encoding="utf-8") as fw:
            for chunk in chunks:
                fw.write(
                    json.dumps(
                        {
                            "id": chunk.id,
                            "document_id": chunk.document_id,
                            "text": chunk.text,
                            "index": chunk.index,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def search(self, *, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        if not self._chunks_path.exists():
            return []
        chunks = self._load_chunks()
        if not chunks:
            return []

        results = self._search_faiss(
            query_vector=query_vector.astype(np.float32),
            top_k=max(0, top_k),
            chunks=chunks,
        )
        if results is not None:
            return results

        if not self._vectors_path.exists():
            return []
        matrix = np.load(self._vectors_path)
        if matrix.size == 0:
            return []
        scores = cosine_similarity_matrix(query_vector.astype(np.float32), matrix)
        order = np.argsort(-scores)[: max(0, top_k)]
        return [
            SearchResult(chunk=chunks[int(i)], score=float(scores[int(i)]))
            for i in order
            if int(i) < len(chunks)
        ]

    def _search_faiss(
        self,
        *,
        query_vector: np.ndarray,
        top_k: int,
        chunks: list[Chunk],
    ) -> list[SearchResult] | None:
        if top_k <= 0:
            return []
        if not self._faiss_path.exists():
            return None
        try:
            import faiss
        except Exception:
            return None

        try:
            index = self._load_faiss_index()
            if index is None:
                return None
            query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query)
            scores, indices = index.search(query, top_k)
            out: list[SearchResult] = []
            if scores.shape[0] == 0 or indices.shape[0] == 0:
                return out
            for score, idx in zip(scores[0], indices[0]):
                pos = int(idx)
                if pos < 0 or pos >= len(chunks):
                    continue
                out.append(SearchResult(chunk=chunks[pos], score=float(score)))
            return out
        except Exception:
            logger.exception(
                "FAISS search failed. Falling back to NumPy-based dense search."
            )
            return None

    def _load_faiss_index(self):
        try:
            import faiss
        except Exception:
            return None
        if not self._faiss_path.exists():
            return None
        current_mtime = self._faiss_path.stat().st_mtime
        if (
            self._cached_index is not None
            and self._cached_index_mtime is not None
            and self._cached_index_mtime == current_mtime
        ):
            return self._cached_index
        index = faiss.read_index(str(self._faiss_path))
        self._cached_index = index
        self._cached_index_mtime = current_mtime
        return index

    def _load_chunks(self) -> list[Chunk]:
        out: list[Chunk] = []
        with self._chunks_path.open("r", encoding="utf-8") as fr:
            for line in fr:
                payload = json.loads(line)
                out.append(
                    Chunk(
                        id=str(payload["id"]),
                        document_id=str(payload["document_id"]),
                        text=str(payload["text"]),
                        index=int(payload["index"]),
                        metadata=dict(payload.get("metadata", {})),
                    )
                )
        return out
