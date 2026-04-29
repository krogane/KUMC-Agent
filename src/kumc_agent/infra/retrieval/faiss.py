from __future__ import annotations

import json
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys

import numpy as np

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.policies.chunk_visibility import is_chunk_allowed_for_answer_context
from kumc_agent.features.indexing.paths import resolve_current_index_dir
from kumc_agent.utils.hashing import cosine_similarity_matrix

logger = logging.getLogger(__name__)

FileSignature = tuple[str, int, int]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class FaissLikeIndex:
    def __init__(self, *, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._cached_index = None
        self._cached_index_signature: FileSignature | None = None
        self._cached_vectors: np.ndarray | None = None
        self._cached_vectors_signature: FileSignature | None = None
        self._cached_chunks: list[Chunk] | None = None
        self._cached_chunks_signature: FileSignature | None = None
        self._faiss_disabled_logged = False
        self._index_dir.mkdir(parents=True, exist_ok=True)

    def build(self, *, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != int(embeddings.shape[0]):
            raise ValueError("Chunk count and embedding row count mismatch")
        matrix = embeddings.astype(np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be 2-dimensional")
        vectors_path = self._write_path("dense_vectors.npy")
        faiss_path = self._write_path("dense_vectors.faiss")
        chunks_path = self._write_path("dense_chunks.jsonl")
        np.save(vectors_path, matrix)
        self._cached_vectors = matrix
        self._cached_vectors_signature = self._file_signature(vectors_path)

        if self._is_faiss_runtime_disabled():
            self._cached_index = None
            self._cached_index_signature = None
        else:
            try:
                import faiss

                normalized = matrix.copy()
                if normalized.shape[0] > 0:
                    faiss.normalize_L2(normalized)
                index = faiss.IndexFlatIP(int(normalized.shape[1]))
                index.add(normalized)
                faiss.write_index(index, str(faiss_path))
                self._cached_index = index
                self._cached_index_signature = self._file_signature(faiss_path)
            except Exception:
                logger.exception(
                    "FAISS index build failed. Falling back to NumPy-based dense search."
                )
                self._cached_index = None
                self._cached_index_signature = None

        with chunks_path.open("w", encoding="utf-8") as fw:
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
        self._cached_chunks = list(chunks)
        self._cached_chunks_signature = self._file_signature(chunks_path)

    def search(self, *, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        chunks = self._load_chunks()
        if not chunks:
            return []

        dense_query = query_vector.astype(np.float32)
        results = self._search_faiss(
            query_vector=dense_query,
            top_k=max(0, top_k),
            chunks=chunks,
        )
        if results is not None:
            return results

        matrix = self._load_vectors()
        if matrix.size == 0:
            return []
        scores = cosine_similarity_matrix(dense_query, matrix)
        order = np.argsort(-scores)[: max(0, top_k)]
        return [
            SearchResult(chunk=chunks[int(i)], score=float(scores[int(i)]))
            for i in order
            if int(i) < len(chunks)
            and is_chunk_allowed_for_answer_context(chunks[int(i)])
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
        if self._is_faiss_runtime_disabled():
            return None
        faiss_path = self._read_path("dense_vectors.faiss")
        if not faiss_path.exists():
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
                chunk = chunks[pos]
                if not is_chunk_allowed_for_answer_context(chunk):
                    continue
                out.append(SearchResult(chunk=chunk, score=float(score)))
            return out
        except Exception:
            logger.exception(
                "FAISS search failed. Falling back to NumPy-based dense search."
            )
            return None

    def _load_faiss_index(self):
        if self._is_faiss_runtime_disabled():
            return None
        try:
            import faiss
        except Exception:
            return None
        faiss_path = self._read_path("dense_vectors.faiss")
        if not faiss_path.exists():
            self._cached_index = None
            self._cached_index_signature = None
            return None
        current_signature = self._file_signature(faiss_path)
        if (
            self._cached_index is not None
            and self._cached_index_signature is not None
            and self._cached_index_signature == current_signature
        ):
            return self._cached_index
        index = faiss.read_index(str(faiss_path))
        self._cached_index = index
        self._cached_index_signature = current_signature
        return index

    def _load_chunks(self) -> list[Chunk]:
        chunks_path = self._read_path("dense_chunks.jsonl")
        if not chunks_path.exists():
            self._cached_chunks = None
            self._cached_chunks_signature = None
            return []
        current_signature = self._file_signature(chunks_path)
        if (
            self._cached_chunks is not None
            and self._cached_chunks_signature is not None
            and self._cached_chunks_signature == current_signature
        ):
            return self._cached_chunks
        out: list[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as fr:
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
        self._cached_chunks = out
        self._cached_chunks_signature = current_signature
        return out

    def _load_vectors(self) -> np.ndarray:
        vectors_path = self._read_path("dense_vectors.npy")
        if not vectors_path.exists():
            self._cached_vectors = None
            self._cached_vectors_signature = None
            return np.empty((0, 0), dtype=np.float32)
        current_signature = self._file_signature(vectors_path)
        if (
            self._cached_vectors is not None
            and self._cached_vectors_signature is not None
            and self._cached_vectors_signature == current_signature
        ):
            return self._cached_vectors
        matrix = np.load(vectors_path)
        self._cached_vectors = matrix
        self._cached_vectors_signature = current_signature
        return matrix

    def _read_path(self, name: str) -> Path:
        return resolve_current_index_dir(self._index_dir) / name

    def _write_path(self, name: str) -> Path:
        return self._index_dir / name

    @staticmethod
    def _file_signature(path: Path) -> FileSignature | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))

    def _is_faiss_runtime_disabled(self) -> bool:
        disabled_env = str(os.getenv("KUMC_DISABLE_FAISS_RUNTIME", "")).strip().lower()
        if disabled_env in {"1", "true", "yes", "on"}:
            return True
        if sys.platform == "darwin" and "torch" in sys.modules:
            if not self._faiss_disabled_logged:
                logger.warning(
                    "FAISS is disabled on macOS because torch is already loaded. "
                    "Using NumPy dense search fallback to avoid libomp conflicts."
                )
                self._faiss_disabled_logged = True
            return True
        return False
