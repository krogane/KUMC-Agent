from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.utils.hashing import cosine_similarity_matrix


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class FaissLikeIndex:
    def __init__(self, *, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._vectors_path = self._index_dir / "dense_vectors.npy"
        self._chunks_path = self._index_dir / "dense_chunks.jsonl"
        self._index_dir.mkdir(parents=True, exist_ok=True)

    def build(self, *, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != int(embeddings.shape[0]):
            raise ValueError("Chunk count and embedding row count mismatch")
        np.save(self._vectors_path, embeddings.astype(np.float32))
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
        if not self._vectors_path.exists() or not self._chunks_path.exists():
            return []
        matrix = np.load(self._vectors_path)
        if matrix.size == 0:
            return []
        chunks = self._load_chunks()
        scores = cosine_similarity_matrix(query_vector.astype(np.float32), matrix)
        order = np.argsort(-scores)[: max(0, top_k)]
        return [
            SearchResult(chunk=chunks[int(i)], score=float(scores[int(i)]))
            for i in order
            if int(i) < len(chunks)
        ]

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
