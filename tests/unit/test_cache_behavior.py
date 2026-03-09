from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever
from kumc_agent.infra.storage.filesystem import FilePromptRepository


class CacheBehaviorTests(unittest.TestCase):
    def test_prompt_repository_cache_and_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = Path(tmp)
            prompt_file = prompt_dir / "sample.md"
            prompt_file.write_text("first", encoding="utf-8")

            repo = FilePromptRepository(prompt_dir)
            first = repo.get("sample")
            second = repo.get("sample")
            self.assertEqual(first, "first")
            self.assertEqual(second, "first")

            prompt_file.write_text("second value", encoding="utf-8")
            third = repo.get("sample")
            self.assertEqual(third, "second value")

    def test_dense_index_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = FaissLikeIndex(index_dir=Path(tmp))
            chunks = [
                Chunk(
                    id="c1",
                    document_id="d1",
                    text="alpha beta",
                    index=0,
                    metadata={},
                ),
                Chunk(
                    id="c2",
                    document_id="d1",
                    text="beta gamma",
                    index=1,
                    metadata={},
                ),
            ]
            embeddings = np.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.3, 0.2, 0.1],
                ],
                dtype=np.float32,
            )
            index.build(chunks=chunks, embeddings=embeddings)

            # Force NumPy fallback path so vector cache is exercised in all envs.
            index._search_faiss = lambda **kwargs: None  # type: ignore[method-assign]

            query = np.array([0.2, 0.2, 0.2], dtype=np.float32)
            _ = index.search(query_vector=query, top_k=2)
            cached_chunks = index._cached_chunks
            cached_vectors = index._cached_vectors
            self.assertIsNotNone(cached_chunks)
            self.assertIsNotNone(cached_vectors)

            _ = index.search(query_vector=query, top_k=2)
            self.assertIs(index._cached_chunks, cached_chunks)
            self.assertIs(index._cached_vectors, cached_vectors)

    def test_sparse_index_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            retriever = SudachiBM25Retriever(index_dir=Path(tmp))
            chunks = [
                Chunk(
                    id="c1",
                    document_id="d1",
                    text="alpha beta",
                    index=0,
                    metadata={},
                ),
                Chunk(
                    id="c2",
                    document_id="d1",
                    text="beta gamma",
                    index=1,
                    metadata={},
                ),
            ]
            retriever.build(chunks)

            _ = retriever.search("alpha", top_k=2)
            cached_bm25 = retriever._cached_bm25
            cached_chunks = retriever._cached_chunks
            cached_tokens = retriever._cached_tokens
            self.assertIsNotNone(cached_bm25)
            self.assertIsNotNone(cached_chunks)
            self.assertIsNotNone(cached_tokens)

            _ = retriever.search("gamma", top_k=2)
            self.assertIs(retriever._cached_bm25, cached_bm25)
            self.assertIs(retriever._cached_chunks, cached_chunks)
            self.assertIs(retriever._cached_tokens, cached_tokens)


if __name__ == "__main__":
    unittest.main()
