from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.features.indexing.embedding_cache import FileIndexEmbeddingCache
from kumc_agent.features.indexing.service import IndexingService


class _Embedder:
    def __init__(self, *, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        batch = [str(text) for text in texts]
        self.calls.append(batch)
        vectors = []
        for text in batch:
            base = float(len(text) % 10)
            vectors.append([base + float(index) for index in range(self.dimensions)])
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text):
        return np.ones(self.dimensions, dtype=np.float32)


class IncrementalEmbeddingTests(unittest.TestCase):
    def _service(
        self,
        root: Path,
        *,
        embedder: _Embedder,
        model: str = "test-model",
        enabled: bool = True,
        force_reembed_on_full_rebuild: bool = True,
    ) -> IndexingService:
        runtime = SimpleNamespace(
            app=SimpleNamespace(
                data_dir=root / "data",
                index_dir=root / "index",
            ),
            providers=SimpleNamespace(
                embeddings=SimpleNamespace(
                    provider="local",
                    model=model,
                    dimensions=embedder.dimensions,
                )
            ),
            indexing=SimpleNamespace(
                embedding_cache=SimpleNamespace(
                    enabled=enabled,
                    compact_after_publish=True,
                    force_reembed_on_full_rebuild=force_reembed_on_full_rebuild,
                )
            ),
        )
        return IndexingService(
            storage=object(),
            embedder=embedder,
            faiss_index=object(),
            bm25_index=object(),
            ingestion_dir=root / "ingestion",
            app_config=runtime,
            embedding_cache=FileIndexEmbeddingCache(root / "cache" / "index_embeddings"),
        )

    def _chunk(
        self,
        chunk_id: str,
        *,
        text: str,
        acl_hash: str = "acl-1",
    ) -> Chunk:
        return Chunk(
            id=chunk_id,
            document_id="doc-1",
            text=text,
            index=0,
            metadata={
                "source_type": "minecraft_wiki",
                "source_kind": "minecraft_wiki",
                "source_item_id": "source-1",
                "minecraft_wiki_title": "丸石",
                "access_scope": {
                    "visibility": "public",
                    "source_acl_hash": acl_hash,
                },
            },
        )

    def test_reuses_cached_vectors_and_embeds_only_changed_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embedder = _Embedder()
            service = self._service(root, embedder=embedder)
            first = service._embed_index_chunks(  # noqa: SLF001
                index_chunks=[self._chunk("c1", text="石を採掘できます。")],
                full_rebuild=False,
            )
            self.assertEqual(first.metadata["embedded_chunks"], 1)
            self.assertEqual(first.metadata["reused_chunks"], 0)
            self.assertEqual(len(embedder.calls), 1)

            second = service._embed_index_chunks(  # noqa: SLF001
                index_chunks=[self._chunk("c1", text="石を採掘できます。", acl_hash="acl-2")],
                full_rebuild=False,
            )
            self.assertEqual(second.metadata["embedded_chunks"], 0)
            self.assertEqual(second.metadata["reused_chunks"], 1)
            self.assertEqual(len(embedder.calls), 1)
            self.assertTrue(np.array_equal(first.matrix, second.matrix))

            third = service._embed_index_chunks(  # noqa: SLF001
                index_chunks=[self._chunk("c1", text="石を素早く採掘できます。")],
                full_rebuild=False,
            )
            self.assertEqual(third.metadata["embedded_chunks"], 1)
            self.assertEqual(third.metadata["reused_chunks"], 0)
            self.assertEqual(len(embedder.calls), 2)
            self.assertEqual(third.matrix.shape, (1, 3))

    def test_model_change_and_full_rebuild_bypass_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_embedder = _Embedder()
            first_service = self._service(root, embedder=first_embedder, model="model-a")
            first_service._embed_index_chunks(  # noqa: SLF001
                index_chunks=[self._chunk("c1", text="同じ本文です。")],
                full_rebuild=False,
            )
            self.assertEqual(len(first_embedder.calls), 1)

            second_embedder = _Embedder()
            second_service = self._service(root, embedder=second_embedder, model="model-b")
            changed_model = second_service._embed_index_chunks(  # noqa: SLF001
                index_chunks=[self._chunk("c1", text="同じ本文です。")],
                full_rebuild=False,
            )
            self.assertEqual(changed_model.metadata["embedded_chunks"], 1)
            self.assertEqual(changed_model.metadata["reused_chunks"], 0)
            self.assertEqual(len(second_embedder.calls), 1)

            full_rebuild = second_service._embed_index_chunks(  # noqa: SLF001
                index_chunks=[self._chunk("c1", text="同じ本文です。")],
                full_rebuild=True,
            )
            self.assertEqual(full_rebuild.metadata["embedded_chunks"], 1)
            self.assertTrue(full_rebuild.metadata["force_reembed"])
            self.assertEqual(len(second_embedder.calls), 2)

    def test_invalid_cache_lines_are_ignored_and_compacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embedder = _Embedder()
            service = self._service(root, embedder=embedder)
            result = service._embed_index_chunks(  # noqa: SLF001
                index_chunks=[
                    self._chunk("c1", text="本文1"),
                    self._chunk("c2", text="本文2"),
                ],
                full_rebuild=False,
            )
            cache_file = next((root / "cache" / "index_embeddings").glob("*.jsonl"))
            with cache_file.open("a", encoding="utf-8") as fw:
                fw.write("{not-json}\n")

            reused = service._embed_index_chunks(  # noqa: SLF001
                index_chunks=[self._chunk("c1", text="本文1")],
                full_rebuild=False,
            )
            self.assertEqual(reused.metadata["cache_invalid"], 1)
            self.assertEqual(reused.metadata["reused_chunks"], 1)

            compact_result = service.compact_embedding_cache((result.cache_keys[0],))
            self.assertEqual(compact_result["status"], "succeeded")
            payloads = [
                json.loads(line)
                for line in cache_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([payload["chunk_id"] for payload in payloads], ["c1"])

    def test_cache_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embedder = _Embedder()
            service = self._service(root, embedder=embedder, enabled=False)
            service._embed_index_chunks(  # noqa: SLF001
                index_chunks=[self._chunk("c1", text="同じ本文です。")],
                full_rebuild=False,
            )
            second = service._embed_index_chunks(  # noqa: SLF001
                index_chunks=[self._chunk("c1", text="同じ本文です。")],
                full_rebuild=False,
            )
            self.assertFalse(second.metadata["enabled"])
            self.assertEqual(len(embedder.calls), 2)


if __name__ == "__main__":
    unittest.main()
