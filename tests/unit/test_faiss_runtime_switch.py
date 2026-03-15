from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex


class FaissRuntimeSwitchTests(unittest.TestCase):
    def test_disable_faiss_by_env_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = FaissLikeIndex(index_dir=Path(tmp))
            with patch.dict("os.environ", {"KUMC_DISABLE_FAISS_RUNTIME": "true"}, clear=False):
                self.assertTrue(index._is_faiss_runtime_disabled())  # noqa: SLF001

    def test_disable_faiss_on_macos_when_torch_loaded(self) -> None:
        if sys.platform != "darwin":
            self.skipTest("macOS specific behavior")
        with tempfile.TemporaryDirectory() as tmp:
            index = FaissLikeIndex(index_dir=Path(tmp))
            with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}, clear=False):
                self.assertTrue(index._is_faiss_runtime_disabled())  # noqa: SLF001

    def test_search_falls_back_to_numpy_when_faiss_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = FaissLikeIndex(index_dir=Path(tmp))
            chunks = [
                Chunk(id="a", document_id="doc", text="alpha", index=0, metadata={}),
                Chunk(id="b", document_id="doc", text="beta", index=1, metadata={}),
            ]
            vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            index.build(chunks=chunks, embeddings=vectors)
            with patch.dict("os.environ", {"KUMC_DISABLE_FAISS_RUNTIME": "1"}, clear=False):
                results = index.search(query_vector=np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].chunk.id, "a")

            payloads = [json.loads(line) for line in (Path(tmp) / "dense_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(payloads), 2)
