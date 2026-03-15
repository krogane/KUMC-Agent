from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.infra.indexing import sparse_normalizer as sparse_normalizer_module
from kumc_agent.infra.retrieval import cross_encoder as cross_encoder_module


class CrossEncoderThreadSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        cross_encoder_module._CROSS_ENCODER_CLIENTS.clear()  # noqa: SLF001

    def test_cross_encoder_client_initializes_once_under_parallel_calls(self) -> None:
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        create_calls = 0
        create_calls_lock = threading.Lock()

        def _fake_create(model_name: str, *, device: str | None = None):  # type: ignore[no-untyped-def]
            _ = model_name
            _ = device
            nonlocal create_calls
            with create_calls_lock:
                create_calls += 1
            return object()

        with patch.object(cross_encoder_module, "_create_cross_encoder", side_effect=_fake_create):
            def _worker() -> object:
                barrier.wait(timeout=5)
                return cross_encoder_module._cross_encoder_client("dummy-model")

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                clients = list(executor.map(lambda _i: _worker(), range(worker_count)))

        self.assertEqual(create_calls, 1)
        self.assertEqual(len({id(client) for client in clients}), 1)

    def test_cross_encoder_client_retries_on_cpu_for_meta_tensor_error(self) -> None:
        devices: list[str | None] = []

        def _fake_create(model_name: str, *, device: str | None = None):  # type: ignore[no-untyped-def]
            _ = model_name
            devices.append(device)
            if device is None:
                raise NotImplementedError(
                    "Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead."
                )
            return {"device": device}

        with patch.object(cross_encoder_module, "_create_cross_encoder", side_effect=_fake_create):
            client = cross_encoder_module._cross_encoder_client("dummy-model")
            cached = cross_encoder_module._cross_encoder_client("dummy-model")

        self.assertEqual(devices, [None, "cpu"])
        self.assertEqual(client, {"device": "cpu"})
        self.assertIs(cached, client)


class SparseNormalizerThreadSafetyTests(unittest.TestCase):
    def test_sparse_normalizer_uses_thread_local_tokenizer(self) -> None:
        worker_count = 6
        barrier = threading.Barrier(worker_count)
        create_count = 0
        create_count_lock = threading.Lock()
        results: list[tuple[str, str]] = []
        results_lock = threading.Lock()

        class _FakeMorph:
            def __init__(self, token: str) -> None:
                self._token = token

            def part_of_speech(self):  # type: ignore[no-untyped-def]
                return ("名詞",)

            def normalized_form(self) -> str:
                return self._token

            def surface(self) -> str:
                return self._token

            def dictionary_form(self) -> str:
                return self._token

        class _FakeTokenizer:
            def __init__(self, token_prefix: str) -> None:
                self._token_prefix = token_prefix

            def tokenize(self, _value: str, _mode):  # type: ignore[no-untyped-def]
                return [_FakeMorph(self._token_prefix)]

        class _FakeDictionary:
            def create(self):  # type: ignore[no-untyped-def]
                nonlocal create_count
                with create_count_lock:
                    create_count += 1
                token_prefix = f"tok-{threading.get_ident()}"
                return _FakeTokenizer(token_prefix)

        sparse_normalizer_module._SUDACHI_TOKENIZER_LOCAL = threading.local()  # noqa: SLF001
        config = sparse_normalizer_module.SparseNormalizerConfig()
        normalizer = sparse_normalizer_module.SparseNormalizer(config=config)

        with patch.object(sparse_normalizer_module.dictionary, "Dictionary", _FakeDictionary):
            def _worker() -> None:
                barrier.wait(timeout=5)
                token1 = normalizer.normalize_tokens("alpha")[0]
                token2 = normalizer.normalize_tokens("beta")[0]
                with results_lock:
                    results.append((token1, token2))

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                list(executor.map(lambda _i: _worker(), range(worker_count)))

        self.assertEqual(len(results), worker_count)
        self.assertTrue(all(token1 == token2 for token1, token2 in results))
        self.assertEqual(len({token1 for token1, _token2 in results}), worker_count)
        self.assertEqual(create_count, worker_count)


if __name__ == "__main__":
    unittest.main()
