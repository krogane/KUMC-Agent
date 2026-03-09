from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.features.indexing.service import IndexingService


class _StubLLM:
    def __init__(self, *, response_text: str) -> None:
        self.response_text = response_text
        self.last_temperature = 0.0
        self.last_max_output_tokens = 0
        self.last_thinking_level = ""
        self.calls = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        thinking_level: str,
    ) -> str:
        self.calls += 1
        self.last_temperature = temperature
        self.last_max_output_tokens = max_output_tokens
        self.last_thinking_level = thinking_level
        return self.response_text


class _ParallelStubLLM(_StubLLM):
    def __init__(self, *, response_text: str, delay_seconds: float = 0.05) -> None:
        super().__init__(response_text=response_text)
        self._delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        thinking_level: str,
    ) -> str:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            time.sleep(self._delay_seconds)
            return super().generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                thinking_level=thinking_level,
            )
        finally:
            with self._lock:
                self._in_flight -= 1


def _runtime_config(*, data_dir: Path, provider: str, summary_batch_size: int = 1) -> object:
    return SimpleNamespace(
        app=SimpleNamespace(data_dir=data_dir),
        indexing=SimpleNamespace(
            chunking=SimpleNamespace(
                summary_llm_provider=provider,
                summary_batch_size=summary_batch_size,
                summary_temperature=0.15,
                summary_max_output_tokens=64,
                summary_thinking_level="minimal",
            )
        ),
    )


class IndexingSummaryChunkingLLMTests(unittest.TestCase):
    def test_summary_uses_configured_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = _StubLLM(response_text="これはLLM要約です。")
            service = IndexingService(
                storage=object(),  # type: ignore[arg-type]
                embedder=object(),  # type: ignore[arg-type]
                faiss_index=object(),  # type: ignore[arg-type]
                bm25_index=object(),  # type: ignore[arg-type]
                raw_dir=Path(tmp),
                app_config=_runtime_config(data_dir=Path(tmp), provider="gemini"),  # type: ignore[arg-type]
                summary_llm=llm,
            )

            summary = service._build_summary_text(  # noqa: SLF001
                text="とても長い本文です。",
                target_characters=20,
            )

            self.assertEqual(summary, "これはLLM要約です。")
            self.assertEqual(llm.calls, 1)
            self.assertEqual(llm.last_temperature, 0.15)
            self.assertEqual(llm.last_max_output_tokens, 64)
            self.assertEqual(llm.last_thinking_level, "minimal")

    def test_summary_falls_back_to_truncation_when_provider_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = _StubLLM(response_text="LLMは使われない")
            service = IndexingService(
                storage=object(),  # type: ignore[arg-type]
                embedder=object(),  # type: ignore[arg-type]
                faiss_index=object(),  # type: ignore[arg-type]
                bm25_index=object(),  # type: ignore[arg-type]
                raw_dir=Path(tmp),
                app_config=_runtime_config(data_dir=Path(tmp), provider="none"),  # type: ignore[arg-type]
                summary_llm=llm,
            )

            summary = service._build_summary_text(  # noqa: SLF001
                text="1234567890abcdef",
                target_characters=8,
            )

            self.assertEqual(summary, "12345678")
            self.assertEqual(llm.calls, 0)

    def test_summary_falls_back_when_llm_returns_error_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = _StubLLM(response_text="Geminiでの回答生成に失敗しました。")
            service = IndexingService(
                storage=object(),  # type: ignore[arg-type]
                embedder=object(),  # type: ignore[arg-type]
                faiss_index=object(),  # type: ignore[arg-type]
                bm25_index=object(),  # type: ignore[arg-type]
                raw_dir=Path(tmp),
                app_config=_runtime_config(data_dir=Path(tmp), provider="gemini"),  # type: ignore[arg-type]
                summary_llm=llm,
            )

            summary = service._build_summary_text(  # noqa: SLF001
                text="abcdefghijklmnop",
                target_characters=6,
            )

            self.assertEqual(summary, "abcdef")

    def test_summary_batch_size_controls_parallelism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = _ParallelStubLLM(response_text="要約")
            service = IndexingService(
                storage=object(),  # type: ignore[arg-type]
                embedder=object(),  # type: ignore[arg-type]
                faiss_index=object(),  # type: ignore[arg-type]
                bm25_index=object(),  # type: ignore[arg-type]
                raw_dir=Path(tmp),
                app_config=_runtime_config(  # type: ignore[arg-type]
                    data_dir=Path(tmp),
                    provider="gemini",
                    summary_batch_size=2,
                ),
                summary_llm=llm,
            )
            first_chunks = [
                Chunk(
                    id=f"chunk-{idx}",
                    document_id="doc-1",
                    text=f"本文{idx}",
                    index=idx,
                    metadata={"source_type": "test"},
                )
                for idx in range(5)
            ]

            result = service._load_or_build_summary_chunks(  # noqa: SLF001
                first_chunks=first_chunks,
                enabled=True,
                target_characters=20,
                should_update=True,
                force=True,
                selected=set(),
            )

            self.assertEqual(len(result), 5)
            self.assertEqual(llm.calls, 5)
            self.assertGreaterEqual(llm.max_in_flight, 2)
            self.assertLessEqual(llm.max_in_flight, 2)


if __name__ == "__main__":
    unittest.main()
