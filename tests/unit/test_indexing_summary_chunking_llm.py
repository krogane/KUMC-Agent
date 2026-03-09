from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


def _runtime_config(*, data_dir: Path, provider: str) -> object:
    return SimpleNamespace(
        app=SimpleNamespace(data_dir=data_dir),
        indexing=SimpleNamespace(
            chunking=SimpleNamespace(
                summary_llm_provider=provider,
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


if __name__ == "__main__":
    unittest.main()
