from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.features.indexing.service import IndexingService
from kumc_agent.infra.indexing.chunks import Chunk as LegacyChunk
from kumc_agent.infra.indexing.chunks import load_chunks, write_chunks
from kumc_agent.infra.indexing.chunking import summery_chunk_jsonl_dir


class _StubLLM:
    def __init__(self, *, response_text: str) -> None:
        self.response_text = response_text
        self.last_temperature = 0.0
        self.last_max_output_tokens = 0
        self.calls = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        self.calls += 1
        self.last_temperature = temperature
        self.last_max_output_tokens = max_output_tokens
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
                summary_characters=20,
                summary_temperature=0.15,
                summary_max_output_tokens=64,
                summary_thinking_level="minimal",
            ),
            stages=SimpleNamespace(summary_enabled=True),
        ),
        minecraft_wiki_rag=SimpleNamespace(
            chunking=SimpleNamespace(
                summary_llm_provider=provider,
                summary_batch_size=summary_batch_size,
                summary_characters=80,
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
                ingestion_dir=Path(tmp),
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

    def test_summary_falls_back_to_truncation_when_provider_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = _StubLLM(response_text="LLMは使われない")
            service = IndexingService(
                storage=object(),  # type: ignore[arg-type]
                embedder=object(),  # type: ignore[arg-type]
                faiss_index=object(),  # type: ignore[arg-type]
                bm25_index=object(),  # type: ignore[arg-type]
                ingestion_dir=Path(tmp),
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
                ingestion_dir=Path(tmp),
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
                ingestion_dir=Path(tmp),
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

    def test_repository_summary_batch_size_controls_parallelism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llm = _ParallelStubLLM(response_text="要約")
            service = IndexingService(
                storage=object(),  # type: ignore[arg-type]
                embedder=object(),  # type: ignore[arg-type]
                faiss_index=object(),  # type: ignore[arg-type]
                bm25_index=object(),  # type: ignore[arg-type]
                ingestion_dir=Path(tmp),
                app_config=_runtime_config(  # type: ignore[arg-type]
                    data_dir=Path(tmp),
                    provider="gemini",
                    summary_batch_size=2,
                ),
                summary_llm=llm,
            )
            repository_chunks = [
                Chunk(
                    id=f"repo-{idx}",
                    document_id="doc-1",
                    text=f"本文{idx}",
                    index=idx,
                    metadata={"source_type": "test", "chunk_id": idx},
                )
                for idx in range(5)
            ]

            result = service._build_repository_index_artifacts(  # noqa: SLF001
                repository_chunks=repository_chunks,
                legacy_cfg=SimpleNamespace(
                    sudachi_mode="C",
                    sparse_use_normalized_form=False,
                    sparse_remove_symbols=False,
                ),
                selected=set(),
            )

            self.assertEqual(len(result.summary_chunks), 5)
            self.assertEqual(llm.calls, 5)
            self.assertGreaterEqual(llm.max_in_flight, 2)
            self.assertLessEqual(llm.max_in_flight, 2)

    def test_minecraft_wiki_summary_batch_size_controls_parallelism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            write_chunks(
                input_dir / "wiki-a.jsonl",
                [
                    LegacyChunk(
                        text=f"本文{idx}",
                        metadata={"chunk_id": idx, "minecraft_wiki_title": "丸石"},
                    )
                    for idx in range(3)
                ],
            )
            write_chunks(
                input_dir / "wiki-b.jsonl",
                [
                    LegacyChunk(
                        text=f"本文{idx}",
                        metadata={"chunk_id": idx, "minecraft_wiki_title": "石"},
                    )
                    for idx in range(2)
                ],
            )
            llm = _ParallelStubLLM(response_text="要約")
            service = object.__new__(IndexingService)
            service._minecraft_wiki_summary_llm = llm
            service._runtime = _runtime_config(
                data_dir=root,
                provider="gemini",
                summary_batch_size=2,
            )

            service._build_minecraft_wiki_summary_chunks(  # noqa: SLF001
                input_chunk_dir=input_dir,
                output_chunk_dir=output_dir,
                skip_existing=False,
                update_existing=True,
                sync_deleted=True,
            )

            output_count = sum(
                len(load_chunks(path)) for path in sorted(output_dir.glob("*.jsonl"))
            )
            self.assertEqual(output_count, 5)
            self.assertEqual(llm.calls, 5)
            self.assertGreaterEqual(llm.max_in_flight, 2)
            self.assertLessEqual(llm.max_in_flight, 2)

    def test_legacy_summary_chunking_parallelizes_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            for file_index in range(4):
                write_chunks(
                    input_dir / f"source-{file_index}.jsonl",
                    [
                        LegacyChunk(
                            text=f"本文{file_index}",
                            metadata={"source_type": "docs", "chunk_id": 0},
                        )
                    ],
                )
            tracker = _ParallelStubLLM(response_text="要約")

            def _fake_run_llm_chunking(**kwargs):  # noqa: ANN001
                tracker.generate(
                    system_prompt="",
                    user_prompt=str(kwargs["prompt"]),
                    temperature=0.0,
                    max_output_tokens=64,
                )
                return ["要約"]

            with patch.dict(
                os.environ,
                {"PROMPT_SUMMERY_CHUNK_DEFAULT_TEMPLATE": "{text}"},
            ), patch(
                "kumc_agent.infra.indexing.chunking._run_llm_chunking",
                side_effect=_fake_run_llm_chunking,
            ):
                summery_chunk_jsonl_dir(
                    input_chunk_dir=input_dir,
                    output_chunk_dir=output_dir,
                    config=SimpleNamespace(
                        summery_provider="gemini",
                        summery_max_retries=1,
                        summery_batch_size=2,
                        summery_characters=20,
                        summery_gemini_model="gemini-test",
                        summery_temperature=0.0,
                        summery_max_output_tokens=64,
                        gemini_api_key="key",
                        gemini_summary_requests_per_minute=100,
                    ),
                    skip_existing=False,
                    update_existing=True,
                    sync_deleted=True,
                )

            output_count = sum(
                len(load_chunks(path)) for path in sorted(output_dir.glob("*.jsonl"))
            )
            self.assertEqual(output_count, 4)
            self.assertEqual(tracker.calls, 4)
            self.assertGreaterEqual(tracker.max_in_flight, 2)
            self.assertLessEqual(tracker.max_in_flight, 2)


if __name__ == "__main__":
    unittest.main()
