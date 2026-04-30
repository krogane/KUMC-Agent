from __future__ import annotations

from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.features.indexing.service import IndexingService


class _StubLLM:
    def __init__(self, *, response_text: str) -> None:
        self.response_text = response_text

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        return self.response_text


class RepositoryBackedIndexArtifactsTests(unittest.TestCase):
    def test_repository_chunks_build_stage_keyword_and_material_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = SimpleNamespace(
                app=SimpleNamespace(
                    data_dir=root / "data",
                    index_dir=root / "data" / "index",
                ),
                indexing=SimpleNamespace(
                    stages=SimpleNamespace(summary_enabled=True),
                    chunking=SimpleNamespace(
                        summary_characters=80,
                        summary_batch_size=1,
                        summary_llm_provider="none",
                        summary_temperature=0.0,
                        summary_max_output_tokens=128,
                    ),
                ),
            )
            legacy_cfg = SimpleNamespace(
                index_dir=runtime.app.index_dir,
                sudachi_mode="B",
                sparse_use_normalized_form=True,
                sparse_remove_symbols=True,
                sparse_bm25_k1=1.5,
                sparse_bm25_b=0.75,
            )
            service = IndexingService(
                storage=object(),
                embedder=object(),
                faiss_index=object(),
                bm25_index=object(),
                ingestion_dir=root / "data" / "ingestion",
                app_config=runtime,
            )
            source_chunk = Chunk(
                id="chunk-1",
                document_id="doc-1",
                text="KUMCの例会は土曜日に開催します。",
                index=0,
                metadata={
                    "source_type": "google_drive",
                    "source_kind": "google_drive",
                    "external_id": "file-1",
                    "source_title": "例会案内",
                    "access_scope": {"visibility": "admin"},
                },
            )

            artifacts = service._build_repository_index_artifacts(  # noqa: SLF001
                repository_chunks=[source_chunk],
                legacy_cfg=legacy_cfg,
                selected=set(),
            )
            service._build_material_catalog_from_repository_chunks(  # noqa: SLF001
                chunks=artifacts.first_chunks,
            )
            service._build_keyword_inverted_indexes_from_repository_artifacts(  # noqa: SLF001
                artifacts=artifacts,
                legacy_cfg=legacy_cfg,
            )
            service._build_material_name_keyword_index(legacy_cfg=legacy_cfg)  # noqa: SLF001

            second_payloads = [
                json.loads(line)
                for line in (
                    root / "data" / "chunks" / "second_rec_chunk" / "google_drive.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(second_payloads[0]["metadata"]["source_file_name"], "file-1")
            self.assertEqual(second_payloads[0]["metadata"]["chunk_stage"], "second_recursive")
            self.assertTrue(
                (root / "data" / "index" / "keyword" / "sparse_second_rec.json").exists()
            )
            catalog = json.loads(
                (root / "data" / "index" / "material_catalog.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(catalog["materials"][0]["source_type"], "google_drive")
            self.assertEqual(catalog["materials"][0]["source_key"], "file-1")
            self.assertTrue(Path(catalog["materials"][0]["raw_path"]).exists())
            self.assertTrue(
                (root / "data" / "index" / "keyword" / "material_names.json").exists()
            )

    def test_unsearchable_repository_parent_excludes_derived_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = SimpleNamespace(
                app=SimpleNamespace(
                    data_dir=root / "data",
                    index_dir=root / "data" / "index",
                ),
                indexing=SimpleNamespace(
                    stages=SimpleNamespace(summary_enabled=True),
                    chunking=SimpleNamespace(
                        summary_characters=80,
                        summary_batch_size=1,
                        summary_llm_provider="gemini",
                        summary_temperature=0.0,
                        summary_max_output_tokens=128,
                    ),
                ),
            )
            legacy_cfg = SimpleNamespace(
                index_dir=runtime.app.index_dir,
                sudachi_mode="B",
                sparse_use_normalized_form=True,
                sparse_remove_symbols=True,
                sparse_bm25_k1=1.5,
                sparse_bm25_b=0.75,
            )
            service = IndexingService(
                storage=object(),
                embedder=object(),
                faiss_index=object(),
                bm25_index=object(),
                ingestion_dir=root / "data" / "ingestion",
                app_config=runtime,
                summary_llm=_StubLLM(
                    response_text='{"searchable": false, "summary": "", "reason": "noise"}'
                ),
            )
            source_chunk = Chunk(
                id="chunk-1",
                document_id="doc-1",
                text="1",
                index=0,
                metadata={
                    "source_type": "docs",
                    "source_kind": "google_drive",
                    "drive_file_id": "file-1",
                    "external_id": "file-1",
                    "source_title": "ノイズ",
                },
            )

            artifacts = service._build_repository_index_artifacts(  # noqa: SLF001
                repository_chunks=[source_chunk],
                legacy_cfg=legacy_cfg,
                selected=set(),
            )

            self.assertEqual(artifacts.first_chunks, [])
            self.assertEqual(artifacts.second_chunks, [])
            self.assertEqual(artifacts.sparse_chunks, [])
            self.assertEqual(artifacts.summary_chunks, [])
            self.assertEqual(artifacts.index_chunks, [])

    def test_repository_notion_chunks_write_source_directory_per_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = SimpleNamespace(
                app=SimpleNamespace(
                    data_dir=root / "data",
                    index_dir=root / "data" / "index",
                ),
                indexing=SimpleNamespace(
                    stages=SimpleNamespace(summary_enabled=False),
                    chunking=SimpleNamespace(
                        summary_characters=80,
                        summary_batch_size=1,
                        summary_llm_provider="none",
                        summary_temperature=0.0,
                        summary_max_output_tokens=128,
                    ),
                ),
            )
            legacy_cfg = SimpleNamespace(
                index_dir=runtime.app.index_dir,
                sudachi_mode="B",
                sparse_use_normalized_form=True,
                sparse_remove_symbols=True,
                sparse_bm25_k1=1.5,
                sparse_bm25_b=0.75,
            )
            service = IndexingService(
                storage=object(),
                embedder=object(),
                faiss_index=object(),
                bm25_index=object(),
                ingestion_dir=root / "data" / "ingestion",
                app_config=runtime,
            )
            page_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            source_chunk = Chunk(
                id="notion-chunk-1",
                document_id="notion-doc-1",
                text="Notion page body",
                index=0,
                metadata={
                    "source_type": "notion",
                    "source_kind": "notion",
                    "external_id": page_id,
                    "notion_page_id": page_id,
                    "notion_title": "Page A",
                    "notion_page_path": "Root / Page A",
                    "access_scope": {"visibility": "public"},
                },
            )

            artifacts = service._build_repository_index_artifacts(  # noqa: SLF001
                repository_chunks=[source_chunk],
                legacy_cfg=legacy_cfg,
                selected=set(),
            )

            self.assertEqual(len(artifacts.second_chunks), 1)
            self.assertTrue(
                (
                    root
                    / "data"
                    / "chunks"
                    / "second_rec_chunk"
                    / "notion"
                    / f"{page_id}.jsonl"
                ).exists()
            )
            self.assertFalse(
                (root / "data" / "chunks" / "second_rec_chunk" / "notion.jsonl").exists()
            )


if __name__ == "__main__":
    unittest.main()
