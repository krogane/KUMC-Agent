from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.infra.indexing.chunking import docs_chunk_dir
from kumc_agent.infra.indexing.chunks import Chunk, load_chunks, write_chunks


class DocsChunkingTests(unittest.TestCase):
    def test_docs_chunk_dir_prefers_normalized_records_and_skips_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "docs"
            normalized_dir = root / "docs_normalized"
            chunk_dir = root / "chunks"
            raw_dir.mkdir()
            normalized_dir.mkdir()
            raw_path = raw_dir / "file-id__sample.md"
            raw_path.write_text("raw text that should not be used", encoding="utf-8")
            raw_path.with_suffix(".md.meta.json").write_text(
                json.dumps(
                    {
                        "drive_file_id": "file-id",
                        "drive_file_name": "sample.pdf",
                        "drive_path": "sample.pdf",
                        "drive_mime_type": "application/pdf",
                        "drive_modified_time": "2026-04-29T00:00:00Z",
                    }
                ),
                encoding="utf-8",
            )
            normalized_path = normalized_dir / "file-id__sample.jsonl"
            write_chunks(
                normalized_path,
                [
                    Chunk(
                        text="本文として使うページ",
                        metadata={
                            "source_type": "docs",
                            "source_file_name": raw_path.name,
                            "drive_file_id": "file-id",
                            "page_number": 2,
                            "index_status": "active",
                        },
                    ),
                    Chunk(
                        text="40",
                        metadata={
                            "source_type": "docs",
                            "source_file_name": raw_path.name,
                            "drive_file_id": "file-id",
                            "page_number": 1,
                            "index_status": "quarantined",
                        },
                    ),
                ],
            )

            docs_chunk_dir(
                ingestion_data_dir=raw_dir,
                structured_data_dir=normalized_dir,
                chunk_dir=chunk_dir,
                chunk_size=100,
                chunk_overlap=0,
                separators=("\n\n", "\n", " ", ""),
                stage="first_recursive",
            )

            chunks = load_chunks(chunk_dir / normalized_path.name)
            self.assertEqual(1, len(chunks))
            self.assertEqual("本文として使うページ", chunks[0].text)
            self.assertEqual(2, chunks[0].metadata["page_number"])
            self.assertEqual("first_recursive", chunks[0].metadata["chunk_stage"])


if __name__ == "__main__":
    unittest.main()
