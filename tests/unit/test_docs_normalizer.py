from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.infra.indexing.docs_normalizer import (
    build_docs_quality_metadata,
    normalize_pdf_pages,
    normalize_pptx_slides,
)


class DocsNormalizerTests(unittest.TestCase):
    def test_pdf_page_records_keep_page_metadata_and_quarantine_page_number_only(self) -> None:
        base_metadata = build_docs_quality_metadata(
            base_metadata={
                "drive_file_id": "file-id",
                "drive_file_name": "回路講座.pdf",
                "drive_path": "docs/2026/04: 資料/回路講座.pdf",
                "drive_mime_type": "application/pdf",
                "drive_modified_time": "2026-04-29T00:00:00Z",
            },
            text="## Page 1\n40\n\n## Page 2\nレッドストーン回路の説明です。",
            extraction_method="pdf_text_ocr",
            page_count=2,
            min_nonempty_characters=20,
        )

        chunks = normalize_pdf_pages(
            [
                {
                    "page_number": 1,
                    "text": "40",
                    "ocr_status": "skipped_model_unavailable",
                    "quality_flags": ["page_number_only"],
                },
                {
                    "page_number": 2,
                    "text": "レッドストーン回路の説明です。",
                    "ocr_status": "not_needed",
                    "quality_flags": [],
                },
            ],
            base_metadata=base_metadata,
            source_file_name="file.md",
            min_nonempty_characters=10,
        )

        self.assertEqual(2, len(chunks))
        self.assertEqual(1, chunks[0].metadata["page_number"])
        self.assertEqual("quarantined", chunks[0].metadata["index_status"])
        self.assertIn("page_number_only", chunks[0].metadata["quality_flags"])
        self.assertEqual(2, chunks[1].metadata["page_number"])
        self.assertEqual("active", chunks[1].metadata["index_status"])

    def test_slide_records_keep_embedded_image_refs(self) -> None:
        chunks = normalize_pptx_slides(
            [
                {
                    "slide_number": 3,
                    "text": "企画概要",
                    "speaker_notes": "補足説明",
                    "embedded_image_refs": ["ppt/media/image1.png"],
                }
            ],
            base_metadata={
                "drive_file_id": "slide-id",
                "drive_file_name": "企画.pptx",
                "source_date": "2026/04/29",
                "index_status": "active",
            },
            source_file_name="slide.md",
            min_nonempty_characters=1,
        )

        self.assertEqual(1, len(chunks))
        self.assertEqual(3, chunks[0].metadata["slide_number"])
        self.assertEqual(["ppt/media/image1.png"], chunks[0].metadata["embedded_image_refs"])
        self.assertIn("Speaker notes", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
