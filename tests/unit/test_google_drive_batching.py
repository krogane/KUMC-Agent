from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import kumc_agent.infra.loaders.google_drive_impl as drive_impl
from kumc_agent.infra.loaders.common import DRIVE_SLIDE_MIME
from kumc_agent.infra.loaders.google_drive_impl import DriveFile, _split_batches


def _drive_file(index: int) -> DriveFile:
    return DriveFile(
        file_id=f"id-{index}",
        name=f"file-{index}",
        mime_type="application/vnd.google-apps.document",
        path=f"path/file-{index}",
        modified_time="2026-03-08T00:00:00.000Z",
    )


class GoogleDriveBatchingTests(unittest.TestCase):
    def test_split_batches_uses_requested_batch_size(self) -> None:
        files = [_drive_file(i) for i in range(45)]
        batches = _split_batches(files, batch_size=20)
        self.assertEqual([20, 20, 5], [len(batch) for batch in batches])

    def test_split_batches_non_positive_is_single_batch(self) -> None:
        files = [_drive_file(i) for i in range(7)]
        batches = _split_batches(files, batch_size=0)
        self.assertEqual(1, len(batches))
        self.assertEqual(7, len(batches[0]))

    def test_split_batches_none_is_single_batch(self) -> None:
        files = [_drive_file(i) for i in range(3)]
        batches = _split_batches(files, batch_size=None)
        self.assertEqual(1, len(batches))
        self.assertEqual(3, len(batches[0]))

    def test_download_drive_markdown_saves_image_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs_dir = root / "docs"
            sheets_dir = root / "sheets"
            docs_dir.mkdir(parents=True)
            sheets_dir.mkdir(parents=True)
            image_file = DriveFile(
                file_id="image-id",
                name="poster.png",
                mime_type="image/png",
                path="folder/poster.png",
                modified_time="2026-03-08T00:00:00.000Z",
            )

            with (
                patch.object(drive_impl, "_build_google_credentials", return_value=object()),
                patch.object(drive_impl, "_build_drive_service", return_value=object()),
                patch.object(drive_impl, "_list_drive_files", return_value=[image_file]),
                patch.object(drive_impl, "_download_file_bytes", return_value=b"png-bytes"),
            ):
                docs_count, sheets_count = drive_impl.download_drive_markdown(
                    drive_folder_id="folder-id",
                    docs_dir=docs_dir,
                    sheets_dir=sheets_dir,
                    google_application_credentials="",
                    pdf_ocr_model_path="",
                    sync_deleted=True,
                )

            self.assertEqual(1, docs_count)
            self.assertEqual(0, sheets_count)
            image_outputs = list((root / "images" / "google_drive").glob("*.png"))
            self.assertEqual(1, len(image_outputs))
            self.assertEqual(b"png-bytes", image_outputs[0].read_bytes())
            metadata = json.loads(
                image_outputs[0].with_suffix(".png.meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual("image-id", metadata["drive_file_id"])

    def test_download_drive_markdown_extracts_pptx_media_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs_dir = root / "docs"
            sheets_dir = root / "sheets"
            docs_dir.mkdir(parents=True)
            sheets_dir.mkdir(parents=True)
            slide_file = DriveFile(
                file_id="slide-id",
                name="slides",
                mime_type=DRIVE_SLIDE_MIME,
                path="folder/slides",
                modified_time="2026-03-08T00:00:00.000Z",
            )

            with (
                patch.object(drive_impl, "_build_google_credentials", return_value=object()),
                patch.object(drive_impl, "_build_drive_service", return_value=object()),
                patch.object(drive_impl, "_list_drive_files", return_value=[slide_file]),
                patch.object(drive_impl, "_download_export_bytes", return_value=_pptx_bytes()),
            ):
                docs_count, sheets_count = drive_impl.download_drive_markdown(
                    drive_folder_id="folder-id",
                    docs_dir=docs_dir,
                    sheets_dir=sheets_dir,
                    google_application_credentials="",
                    pdf_ocr_model_path="",
                    sync_deleted=True,
                )

            self.assertEqual(1, docs_count)
            self.assertEqual(0, sheets_count)
            image_outputs = list((root / "images" / "google_drive").glob("*.png"))
            self.assertEqual(1, len(image_outputs))
            metadata = json.loads(
                image_outputs[0].with_suffix(".png.meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual("slide-id", metadata["drive_file_id"])
            self.assertEqual("ppt/media/image1.png", metadata["pptx_media_path"])
            self.assertIn("新歓スライド", metadata["surrounding_text"])

def _pptx_bytes() -> bytes:
    buffer = io.BytesIO()
    slide_xml = (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>新歓スライド</a:t></a:r></a:p>"
        "</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
    )
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide_xml)
        archive.writestr("ppt/media/image1.png", b"png-bytes")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
