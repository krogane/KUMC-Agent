from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import kumc_agent.infra.loaders.google_drive_impl as drive_impl
from kumc_agent.infra.loaders.common import DRIVE_SLIDE_MIME

_SLIDES_PPTX_EXPORT_MIME = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)


class _FakeResp:
    def __init__(self, status: int) -> None:
        self.status = status


class _FakeHttpError(Exception):
    def __init__(self, *, status: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.resp = _FakeResp(status)
        self.content = json.dumps(
            {
                "error": {
                    "errors": [
                        {
                            "reason": reason,
                            "message": message,
                        }
                    ]
                }
            }
        ).encode("utf-8")


def _slide_file() -> drive_impl.DriveFile:
    return drive_impl.DriveFile(
        file_id="slide-file-id",
        name="sample-slides",
        mime_type=DRIVE_SLIDE_MIME,
        path="folder/sample-slides",
        modified_time="2026-03-09T19:53:54.822Z",
    )


class GoogleDriveSlidesFallbackTests(unittest.TestCase):
    def test_slides_export_size_limit_falls_back_to_text_plain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp) / "docs"
            sheets_dir = Path(tmp) / "sheets"
            docs_dir.mkdir(parents=True, exist_ok=True)
            sheets_dir.mkdir(parents=True, exist_ok=True)

            requested_mimes: list[str] = []

            def _fake_download_export_bytes(
                service: object,
                *,
                file_id: str,
                mime_type: str,
                max_retries: int,
                initial_delay_seconds: float,
                max_delay_seconds: float,
                backoff_multiplier: float,
            ) -> bytes:
                del service
                del file_id
                del max_retries
                del initial_delay_seconds
                del max_delay_seconds
                del backoff_multiplier
                requested_mimes.append(mime_type)
                if mime_type == _SLIDES_PPTX_EXPORT_MIME:
                    raise _FakeHttpError(
                        status=403,
                        reason="exportSizeLimitExceeded",
                        message="This file is too large to be exported.",
                    )
                if mime_type == "text/plain":
                    return b"fallback slide text"
                raise AssertionError(f"Unexpected export mime_type: {mime_type}")

            with (
                patch.object(
                    drive_impl, "_build_google_credentials", return_value=object()
                ),
                patch.object(drive_impl, "_build_drive_service", return_value=object()),
                patch.object(drive_impl, "_list_drive_files", return_value=[_slide_file()]),
                patch.object(
                    drive_impl,
                    "_download_export_bytes",
                    side_effect=_fake_download_export_bytes,
                ),
            ):
                docs_count, sheets_count = drive_impl.download_drive_markdown(
                    drive_folder_id="folder-id",
                    docs_dir=docs_dir,
                    sheets_dir=sheets_dir,
                    google_application_credentials="",
                    pdf_ocr_model_path="",
                    sync_deleted=False,
                )

            self.assertEqual(
                [_SLIDES_PPTX_EXPORT_MIME, "text/plain"],
                requested_mimes,
            )
            self.assertEqual(1, docs_count)
            self.assertEqual(0, sheets_count)
            outputs = list(docs_dir.glob("*.md"))
            self.assertEqual(1, len(outputs))
            self.assertEqual(
                "fallback slide text",
                outputs[0].read_text(encoding="utf-8"),
            )

    def test_non_size_limit_error_does_not_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp) / "docs"
            sheets_dir = Path(tmp) / "sheets"
            docs_dir.mkdir(parents=True, exist_ok=True)
            sheets_dir.mkdir(parents=True, exist_ok=True)

            requested_mimes: list[str] = []

            def _fake_download_export_bytes(
                service: object,
                *,
                file_id: str,
                mime_type: str,
                max_retries: int,
                initial_delay_seconds: float,
                max_delay_seconds: float,
                backoff_multiplier: float,
            ) -> bytes:
                del service
                del file_id
                del max_retries
                del initial_delay_seconds
                del max_delay_seconds
                del backoff_multiplier
                requested_mimes.append(mime_type)
                raise _FakeHttpError(
                    status=403,
                    reason="insufficientFilePermissions",
                    message="The user does not have sufficient permissions.",
                )

            with (
                patch.object(
                    drive_impl, "_build_google_credentials", return_value=object()
                ),
                patch.object(drive_impl, "_build_drive_service", return_value=object()),
                patch.object(drive_impl, "_list_drive_files", return_value=[_slide_file()]),
                patch.object(
                    drive_impl,
                    "_download_export_bytes",
                    side_effect=_fake_download_export_bytes,
                ),
            ):
                docs_count, sheets_count = drive_impl.download_drive_markdown(
                    drive_folder_id="folder-id",
                    docs_dir=docs_dir,
                    sheets_dir=sheets_dir,
                    google_application_credentials="",
                    pdf_ocr_model_path="",
                    sync_deleted=False,
                )

            self.assertEqual([_SLIDES_PPTX_EXPORT_MIME], requested_mimes)
            self.assertEqual(0, docs_count)
            self.assertEqual(0, sheets_count)
            self.assertEqual([], list(docs_dir.glob("*.md")))


if __name__ == "__main__":
    unittest.main()
