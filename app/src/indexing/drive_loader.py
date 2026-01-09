from __future__ import annotations

import io
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indexing.constants import (
    DRIVE_DOC_MIME,
    DRIVE_SHEET_MIME,
    FILE_ID_SEPARATOR,
    GOOGLE_SCOPES,
)
from indexing.utils import ensure_dir, sanitize_filename

logger = logging.getLogger(__name__)

_DRIVE_IMAGE_PLACEHOLDER_RE = re.compile(r"\[image\d+\]:\s+[^>\n]*>")


@dataclass(frozen=True)
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    path: str


def _build_google_credentials() -> Any:
    try:
        import google.auth
        from google.oauth2.service_account import Credentials
    except ImportError as e:
        raise RuntimeError(
            "google-auth is required for Google API credentials."
        ) from e

    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_FILE"
    )
    if sa_path:
        return Credentials.from_service_account_file(sa_path, scopes=GOOGLE_SCOPES)

    creds, _ = google.auth.default(scopes=GOOGLE_SCOPES)
    return creds


def _build_drive_service(creds: Any) -> Any:
    try:
        from googleapiclient.discovery import build
    except ImportError as e:
        raise RuntimeError(
            "google-api-python-client is required for Google Drive access."
        ) from e

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_drive_files(service: Any, folder_id: str) -> list[DriveFile]:
    files: list[DriveFile] = []
    stack: list[tuple[str, str]] = [(folder_id, "")]

    while stack:
        current_id, current_path = stack.pop()
        page_token: str | None = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{current_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                )
                .execute()
            )

            for item in response.get("files", []):
                mime_type = item.get("mimeType", "")
                name = item.get("name", "")
                file_id = item.get("id", "")
                if not file_id or not name:
                    continue

                if mime_type == "application/vnd.google-apps.folder":
                    next_path = f"{current_path}/{name}" if current_path else name
                    stack.append((file_id, next_path))
                    continue

                if mime_type in (DRIVE_DOC_MIME, DRIVE_SHEET_MIME):
                    file_path = f"{current_path}/{name}" if current_path else name
                    files.append(
                        DriveFile(
                            file_id=file_id,
                            name=name,
                            mime_type=mime_type,
                            path=file_path,
                        )
                    )

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return files


def _download_export_bytes(service: Any, *, file_id: str, mime_type: str) -> bytes:
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as e:
        raise RuntimeError(
            "google-api-python-client is required to download Drive files."
        ) from e

    request = service.files().export_media(
        fileId=file_id, mimeType=mime_type,
    )
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def _build_output_filename(drive_file: DriveFile, *, extension: str) -> str:
    safe_path = sanitize_filename(drive_file.path.replace("/", "__"))
    return f"{drive_file.file_id}{FILE_ID_SEPARATOR}{safe_path}{extension}"


def _write_drive_metadata(out_path: Path, drive_file: DriveFile) -> None:
    metadata = {
        "drive_file_id": drive_file.file_id,
        "drive_file_name": drive_file.name,
        "drive_path": drive_file.path,
        "drive_mime_type": drive_file.mime_type,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def _strip_drive_image_placeholders(text: str) -> str:
    cleaned, _ = _DRIVE_IMAGE_PLACEHOLDER_RE.subn("", text)
    return cleaned


def download_drive_markdown(
    *,
    folder_id: str,
    docs_dir: Path,
    sheets_dir: Path,
) -> tuple[int, int]:
    ensure_dir(docs_dir)
    ensure_dir(sheets_dir)

    creds = _build_google_credentials()
    drive_service = _build_drive_service(creds)

    drive_files = _list_drive_files(drive_service, folder_id)
    if not drive_files:
        logger.warning("No Google Docs/Sheets files found under folder ID: %s", folder_id)
        return 0, 0

    docs_count = 0
    sheets_count = 0

    for drive_file in drive_files:
        try:
            if drive_file.mime_type == DRIVE_DOC_MIME:
                content = _download_export_bytes(
                    drive_service, file_id=drive_file.file_id, mime_type="text/markdown"
                )
                text = content.decode("utf-8", errors="replace")
                text = _strip_drive_image_placeholders(text)
                out_path = docs_dir / _build_output_filename(drive_file, extension=".md")
                out_path.write_text(text, encoding="utf-8")
                _write_drive_metadata(out_path, drive_file)
                docs_count += 1
                logger.info("Downloaded doc: %s", drive_file.path)
            elif drive_file.mime_type == DRIVE_SHEET_MIME:
                csv_bytes = _download_export_bytes(
                    drive_service, file_id=drive_file.file_id, mime_type="text/csv"
                )
                csv_text = csv_bytes.decode("utf-8", errors="replace")
                out_path = sheets_dir / _build_output_filename(drive_file, extension=".csv")
                out_path.write_text(csv_text, encoding="utf-8")
                _write_drive_metadata(out_path, drive_file)
                sheets_count += 1
                logger.info("Downloaded sheet: %s", drive_file.path)
        except Exception:
            logger.exception(
                "Failed to download %s (%s)", drive_file.path, drive_file.file_id
            )

    logger.info("Downloaded %d docs and %d sheets", docs_count, sheets_count)
    return docs_count, sheets_count
