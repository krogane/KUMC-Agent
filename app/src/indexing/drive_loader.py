from __future__ import annotations

import io
import json
import logging
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
    modified_time: str


def _build_google_credentials(
    *, application_credentials: str
) -> Any:
    try:
        import google.auth
        from google.oauth2.service_account import Credentials
    except ImportError as e:
        raise RuntimeError(
            "google-auth is required for Google API credentials."
        ) from e

    sa_path = application_credentials
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


def _list_drive_files(
    service: Any, drive_folder_id: str, max_files: int | None = None
) -> list[DriveFile]:
    files: list[DriveFile] = []
    limit = max_files if max_files is not None and max_files > 0 else None
    stack: list[tuple[str, str]] = [(drive_folder_id, "")]

    while stack:
        current_id, current_path = stack.pop()
        page_token: str | None = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{current_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
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
                modified_time = item.get("modifiedTime", "")
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
                            modified_time=modified_time,
                        )
                    )
                    if limit is not None and len(files) >= limit:
                        return files

            page_token = response.get("nextPageToken")
            if not page_token:
                break
            if limit is not None and len(files) >= limit:
                return files

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


def _read_drive_metadata(out_path: Path) -> dict[str, str]:
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read metadata sidecar %s: %s", meta_path.name, exc)
        return {}

    if not isinstance(data, dict):
        logger.warning("Invalid metadata sidecar %s: expected object", meta_path.name)
        return {}

    metadata: dict[str, str] = {}
    for key in (
        "drive_file_id",
        "drive_file_name",
        "drive_path",
        "drive_mime_type",
        "drive_modified_time",
    ):
        value = data.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value
    return metadata


def _is_drive_file_up_to_date(out_path: Path, drive_file: DriveFile) -> bool:
    if not out_path.exists():
        return False

    metadata = _read_drive_metadata(out_path)
    if not metadata:
        return False
    if metadata.get("drive_file_id") != drive_file.file_id:
        return False

    stored_modified_time = metadata.get("drive_modified_time")
    if not stored_modified_time:
        return False
    return stored_modified_time == drive_file.modified_time


def _cleanup_drive_duplicates(
    *, out_dir: Path, drive_file: DriveFile, keep_path: Path
) -> None:
    prefix = f"{drive_file.file_id}{FILE_ID_SEPARATOR}"
    keep_meta = keep_path.with_suffix(keep_path.suffix + ".meta.json")
    for path in out_dir.glob(f"{prefix}*"):
        if path == keep_path or path == keep_meta:
            continue
        if path.is_dir():
            logger.warning("Skip unexpected directory in Drive export cleanup: %s", path)
            continue
        try:
            path.unlink()
            logger.info("Removed stale Drive export %s", path.name)
        except Exception as exc:
            logger.warning("Failed to remove stale Drive export %s: %s", path.name, exc)


def _write_drive_metadata(out_path: Path, drive_file: DriveFile) -> None:
    metadata = {
        "drive_file_id": drive_file.file_id,
        "drive_file_name": drive_file.name,
        "drive_path": drive_file.path,
        "drive_mime_type": drive_file.mime_type,
        "drive_modified_time": drive_file.modified_time,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def _strip_drive_image_placeholders(text: str) -> str:
    cleaned, _ = _DRIVE_IMAGE_PLACEHOLDER_RE.subn("", text)
    return cleaned


def download_drive_markdown(
    *,
    drive_folder_id: str,
    docs_dir: Path,
    sheets_dir: Path,
    google_application_credentials: str,
    drive_max_files: int | None = None,
    skip_existing: bool = False,
) -> tuple[int, int]:
    ensure_dir(docs_dir)
    ensure_dir(sheets_dir)

    creds = _build_google_credentials(
        application_credentials=google_application_credentials,
    )
    drive_service = _build_drive_service(creds)

    drive_files = _list_drive_files(
        drive_service, drive_folder_id, max_files=drive_max_files
    )
    if not drive_files:
        logger.warning("No Google Docs/Sheets files found under folder ID: %s", drive_folder_id)
        return 0, 0
    if drive_max_files is not None and drive_max_files > 0:
        logger.info("Limiting Drive downloads to first %d files", drive_max_files)

    docs_count = 0
    sheets_count = 0

    for drive_file in drive_files:
        try:
            if drive_file.mime_type == DRIVE_DOC_MIME:
                out_path = docs_dir / _build_output_filename(
                    drive_file, extension=".md"
                )
                _cleanup_drive_duplicates(
                    out_dir=docs_dir, drive_file=drive_file, keep_path=out_path
                )
                if skip_existing and _is_drive_file_up_to_date(out_path, drive_file):
                    logger.info("Skip download (up-to-date): %s", out_path.name)
                    continue
                content = _download_export_bytes(
                    drive_service, file_id=drive_file.file_id, mime_type="text/markdown"
                )
                text = content.decode("utf-8", errors="replace")
                text = _strip_drive_image_placeholders(text)
                out_path.write_text(text, encoding="utf-8")
                _write_drive_metadata(out_path, drive_file)
                docs_count += 1
                logger.info("Downloaded doc: %s", drive_file.path)
            elif drive_file.mime_type == DRIVE_SHEET_MIME:
                out_path = sheets_dir / _build_output_filename(
                    drive_file, extension=".csv"
                )
                _cleanup_drive_duplicates(
                    out_dir=sheets_dir, drive_file=drive_file, keep_path=out_path
                )
                if skip_existing and _is_drive_file_up_to_date(out_path, drive_file):
                    logger.info("Skip download (up-to-date): %s", out_path.name)
                    continue
                csv_bytes = _download_export_bytes(
                    drive_service, file_id=drive_file.file_id, mime_type="text/csv"
                )
                csv_text = csv_bytes.decode("utf-8", errors="replace")
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
