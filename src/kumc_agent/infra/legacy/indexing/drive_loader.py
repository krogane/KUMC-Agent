from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from csv import writer
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from kumc_agent.infra.legacy.indexing.constants import (
    DRIVE_DOC_MIME,
    DRIVE_EXCEL_MIMES,
    DRIVE_PDF_MIME,
    DRIVE_POWERPOINT_MIMES,
    DRIVE_SHEET_MIME,
    DRIVE_SLIDE_MIME,
    DRIVE_WORD_MIMES,
    FILE_ID_SEPARATOR,
    GOOGLE_SCOPES,
)
from kumc_agent.infra.legacy.indexing.utils import ensure_dir, sanitize_filename

logger = logging.getLogger(__name__)

_DRIVE_IMAGE_PLACEHOLDER_RE = re.compile(r"\[image\d+\]:\s+[^>\n]*>")
_OFFICE_MIMES = (
    set(DRIVE_WORD_MIMES) | set(DRIVE_EXCEL_MIMES) | set(DRIVE_POWERPOINT_MIMES)
)
_SUPPORTED_DRIVE_MIMES = {
    DRIVE_DOC_MIME,
    DRIVE_SHEET_MIME,
    DRIVE_SLIDE_MIME,
    DRIVE_PDF_MIME,
    *_OFFICE_MIMES,
}


@dataclass(frozen=True)
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    path: str
    modified_time: str


def _extract_drive_file_id(filename: str) -> str | None:
    if FILE_ID_SEPARATOR not in filename:
        return None
    prefix, _ = filename.split(FILE_ID_SEPARATOR, 1)
    return prefix or None


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

                if mime_type in _SUPPORTED_DRIVE_MIMES:
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


def _download_file_bytes(service: Any, *, file_id: str) -> bytes:
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as e:
        raise RuntimeError(
            "google-api-python-client is required to download Drive files."
        ) from e

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def _excel_column_index(cell_ref: str) -> int | None:
    match = re.match(r"^([A-Za-z]+)\d+$", cell_ref)
    if not match:
        return None
    column = match.group(1).upper()
    index = 0
    for char in column:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    shared_path = "xl/sharedStrings.xml"
    if shared_path not in archive.namelist():
        return []

    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(archive.read(shared_path))
    values: list[str] = []
    for item in root.findall(".//s:si", ns):
        text = "".join(node.text or "" for node in item.findall(".//s:t", ns))
        values.append(text)
    return values


def _sheet_xml_sort_key(path: str) -> int:
    match = re.search(r"sheet(\d+)\.xml$", path)
    if not match:
        return 0
    return int(match.group(1))


def _xlsx_cell_value(
    cell: ET.Element, *, shared_strings: list[str], ns: dict[str, str]
) -> str:
    cell_type = cell.get("t", "")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//s:t", ns))

    value_node = cell.find("s:v", ns)
    value = value_node.text if value_node is not None and value_node.text else ""
    if not value:
        return ""

    if cell_type == "s":
        try:
            idx = int(value)
        except ValueError:
            return value
        return shared_strings[idx] if 0 <= idx < len(shared_strings) else value

    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"

    return value


def _extract_xlsx_csv_text(xlsx_bytes: bytes) -> str:
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        workbook_path = "xl/workbook.xml"
        sheet_names: list[str] = []
        if workbook_path in archive.namelist():
            workbook_root = ET.fromstring(archive.read(workbook_path))
            for sheet in workbook_root.findall(".//s:sheet", ns):
                name = sheet.get("name")
                if name:
                    sheet_names.append(name)

        sheet_xml_paths = sorted(
            (
                path
                for path in archive.namelist()
                if re.match(r"^xl/worksheets/sheet\d+\.xml$", path)
            ),
            key=_sheet_xml_sort_key,
        )
        if not sheet_xml_paths:
            return ""

        sections: list[str] = []
        for idx, sheet_path in enumerate(sheet_xml_paths):
            sheet_root = ET.fromstring(archive.read(sheet_path))
            sheet_name = (
                sheet_names[idx]
                if idx < len(sheet_names)
                else f"Sheet{idx + 1}"
            )
            rows: list[list[str]] = []
            for row in sheet_root.findall(".//s:sheetData/s:row", ns):
                row_values: dict[int, str] = {}
                max_col = 0
                fallback_col = 1
                for cell in row.findall("s:c", ns):
                    cell_ref = cell.get("r", "")
                    col_idx = _excel_column_index(cell_ref) or fallback_col
                    row_values[col_idx] = _xlsx_cell_value(
                        cell, shared_strings=shared_strings, ns=ns
                    )
                    if col_idx > max_col:
                        max_col = col_idx
                    fallback_col = col_idx + 1

                if max_col == 0:
                    continue
                rows.append([row_values.get(i, "") for i in range(1, max_col + 1)])

            sheet_buffer = io.StringIO()
            csv_writer = writer(sheet_buffer)
            for row in rows:
                csv_writer.writerow(row)
            csv_body = sheet_buffer.getvalue().rstrip()
            if csv_body:
                sections.append(f"# sheet: {sheet_name}\n{csv_body}")
            else:
                sections.append(f"# sheet: {sheet_name}")

        return "\n\n".join(sections).strip() + "\n"


def _extract_docx_text(docx_bytes: bytes) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        xml_paths = sorted(
            path
            for path in archive.namelist()
            if re.match(r"^word/(document|header\d+|footer\d+)\.xml$", path)
        )
        lines: list[str] = []
        for xml_path in xml_paths:
            root = ET.fromstring(archive.read(xml_path))
            for para in root.findall(".//w:p", ns):
                text = "".join(node.text or "" for node in para.findall(".//w:t", ns))
                text = text.strip()
                if text:
                    lines.append(text)
        return "\n\n".join(lines).strip() + "\n" if lines else ""


def _extract_pptx_text(pptx_bytes: bytes) -> str:
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(pptx_bytes)) as archive:
        slide_paths = sorted(
            path
            for path in archive.namelist()
            if re.match(r"^ppt/slides/slide\d+\.xml$", path)
        )
        sections: list[str] = []
        for idx, slide_path in enumerate(slide_paths, start=1):
            root = ET.fromstring(archive.read(slide_path))
            slide_lines = [
                node.text.strip()
                for node in root.findall(".//a:t", ns)
                if node.text and node.text.strip()
            ]
            if not slide_lines:
                continue
            sections.append(f"## Slide {idx}\n" + "\n".join(slide_lines))
        return "\n\n".join(sections).strip() + "\n" if sections else ""


@lru_cache(maxsize=2)
def _load_pdf_ocr_pipeline(model_path: str) -> Any:
    if not model_path:
        raise RuntimeError(
            "PDF_OCR_MODEL is empty. Set PDF_OCR_MODEL to a local OCR model path."
        )
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as e:
        raise RuntimeError(
            "transformers is required for OCR on image-based PDFs."
        ) from e

    last_error: Exception | None = None
    for task in ("image-text-to-text", "image-to-text"):
        try:
            return hf_pipeline(
                task=task,
                model=model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to load OCR model from {model_path}. "
        "Check PDF_OCR_MODEL and local model files."
    ) from last_error


def _extract_generated_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("generated_text", "text", "answer"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    if isinstance(payload, list):
        parts = [_extract_generated_text(item).strip() for item in payload]
        non_empty = [part for part in parts if part]
        return "\n".join(non_empty)
    return ""


def _extract_pdf_page_text_with_ocr(*, page: Any, ocr_model_path: str) -> str:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "PyMuPDF is required for PDF parsing."
        ) from e
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "Pillow is required for OCR image preprocessing."
        ) from e

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image_bytes = pix.tobytes("png")
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB")
    try:
        ocr_pipeline = _load_pdf_ocr_pipeline(ocr_model_path)
        try:
            result = ocr_pipeline(rgb_image, max_new_tokens=2048)
        except TypeError:
            result = ocr_pipeline(rgb_image)
        return _extract_generated_text(result).strip()
    finally:
        rgb_image.close()


def _extract_pdf_text(pdf_bytes: bytes, *, ocr_model_path: str) -> str:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "PyMuPDF is required for PDF parsing."
        ) from e

    sections: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_index, page in enumerate(document, start=1):
            page_text = (page.get_text("text") or "").strip()
            if not page_text:
                page_text = _extract_pdf_page_text_with_ocr(
                    page=page,
                    ocr_model_path=ocr_model_path,
                )
            if not page_text:
                continue
            sections.append(f"## Page {page_index}\n{page_text}")

    return "\n\n".join(sections).strip() + "\n" if sections else ""


def _is_doc_like_mime(mime_type: str) -> bool:
    return mime_type in {
        DRIVE_DOC_MIME,
        DRIVE_SLIDE_MIME,
        DRIVE_PDF_MIME,
        *DRIVE_WORD_MIMES,
        *DRIVE_POWERPOINT_MIMES,
    }


def _is_sheet_like_mime(mime_type: str) -> bool:
    return mime_type in {
        DRIVE_SHEET_MIME,
        *DRIVE_EXCEL_MIMES,
    }


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


def _cleanup_missing_drive_files(
    *,
    out_dir: Path,
    extension: str,
    valid_file_ids: set[str],
) -> None:
    for path in out_dir.glob(f"*{extension}"):
        file_id = _extract_drive_file_id(path.name)
        if not file_id:
            continue
        if file_id in valid_file_ids:
            continue
        try:
            path.unlink()
            logger.info("Removed deleted Drive export %s", path.name)
        except Exception as exc:
            logger.warning("Failed to remove deleted Drive export %s: %s", path.name, exc)
            continue

        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not meta_path.exists():
            continue
        try:
            meta_path.unlink()
            logger.info("Removed deleted Drive metadata %s", meta_path.name)
        except Exception as exc:
            logger.warning(
                "Failed to remove deleted Drive metadata %s: %s",
                meta_path.name,
                exc,
            )


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
    pdf_ocr_model_path: str,
    drive_max_files: int | None = None,
    skip_existing: bool = False,
    update_existing: bool = True,
    sync_deleted: bool = False,
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
        logger.warning(
            "No supported Google Drive files found under folder ID: %s",
            drive_folder_id,
        )
        return 0, 0
    if drive_max_files is not None and drive_max_files > 0:
        logger.info("Limiting Drive downloads to first %d files", drive_max_files)

    if sync_deleted:
        valid_doc_ids = {
            drive_file.file_id
            for drive_file in drive_files
            if _is_doc_like_mime(drive_file.mime_type)
        }
        valid_sheet_ids = {
            drive_file.file_id
            for drive_file in drive_files
            if _is_sheet_like_mime(drive_file.mime_type)
        }
        _cleanup_missing_drive_files(
            out_dir=docs_dir,
            extension=".md",
            valid_file_ids=valid_doc_ids,
        )
        _cleanup_missing_drive_files(
            out_dir=sheets_dir,
            extension=".csv",
            valid_file_ids=valid_sheet_ids,
        )

    docs_count = 0
    sheets_count = 0

    for drive_file in drive_files:
        try:
            if _is_doc_like_mime(drive_file.mime_type):
                out_path = docs_dir / _build_output_filename(
                    drive_file, extension=".md"
                )
                _cleanup_drive_duplicates(
                    out_dir=docs_dir, drive_file=drive_file, keep_path=out_path
                )
                if skip_existing and out_path.exists():
                    if not update_existing:
                        logger.info("Skip download (exists): %s", out_path.name)
                        continue
                    if _is_drive_file_up_to_date(out_path, drive_file):
                        logger.info("Skip download (up-to-date): %s", out_path.name)
                        continue
                if drive_file.mime_type == DRIVE_DOC_MIME:
                    content = _download_export_bytes(
                        drive_service,
                        file_id=drive_file.file_id,
                        mime_type="text/markdown",
                    )
                    text = content.decode("utf-8", errors="replace")
                    text = _strip_drive_image_placeholders(text)
                elif drive_file.mime_type == DRIVE_SLIDE_MIME:
                    content = _download_export_bytes(
                        drive_service,
                        file_id=drive_file.file_id,
                        mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                    text = _extract_pptx_text(content)
                elif drive_file.mime_type in DRIVE_WORD_MIMES:
                    if drive_file.mime_type == "application/msword":
                        logger.warning(
                            "Skip unsupported binary Word format (.doc): %s",
                            drive_file.path,
                        )
                        continue
                    content = _download_file_bytes(
                        drive_service, file_id=drive_file.file_id
                    )
                    text = _extract_docx_text(content)
                elif drive_file.mime_type in DRIVE_POWERPOINT_MIMES:
                    if drive_file.mime_type == "application/vnd.ms-powerpoint":
                        logger.warning(
                            "Skip unsupported binary PowerPoint format (.ppt): %s",
                            drive_file.path,
                        )
                        continue
                    content = _download_file_bytes(
                        drive_service, file_id=drive_file.file_id
                    )
                    text = _extract_pptx_text(content)
                elif drive_file.mime_type == DRIVE_PDF_MIME:
                    content = _download_file_bytes(
                        drive_service, file_id=drive_file.file_id
                    )
                    text = _extract_pdf_text(
                        content,
                        ocr_model_path=pdf_ocr_model_path,
                    )
                else:
                    continue
                if not text.strip():
                    logger.warning("Skip empty extracted text: %s", drive_file.path)
                    continue
                out_path.write_text(text, encoding="utf-8")
                _write_drive_metadata(out_path, drive_file)
                docs_count += 1
                logger.info("Downloaded doc-like file: %s", drive_file.path)
            elif _is_sheet_like_mime(drive_file.mime_type):
                out_path = sheets_dir / _build_output_filename(
                    drive_file, extension=".csv"
                )
                _cleanup_drive_duplicates(
                    out_dir=sheets_dir, drive_file=drive_file, keep_path=out_path
                )
                if skip_existing and out_path.exists():
                    if not update_existing:
                        logger.info("Skip download (exists): %s", out_path.name)
                        continue
                    if _is_drive_file_up_to_date(out_path, drive_file):
                        logger.info("Skip download (up-to-date): %s", out_path.name)
                        continue
                if drive_file.mime_type == DRIVE_SHEET_MIME:
                    csv_bytes = _download_export_bytes(
                        drive_service, file_id=drive_file.file_id, mime_type="text/csv"
                    )
                    csv_text = csv_bytes.decode("utf-8", errors="replace")
                elif drive_file.mime_type in DRIVE_EXCEL_MIMES:
                    if drive_file.mime_type == "application/vnd.ms-excel":
                        logger.warning(
                            "Skip unsupported binary Excel format (.xls): %s",
                            drive_file.path,
                        )
                        continue
                    xlsx_bytes = _download_file_bytes(
                        drive_service, file_id=drive_file.file_id
                    )
                    csv_text = _extract_xlsx_csv_text(xlsx_bytes)
                else:
                    continue
                if not csv_text.strip():
                    logger.warning("Skip empty extracted sheet data: %s", drive_file.path)
                    continue
                out_path.write_text(csv_text, encoding="utf-8")
                _write_drive_metadata(out_path, drive_file)
                sheets_count += 1
                logger.info("Downloaded sheet-like file: %s", drive_file.path)
        except Exception:
            logger.exception(
                "Failed to download %s (%s)", drive_file.path, drive_file.file_id
            )

    logger.info(
        "Downloaded %d doc-like and %d sheet-like files",
        docs_count,
        sheets_count,
    )
    return docs_count, sheets_count
