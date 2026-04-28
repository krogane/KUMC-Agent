from __future__ import annotations

import importlib.util
import io
import inspect
import json
import logging
import mimetypes
import os
import re
import sys
import tempfile
import time
import types
import zipfile
from csv import writer
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

from kumc_agent.infra.loaders.common import (
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
from kumc_agent.infra.loaders.common import ensure_dir, sanitize_filename

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
_DRIVE_IMAGE_MIME_PREFIX = "image/"
_PP_OCR_V5_MOBILE_PREFIX = "PP-OCRv5_mobile"
_PP_OCR_V5_TEXTLINE_CLS_NAMES = (
    "PP-LCNet_x0_25_textline_ori",
    "PP-LCNet_x1_0_textline_ori",
)


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

                if mime_type in _SUPPORTED_DRIVE_MIMES or _is_drive_image_mime(mime_type):
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


def _split_batches(
    items: list[DriveFile], *, batch_size: int | None
) -> list[list[DriveFile]]:
    if not items:
        return []
    if batch_size is None:
        return [items]
    size = int(batch_size)
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _is_drive_image_mime(mime_type: str) -> bool:
    return str(mime_type or "").lower().startswith(_DRIVE_IMAGE_MIME_PREFIX)


def _download_export_bytes(
    service: Any,
    *,
    file_id: str,
    mime_type: str,
    max_retries: int,
    initial_delay_seconds: float,
    max_delay_seconds: float,
    backoff_multiplier: float,
) -> bytes:
    return _download_drive_bytes_with_retry(
        service,
        file_id=file_id,
        operation_name=f"export({mime_type})",
        request_builder=lambda files_api: files_api.export_media(
            fileId=file_id,
            mimeType=mime_type,
        ),
        max_retries=max_retries,
        initial_delay_seconds=initial_delay_seconds,
        max_delay_seconds=max_delay_seconds,
        backoff_multiplier=backoff_multiplier,
    )


def _download_file_bytes(
    service: Any,
    *,
    file_id: str,
    max_retries: int,
    initial_delay_seconds: float,
    max_delay_seconds: float,
    backoff_multiplier: float,
) -> bytes:
    return _download_drive_bytes_with_retry(
        service,
        file_id=file_id,
        operation_name="get_media",
        request_builder=lambda files_api: files_api.get_media(
            fileId=file_id,
            supportsAllDrives=True,
        ),
        max_retries=max_retries,
        initial_delay_seconds=initial_delay_seconds,
        max_delay_seconds=max_delay_seconds,
        backoff_multiplier=backoff_multiplier,
    )


def _drive_error_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "resp", None)
    status = getattr(response, "status", None)
    return status if isinstance(status, int) else None


def _drive_error_payload(exc: Exception) -> dict[str, Any] | None:
    raw_content = getattr(exc, "content", None)
    if raw_content is None:
        return None
    if isinstance(raw_content, (bytes, bytearray)):
        content = raw_content.decode("utf-8", errors="replace")
    else:
        content = str(raw_content)
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _drive_error_reasons(exc: Exception) -> set[str]:
    payload = _drive_error_payload(exc)
    if payload is None:
        return set()
    error = payload.get("error")
    if not isinstance(error, dict):
        return set()

    reasons: set[str] = set()
    errors = error.get("errors")
    if isinstance(errors, list):
        for item in errors:
            if not isinstance(item, dict):
                continue
            reason = item.get("reason")
            if isinstance(reason, str) and reason:
                reasons.add(reason)
    reason = error.get("reason")
    if isinstance(reason, str) and reason:
        reasons.add(reason)
    return reasons


def _is_export_size_limit_exceeded(exc: Exception) -> bool:
    if _drive_error_status_code(exc) != 403:
        return False
    reasons = _drive_error_reasons(exc)
    if "exportSizeLimitExceeded" in reasons:
        return True
    message = str(exc).lower()
    return (
        "exportsizelimitexceeded" in message
        or "too large to be exported" in message
    )


def _is_retryable_drive_exception(
    exc: Exception,
    *,
    http_error_type: type[Any],
) -> bool:
    if isinstance(exc, http_error_type):
        status = _drive_error_status_code(exc)
        return status == 429 or (status is not None and status >= 500)
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def _download_drive_bytes_with_retry(
    service: Any,
    *,
    file_id: str,
    operation_name: str,
    request_builder: Callable[[Any], Any],
    max_retries: int = 3,
    initial_delay_seconds: float = 0.5,
    max_delay_seconds: float = 8.0,
    backoff_multiplier: float = 2.0,
) -> bytes:
    try:
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as e:
        raise RuntimeError(
            "google-api-python-client is required to download Drive files."
        ) from e

    retries = max(0, int(max_retries))
    delay = max(0.0, float(initial_delay_seconds))
    max_delay = max(0.0, float(max_delay_seconds))
    multiplier = max(1.0, float(backoff_multiplier))
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        request = request_builder(service.files())
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        try:
            while not done:
                _, done = downloader.next_chunk()
            if attempt > 1:
                logger.info(
                    "Drive %s succeeded on retry for file %s (attempt %d/%d).",
                    operation_name,
                    file_id,
                    attempt,
                    attempts,
                )
            return buffer.getvalue()
        except Exception as exc:
            is_retryable = _is_retryable_drive_exception(
                exc,
                http_error_type=HttpError,
            )
            if not is_retryable or attempt >= attempts:
                raise
            status = _drive_error_status_code(exc)
            sleep_seconds = min(delay, max_delay) if max_delay > 0 else delay
            logger.warning(
                (
                    "Drive %s failed for file %s (status=%s, attempt %d/%d). "
                    "Retrying in %.2fs."
                ),
                operation_name,
                file_id,
                status,
                attempt,
                attempts,
                sleep_seconds,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if delay > 0:
                next_delay = delay * multiplier
                delay = min(next_delay, max_delay) if max_delay > 0 else next_delay

    raise RuntimeError(
        f"Unreachable retry loop while downloading Drive file {file_id}."
    )


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


def _extract_pptx_images(
    pptx_bytes: bytes,
    *,
    drive_file: DriveFile,
    images_dir: Path,
    surrounding_text: str,
) -> int:
    ensure_dir(images_dir)
    count = 0
    safe_path = sanitize_filename(drive_file.path.replace("/", "__"))
    with zipfile.ZipFile(io.BytesIO(pptx_bytes)) as archive:
        media_paths = sorted(
            path
            for path in archive.namelist()
            if path.startswith("ppt/media/")
            and Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
        )
        for index, media_path in enumerate(media_paths):
            data = archive.read(media_path)
            suffix = Path(media_path).suffix.lower() or ".img"
            out_path = images_dir / f"{drive_file.file_id}{FILE_ID_SEPARATOR}{safe_path}__ppt_media_{index}{suffix}"
            if not out_path.exists() or out_path.read_bytes() != data:
                out_path.write_bytes(data)
            metadata = _drive_metadata_payload(drive_file)
            metadata.update(
                {
                    "drive_embedded_source": "pptx_media",
                    "pptx_media_path": media_path,
                    "image_index": index,
                    "surrounding_text": surrounding_text,
                }
            )
            out_path.with_suffix(out_path.suffix + ".meta.json").write_text(
                json.dumps(metadata, ensure_ascii=False),
                encoding="utf-8",
            )
            count += 1
    return count


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _move_inputs_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        moved_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            moved_inputs[key] = value.to(device) if hasattr(value, "to") else value
        return moved_inputs
    return inputs


def _looks_like_pp_ocr_v5_mobile_path(model_path: str) -> bool:
    return _PP_OCR_V5_MOBILE_PREFIX.lower() in str(model_path or "").lower()


def _is_paddle_inference_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "inference.pdiparams").exists()
        and (path / "inference.yml").exists()
    )


def _ensure_paddle_cache_home() -> None:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if os.environ.get("PADDLE_PDX_CACHE_HOME"):
        return
    cache_dir = Path(tempfile.gettempdir()) / "kumc-agent-paddlex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_dir)
    os.environ.setdefault("PADDLE_HOME", str(cache_dir / "paddle"))


def _ensure_paddlex_langchain_compat() -> None:
    try:
        if importlib.util.find_spec("langchain.docstore.document") is not None:
            return
    except ModuleNotFoundError:
        pass
    try:
        import langchain
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return

    docstore_module = types.ModuleType("langchain.docstore")
    document_module = types.ModuleType("langchain.docstore.document")
    text_splitter_module = types.ModuleType("langchain.text_splitter")

    document_module.Document = Document
    text_splitter_module.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    docstore_module.document = document_module

    setattr(langchain, "docstore", docstore_module)
    setattr(langchain, "text_splitter", text_splitter_module)
    sys.modules["langchain.docstore"] = docstore_module
    sys.modules["langchain.docstore.document"] = document_module
    sys.modules["langchain.text_splitter"] = text_splitter_module


def _resolve_pp_ocr_v5_mobile_dirs(
    model_path: str,
) -> tuple[Path, Path, Path | None] | None:
    raw = Path(model_path).expanduser()
    det_suffix = f"{_PP_OCR_V5_MOBILE_PREFIX}_det"
    rec_suffix = f"{_PP_OCR_V5_MOBILE_PREFIX}_rec"

    candidate_pairs: list[tuple[Path, Path]] = []
    if raw.name == _PP_OCR_V5_MOBILE_PREFIX:
        candidate_pairs.extend(
            [
                (raw.with_name(det_suffix), raw.with_name(rec_suffix)),
                (raw / det_suffix, raw / rec_suffix),
            ]
        )
    if raw.name.endswith("_det"):
        candidate_pairs.append((raw, raw.with_name(raw.name[:-4] + "_rec")))
    if raw.name.endswith("_rec"):
        candidate_pairs.append((raw.with_name(raw.name[:-4] + "_det"), raw))
    if raw.is_dir():
        candidate_pairs.extend(
            [
                (raw / det_suffix, raw / rec_suffix),
                (raw.parent / det_suffix, raw.parent / rec_suffix),
            ]
        )
    candidate_pairs.append((Path(f"{model_path}_det"), Path(f"{model_path}_rec")))

    seen_pairs: set[tuple[str, str]] = set()
    for det_dir, rec_dir in candidate_pairs:
        key = (str(det_dir), str(rec_dir))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        if not (_is_paddle_inference_dir(det_dir) and _is_paddle_inference_dir(rec_dir)):
            continue

        cls_dir: Path | None = None
        parent_candidates = [det_dir.parent, rec_dir.parent, raw, raw.parent]
        checked: set[str] = set()
        for parent in parent_candidates:
            parent_key = str(parent)
            if parent_key in checked:
                continue
            checked.add(parent_key)
            for cls_name in _PP_OCR_V5_TEXTLINE_CLS_NAMES:
                maybe_cls_dir = parent / cls_name
                if _is_paddle_inference_dir(maybe_cls_dir):
                    cls_dir = maybe_cls_dir
                    break
            if cls_dir is not None:
                break

        return det_dir, rec_dir, cls_dir

    return None


class _PaddlePdfOcrRunner:
    def __init__(self, *, ocr: Any, use_cls: bool) -> None:
        self._ocr = ocr
        self._use_cls = use_cls

    def __call__(self, image: Any, max_new_tokens: int = 2048) -> list[dict[str, str]]:
        del max_new_tokens
        try:
            import numpy as np
        except ImportError as e:
            raise RuntimeError("numpy is required for PaddleOCR inference.") from e

        image_array = np.array(image)
        predict = getattr(self._ocr, "predict", None)
        if callable(predict):
            result = predict(image_array)
        elif hasattr(self._ocr, "ocr"):
            result = self._ocr.ocr(image_array, cls=self._use_cls)
        else:
            raise RuntimeError(
                "PaddleOCR runner does not expose `predict` or `ocr` methods."
            )

        return [{"generated_text": _extract_generated_text(result).strip()}]


class _DirectPdfOcrRunner:
    def __init__(self, *, model: Any, processor: Any, device: str) -> None:
        self._model = model
        self._processor = processor
        self._device = device
        image_processor = getattr(processor, "image_processor", None)
        self._min_pixels = getattr(image_processor, "min_pixels", None)
        self._max_pixels = 1280 * 28 * 28

    def __call__(self, image: Any, max_new_tokens: int = 2048) -> list[dict[str, str]]:
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("torch is required for direct OCR model inference.") from e

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ]

        apply_chat_template = getattr(self._processor, "apply_chat_template", None)
        if callable(apply_chat_template):
            kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            if isinstance(self._min_pixels, int) and self._min_pixels > 0:
                kwargs["images_kwargs"] = {
                    "size": {
                        "shortest_edge": self._min_pixels,
                        "longest_edge": self._max_pixels,
                    }
                }
            inputs = apply_chat_template(messages, **kwargs)
        else:
            inputs = self._processor(
                text=["OCR:"],
                images=[image],
                return_tensors="pt",
            )

        inputs = _move_inputs_to_device(inputs, self._device)

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = getattr(outputs, "sequences", outputs)
        input_ids = None
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
        else:
            input_ids = getattr(inputs, "input_ids", None)
        if (
            input_ids is not None
            and hasattr(generated_ids, "ndim")
            and generated_ids.ndim >= 2
            and generated_ids.shape[-1] > input_ids.shape[-1]
        ):
            generated_ids = generated_ids[:, input_ids.shape[-1] :]

        if hasattr(self._processor, "batch_decode"):
            decoded = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            text = decoded[0] if decoded else ""
        elif hasattr(self._processor, "decode"):
            first_item = generated_ids[0] if hasattr(generated_ids, "__getitem__") else generated_ids
            text = self._processor.decode(first_item, skip_special_tokens=True)
        else:
            text = ""
        return [{"generated_text": text}]


def _load_pdf_ocr_paddle_runner(model_path: str) -> Any:
    resolved_dirs = _resolve_pp_ocr_v5_mobile_dirs(model_path)
    if resolved_dirs is None:
        raise RuntimeError(
            "PP-OCRv5_mobile requires local det/rec model directories. "
            "Expected either '<path>_det + <path>_rec' or "
            "'<dir>/PP-OCRv5_mobile_det + <dir>/PP-OCRv5_mobile_rec'."
        )

    det_dir, rec_dir, cls_dir = resolved_dirs
    _ensure_paddle_cache_home()
    _ensure_paddlex_langchain_compat()
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise RuntimeError(
            "paddleocr is required for PP-OCRv5_mobile inference. "
            "Install paddleocr (and paddlepaddle for your environment)."
        ) from e

    signature = inspect.signature(PaddleOCR.__init__)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )

    def _accepts(name: str) -> bool:
        return accepts_var_kwargs or name in parameters

    use_modern_args = (
        accepts_var_kwargs
        or "text_detection_model_dir" in parameters
        or "text_recognition_model_dir" in parameters
    )

    kwargs: dict[str, Any] = {}
    if use_modern_args:
        if _accepts("text_detection_model_dir"):
            kwargs["text_detection_model_dir"] = str(det_dir)
        if _accepts("text_detection_model_name"):
            kwargs["text_detection_model_name"] = det_dir.name
        if _accepts("text_recognition_model_dir"):
            kwargs["text_recognition_model_dir"] = str(rec_dir)
        if _accepts("text_recognition_model_name"):
            kwargs["text_recognition_model_name"] = rec_dir.name
        if cls_dir is not None and _accepts("textline_orientation_model_dir"):
            kwargs["textline_orientation_model_dir"] = str(cls_dir)
        if cls_dir is not None and _accepts("textline_orientation_model_name"):
            kwargs["textline_orientation_model_name"] = cls_dir.name
    else:
        if _accepts("det_model_dir"):
            kwargs["det_model_dir"] = str(det_dir)
        if _accepts("rec_model_dir"):
            kwargs["rec_model_dir"] = str(rec_dir)
        if cls_dir is not None and _accepts("cls_model_dir"):
            kwargs["cls_model_dir"] = str(cls_dir)

    if _accepts("use_doc_orientation_classify"):
        kwargs["use_doc_orientation_classify"] = False
    if _accepts("use_doc_unwarping"):
        kwargs["use_doc_unwarping"] = False
    if _accepts("show_log"):
        kwargs["show_log"] = False
    if _accepts("use_gpu"):
        kwargs["use_gpu"] = False
    if _accepts("device"):
        kwargs["device"] = "cpu"

    use_cls = cls_dir is not None
    if use_modern_args:
        if _accepts("use_textline_orientation"):
            kwargs["use_textline_orientation"] = use_cls
    else:
        if _accepts("use_angle_cls"):
            kwargs["use_angle_cls"] = use_cls

    load_kwargs = dict(kwargs)
    while True:
        try:
            return _PaddlePdfOcrRunner(ocr=PaddleOCR(**load_kwargs), use_cls=use_cls)
        except Exception as exc:
            unknown_arg_match = re.search(r"Unknown argument:\s*([a-zA-Z0-9_]+)", str(exc))
            if unknown_arg_match is None:
                raise
            unknown_arg = unknown_arg_match.group(1)
            if unknown_arg not in load_kwargs:
                raise
            load_kwargs.pop(unknown_arg, None)


def _load_pdf_ocr_direct_runner(model_path: str) -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as e:
        raise RuntimeError(
            "transformers and torch are required for direct OCR model inference."
        ) from e

    model_classes: list[Any] = []
    try:
        from transformers import AutoModelForImageTextToText

        model_classes.append(AutoModelForImageTextToText)
    except Exception:
        pass
    model_classes.append(AutoModelForCausalLM)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_errors: list[str] = []
    for model_class in model_classes:
        load_kwargs_base = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        attempts: list[dict[str, Any]] = [load_kwargs_base]
        if device == "cpu":
            attempts = [
                {**load_kwargs_base, "torch_dtype": torch.float32},
                load_kwargs_base,
            ]
        for load_kwargs in attempts:
            try:
                model = model_class.from_pretrained(model_path, **load_kwargs)
                model = model.to(device)
                model.eval()
                return _DirectPdfOcrRunner(
                    model=model,
                    processor=processor,
                    device=device,
                )
            except Exception as exc:
                model_errors.append(
                    f"{model_class.__name__}({load_kwargs}): {_format_exception(exc)}"
                )

    joined_errors = "\n".join(model_errors)
    dependency_hint = (
        "\nHint: install missing optional dependency `einops` "
        "if it is reported in the errors."
        if "einops" in joined_errors
        else ""
    )
    raise RuntimeError(
        "Failed to load OCR model via direct transformers APIs.\n"
        f"{joined_errors}{dependency_hint}"
    )


@lru_cache(maxsize=2)
def _load_pdf_ocr_pipeline(model_path: str) -> Any:
    if not model_path:
        raise RuntimeError(
            "PDF_OCR_MODEL is empty. Set PDF_OCR_MODEL to a local OCR model path."
        )
    errors: list[str] = []
    if _looks_like_pp_ocr_v5_mobile_path(model_path):
        try:
            return _load_pdf_ocr_paddle_runner(model_path)
        except Exception as exc:
            errors.append(f"paddle_load: {_format_exception(exc)}")
            raise RuntimeError(
                f"Failed to load OCR model from {model_path}. "
                "PP-OCRv5_mobile requires PaddleOCR-compatible local model files.\n"
                + "\n".join(errors)
            ) from exc

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as e:
        raise RuntimeError(
            "transformers is required for OCR on image-based PDFs."
        ) from e

    for task in ("image-text-to-text", "image-to-text"):
        try:
            return hf_pipeline(
                task=task,
                model=model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:
            errors.append(f"pipeline[{task}]: {_format_exception(exc)}")

    try:
        return _load_pdf_ocr_direct_runner(model_path)
    except Exception as exc:
        errors.append(f"direct_load: {_format_exception(exc)}")
        raise RuntimeError(
            f"Failed to load OCR model from {model_path}. "
            "Check PDF_OCR_MODEL, local model files, and OCR dependencies.\n"
            + "\n".join(errors)
        ) from exc


def _extract_generated_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("generated_text", "text", "answer", "rec_text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        rec_texts = payload.get("rec_texts")
        if isinstance(rec_texts, list):
            values = [
                str(item).strip()
                for item in rec_texts
                if isinstance(item, str) and str(item).strip()
            ]
            if values:
                return "\n".join(values)
        nested_result = payload.get("res")
        if nested_result is not None:
            nested_text = _extract_generated_text(nested_result).strip()
            if nested_text:
                return nested_text
        return ""
    if isinstance(payload, (list, tuple)):
        if (
            len(payload) == 2
            and isinstance(payload[1], (list, tuple))
            and payload[1]
            and isinstance(payload[1][0], str)
            and payload[1][0].strip()
        ):
            return payload[1][0]
        parts = [_extract_generated_text(item).strip() for item in payload]
        non_empty = [part for part in parts if part]
        return "\n".join(non_empty)
    result_attr = getattr(payload, "res", None)
    if result_attr is not None:
        return _extract_generated_text(result_attr).strip()
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


def _cleanup_missing_drive_image_files(
    *,
    out_dir: Path,
    valid_file_ids: set[str],
) -> None:
    if not out_dir.exists():
        return
    for path in out_dir.glob("*"):
        if path.is_dir() or path.name.endswith(".meta.json"):
            continue
        file_id = _extract_drive_file_id(path.name)
        if not file_id or file_id in valid_file_ids:
            continue
        try:
            path.unlink()
            logger.info("Removed deleted Drive image %s", path.name)
        except Exception as exc:
            logger.warning("Failed to remove deleted Drive image %s: %s", path.name, exc)
            continue

        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not meta_path.exists():
            continue
        try:
            meta_path.unlink()
            logger.info("Removed deleted Drive image metadata %s", meta_path.name)
        except Exception as exc:
            logger.warning(
                "Failed to remove deleted Drive image metadata %s: %s",
                meta_path.name,
                exc,
            )


def _drive_metadata_payload(drive_file: DriveFile) -> dict[str, str]:
    return {
        "drive_file_id": drive_file.file_id,
        "drive_file_name": drive_file.name,
        "drive_name": drive_file.name,
        "drive_path": drive_file.path,
        "drive_mime_type": drive_file.mime_type,
        "drive_modified_time": drive_file.modified_time,
        "drive_url": f"https://drive.google.com/file/d/{drive_file.file_id}/view",
    }


def _write_drive_metadata(out_path: Path, drive_file: DriveFile) -> None:
    metadata = _drive_metadata_payload(drive_file)
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
    drive_batch_size: int | None = 20,
    drive_download_max_retries: int = 3,
    drive_download_retry_initial_delay_seconds: float = 0.5,
    drive_download_retry_max_delay_seconds: float = 8.0,
    drive_download_retry_backoff_multiplier: float = 2.0,
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

    images_dir = docs_dir.parent / "images" / "google_drive"
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
        valid_image_ids = {
            drive_file.file_id
            for drive_file in drive_files
            if _is_drive_image_mime(drive_file.mime_type)
            or drive_file.mime_type == DRIVE_SLIDE_MIME
            or drive_file.mime_type in DRIVE_POWERPOINT_MIMES
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
        _cleanup_missing_drive_image_files(
            out_dir=images_dir,
            valid_file_ids=valid_image_ids,
        )

    docs_count = 0
    sheets_count = 0
    ensure_dir(images_dir)
    download_retry_options = {
        "max_retries": drive_download_max_retries,
        "initial_delay_seconds": drive_download_retry_initial_delay_seconds,
        "max_delay_seconds": drive_download_retry_max_delay_seconds,
        "backoff_multiplier": drive_download_retry_backoff_multiplier,
    }

    batches = _split_batches(drive_files, batch_size=drive_batch_size)
    if len(batches) > 1:
        logger.info(
            "Downloading Google Drive files in %d batches (batch_size=%d).",
            len(batches),
            len(batches[0]),
        )

    for batch_index, batch in enumerate(batches, start=1):
        logger.info(
            "Processing Google Drive batch %d/%d (%d files).",
            batch_index,
            len(batches),
            len(batch),
        )
        for drive_file in batch:
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
                            **download_retry_options,
                        )
                        text = content.decode("utf-8", errors="replace")
                        text = _strip_drive_image_placeholders(text)
                    elif drive_file.mime_type == DRIVE_SLIDE_MIME:
                        try:
                            content = _download_export_bytes(
                                drive_service,
                                file_id=drive_file.file_id,
                                mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                **download_retry_options,
                            )
                            text = _extract_pptx_text(content)
                            _extract_pptx_images(
                                content,
                                drive_file=drive_file,
                                images_dir=images_dir,
                                surrounding_text=text,
                            )
                        except Exception as exc:
                            if not _is_export_size_limit_exceeded(exc):
                                raise
                            logger.warning(
                                (
                                    "Slides export exceeds Drive size limit; "
                                    "falling back to text/plain export: %s (%s)"
                                ),
                                drive_file.path,
                                drive_file.file_id,
                            )
                            content = _download_export_bytes(
                                drive_service,
                                file_id=drive_file.file_id,
                                mime_type="text/plain",
                                **download_retry_options,
                            )
                            text = content.decode("utf-8", errors="replace")
                    elif drive_file.mime_type in DRIVE_WORD_MIMES:
                        if drive_file.mime_type == "application/msword":
                            logger.warning(
                                "Skip unsupported binary Word format (.doc): %s",
                                drive_file.path,
                            )
                            continue
                        content = _download_file_bytes(
                            drive_service,
                            file_id=drive_file.file_id,
                            **download_retry_options,
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
                            drive_service,
                            file_id=drive_file.file_id,
                            **download_retry_options,
                        )
                        text = _extract_pptx_text(content)
                        _extract_pptx_images(
                            content,
                            drive_file=drive_file,
                            images_dir=images_dir,
                            surrounding_text=text,
                        )
                    elif drive_file.mime_type == DRIVE_PDF_MIME:
                        content = _download_file_bytes(
                            drive_service,
                            file_id=drive_file.file_id,
                            **download_retry_options,
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
                            drive_service,
                            file_id=drive_file.file_id,
                            mime_type="text/csv",
                            **download_retry_options,
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
                            drive_service,
                            file_id=drive_file.file_id,
                            **download_retry_options,
                        )
                        csv_text = _extract_xlsx_csv_text(xlsx_bytes)
                    else:
                        continue
                    if not csv_text.strip():
                        logger.warning(
                            "Skip empty extracted sheet data: %s", drive_file.path
                        )
                        continue
                    out_path.write_text(csv_text, encoding="utf-8")
                    _write_drive_metadata(out_path, drive_file)
                    sheets_count += 1
                    logger.info("Downloaded sheet-like file: %s", drive_file.path)
                elif _is_drive_image_mime(drive_file.mime_type):
                    extension = (
                        mimetypes.guess_extension(drive_file.mime_type)
                        or Path(drive_file.name).suffix
                        or ".img"
                    )
                    out_path = images_dir / _build_output_filename(
                        drive_file, extension=extension
                    )
                    _cleanup_drive_duplicates(
                        out_dir=images_dir, drive_file=drive_file, keep_path=out_path
                    )
                    if skip_existing and out_path.exists():
                        if not update_existing:
                            logger.info(
                                "Skip Drive image download (exists): %s",
                                out_path.name,
                            )
                            continue
                        if _is_drive_file_up_to_date(out_path, drive_file):
                            logger.info(
                                "Skip Drive image download (up-to-date): %s",
                                out_path.name,
                            )
                            continue
                    content = _download_file_bytes(
                        drive_service,
                        file_id=drive_file.file_id,
                        **download_retry_options,
                    )
                    out_path.write_bytes(content)
                    _write_drive_metadata(out_path, drive_file)
                    docs_count += 1
                    logger.info("Downloaded Drive image file: %s", drive_file.path)
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
