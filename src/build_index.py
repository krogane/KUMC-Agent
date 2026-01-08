from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import AppConfig, EmbeddingFactory

logger = logging.getLogger(__name__)

# コンフィグ
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
MODEL_NAME: str = "intfloat/multilingual-e5-small"

DOCS_CHUNK_SIZE: int = CHUNK_SIZE
DOCS_CHUNK_OVERLAP: int = CHUNK_OVERLAP
SHEETS_CHUNK_SIZE: int = CHUNK_SIZE
SHEETS_CHUNK_OVERLAP: int = CHUNK_OVERLAP

DOCS_SEPARATORS: Sequence[str] = (
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    " ",
    "",
)
SHEETS_SEPARATORS: Sequence[str] = (
    "\n|",
    "\n\n",
    "\n",
    " ",
    "",
)

LOG_LEVEL: str = "INFO"

DRIVE_DOC_MIME: str = "application/vnd.google-apps.document"
DRIVE_SHEET_MIME: str = "application/vnd.google-apps.spreadsheet"
DRIVE_FOLDER_ID_ENV: str = "FOLDER_ID"
DRIVE_FOLDER_ID_FALLBACK_ENV: str = "GOOGLE_DRIVE_FOLDER_ID"
GOOGLE_SCOPES: Sequence[str] = (
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
)
FILE_ID_SEPARATOR: str = "__"


@dataclass(frozen=True)
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    path: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return sanitized or "drive_file"


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


def _build_sheets_service(creds: Any) -> Any:
    try:
        from googleapiclient.discovery import build
    except ImportError as e:
        raise RuntimeError(
            "google-api-python-client is required for Google Sheets access."
        ) from e

    return build("sheets", "v4", credentials=creds, cache_discovery=False)


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


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _rows_to_markdown(
    rows: Sequence[Sequence[Any]],
    *,
    title: str | None = None,
    heading_level: int = 1,
) -> str:
    normalized_rows: list[list[str]] = [
        ["" if cell is None else str(cell) for cell in row] for row in rows
    ]

    if not normalized_rows:
        return f"{'#' * heading_level} {title}\n" if title else ""

    max_cols = max(len(row) for row in normalized_rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in normalized_rows]

    header = [_escape_markdown_cell(cell) for cell in normalized[0]]
    separator = ["---"] * len(header)

    lines = [
        f"| {' | '.join(header)} |",
        f"| {' | '.join(separator)} |",
    ]

    for row in normalized[1:]:
        escaped = [_escape_markdown_cell(cell) for cell in row]
        lines.append(f"| {' | '.join(escaped)} |")

    table = "\n".join(lines)
    if title:
        return f"{'#' * heading_level} {title}\n\n{table}"
    return table


def _csv_to_markdown(csv_text: str, *, title: str | None = None) -> str:
    csv_text = csv_text.lstrip("\ufeff")
    reader = csv.reader(io.StringIO(csv_text))
    rows = [row for row in reader if row]
    return _rows_to_markdown(rows, title=title, heading_level=1)


def _download_sheet_markdown(*, sheets_service: Any, drive_file: DriveFile) -> str:
    spreadsheet = (
        sheets_service.spreadsheets()
        .get(spreadsheetId=drive_file.file_id)
        .execute()
    )
    sheets = spreadsheet.get("sheets", [])
    if not sheets:
        return f"# {drive_file.name}\n"

    sections = [f"# {drive_file.name}"]
    for sheet in sheets:
        title = sheet.get("properties", {}).get("title")
        if not title:
            continue
        values = (
            sheets_service.spreadsheets()
            .values()
            .get(spreadsheetId=drive_file.file_id, range=title)
            .execute()
            .get("values", [])
        )
        sections.append(_rows_to_markdown(values, title=title, heading_level=2))

    return "\n\n".join(section for section in sections if section).strip()


def _build_output_filename(drive_file: DriveFile) -> str:
    safe_path = _sanitize_filename(drive_file.path.replace("/", "__"))
    return f"{drive_file.file_id}{FILE_ID_SEPARATOR}{safe_path}.md"


def download_drive_markdown(
    *,
    folder_id: str,
    docs_dir: Path,
    sheets_dir: Path,
) -> tuple[int, int]:
    _ensure_dir(docs_dir)
    _ensure_dir(sheets_dir)

    creds = _build_google_credentials()
    drive_service = _build_drive_service(creds)
    sheets_service: Any | None = None
    try:
        sheets_service = _build_sheets_service(creds)
    except RuntimeError:
        logger.info("Google Sheets API unavailable; falling back to CSV exports.")

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
                out_path = docs_dir / _build_output_filename(drive_file)
                out_path.write_text(text, encoding="utf-8")
                docs_count += 1
                logger.info("Downloaded doc: %s", drive_file.path)
            elif drive_file.mime_type == DRIVE_SHEET_MIME:
                markdown: str | None = None
                if sheets_service is not None:
                    try:
                        markdown = _download_sheet_markdown(
                            sheets_service=sheets_service, drive_file=drive_file
                        )
                    except Exception:
                        logger.exception(
                            "Sheets API failed for %s; falling back to CSV export.",
                            drive_file.path,
                        )
                        markdown = None

                if markdown is None:
                    csv_bytes = _download_export_bytes(
                        drive_service, file_id=drive_file.file_id, mime_type="text/csv"
                    )
                    csv_text = csv_bytes.decode("utf-8", errors="replace")
                    markdown = _csv_to_markdown(csv_text, title=drive_file.name)

                out_path = sheets_dir / _build_output_filename(drive_file)
                out_path.write_text(markdown, encoding="utf-8")
                sheets_count += 1
                logger.info("Downloaded sheet: %s", drive_file.path)
        except Exception:
            logger.exception(
                "Failed to download %s (%s)", drive_file.path, drive_file.file_id
            )

    logger.info("Downloaded %d docs and %d sheets", docs_count, sheets_count)
    return docs_count, sheets_count


def _extract_drive_file_id(filename: str) -> str | None:
    if FILE_ID_SEPARATOR not in filename:
        return None
    prefix, _ = filename.split(FILE_ID_SEPARATOR, 1)
    return prefix or None


def _build_splitter(
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
    )


def chunk_markdown_to_jsonl(
    *,
    raw_data_dir: Path,
    chunk_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
    source_type: str,
) -> None:
    _ensure_dir(chunk_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    md_files = sorted(raw_data_dir.rglob("*.md"))
    if not md_files:
        logger.warning("No .md files found under %s", raw_data_dir)
        return

    splitter = _build_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    for f in md_files:
        text = f.read_text(encoding="utf-8")
        rel_path = f.relative_to(raw_data_dir)
        metadata = {
            "source": f.name,
            "source_path": str(rel_path),
            "source_type": source_type,
        }
        drive_file_id = _extract_drive_file_id(f.name)
        if drive_file_id:
            metadata["drive_file_id"] = drive_file_id

        docs = splitter.split_documents(
            [Document(page_content=text, metadata=metadata)]
        )

        safe_rel = _sanitize_filename(str(rel_path).replace(os.sep, "__"))
        out_path = chunk_dir / f"{safe_rel}.jsonl"
        with out_path.open("w", encoding="utf-8") as fw:
            for i, doc in enumerate(docs):
                record = {
                    "text": doc.page_content,
                    "metadata": {
                        **(doc.metadata or {}),
                        "chunk_id": i,
                    },
                }
                fw.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Chunked %s -> %s (%d chunks)", f.name, out_path.name, len(docs))


def load_documents_from_jsonl(chunk_dir: Path) -> list[Document]:
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")

    docs: list[Document] = []
    jsonl_files = sorted(chunk_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No .jsonl chunk files found under %s", chunk_dir)
        return docs

    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as fr:
            for line_no, line in enumerate(fr, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in {path.name} at line {line_no}: {e}"
                    ) from e

                text = obj.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"Missing/invalid 'text' in {path.name} line {line_no}")

                metadata = obj.get("metadata")
                if metadata is None:
                    metadata = {}
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"Missing/invalid 'metadata' (must be dict) in {path.name} line {line_no}"
                    )

                # Persist file context
                metadata.setdefault("chunk_file", path.name)
                metadata.setdefault("line_no", line_no)

                docs.append(Document(page_content=text, metadata=metadata))

    logger.info("Loaded %d documents from %d chunk files", len(docs), len(jsonl_files))
    return docs


def load_documents_from_jsonl_dirs(chunk_dirs: Iterable[Path]) -> list[Document]:
    docs: list[Document] = []
    for chunk_dir in chunk_dirs:
        docs.extend(load_documents_from_jsonl(chunk_dir))
    return docs


def build_faiss_index(*, docs: list[Document], model_name: str, index_dir: Path) -> None:
    if not docs:
        logger.warning("No documents to index. Skipping FAISS build.")
        return

    _ensure_dir(index_dir)

    embeddings = EmbeddingFactory(model_name).get_embeddings()

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local(str(index_dir))
    logger.info("Saved FAISS index to %s", index_dir)


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = AppConfig.from_here(
        embedding_model_name=MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    load_dotenv(cfg.base_dir / ".env")
    folder_id = os.getenv(DRIVE_FOLDER_ID_ENV) or os.getenv(
        DRIVE_FOLDER_ID_FALLBACK_ENV
    )
    if not folder_id:
        raise RuntimeError(
            f"{DRIVE_FOLDER_ID_ENV} is not set. Please set it in .env or your environment."
        )

    raw_docs_dir = cfg.raw_data_dir / "docs"
    raw_sheets_dir = cfg.raw_data_dir / "sheets"
    chunk_docs_dir = cfg.chunk_dir / "docs"
    chunk_sheets_dir = cfg.chunk_dir / "sheets"

    logger.info("BASE_DIR      : %s", cfg.base_dir)
    logger.info("RAW_DOCS_DIR  : %s", raw_docs_dir)
    logger.info("RAW_SHEETS_DIR: %s", raw_sheets_dir)
    logger.info("CHUNK_DOCS_DIR: %s", chunk_docs_dir)
    logger.info("CHUNK_SHEETS_DIR: %s", chunk_sheets_dir)
    logger.info("INDEX_DIR     : %s", cfg.index_dir)
    logger.info("DOCS_CHUNK    : %d / %d", DOCS_CHUNK_SIZE, DOCS_CHUNK_OVERLAP)
    logger.info("SHEETS_CHUNK  : %d / %d", SHEETS_CHUNK_SIZE, SHEETS_CHUNK_OVERLAP)
    logger.info("MODEL         : %s", cfg.embedding_model_name)
    logger.info("DRIVE_FOLDER  : %s", folder_id)

    download_drive_markdown(
        folder_id=folder_id,
        docs_dir=raw_docs_dir,
        sheets_dir=raw_sheets_dir,
    )

    chunk_markdown_to_jsonl(
        raw_data_dir=raw_docs_dir,
        chunk_dir=chunk_docs_dir,
        chunk_size=DOCS_CHUNK_SIZE,
        chunk_overlap=DOCS_CHUNK_OVERLAP,
        separators=DOCS_SEPARATORS,
        source_type="docs",
    )
    chunk_markdown_to_jsonl(
        raw_data_dir=raw_sheets_dir,
        chunk_dir=chunk_sheets_dir,
        chunk_size=SHEETS_CHUNK_SIZE,
        chunk_overlap=SHEETS_CHUNK_OVERLAP,
        separators=SHEETS_SEPARATORS,
        source_type="sheets",
    )

    docs = load_documents_from_jsonl_dirs([chunk_docs_dir, chunk_sheets_dir])
    build_faiss_index(
        docs=docs,
        model_name=cfg.embedding_model_name,
        index_dir=cfg.index_dir,
    )


if __name__ == "__main__":
    main()
