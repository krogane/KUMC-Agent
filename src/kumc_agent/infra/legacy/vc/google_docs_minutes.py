from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from kumc_agent.infra.legacy.indexing.constants import DRIVE_DOC_MIME

_JST = ZoneInfo("Asia/Tokyo")
_FOLDER_MIME = "application/vnd.google-apps.folder"

# Docs update requires write scopes.
_GOOGLE_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
)


@dataclass(frozen=True)
class MinutesDocument:
    doc_id: str
    name: str
    modified_time: str
    web_view_link: str


class GoogleDocsMinutesClient:
    def __init__(
        self,
        *,
        drive_folder_id: str,
        google_application_credentials: str,
        minutes_drive_dir: str,
    ) -> None:
        self._drive_folder_id = (drive_folder_id or "").strip()
        self._google_application_credentials = (
            google_application_credentials or ""
        ).strip()
        self._minutes_drive_dir = (minutes_drive_dir or "議事録").strip().strip("/")
        self._drive_service: Any | None = None
        self._docs_service: Any | None = None
        self._minutes_folder_id: str | None = None

    def resolve_today_minutes_document(
        self,
        *,
        cached_doc_id: str = "",
        now: datetime | None = None,
    ) -> MinutesDocument | None:
        today = now.astimezone(_JST) if now is not None else datetime.now(_JST)
        token = today.strftime("%Y%m%d")
        expected_names = (f"{token}議事録", f"{token} 議事録")

        cached = (cached_doc_id or "").strip()
        if cached:
            doc = self._get_doc_metadata(cached)
            if doc is not None:
                return doc

        folder_id = self._resolve_minutes_folder_id()
        if not folder_id:
            return None

        docs = self._list_minutes_candidates(
            folder_id=folder_id,
            expected_names=expected_names,
        )
        if not docs:
            return None
        return docs[0]

    def export_markdown(self, *, doc_id: str) -> str:
        payload = self._download_export_bytes(doc_id=doc_id, mime_type="text/markdown")
        return payload.decode("utf-8", errors="replace")

    def export_pdf(self, *, doc_id: str) -> bytes:
        return self._download_export_bytes(doc_id=doc_id, mime_type="application/pdf")

    def replace_document_with_plain_text(self, *, doc_id: str, text: str) -> None:
        docs = self._docs()
        doc = docs.documents().get(documentId=doc_id).execute()
        body = doc.get("body") or {}
        content = body.get("content") or []
        end_index = 1
        if content:
            last = content[-1] or {}
            value = int(last.get("endIndex") or 1)
            end_index = max(1, value)

        requests: list[dict[str, Any]] = []
        if end_index > 1:
            requests.append(
                {
                    "deleteContentRange": {
                        "range": {
                            "startIndex": 1,
                            "endIndex": end_index - 1,
                        }
                    }
                }
            )
        if text:
            requests.append(
                {
                    "insertText": {
                        "location": {"index": 1},
                        "text": text,
                    }
                }
            )
        if not requests:
            return
        docs.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": requests},
        ).execute()

    @staticmethod
    def pdf_to_png_pages(pdf_bytes: bytes, *, dpi: int = 144) -> list[bytes]:
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for PDF rendering.") from exc

        scale = max(0.1, float(dpi) / 72.0)
        output: list[bytes] = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            for page in document:
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                output.append(pix.tobytes("png"))
        return output

    def _resolve_minutes_folder_id(self) -> str | None:
        if self._minutes_folder_id:
            return self._minutes_folder_id
        root_id = self._drive_folder_id
        if not root_id:
            return None

        current_ids = [root_id]
        if self._minutes_drive_dir:
            for segment in [part for part in self._minutes_drive_dir.split("/") if part]:
                next_ids: list[str] = []
                for parent_id in current_ids:
                    next_ids.extend(self._find_child_folder_ids(parent_id=parent_id, name=segment))
                if not next_ids:
                    return None
                current_ids = next_ids

        # If multiple folders match the same path, keep first one returned by API.
        self._minutes_folder_id = current_ids[0]
        return self._minutes_folder_id

    def _find_child_folder_ids(self, *, parent_id: str, name: str) -> list[str]:
        drive = self._drive()
        page_token: str | None = None
        folder_ids: list[str] = []
        while True:
            response = (
                drive.files()
                .list(
                    q=(
                        f"'{parent_id}' in parents and trashed = false and "
                        f"mimeType = '{_FOLDER_MIME}' and name = '{self._escape_query(name)}'"
                    ),
                    fields="nextPageToken, files(id)",
                    pageToken=page_token,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                )
                .execute()
            )
            for item in response.get("files", []):
                folder_id = str(item.get("id") or "").strip()
                if folder_id:
                    folder_ids.append(folder_id)
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return folder_ids

    def _list_minutes_candidates(
        self,
        *,
        folder_id: str,
        expected_names: tuple[str, str],
    ) -> list[MinutesDocument]:
        drive = self._drive()
        escaped_names = [self._escape_query(name) for name in expected_names]
        name_query = " or ".join([f"name = '{name}'" for name in escaped_names])

        page_token: str | None = None
        rows: list[MinutesDocument] = []
        while True:
            response = (
                drive.files()
                .list(
                    q=(
                        f"'{folder_id}' in parents and trashed = false and "
                        f"mimeType = '{DRIVE_DOC_MIME}' and ({name_query})"
                    ),
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)",
                    pageToken=page_token,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                )
                .execute()
            )
            for item in response.get("files", []):
                doc = self._minutes_doc_from_file(item)
                if doc is not None:
                    rows.append(doc)
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return sorted(
            rows,
            key=lambda item: self._candidate_sort_key(
                item=item,
                expected_names=expected_names,
            ),
        )

    def _get_doc_metadata(self, doc_id: str) -> MinutesDocument | None:
        drive = self._drive()
        try:
            item = (
                drive.files()
                .get(
                    fileId=doc_id,
                    fields="id, name, mimeType, modifiedTime, webViewLink",
                    supportsAllDrives=True,
                )
                .execute()
            )
        except Exception:
            return None
        return self._minutes_doc_from_file(item)

    def _minutes_doc_from_file(self, item: Any) -> MinutesDocument | None:
        if not isinstance(item, dict):
            return None
        if str(item.get("mimeType") or "") != DRIVE_DOC_MIME:
            return None
        doc_id = str(item.get("id") or "").strip()
        if not doc_id:
            return None
        name = str(item.get("name") or "").strip()
        modified_time = str(item.get("modifiedTime") or "").strip()
        web_view_link = str(item.get("webViewLink") or "").strip()
        if not web_view_link:
            web_view_link = f"https://docs.google.com/document/d/{doc_id}/edit"
        return MinutesDocument(
            doc_id=doc_id,
            name=name,
            modified_time=modified_time,
            web_view_link=web_view_link,
        )

    def _download_export_bytes(self, *, doc_id: str, mime_type: str) -> bytes:
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as exc:
            raise RuntimeError(
                "google-api-python-client is required to export Google Docs files."
            ) from exc

        request = self._drive().files().export_media(
            fileId=doc_id,
            mimeType=mime_type,
        )
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()

    def _drive(self):
        if self._drive_service is None:
            self._build_services()
        return self._drive_service

    def _docs(self):
        if self._docs_service is None:
            self._build_services()
        return self._docs_service

    def _build_services(self) -> None:
        if self._drive_service is not None and self._docs_service is not None:
            return
        creds = self._build_google_credentials(
            application_credentials=self._google_application_credentials
        )
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError(
                "google-api-python-client is required for Google Drive/Docs access."
            ) from exc
        self._drive_service = build(
            "drive",
            "v3",
            credentials=creds,
            cache_discovery=False,
        )
        self._docs_service = build(
            "docs",
            "v1",
            credentials=creds,
            cache_discovery=False,
        )

    @staticmethod
    def _build_google_credentials(*, application_credentials: str) -> Any:
        try:
            import google.auth
            from google.oauth2.service_account import Credentials
        except ImportError as exc:
            raise RuntimeError(
                "google-auth is required for Google API credentials."
            ) from exc

        if application_credentials:
            return Credentials.from_service_account_file(
                application_credentials,
                scopes=_GOOGLE_SCOPES,
            )
        creds, _ = google.auth.default(scopes=_GOOGLE_SCOPES)
        return creds

    @staticmethod
    def _escape_query(value: str) -> str:
        return (value or "").replace("'", "\\'")

    @staticmethod
    def _name_priority(name: str, expected_names: tuple[str, str]) -> int:
        normalized = (name or "").strip()
        if normalized == expected_names[0]:
            return 0
        if normalized == expected_names[1]:
            return 1
        return 99

    @classmethod
    def _candidate_sort_key(
        cls,
        *,
        item: MinutesDocument,
        expected_names: tuple[str, str],
    ) -> tuple[int, float]:
        return (
            cls._name_priority(item.name, expected_names),
            -cls._modified_unix(item.modified_time),
        )

    @staticmethod
    def _modified_unix(modified_time: str) -> float:
        value = (modified_time or "").strip()
        if not value:
            return 0.0
        # Google Drive modifiedTime uses RFC3339 (e.g. 2026-02-20T12:34:56.000Z).
        iso = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(iso).timestamp()
        except Exception:
            return 0.0
