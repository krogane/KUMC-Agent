from __future__ import annotations

from typing import Sequence

DOCS_SEPARATORS: Sequence[str] = (
    "\n## ",
    "\n### ",
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
MESSAGE_SEPARATORS: Sequence[str] = (
    "\n",
    " ",
    "",
)


DRIVE_DOC_MIME: str = "application/vnd.google-apps.document"
DRIVE_SHEET_MIME: str = "application/vnd.google-apps.spreadsheet"
DRIVE_SLIDE_MIME: str = "application/vnd.google-apps.presentation"
DRIVE_PDF_MIME: str = "application/pdf"
DRIVE_WORD_MIMES: Sequence[str] = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
)
DRIVE_EXCEL_MIMES: Sequence[str] = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
)
DRIVE_POWERPOINT_MIMES: Sequence[str] = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-powerpoint",
)
GOOGLE_SCOPES: Sequence[str] = (
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/documents",
)
FILE_ID_SEPARATOR: str = "__"
