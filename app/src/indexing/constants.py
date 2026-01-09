from __future__ import annotations

from typing import Sequence

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


DRIVE_DOC_MIME: str = "application/vnd.google-apps.document"
DRIVE_SHEET_MIME: str = "application/vnd.google-apps.spreadsheet"
DRIVE_FOLDER_ID_ENV: str = "FOLDER_ID"
GOOGLE_SCOPES: Sequence[str] = (
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
)
FILE_ID_SEPARATOR: str = "__"
