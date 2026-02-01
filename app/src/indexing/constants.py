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
GOOGLE_SCOPES: Sequence[str] = (
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
)
FILE_ID_SEPARATOR: str = "__"
