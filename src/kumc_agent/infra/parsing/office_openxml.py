from __future__ import annotations

from pathlib import Path


def parse_office_file(path: Path) -> str:
    # Parsing is delegated to Google Drive loader export flow in current architecture.
    return path.read_text(encoding="utf-8", errors="ignore")
