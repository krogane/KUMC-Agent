from __future__ import annotations

from pathlib import Path


def parse_pdf_with_ocr(path: Path) -> str:
    # OCR pipeline is delegated to legacy Drive loader. This is a stub entrypoint.
    return path.read_text(encoding="utf-8", errors="ignore") if path.suffix == ".txt" else ""
