from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_SRC = Path(__file__).resolve().parents[1]
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from config import AppConfig
from indexing.chunking import (
    chunk_markdown_to_jsonl,
    llm_chunk_jsonl_dir,
    load_documents_from_jsonl_dirs,
)
from indexing.constants import (
    DOCS_SEPARATORS,
    DRIVE_FOLDER_ID_ENV,
    SHEETS_SEPARATORS,
)
from indexing.drive_loader import download_drive_markdown
from indexing.faiss_index import build_faiss_index

logger = logging.getLogger(__name__)


def _reset_output_dirs(
    *,
    raw_data_dir: Path,
    chunk_dir: Path,
    llm_chunk_dir: Path,
    index_dir: Path,
) -> None:
    def _clear_dir_contents(target: Path) -> None:
        if not target.exists():
            return
        for entry in target.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        logger.info("Cleared contents of %s", target)

    for parent in (raw_data_dir, chunk_dir, llm_chunk_dir):
        for name in ("docs", "sheets"):
            target = parent / name
            if target.exists():
                _clear_dir_contents(target)

    if index_dir.exists():
        _clear_dir_contents(index_dir)


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    resolved_base = Path(__file__).resolve().parents[3]
    load_dotenv(resolved_base / ".env", override=True)
    cfg = AppConfig.from_here(base_dir=resolved_base)
    folder_id = os.getenv(DRIVE_FOLDER_ID_ENV)
    if not folder_id:
        raise RuntimeError(
            f"{DRIVE_FOLDER_ID_ENV} is not set. Please set it in .env or your environment."
        )

    raw_docs_dir = cfg.raw_data_dir / "docs"
    raw_sheets_dir = cfg.raw_data_dir / "sheets"
    chunk_docs_dir = cfg.chunk_dir / "docs"
    chunk_sheets_dir = cfg.chunk_dir / "sheets"
    llm_chunk_docs_dir = cfg.llm_chunk_dir / "docs"
    llm_chunk_sheets_dir = cfg.llm_chunk_dir / "sheets"

    logger.info("DOCS_CHUNK    : %d / %d", cfg.chunk_size, cfg.chunk_overlap)
    logger.info("SHEETS_CHUNK  : %d / %d", cfg.chunk_size, cfg.chunk_overlap)
    logger.info("MODEL         : %s", cfg.embedding_model_name)
    logger.info("DRIVE_FOLDER  : %s", folder_id)
    logger.info(
        "LLM_CHUNKING  : %s",
        "enabled" if cfg.llm_chunking_enabled else "disabled",
    )
    if cfg.llm_chunking_enabled:
        logger.info("LLM_PROVIDER  : %s", cfg.llm_chunk_provider)
        logger.info("LLM_MODEL     : %s", cfg.llm_chunk_model)
        logger.info("LLM_RETRY     : %d", cfg.llm_chunk_max_retries)
    logger.info(
        "LLM_CHUNK     : %d %s",
        cfg.llm_chunk_size,
    )
    logger.info("LLM_TEMP      : %s", cfg.llm_chunk_temperature)

    _reset_output_dirs(
        raw_data_dir=cfg.raw_data_dir,
        chunk_dir=cfg.chunk_dir,
        llm_chunk_dir=cfg.llm_chunk_dir,
        index_dir=cfg.index_dir,
    )

    download_drive_markdown(
        folder_id=folder_id,
        docs_dir=raw_docs_dir,
        sheets_dir=raw_sheets_dir,
    )

    chunk_markdown_to_jsonl(
        raw_data_dir=raw_docs_dir,
        chunk_dir=chunk_docs_dir,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=DOCS_SEPARATORS,
        source_type="docs",
    )
    chunk_markdown_to_jsonl(
        raw_data_dir=raw_sheets_dir,
        chunk_dir=chunk_sheets_dir,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=SHEETS_SEPARATORS,
        source_type="sheets",
        file_extensions=(".csv",),
    )

    if cfg.llm_chunking_enabled:
        llm_chunk_jsonl_dir(
            input_chunk_dir=chunk_docs_dir,
            output_chunk_dir=llm_chunk_docs_dir,
            config=cfg,
        )
        llm_chunk_jsonl_dir(
            input_chunk_dir=chunk_sheets_dir,
            output_chunk_dir=llm_chunk_sheets_dir,
            config=cfg,
        )
        docs = load_documents_from_jsonl_dirs(
            [llm_chunk_docs_dir, llm_chunk_sheets_dir]
        )
    else:
        docs = load_documents_from_jsonl_dirs([chunk_docs_dir, chunk_sheets_dir])
    build_faiss_index(
        docs=docs,
        model_name=cfg.embedding_model_name,
        index_dir=cfg.index_dir,
    )


if __name__ == "__main__":
    main()
