from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_SRC = Path(__file__).resolve().parents[1]
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from config import AppConfig
from indexing.chunking import proposition_chunk_jsonl_dir, recursive_chunk_dir
from indexing.chunks import load_chunks_from_dirs
from indexing.constants import (
    DOCS_SEPARATORS,
    SHEETS_SEPARATORS,
)
from indexing.drive_loader import download_drive_markdown
from indexing.faiss_index import build_faiss_index
from indexing.raptor import raptor_chunk_global

logger = logging.getLogger(__name__)


def _clear_dir_contents(target: Path) -> None:
    if not target.exists():
        return
    for entry in target.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    logger.info("Cleared contents of %s", target)


def _reset_output_dirs(cfg: AppConfig) -> None:
    if cfg.clear_raw_data:
        for name in ("docs", "sheets"):
            target = cfg.raw_data_dir / name
            if target.exists():
                _clear_dir_contents(target)

    if cfg.clear_rec_chunk_data:
        for name in ("docs", "sheets"):
            target = cfg.rec_chunk_dir / name
            if target.exists():
                _clear_dir_contents(target)

    if cfg.clear_prop_chunk_data:
        for name in ("docs", "sheets"):
            target = cfg.prop_chunk_dir / name
            if target.exists():
                _clear_dir_contents(target)

    if cfg.clear_raptor_chunk_data and cfg.raptor_chunk_dir.exists():
        _clear_dir_contents(cfg.raptor_chunk_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    resolved_base = Path(__file__).resolve().parents[3]
    load_dotenv(resolved_base / ".env", override=True)
    cfg = AppConfig.from_here(base_dir=resolved_base)
    drive_folder_id = cfg.drive_folder_id
    if not drive_folder_id:
        raise RuntimeError(
            f"{drive_folder_id} is not set. Please set it in .env or your environment."
        )

    raw_docs_dir = cfg.raw_data_dir / "docs"
    raw_sheets_dir = cfg.raw_data_dir / "sheets"
    rec_docs_dir = cfg.rec_chunk_dir / "docs"
    rec_sheets_dir = cfg.rec_chunk_dir / "sheets"
    prop_docs_dir = cfg.prop_chunk_dir / "docs"
    prop_sheets_dir = cfg.prop_chunk_dir / "sheets"

    logger.info(
        "RECURSIVE_CFG : size=%d overlap=%d min_tokens=%d",
        cfg.rec_chunk_size,
        cfg.rec_chunk_overlap,
        cfg.rec_min_chunk_tokens,
    )
    logger.info("MODEL         : %s", cfg.embedding_model)
    logger.info("DRIVE_FOLDER  : %s", drive_folder_id)
    logger.info(
        "DRIVE_MAX     : %s",
        cfg.drive_max_files if cfg.drive_max_files > 0 else "unlimited",
    )
    logger.info("CLEAR_RAW     : %s", "yes" if cfg.clear_raw_data else "no")
    logger.info("CLEAR_REC     : %s", "yes" if cfg.clear_rec_chunk_data else "no")
    logger.info("CLEAR_PROP    : %s", "yes" if cfg.clear_prop_chunk_data else "no")
    logger.info("CLEAR_RAPTOR  : %s", "yes" if cfg.clear_raptor_chunk_data else "no")
    logger.info(
        "PROP_CHUNKING : %s",
        "enabled" if cfg.prop_chunk_enabled else "disabled",
    )
    if cfg.prop_chunk_enabled:
        logger.info(
            "PROP_CHUNK_LLM: %s / %s",
            cfg.prop_chunk_provider,
            cfg.prop_chunk_model,
        )
        logger.info(
            "PROP_CHUNK_CFG: size=%d temp=%s retries=%d",
            cfg.prop_chunk_size,
            cfg.prop_chunk_temperature,
            cfg.prop_chunk_max_retries,
        )
    logger.info(
        "RAPTOR        : %s",
        "enabled" if cfg.raptor_enabled else "disabled",
    )
    if cfg.raptor_enabled:
        logger.info(
            "RAPTOR_CFG    : cluster_max=%d summary_max=%d stop=%d k_max=%d method=%s",
            cfg.raptor_cluster_max_tokens,
            cfg.raptor_summary_max_tokens,
            cfg.raptor_stop_chunk_count,
            cfg.raptor_k_max,
            cfg.raptor_k_selection,
        )
        logger.info("RAPTOR_EMBED  : %s", cfg.raptor_embedding_model)
        logger.info(
            "RAPTOR_LLM    : %s / %s",
            cfg.raptor_summary_provider,
            cfg.raptor_summary_model,
        )

    _reset_output_dirs(cfg)

    download_drive_markdown(
        drive_folder_id=drive_folder_id,
        docs_dir=raw_docs_dir,
        sheets_dir=raw_sheets_dir,
        google_application_credentials=cfg.google_application_credentials,
        drive_max_files=cfg.drive_max_files,
        skip_existing=not cfg.clear_raw_data,
    )

    recursive_chunk_dir(
        raw_data_dir=raw_docs_dir,
        chunk_dir=rec_docs_dir,
        rec_chunk_size=cfg.rec_chunk_size,
        rec_chunk_overlap=cfg.rec_chunk_overlap,
        rec_min_chunk_tokens=cfg.rec_min_chunk_tokens,
        tokenizer_model=cfg.embedding_model,
        separators=DOCS_SEPARATORS,
        source_type="docs",
        skip_existing=not cfg.clear_rec_chunk_data,
    )
    recursive_chunk_dir(
        raw_data_dir=raw_sheets_dir,
        chunk_dir=rec_sheets_dir,
        rec_chunk_size=cfg.rec_chunk_size,
        rec_chunk_overlap=cfg.rec_chunk_overlap,
        rec_min_chunk_tokens=cfg.rec_min_chunk_tokens,
        tokenizer_model=cfg.embedding_model,
        separators=SHEETS_SEPARATORS,
        source_type="sheets",
        file_extensions=(".csv",),
        skip_existing=not cfg.clear_rec_chunk_data,
    )

    if cfg.prop_chunk_enabled:
        proposition_chunk_jsonl_dir(
            input_chunk_dir=rec_docs_dir,
            output_chunk_dir=prop_docs_dir,
            config=cfg,
            skip_existing=not cfg.clear_prop_chunk_data,
        )
        proposition_chunk_jsonl_dir(
            input_chunk_dir=rec_sheets_dir,
            output_chunk_dir=prop_sheets_dir,
            config=cfg,
            skip_existing=not cfg.clear_prop_chunk_data,
        )

    if cfg.raptor_enabled:
        raptor_input_dirs = (
            [prop_docs_dir, prop_sheets_dir]
            if cfg.prop_chunk_enabled
            else [rec_docs_dir, rec_sheets_dir]
        )
        raptor_chunk_global(
            input_chunk_dirs=raptor_input_dirs,
            output_chunk_dir=cfg.raptor_chunk_dir,
            config=cfg,
            skip_existing=not cfg.clear_raptor_chunk_data,
        )

    index_chunks = []
    index_chunks.extend(load_chunks_from_dirs([rec_docs_dir, rec_sheets_dir]))
    if cfg.prop_chunk_enabled:
        index_chunks.extend(load_chunks_from_dirs([prop_docs_dir, prop_sheets_dir]))
    if cfg.raptor_enabled:
        index_chunks.extend(load_chunks_from_dirs([cfg.raptor_chunk_dir]))

    build_faiss_index(
        chunks=index_chunks,
        model_name=cfg.embedding_model,
        index_dir=cfg.index_dir,
    )


if __name__ == "__main__":
    main()
