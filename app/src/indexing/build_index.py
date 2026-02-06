from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_SRC = Path(__file__).resolve().parents[1]
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from config import AppConfig
from indexing.chunking import (
    message_chunk_jsonl_dir,
    proposition_chunk_jsonl_dir,
    recursive_chunk_dir,
    recursive_chunk_jsonl_dir,
    sparse_chunk_jsonl_dir,
    summery_chunk_jsonl_dir,
)
from indexing.chunks import load_chunks_from_dirs
from indexing.constants import DOCS_SEPARATORS, MESSAGE_SEPARATORS, SHEETS_SEPARATORS
from indexing.discord_loader import download_discord_messages
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
        for name in ("docs", "sheets", "messages"):
            target = cfg.raw_data_dir / name
            if target.exists():
                _clear_dir_contents(target)

    if cfg.clear_first_rec_chunk_data:
        for name in ("docs", "sheets", "messages"):
            target = cfg.first_rec_chunk_dir / name
            if target.exists():
                _clear_dir_contents(target)

    if cfg.clear_second_rec_chunk_data:
        for name in ("docs", "sheets", "messages"):
            target = cfg.second_rec_chunk_dir / name
            if target.exists():
                _clear_dir_contents(target)
            sparse_target = cfg.sparse_second_rec_chunk_dir / name
            if sparse_target.exists():
                _clear_dir_contents(sparse_target)

    if cfg.clear_summery_chunk_data:
        for name in ("docs", "sheets", "messages"):
            target = cfg.summery_chunk_dir / name
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
    raw_messages_dir = cfg.raw_data_dir / "messages"
    first_rec_docs_dir = cfg.first_rec_chunk_dir / "docs"
    first_rec_sheets_dir = cfg.first_rec_chunk_dir / "sheets"
    first_rec_messages_dir = cfg.first_rec_chunk_dir / "messages"
    second_rec_docs_dir = cfg.second_rec_chunk_dir / "docs"
    second_rec_sheets_dir = cfg.second_rec_chunk_dir / "sheets"
    second_rec_messages_dir = cfg.second_rec_chunk_dir / "messages"
    sparse_second_rec_docs_dir = cfg.sparse_second_rec_chunk_dir / "docs"
    sparse_second_rec_sheets_dir = cfg.sparse_second_rec_chunk_dir / "sheets"
    sparse_second_rec_messages_dir = cfg.sparse_second_rec_chunk_dir / "messages"
    summery_docs_dir = cfg.summery_chunk_dir / "docs"
    summery_sheets_dir = cfg.summery_chunk_dir / "sheets"
    summery_messages_dir = cfg.summery_chunk_dir / "messages"
    prop_docs_dir = cfg.prop_chunk_dir / "docs"
    prop_sheets_dir = cfg.prop_chunk_dir / "sheets"

    def _llm_label(provider: str, gemini_model: str, llama_model_path: str, llama_model: str) -> str:
        if (provider or "").lower() == "gemini":
            return gemini_model
        if llama_model_path:
            return Path(llama_model_path).name
        return llama_model

    logger.info(
        "FIRST_REC_CFG : size=%d overlap=%d",
        cfg.first_rec_chunk_size,
        cfg.first_rec_chunk_overlap,
    )
    logger.info("MODEL         : %s", cfg.embedding_model)
    logger.info("DRIVE_FOLDER  : %s", drive_folder_id)
    logger.info(
        "DRIVE_MAX     : %s",
        cfg.drive_max_files if cfg.drive_max_files > 0 else "unlimited",
    )
    logger.info("CLEAR_RAW     : %s", "yes" if cfg.clear_raw_data else "no")
    logger.info(
        "CLEAR_FIRST_REC : %s",
        "yes" if cfg.clear_first_rec_chunk_data else "no",
    )
    logger.info(
        "CLEAR_SECOND_REC: %s",
        "yes" if cfg.clear_second_rec_chunk_data else "no",
    )
    logger.info(
        "CLEAR_SUMMERY   : %s",
        "yes" if cfg.clear_summery_chunk_data else "no",
    )
    logger.info("CLEAR_PROP    : %s", "yes" if cfg.clear_prop_chunk_data else "no")
    logger.info("CLEAR_RAPTOR  : %s", "yes" if cfg.clear_raptor_chunk_data else "no")
    logger.info("UPDATE_RAW    : %s", "yes" if cfg.update_raw_data else "no")
    logger.info(
        "UPDATE_FIRST_REC: %s",
        "yes" if cfg.update_first_rec_chunk_data else "no",
    )
    logger.info(
        "UPDATE_SECOND_REC: %s",
        "yes" if cfg.update_second_rec_chunk_data else "no",
    )
    logger.info(
        "UPDATE_SPARSE_SECOND_REC: %s",
        "yes" if cfg.update_sparse_second_rec_chunk_data else "no",
    )
    logger.info(
        "UPDATE_SUMMERY : %s",
        "yes" if cfg.update_summery_chunk_data else "no",
    )
    logger.info("UPDATE_PROP   : %s", "yes" if cfg.update_prop_chunk_data else "no")
    logger.info(
        "UPDATE_RAPTOR : %s",
        "yes" if cfg.update_raptor_chunk_data else "no",
    )
    logger.info(
        "SECOND_REC    : %s",
        "enabled" if cfg.second_rec_enabled else "disabled",
    )
    if cfg.second_rec_enabled:
        logger.info(
            "SECOND_REC_CFG: size=%d overlap=%d",
            cfg.second_rec_chunk_size,
            cfg.second_rec_chunk_overlap,
        )

    logger.info(
        "SUMMERY       : %s",
        "enabled" if cfg.summery_enabled else "disabled",
    )
    if cfg.summery_enabled:
        logger.info(
            "SUMMERY_LLM   : %s / %s",
            cfg.summery_provider,
            _llm_label(
                cfg.summery_provider,
                cfg.summery_gemini_model,
                cfg.summery_llama_model_path,
                cfg.summery_llama_model,
            ),
        )
        logger.info(
            "SUMMERY_CFG   : chars=%d temp=%s retries=%d",
            cfg.summery_characters,
            cfg.summery_temperature,
            cfg.summery_max_retries,
        )

    logger.info(
        "PROP_CHUNKING : %s",
        "enabled" if cfg.prop_enabled else "disabled",
    )
    if cfg.prop_enabled:
        logger.info(
            "PROP_CHUNK_LLM: %s / %s",
            cfg.prop_provider,
            _llm_label(
                cfg.prop_provider,
                cfg.prop_gemini_model,
                cfg.prop_llama_model_path,
                cfg.prop_llama_model,
            ),
        )
        logger.info(
            "PROP_CHUNK_CFG: temp=%s retries=%d",
            cfg.prop_temperature,
            cfg.prop_max_retries,
        )
    logger.info(
        "RAPTOR        : %s",
        "enabled" if cfg.raptor_enabled else "disabled",
    )
    if cfg.raptor_enabled:
        logger.info(
            "RAPTOR_CFG    : cluster_max=%d summary_max=%d stop=%d k_max=%d method=%s",
            cfg.raptor_cluster_max_tokens,
            cfg.raptor_summery_max_tokens,
            cfg.raptor_stop_chunk_count,
            cfg.raptor_k_max,
            cfg.raptor_k_selection,
        )
        logger.info("RAPTOR_EMBED  : %s", cfg.raptor_embedding_model)
        logger.info(
            "RAPTOR_LLM    : %s / %s",
            cfg.raptor_summery_provider,
            _llm_label(
                cfg.raptor_summery_provider,
                cfg.raptor_summery_gemini_model,
                cfg.raptor_summery_llama_model_path,
                cfg.raptor_summery_llama_model,
            ),
        )

    _reset_output_dirs(cfg)

    if cfg.discord_bot_token:
        allowed_ids = (
            set(cfg.discord_guild_allow_list)
            if cfg.discord_guild_allow_list
            else None
        )
        logger.info("DISCORD      : fetching messages")
        try:
            stats = asyncio.run(
                download_discord_messages(
                    token=cfg.discord_bot_token,
                    output_dir=raw_messages_dir,
                    allowed_guild_ids=allowed_ids,
                )
            )
            logger.info(
                "DISCORD      : fetched messages=%d channels=%d guilds=%d",
                stats.messages,
                stats.channels,
                stats.guilds,
            )
        except Exception:
            logger.exception("DISCORD      : failed to fetch messages")
    else:
        logger.warning("DISCORD      : DISCORD_BOT_TOKEN not set, skipping")

    download_drive_markdown(
        drive_folder_id=drive_folder_id,
        docs_dir=raw_docs_dir,
        sheets_dir=raw_sheets_dir,
        google_application_credentials=cfg.google_application_credentials,
        drive_max_files=cfg.drive_max_files,
        skip_existing=not cfg.clear_raw_data,
        update_existing=cfg.update_raw_data,
        sync_deleted=cfg.update_raw_data,
    )

    if raw_messages_dir.exists():
        message_chunk_jsonl_dir(
            raw_messages_dir=raw_messages_dir,
            chunk_dir=first_rec_messages_dir,
            chunk_size=cfg.first_rec_chunk_size,
            chunk_overlap=cfg.first_rec_chunk_overlap,
            stage="first_recursive",
            skip_existing=not cfg.clear_first_rec_chunk_data,
            update_existing=cfg.update_first_rec_chunk_data,
            sync_deleted=cfg.update_first_rec_chunk_data,
        )

    recursive_chunk_dir(
        raw_data_dir=raw_docs_dir,
        chunk_dir=first_rec_docs_dir,
        chunk_size=cfg.first_rec_chunk_size,
        chunk_overlap=cfg.first_rec_chunk_overlap,
        separators=DOCS_SEPARATORS,
        source_type="docs",
        stage="first_recursive",
        skip_existing=not cfg.clear_first_rec_chunk_data,
        update_existing=cfg.update_first_rec_chunk_data,
        sync_deleted=cfg.update_first_rec_chunk_data,
    )
    recursive_chunk_dir(
        raw_data_dir=raw_sheets_dir,
        chunk_dir=first_rec_sheets_dir,
        chunk_size=cfg.first_rec_chunk_size,
        chunk_overlap=cfg.first_rec_chunk_overlap,
        separators=SHEETS_SEPARATORS,
        source_type="sheets",
        stage="first_recursive",
        file_extensions=(".csv",),
        skip_existing=not cfg.clear_first_rec_chunk_data,
        update_existing=cfg.update_first_rec_chunk_data,
        sync_deleted=cfg.update_first_rec_chunk_data,
    )

    if cfg.second_rec_enabled:
        recursive_chunk_jsonl_dir(
            input_chunk_dir=first_rec_docs_dir,
            output_chunk_dir=second_rec_docs_dir,
            chunk_size=cfg.second_rec_chunk_size,
            chunk_overlap=cfg.second_rec_chunk_overlap,
            separators=DOCS_SEPARATORS,
            stage="second_recursive",
            skip_existing=not cfg.clear_second_rec_chunk_data,
            update_existing=cfg.update_second_rec_chunk_data,
            sync_deleted=cfg.update_second_rec_chunk_data,
        )
        recursive_chunk_jsonl_dir(
            input_chunk_dir=first_rec_sheets_dir,
            output_chunk_dir=second_rec_sheets_dir,
            chunk_size=cfg.second_rec_chunk_size,
            chunk_overlap=cfg.second_rec_chunk_overlap,
            separators=SHEETS_SEPARATORS,
            stage="second_recursive",
            skip_existing=not cfg.clear_second_rec_chunk_data,
            update_existing=cfg.update_second_rec_chunk_data,
            sync_deleted=cfg.update_second_rec_chunk_data,
        )
        if first_rec_messages_dir.exists():
            recursive_chunk_jsonl_dir(
                input_chunk_dir=first_rec_messages_dir,
                output_chunk_dir=second_rec_messages_dir,
                chunk_size=cfg.second_rec_chunk_size,
                chunk_overlap=cfg.second_rec_chunk_overlap,
                separators=MESSAGE_SEPARATORS,
                stage="second_recursive",
                skip_existing=not cfg.clear_second_rec_chunk_data,
                update_existing=cfg.update_second_rec_chunk_data,
                sync_deleted=cfg.update_second_rec_chunk_data,
            )
        sparse_chunk_jsonl_dir(
            input_chunk_dir=second_rec_docs_dir,
            output_chunk_dir=sparse_second_rec_docs_dir,
            config=cfg,
            skip_existing=not cfg.clear_second_rec_chunk_data,
            update_existing=cfg.update_sparse_second_rec_chunk_data,
            sync_deleted=cfg.update_sparse_second_rec_chunk_data,
        )
        sparse_chunk_jsonl_dir(
            input_chunk_dir=second_rec_sheets_dir,
            output_chunk_dir=sparse_second_rec_sheets_dir,
            config=cfg,
            skip_existing=not cfg.clear_second_rec_chunk_data,
            update_existing=cfg.update_sparse_second_rec_chunk_data,
            sync_deleted=cfg.update_sparse_second_rec_chunk_data,
        )
        if second_rec_messages_dir.exists():
            sparse_chunk_jsonl_dir(
                input_chunk_dir=second_rec_messages_dir,
                output_chunk_dir=sparse_second_rec_messages_dir,
                config=cfg,
                skip_existing=not cfg.clear_second_rec_chunk_data,
                update_existing=cfg.update_sparse_second_rec_chunk_data,
                sync_deleted=cfg.update_sparse_second_rec_chunk_data,
            )

    if cfg.summery_enabled:
        summery_chunk_jsonl_dir(
            input_chunk_dir=first_rec_docs_dir,
            output_chunk_dir=summery_docs_dir,
            config=cfg,
            skip_existing=not cfg.clear_summery_chunk_data,
            update_existing=cfg.update_summery_chunk_data,
            sync_deleted=cfg.update_summery_chunk_data,
        )
        summery_chunk_jsonl_dir(
            input_chunk_dir=first_rec_sheets_dir,
            output_chunk_dir=summery_sheets_dir,
            config=cfg,
            skip_existing=not cfg.clear_summery_chunk_data,
            update_existing=cfg.update_summery_chunk_data,
            sync_deleted=cfg.update_summery_chunk_data,
        )
        if first_rec_messages_dir.exists():
            summery_chunk_jsonl_dir(
                input_chunk_dir=first_rec_messages_dir,
                output_chunk_dir=summery_messages_dir,
                config=cfg,
                skip_existing=not cfg.clear_summery_chunk_data,
                update_existing=cfg.update_summery_chunk_data,
                sync_deleted=cfg.update_summery_chunk_data,
            )

    if cfg.prop_enabled:
        if not cfg.second_rec_enabled:
            logger.warning(
                "Proposition chunking is enabled but SECOND_REC is disabled. Skipping."
            )
        else:
            proposition_chunk_jsonl_dir(
                input_chunk_dir=second_rec_docs_dir,
                output_chunk_dir=prop_docs_dir,
                config=cfg,
                skip_existing=not cfg.clear_prop_chunk_data,
                update_existing=cfg.update_prop_chunk_data,
                sync_deleted=cfg.update_prop_chunk_data,
            )
            proposition_chunk_jsonl_dir(
                input_chunk_dir=second_rec_sheets_dir,
                output_chunk_dir=prop_sheets_dir,
                config=cfg,
                skip_existing=not cfg.clear_prop_chunk_data,
                update_existing=cfg.update_prop_chunk_data,
                sync_deleted=cfg.update_prop_chunk_data,
            )

    if cfg.raptor_enabled:
        if cfg.summery_enabled:
            raptor_input_dirs = [summery_docs_dir, summery_sheets_dir]
        elif cfg.second_rec_enabled:
            raptor_input_dirs = [second_rec_docs_dir, second_rec_sheets_dir]
        else:
            raptor_input_dirs = [first_rec_docs_dir, first_rec_sheets_dir]
        raptor_chunk_global(
            input_chunk_dirs=raptor_input_dirs,
            output_chunk_dir=cfg.raptor_chunk_dir,
            config=cfg,
            skip_existing=not cfg.clear_raptor_chunk_data,
            update_existing=cfg.update_raptor_chunk_data,
            sync_deleted=cfg.update_raptor_chunk_data,
        )

    index_chunks = []
    base_chunk_dirs = []
    if cfg.second_rec_enabled:
        base_chunk_dirs.extend([second_rec_docs_dir, second_rec_sheets_dir])
        if second_rec_messages_dir.exists():
            base_chunk_dirs.append(second_rec_messages_dir)
    else:
        base_chunk_dirs.extend([first_rec_docs_dir, first_rec_sheets_dir])
        if first_rec_messages_dir.exists():
            base_chunk_dirs.append(first_rec_messages_dir)
    if base_chunk_dirs:
        index_chunks.extend(load_chunks_from_dirs(base_chunk_dirs))
    if cfg.prop_enabled and cfg.second_rec_enabled:
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
