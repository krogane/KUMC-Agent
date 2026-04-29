from __future__ import annotations

from pathlib import Path

from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.domain.ports.connectors import SourceConnector
from kumc_agent.infra.connectors.base import LoaderBackedConnector
from kumc_agent.infra.connectors.file_scanner import (
    iter_raw_files,
    iter_structured_jsonl_records,
)
from kumc_agent.infra.connectors.minecraft_wiki import MinecraftWikiConnector
from kumc_agent.infra.loaders.crafters_colony import CraftersColonyLoader
from kumc_agent.infra.loaders.discord import DiscordLoader
from kumc_agent.infra.loaders.google_drive import GoogleDriveLoader
from kumc_agent.infra.loaders.hatenablog import HatenaBlogLoader
from kumc_agent.infra.loaders.notion import NotionLoader
from kumc_agent.infra.loaders.x import XPostsLoader


def build_source_connectors(config: RuntimeConfig) -> dict[str, SourceConnector]:
    ingestion_dir = config.app.ingestion_dir
    connectors: dict[str, SourceConnector] = {}
    if config.features.sources.drive:
        drive_loader = GoogleDriveLoader(
            folder_id=config.integrations.drive.folder_id,
            credentials_path=config.integrations.drive.google_application_credentials,
            ingestion_dir=ingestion_dir,
            max_files=config.integrations.drive.max_files,
            batch_size=config.integrations.drive.batch_size,
            download_max_retries=config.integrations.drive.download_max_retries,
            download_retry_initial_delay_seconds=(
                config.integrations.drive.download_retry_initial_delay_seconds
            ),
            download_retry_max_delay_seconds=(
                config.integrations.drive.download_retry_max_delay_seconds
            ),
            download_retry_backoff_multiplier=(
                config.integrations.drive.download_retry_backoff_multiplier
            ),
            pdf_ocr_model_path=config.integrations.drive.pdf_ocr_model_path,
            docs_quality_min_nonempty_characters=(
                config.indexing.docs_quality.min_nonempty_characters
            ),
            docs_quality_quarantine_low_information=(
                config.indexing.docs_quality.quarantine_low_information
            ),
        )
        connectors["google_drive"] = LoaderBackedConnector(
            source_kind="google_drive",
            loader=drive_loader,
            raw_items=lambda: [
                *_google_drive_doc_raw_items(ingestion_dir),
                *iter_raw_files(
                    source_kind="google_drive",
                    root_dir=ingestion_dir / "sheets",
                    extensions={".csv"},
                    default_visibility="admin",
                ),
            ],
            normalized_format="markdown",
        )
    if config.features.sources.discord:
        connectors["discord"] = LoaderBackedConnector(
            source_kind="discord",
            loader=DiscordLoader(
                bot_token=config.integrations.discord.bot_token,
                ingestion_dir=ingestion_dir,
                allow_guild_ids=config.security.discord_guild_allow_list,
            ),
            raw_items=lambda: iter_raw_files(
                source_kind="discord",
                root_dir=ingestion_dir / "messages",
                extensions={".jsonl"},
                default_visibility="guild",
            ),
            normalized_format="plain",
        )
    if config.features.sources.notion:
        connectors["notion"] = LoaderBackedConnector(
            source_kind="notion",
            loader=NotionLoader(
                api_token=config.integrations.notion.api_token,
                database_ids=config.integrations.notion.database_ids,
                page_ids=config.integrations.notion.page_ids,
                ingestion_dir=ingestion_dir,
            ),
            raw_items=lambda: iter_raw_files(
                source_kind="notion",
                root_dir=ingestion_dir / "notion",
                extensions={".md"},
                default_visibility="admin",
            ),
            normalized_format="markdown",
        )
    if config.features.sources.hatenablog:
        connectors["hatenablog"] = LoaderBackedConnector(
            source_kind="hatenablog",
            loader=HatenaBlogLoader(
                ingestion_dir=ingestion_dir,
                blog_url=config.integrations.hatenablog.blog_url,
            ),
            raw_items=lambda: iter_raw_files(
                source_kind="hatenablog",
                root_dir=ingestion_dir / "hatenablog",
                extensions={".md"},
                default_visibility="public",
            ),
            normalized_format="markdown",
        )
    if config.features.sources.x:
        connectors["x"] = LoaderBackedConnector(
            source_kind="x",
            loader=XPostsLoader(ingestion_dir=ingestion_dir),
            raw_items=lambda: iter_raw_files(
                source_kind="x",
                root_dir=ingestion_dir / "x",
                extensions={".jsonl"},
                default_visibility="public",
            ),
            normalized_format="plain",
        )
    if config.features.sources.crafters_colony:
        connectors["crafters_colony"] = LoaderBackedConnector(
            source_kind="crafters_colony",
            loader=CraftersColonyLoader(
                ingestion_dir=ingestion_dir,
                author_url=config.integrations.crafters_colony.author_url,
                max_pages=config.integrations.crafters_colony.max_pages,
                max_articles=config.integrations.crafters_colony.max_articles,
            ),
            raw_items=lambda: iter_raw_files(
                source_kind="crafters_colony",
                root_dir=ingestion_dir / "crafters_colony",
                extensions={".md"},
                default_visibility="public",
            ),
            normalized_format="markdown",
        )
    if config.features.sources.minecraft_wiki:
        connectors["minecraft_wiki"] = MinecraftWikiConnector(
            ingestion_dir=Path(ingestion_dir) / "minecraft_wiki",
            page_titles=tuple(config.integrations.minecraft_wiki.page_titles),
            api_url=config.integrations.minecraft_wiki.api_url,
            page_url_base=config.integrations.minecraft_wiki.page_url_base,
            max_pages=config.integrations.minecraft_wiki.max_pages,
            rate_limit_per_minute=(
                config.integrations.minecraft_wiki.rate_limit_per_minute
            ),
            request_interval_seconds=(
                config.integrations.minecraft_wiki.request_interval_seconds
            ),
            namespaces=tuple(config.integrations.minecraft_wiki.namespaces),
            full_backfill_enabled=(
                config.integrations.minecraft_wiki.full_backfill_enabled
            ),
            acquisition_mode=config.integrations.minecraft_wiki.acquisition_mode,
            category_sample_categories=dict(
                config.integrations.minecraft_wiki.category_sample.categories
            ),
            category_sample_per_category=(
                config.integrations.minecraft_wiki.category_sample.per_category_limit
            ),
        )
    return connectors


def _google_drive_doc_raw_items(ingestion_dir: Path):
    structured = iter_structured_jsonl_records(
        source_kind="google_drive",
        root_dir=ingestion_dir / "docs_normalized",
        default_visibility="admin",
    )
    structured_drive_ids = {
        str(item.metadata.get("drive_file_id") or "").strip()
        for item in structured
        if str(item.metadata.get("drive_file_id") or "").strip()
    }
    fallback = [
        item
        for item in iter_raw_files(
            source_kind="google_drive",
            root_dir=ingestion_dir / "docs",
            extensions={".md"},
            default_visibility="admin",
        )
        if str(item.metadata.get("drive_file_id") or "").strip()
        not in structured_drive_ids
    ]
    return [*structured, *fallback]
