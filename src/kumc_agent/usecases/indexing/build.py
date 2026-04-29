from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
from pathlib import Path

from kumc_agent.domain.models.source import BackfillScope
from kumc_agent.features.indexing.embedding_cache import IndexEmbeddingCacheKey
from kumc_agent.features.indexing.service import IndexBuildResult, IndexingService
from kumc_agent.infra.loaders.crafters_colony import CraftersColonyLoader
from kumc_agent.infra.loaders.discord import DiscordLoader
from kumc_agent.infra.loaders.google_drive import GoogleDriveLoader
from kumc_agent.infra.loaders.hatenablog import HatenaBlogLoader
from kumc_agent.infra.loaders.notion import NotionLoader
from kumc_agent.infra.loaders.x import XPostsLoader


@dataclass(frozen=True)
class BuildIndexRequest:
    refresh_sources: bool = True
    full_rebuild: bool = False
    stage_selection: tuple[str, ...] | None = None
    allow_cancel: bool = False
    cancel_event: threading.Event | None = None
    index_dir: Path | None = None
    prefer_ingestion_repository: bool = False


class BuildIndexUsecase:
    def __init__(
        self,
        *,
        indexing_service: IndexingService,
        drive_loader: GoogleDriveLoader | None,
        discord_loader: DiscordLoader | None,
        hatenablog_loader: HatenaBlogLoader | None,
        crafters_colony_loader: CraftersColonyLoader | None,
        x_loader: XPostsLoader | None,
        notion_loader: NotionLoader | None = None,
        ingestion_service: object | None = None,
    ) -> None:
        self._indexing_service = indexing_service
        self._drive_loader = drive_loader
        self._discord_loader = discord_loader
        self._hatenablog_loader = hatenablog_loader
        self._crafters_colony_loader = crafters_colony_loader
        self._x_loader = x_loader
        self._notion_loader = notion_loader
        self._ingestion_service = ingestion_service

    def execute(self, request: BuildIndexRequest) -> IndexBuildResult:
        loaded = 0
        if request.refresh_sources:
            for loader in (
                self._discord_loader,
                self._drive_loader,
                self._hatenablog_loader,
                self._crafters_colony_loader,
                self._x_loader,
                self._notion_loader,
            ):
                if loader is None:
                    continue
                loaded += loader.load()
            loaded += self._refresh_minecraft_wiki_source(force=request.full_rebuild)
        return self._indexing_service.build(
            loaded_sources=loaded,
            full_rebuild=request.full_rebuild,
            stage_selection=request.stage_selection,
            allow_cancel=request.allow_cancel,
            cancel_event=request.cancel_event,
            index_dir=request.index_dir,
            prefer_ingestion_repository=request.prefer_ingestion_repository,
        )

    def compact_embedding_cache(
        self,
        active_keys: tuple[IndexEmbeddingCacheKey, ...],
    ) -> dict[str, object]:
        return self._indexing_service.compact_embedding_cache(active_keys)

    def commit_staged_side_effects(self, index_dir: Path) -> dict[str, object]:
        commit = getattr(self._indexing_service, "commit_staged_side_effects", None)
        if not callable(commit):
            return {}
        return commit(index_dir)

    def _refresh_minecraft_wiki_source(self, *, force: bool) -> int:
        if self._ingestion_service is None:
            return 0
        backfill_many = getattr(self._ingestion_service, "backfill_many", None)
        if not callable(backfill_many):
            return 0
        available_sources = getattr(self._ingestion_service, "available_sources", None)
        if callable(available_sources) and "minecraft_wiki" not in set(available_sources()):
            return 0

        async def _run() -> int:
            results = await backfill_many(
                source_kinds=("minecraft_wiki",),
                scope=BackfillScope(force=force),
            )
            return sum(int(getattr(result, "seen", 0) or 0) for result in results)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(_run())).result()
