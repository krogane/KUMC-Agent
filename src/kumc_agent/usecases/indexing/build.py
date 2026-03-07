from __future__ import annotations

from dataclasses import dataclass
import threading

from kumc_agent.features.indexing.service import IndexBuildResult, IndexingService
from kumc_agent.infra.loaders.crafters_colony import CraftersColonyLoader
from kumc_agent.infra.loaders.discord import DiscordLoader
from kumc_agent.infra.loaders.google_drive import GoogleDriveLoader
from kumc_agent.infra.loaders.hatenablog import HatenaBlogLoader


@dataclass(frozen=True)
class BuildIndexRequest:
    refresh_sources: bool = True
    full_rebuild: bool = False
    stage_selection: tuple[str, ...] | None = None
    allow_cancel: bool = False
    cancel_event: threading.Event | None = None


class BuildIndexUsecase:
    def __init__(
        self,
        *,
        indexing_service: IndexingService,
        drive_loader: GoogleDriveLoader | None,
        discord_loader: DiscordLoader | None,
        hatenablog_loader: HatenaBlogLoader | None,
        crafters_colony_loader: CraftersColonyLoader | None,
    ) -> None:
        self._indexing_service = indexing_service
        self._drive_loader = drive_loader
        self._discord_loader = discord_loader
        self._hatenablog_loader = hatenablog_loader
        self._crafters_colony_loader = crafters_colony_loader

    def execute(self, request: BuildIndexRequest) -> IndexBuildResult:
        loaded = 0
        if request.refresh_sources:
            for loader in (
                self._discord_loader,
                self._drive_loader,
                self._hatenablog_loader,
                self._crafters_colony_loader,
            ):
                if loader is None:
                    continue
                loaded += loader.load()
        return self._indexing_service.build(
            loaded_sources=loaded,
            full_rebuild=request.full_rebuild,
            stage_selection=request.stage_selection,
            allow_cancel=request.allow_cancel,
            cancel_event=request.cancel_event,
        )
