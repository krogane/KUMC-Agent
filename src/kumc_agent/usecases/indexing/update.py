from __future__ import annotations

from dataclasses import dataclass
import threading

from kumc_agent.features.indexing.service import IndexBuildResult, IndexingService
from kumc_agent.usecases.indexing.build import BuildIndexUsecase, BuildIndexRequest


@dataclass(frozen=True)
class UpdateIndexRequest:
    refresh_sources: bool = True
    full_rebuild: bool = False
    stage_selection: tuple[str, ...] | None = None
    allow_cancel: bool = False
    cancel_event: threading.Event | None = None


class UpdateIndexUsecase:
    def __init__(self, *, build_usecase: BuildIndexUsecase, indexing_service: IndexingService) -> None:
        self._build_usecase = build_usecase
        self._indexing_service = indexing_service

    def execute(self, request: UpdateIndexRequest) -> IndexBuildResult:
        # For now, update pipeline reuses build pipeline and overwrites index artifacts.
        return self._build_usecase.execute(
            BuildIndexRequest(
                refresh_sources=request.refresh_sources,
                full_rebuild=request.full_rebuild,
                stage_selection=request.stage_selection,
                allow_cancel=request.allow_cancel,
                cancel_event=request.cancel_event,
            )
        )
