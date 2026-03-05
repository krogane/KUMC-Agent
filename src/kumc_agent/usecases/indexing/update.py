from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.features.indexing.service import IndexBuildResult, IndexingService
from kumc_agent.usecases.indexing.build import BuildIndexUsecase, BuildIndexRequest


@dataclass(frozen=True)
class UpdateIndexRequest:
    refresh_sources: bool = True


class UpdateIndexUsecase:
    def __init__(self, *, build_usecase: BuildIndexUsecase, indexing_service: IndexingService) -> None:
        self._build_usecase = build_usecase
        self._indexing_service = indexing_service

    def execute(self, request: UpdateIndexRequest) -> IndexBuildResult:
        # For now, update pipeline reuses build pipeline and overwrites index artifacts.
        return self._build_usecase.execute(BuildIndexRequest(refresh_sources=request.refresh_sources))
