from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from kumc_agent.domain.models.source import (
    BackfillScope,
    NormalizedDocument,
    SourceDeleteItem,
    SourceRawItem,
    SyncCursor,
)


class SourceConnector(Protocol):
    source_kind: str

    async def backfill(self, scope: BackfillScope) -> AsyncIterator[SourceRawItem]:
        ...

    async def poll_changes(
        self,
        cursor: SyncCursor,
    ) -> AsyncIterator[SourceRawItem | SourceDeleteItem]:
        ...

    async def fetch_item(self, external_id: str) -> SourceRawItem:
        ...

    async def normalize(self, raw: SourceRawItem) -> NormalizedDocument:
        ...
