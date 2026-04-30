from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Callable

from kumc_agent.domain.models.source import (
    BackfillScope,
    NormalizedDocument,
    SourceDeleteItem,
    SourceRawItem,
    SyncCursor,
)
from kumc_agent.utils.hashing import stable_hash


@dataclass
class LoaderBackedConnector:
    source_kind: str
    loader: object
    raw_items: Callable[[], list[SourceRawItem | SourceDeleteItem]]
    normalized_format: str = "markdown"
    supports_incremental: bool = False
    _last_sync_metadata: dict[str, object] | None = None

    async def backfill(
        self,
        scope: BackfillScope,
    ) -> AsyncIterator[SourceRawItem | SourceDeleteItem]:
        load = getattr(self.loader, "load", None)
        if callable(load):
            result = await asyncio.to_thread(load)
            self._last_sync_metadata = self._sync_metadata_from_loader(result)
        count = 0
        for item in self.raw_items():
            if scope.limit is not None and count >= scope.limit:
                break
            count += 1
            yield item

    def sync_metadata(self) -> dict[str, object]:
        return dict(self._last_sync_metadata or {})

    def _sync_metadata_from_loader(self, load_result: object) -> dict[str, object]:
        metadata_fn = getattr(self.loader, "sync_metadata", None)
        if callable(metadata_fn):
            value = metadata_fn()
            if isinstance(value, dict):
                return {str(key): item for key, item in value.items()}
        return {"loaded": int(load_result)} if isinstance(load_result, int) else {}

    async def poll_changes(
        self,
        cursor: SyncCursor,
    ) -> AsyncIterator[SourceRawItem | SourceDeleteItem]:
        async for item in self.backfill(BackfillScope()):
            yield item

    async def fetch_item(self, external_id: str) -> SourceRawItem:
        for item in self.raw_items():
            if not isinstance(item, SourceRawItem):
                continue
            if item.external_id == external_id:
                return item
        raise KeyError(f"Source item not found: {self.source_kind}:{external_id}")

    async def normalize(self, raw: SourceRawItem) -> NormalizedDocument:
        source_item_id = stable_hash(f"{raw.source_kind}:{raw.external_id}")
        return NormalizedDocument(
            id=stable_hash(f"document:{source_item_id}:{raw.checksum}"),
            source_item_id=source_item_id,
            source_kind=raw.source_kind,
            external_id=raw.external_id,
            version=1,
            title=raw.title,
            normalized_text=raw.text,
            normalized_format=self.normalized_format,
            language="ja",
            access_scope=raw.access_scope,
            checksum=raw.checksum,
            metadata=dict(raw.metadata),
        )
