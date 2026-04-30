from __future__ import annotations

from pathlib import Path


class NotionLoader:
    def __init__(
        self,
        *,
        api_token: str,
        database_ids: list[str],
        page_ids: list[str] | None = None,
        ingestion_dir: Path,
        default_visibility: str = "public",
    ) -> None:
        self._api_token = api_token
        self._database_ids = list(database_ids)
        self._page_ids = list(page_ids or [])
        self._ingestion_dir = ingestion_dir
        self._default_visibility = default_visibility
        self._last_sync_metadata: dict[str, object] = {}

    def load(self) -> int:
        token = (self._api_token or "").strip()
        database_ids = [
            str(value).strip() for value in self._database_ids if str(value).strip()
        ]
        page_ids = [
            str(value).strip() for value in self._page_ids if str(value).strip()
        ]
        if not token or (not database_ids and not page_ids):
            self._last_sync_metadata = {"pages_seen": 0, "pages_updated": 0}
            return 0

        from kumc_agent.infra.loaders.notion_impl import download_notion_database_pages

        output_dir = self._ingestion_dir / "notion"
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = download_notion_database_pages(
            api_token=token,
            database_ids=database_ids,
            page_ids=page_ids,
            output_dir=output_dir,
            skip_existing=True,
            update_existing=True,
            sync_deleted=True,
            default_visibility=self._default_visibility,
            return_stats=True,
        )
        self._last_sync_metadata = (
            stats.as_dict() if hasattr(stats, "as_dict") else {"pages_updated": int(stats)}
        )
        return int(getattr(stats, "pages_updated", stats))

    def sync_metadata(self) -> dict[str, object]:
        return dict(self._last_sync_metadata)
