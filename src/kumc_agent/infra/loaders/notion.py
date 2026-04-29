from __future__ import annotations

from pathlib import Path


class NotionLoader:
    def __init__(
        self,
        *,
        api_token: str,
        database_ids: list[str],
        ingestion_dir: Path,
    ) -> None:
        self._api_token = api_token
        self._database_ids = list(database_ids)
        self._ingestion_dir = ingestion_dir

    def load(self) -> int:
        token = (self._api_token or "").strip()
        database_ids = [str(value).strip() for value in self._database_ids if str(value).strip()]
        if not token or not database_ids:
            return 0

        from kumc_agent.infra.loaders.notion_impl import download_notion_database_pages

        output_dir = self._ingestion_dir / "notion"
        output_dir.mkdir(parents=True, exist_ok=True)
        return int(
            download_notion_database_pages(
                api_token=token,
                database_ids=database_ids,
                output_dir=output_dir,
                skip_existing=True,
                update_existing=True,
                sync_deleted=True,
            )
        )
