from __future__ import annotations

from pathlib import Path


class CraftersColonyLoader:
    def __init__(
        self,
        *,
        raw_dir: Path,
        author_url: str,
        max_pages: int,
        max_articles: int,
    ) -> None:
        self._raw_dir = raw_dir
        self._author_url = author_url
        self._max_pages = max_pages
        self._max_articles = max_articles

    def load(self) -> int:
        if not self._author_url:
            return 0
        from kumc_agent.infra.loaders.crafters_colony_impl import (
            download_crafters_colony_articles,
        )

        output_dir = self._raw_dir / "crafters_colony"
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded = download_crafters_colony_articles(
            author_url=self._author_url,
            output_dir=output_dir,
            max_pages=self._max_pages,
            max_articles=self._max_articles,
            skip_existing=True,
            update_existing=True,
            sync_deleted=True,
        )
        return int(downloaded)
