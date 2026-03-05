from __future__ import annotations

from pathlib import Path


class HatenaBlogLoader:
    def __init__(self, *, raw_dir: Path, blog_url: str = "https://kumc.hatenablog.com/") -> None:
        self._raw_dir = raw_dir
        self._blog_url = blog_url

    def load(self) -> int:
        from kumc_agent.infra.legacy.indexing.hatenablog_loader import (
            download_hatenablog_articles,
        )

        output_dir = self._raw_dir / "hatenablog"
        output_dir.mkdir(parents=True, exist_ok=True)
        download_hatenablog_articles(
            blog_url=self._blog_url,
            output_dir=output_dir,
            skip_existing=False,
            update_existing=True,
            sync_deleted=True,
        )
        return 1
