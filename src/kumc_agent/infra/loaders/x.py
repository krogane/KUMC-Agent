from __future__ import annotations

from pathlib import Path


class XPostsLoader:
    def __init__(self, *, ingestion_dir: Path) -> None:
        self._ingestion_dir = ingestion_dir

    def load(self) -> int:
        from kumc_agent.infra.loaders.x_impl import convert_x_tweets_js_to_jsonl

        output_dir = self._ingestion_dir / "x"
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = convert_x_tweets_js_to_jsonl(
            raw_x_dir=output_dir,
            output_path=output_dir / "posts.jsonl",
            skip_existing=True,
            update_existing=True,
            sync_deleted=True,
        )
        return int(stats.posts)
