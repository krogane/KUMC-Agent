from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.infra.loaders.crafters_colony import CraftersColonyLoader
from kumc_agent.infra.loaders.google_drive import GoogleDriveLoader
from kumc_agent.infra.loaders.hatenablog import HatenaBlogLoader
from kumc_agent.infra.loaders.notion import NotionLoader
from kumc_agent.infra.loaders.x import XPostsLoader
from kumc_agent.infra.loaders.x_impl import XConvertStats


class RawLoaderUpdatePolicyTests(unittest.TestCase):
    def test_google_drive_loader_enables_up_to_date_skip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            loader = GoogleDriveLoader(
                folder_id="folder-id",
                credentials_path="/tmp/creds.json",
                ingestion_dir=Path(td),
                max_files=100,
                batch_size=20,
                download_max_retries=3,
                download_retry_initial_delay_seconds=0.5,
                download_retry_max_delay_seconds=8.0,
                download_retry_backoff_multiplier=2.0,
                pdf_ocr_model_path="/tmp/ocr-model",
            )

            with patch(
                "kumc_agent.infra.loaders.google_drive_impl.download_drive_markdown",
                return_value=(2, 3),
            ) as mocked:
                loaded = loader.load()

            self.assertEqual(5, loaded)
            mocked.assert_called_once()
            kwargs = mocked.call_args.kwargs
            self.assertTrue(kwargs["skip_existing"])
            self.assertTrue(kwargs["update_existing"])
            self.assertTrue(kwargs["sync_deleted"])

    def test_hatenablog_loader_enables_up_to_date_skip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            loader = HatenaBlogLoader(
                ingestion_dir=Path(td),
                blog_url="https://example.com/",
            )
            with patch(
                "kumc_agent.infra.loaders.hatenablog_impl.download_hatenablog_articles",
                return_value=7,
            ) as mocked:
                loaded = loader.load()

            self.assertEqual(7, loaded)
            mocked.assert_called_once()
            kwargs = mocked.call_args.kwargs
            self.assertTrue(kwargs["skip_existing"])
            self.assertTrue(kwargs["update_existing"])
            self.assertTrue(kwargs["sync_deleted"])

    def test_crafters_colony_loader_enables_up_to_date_skip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            loader = CraftersColonyLoader(
                ingestion_dir=Path(td),
                author_url="https://example.com",
                max_pages=3,
                max_articles=20,
            )
            with patch(
                "kumc_agent.infra.loaders.crafters_colony_impl.download_crafters_colony_articles",
                return_value=4,
            ) as mocked:
                loaded = loader.load()

            self.assertEqual(4, loaded)
            mocked.assert_called_once()
            kwargs = mocked.call_args.kwargs
            self.assertTrue(kwargs["skip_existing"])
            self.assertTrue(kwargs["update_existing"])
            self.assertTrue(kwargs["sync_deleted"])

    def test_x_loader_enables_up_to_date_skip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            loader = XPostsLoader(ingestion_dir=Path(td))
            with patch(
                "kumc_agent.infra.loaders.x_impl.convert_x_tweets_js_to_jsonl",
                return_value=XConvertStats(files=1, posts=9, skipped_invalid=0),
            ) as mocked:
                loaded = loader.load()

            self.assertEqual(9, loaded)
            mocked.assert_called_once()
            kwargs = mocked.call_args.kwargs
            self.assertTrue(kwargs["skip_existing"])
            self.assertTrue(kwargs["update_existing"])
            self.assertTrue(kwargs["sync_deleted"])

    def test_notion_loader_enables_up_to_date_skip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            loader = NotionLoader(
                api_token="token",
                database_ids=["db-1"],
                ingestion_dir=Path(td),
            )
            with patch(
                "kumc_agent.infra.loaders.notion_impl.download_notion_database_pages",
                return_value=6,
            ) as mocked:
                loaded = loader.load()

            self.assertEqual(6, loaded)
            mocked.assert_called_once()
            kwargs = mocked.call_args.kwargs
            self.assertTrue(kwargs["skip_existing"])
            self.assertTrue(kwargs["update_existing"])
            self.assertTrue(kwargs["sync_deleted"])


if __name__ == "__main__":
    unittest.main()
