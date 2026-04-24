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

from kumc_agent.infra.loaders import notion_impl
from kumc_agent.infra.loaders.notion_impl import NotionPage


class NotionLoaderImplTests(unittest.TestCase):
    def test_skip_when_page_is_up_to_date(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            page = NotionPage(
                page_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                title="Page A",
                url="https://www.notion.so/workspace/page-a",
                last_edited_time="2026-01-01T00:00:00Z",
                created_time="2026-01-01T00:00:00Z",
            )

            with (
                patch.object(notion_impl, "_list_database_pages", return_value=[page]),
                patch.object(notion_impl, "_render_page_markdown", return_value="# Page A\n\nbody\n") as render_mock,
            ):
                first = notion_impl.download_notion_database_pages(
                    api_token="token",
                    database_ids=["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"],
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                    sync_deleted=True,
                )
                second = notion_impl.download_notion_database_pages(
                    api_token="token",
                    database_ids=["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"],
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                    sync_deleted=True,
                )

            self.assertEqual(first, 1)
            self.assertEqual(second, 0)
            self.assertEqual(render_mock.call_count, 1)

    def test_rebuild_when_last_edited_time_changed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            page_v1 = NotionPage(
                page_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                title="Page A",
                url="https://www.notion.so/workspace/page-a",
                last_edited_time="2026-01-01T00:00:00Z",
                created_time="2026-01-01T00:00:00Z",
            )
            page_v2 = NotionPage(
                page_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                title="Page A",
                url="https://www.notion.so/workspace/page-a",
                last_edited_time="2026-01-02T00:00:00Z",
                created_time="2026-01-01T00:00:00Z",
            )

            with (
                patch.object(notion_impl, "_list_database_pages", side_effect=[[page_v1], [page_v2]]),
                patch.object(notion_impl, "_render_page_markdown", return_value="# Page A\n\nbody\n") as render_mock,
            ):
                first = notion_impl.download_notion_database_pages(
                    api_token="token",
                    database_ids=["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"],
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                    sync_deleted=True,
                )
                second = notion_impl.download_notion_database_pages(
                    api_token="token",
                    database_ids=["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"],
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                    sync_deleted=True,
                )

            self.assertEqual(first, 1)
            self.assertEqual(second, 1)
            self.assertEqual(render_mock.call_count, 2)

    def test_sync_deleted_removes_stale_page_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            page_a = NotionPage(
                page_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                title="Page A",
                url="https://www.notion.so/workspace/page-a",
                last_edited_time="2026-01-01T00:00:00Z",
                created_time="2026-01-01T00:00:00Z",
            )
            page_b = NotionPage(
                page_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
                title="Page B",
                url="https://www.notion.so/workspace/page-b",
                last_edited_time="2026-01-01T00:00:00Z",
                created_time="2026-01-01T00:00:00Z",
            )

            db_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
            with (
                patch.object(notion_impl, "_list_database_pages", side_effect=[[page_a, page_b], [page_a]]),
                patch.object(notion_impl, "_render_page_markdown", return_value="# Page\n\nbody\n"),
            ):
                notion_impl.download_notion_database_pages(
                    api_token="token",
                    database_ids=[db_id],
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                    sync_deleted=True,
                )
                notion_impl.download_notion_database_pages(
                    api_token="token",
                    database_ids=[db_id],
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                    sync_deleted=True,
                )

            db_dir = output_dir / db_id.replace("-", "")
            remaining = sorted(path.name for path in db_dir.glob("*.md"))
            self.assertEqual(len(remaining), 1)
            self.assertIn("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", remaining[0])


if __name__ == "__main__":
    unittest.main()
