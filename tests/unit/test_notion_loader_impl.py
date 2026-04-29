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
            self.assertEqual(render_mock.call_count, 2)

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

    def test_downloads_standalone_page_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            page = NotionPage(
                page_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                title="Standalone Page",
                url="https://www.notion.so/workspace/standalone-page",
                last_edited_time="2026-01-01T00:00:00Z",
                created_time="2026-01-01T00:00:00Z",
            )

            with (
                patch.object(notion_impl, "_retrieve_page", return_value=page) as retrieve_mock,
                patch.object(
                    notion_impl,
                    "_render_page_markdown",
                    return_value="# Standalone Page\n\nbody\n",
                ) as render_mock,
            ):
                first = notion_impl.download_notion_database_pages(
                    api_token="token",
                    database_ids=[],
                    page_ids=["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                    sync_deleted=True,
                )
                second = notion_impl.download_notion_database_pages(
                    api_token="token",
                    database_ids=[],
                    page_ids=["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                    sync_deleted=True,
                )

            self.assertEqual(first, 1)
            self.assertEqual(second, 0)
            self.assertEqual(retrieve_mock.call_count, 2)
            self.assertEqual(render_mock.call_count, 2)
            pages_dir = output_dir / "pages"
            files = sorted(path.name for path in pages_dir.glob("*.md"))
            self.assertEqual(len(files), 1)
            self.assertIn("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", files[0])
            metadata = notion_impl._read_page_metadata(pages_dir / files[0])
            self.assertEqual(metadata["notion_database_id"], "")
            self.assertEqual(metadata["notion_page_id"], "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    def test_recursively_downloads_linked_pages(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            parent = NotionPage(
                page_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                title="Parent",
                url="https://www.notion.so/workspace/parent",
                last_edited_time="2026-01-01T00:00:00Z",
                created_time="2026-01-01T00:00:00Z",
            )
            child = NotionPage(
                page_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
                title="Child",
                url="https://www.notion.so/workspace/child",
                last_edited_time="2026-01-01T00:00:00Z",
                created_time="2026-01-01T00:00:00Z",
            )

            def retrieve_page(*, api_token: str, page_id: str) -> NotionPage:
                if page_id.replace("-", "") == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa":
                    return parent
                return child

            def render_page(*, api_token: str, page: NotionPage, references: set[tuple[str, str]] | None = None) -> str:
                if page.page_id == parent.page_id and references is not None:
                    references.add(("page", child.page_id))
                return f"# {page.title}\n\nbody\n"

            with (
                patch.object(notion_impl, "_retrieve_page", side_effect=retrieve_page),
                patch.object(notion_impl, "_render_page_markdown", side_effect=render_page),
            ):
                updated = notion_impl.download_notion_database_pages(
                    api_token="token",
                    database_ids=[],
                    page_ids=[parent.page_id],
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                    sync_deleted=True,
                )

            self.assertEqual(updated, 2)
            files = sorted(path.name for path in (output_dir / "pages").glob("*.md"))
            self.assertEqual(len(files), 2)
            self.assertIn("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa__Parent.md", files)
            self.assertIn("cccccccccccccccccccccccccccccccc__Child.md", files)

    def test_recursively_downloads_linked_database_pages(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            parent = NotionPage(
                page_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                title="Parent",
                url="https://www.notion.so/workspace/parent",
                last_edited_time="2026-01-01T00:00:00Z",
                created_time="2026-01-01T00:00:00Z",
            )
            db_page = NotionPage(
                page_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
                title="DB Page",
                url="https://www.notion.so/workspace/db-page",
                last_edited_time="2026-01-01T00:00:00Z",
                created_time="2026-01-01T00:00:00Z",
            )
            db_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

            def render_page(*, api_token: str, page: NotionPage, references: set[tuple[str, str]] | None = None) -> str:
                if page.page_id == parent.page_id and references is not None:
                    references.add(("database", db_id))
                return f"# {page.title}\n\nbody\n"

            with (
                patch.object(notion_impl, "_retrieve_page", return_value=parent),
                patch.object(notion_impl, "_list_database_pages", return_value=[db_page]),
                patch.object(notion_impl, "_render_page_markdown", side_effect=render_page),
            ):
                updated = notion_impl.download_notion_database_pages(
                    api_token="token",
                    database_ids=[],
                    page_ids=[parent.page_id],
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                    sync_deleted=True,
                )

            self.assertEqual(updated, 2)
            db_files = sorted(path.name for path in (output_dir / db_id.replace("-", "")).glob("*.md"))
            self.assertEqual(db_files, ["dddddddddddddddddddddddddddddddd__DB_Page.md"])

    def test_collects_only_notion_page_urls_from_rich_text_links(self) -> None:
        references = notion_impl._collect_block_references(
            {
                "id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "plain_text": "external",
                            "href": "https://example.com/ffffffffffffffffffffffffffffffff",
                        },
                        {
                            "plain_text": "notion",
                            "href": "https://www.notion.so/ws/Page-99999999999999999999999999999999",
                        },
                    ],
                },
            }
        )

        self.assertEqual(references, {("page", "99999999999999999999999999999999")})


if __name__ == "__main__":
    unittest.main()
