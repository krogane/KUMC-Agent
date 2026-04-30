from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import kumc_agent.infra.loaders.crafters_colony_impl as crafters_impl
from kumc_agent.infra.connectors.file_scanner import iter_raw_files


class CraftersColonyImageTests(unittest.TestCase):
    def test_html_to_markdown_preserves_article_images(self) -> None:
        markdown = crafters_impl._html_to_markdown(
            """
            <article>
              <p>イベントの紹介です。</p>
              <img src="/uploads/poster.png" alt="新歓ポスター">
              <p>詳細は後日案内します。</p>
            </article>
            """,
            base_url="https://crafters.example/articles/1",
        )

        self.assertIn("イベントの紹介です。", markdown)
        self.assertIn(
            "![新歓ポスター](https://crafters.example/uploads/poster.png)",
            markdown,
        )
        self.assertIn("詳細は後日案内します。", markdown)

    def test_article_body_extraction_removes_ui_noise_without_dropping_body_images(self) -> None:
        body_html = crafters_impl._extract_article_body_html(
            """
            <article>
              <div class="entry-content">
                <h1>京大トレジャーラン created by KUMC</h1>
                <p>配布対象: Java Edition</p>
                <img src="/uploads/body.png" alt="本文画像">
                <img class="post-ratings-image" src="/wp-content/rating.gif" alt="rating">
                <p>説明文はここに残ります。</p>
              </div>
              <div id="comments">
                <h2>コメントを書き込む</h2>
                <p>メールアドレスが公開されることはありません</p>
              </div>
              <aside class="sidebar widget">
                <h2>キーワードからさがす</h2>
                <p>サイト運営</p>
              </aside>
            </article>
            """
        )
        markdown = crafters_impl._html_to_markdown(
            body_html,
            base_url="https://crafters.example/12345/",
        )
        document = crafters_impl._build_markdown(
            crafters_impl.CraftersColonyEntry(
                article_url="https://crafters.example/12345/",
                title="京大トレジャーラン created by KUMC",
                published_at="2024-01-01T00:00:00+09:00",
                updated_at="2024-01-02T00:00:00+09:00",
                article_id="12345",
            ),
            body_markdown=markdown,
        )

        self.assertIn("配布対象: Java Edition", document)
        self.assertIn("説明文はここに残ります。", document)
        self.assertIn(
            "![本文画像](https://crafters.example/uploads/body.png)",
            document,
        )
        self.assertNotIn("rating.gif", document)
        self.assertNotIn("コメントを書き込む", document)
        self.assertNotIn("メールアドレスが公開されることはありません", document)
        self.assertNotIn("キーワードからさがす", document)
        self.assertNotIn("サイト運営", document)
        self.assertEqual(document.count("# 京大トレジャーラン created by KUMC"), 1)

    def test_download_writes_article_id_updated_at_and_uses_updated_revision(self) -> None:
        first_html = """
        <html>
          <head>
            <meta property="og:title" content="京大トレジャーラン created by KUMC">
            <meta property="article:published_time" content="2024-01-01T00:00:00+09:00">
            <meta property="article:modified_time" content="2024-01-02T03:04:05+09:00">
          </head>
          <body>
            <article>
              <div class="entry-content">
                <p>初回の本文です。</p>
                <img src="/uploads/body.png" alt="本文画像">
              </div>
            </article>
          </body>
        </html>
        """
        second_html = first_html.replace(
            "2024-01-02T03:04:05+09:00",
            "2024-01-03T03:04:05+09:00",
        ).replace("初回の本文です。", "更新後の本文です。")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "crafters_colony"
            with (
                patch(
                    "kumc_agent.infra.loaders.crafters_colony_impl._collect_article_urls",
                    return_value=["https://crafters.example/12345/"],
                ),
                patch(
                    "kumc_agent.infra.loaders.crafters_colony_impl._http_get_text",
                    return_value=first_html,
                ),
            ):
                downloaded = crafters_impl.download_crafters_colony_articles(
                    author_url="https://crafters.example/member/kumc/",
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                )
            self.assertEqual(downloaded, 1)

            markdown_path = next(output_dir.glob("*.md"))
            metadata_path = markdown_path.with_suffix(markdown_path.suffix + ".meta.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["crafters_colony_article_id"], "12345")
            self.assertEqual(
                metadata["crafters_colony_updated_at"],
                "2024-01-02T03:04:05+09:00",
            )
            self.assertTrue(metadata["crafters_colony_content_checksum"])

            items = iter_raw_files(
                source_kind="crafters_colony",
                root_dir=output_dir,
                extensions={".md"},
                default_visibility="public",
            )
            self.assertEqual(items[0].external_id, "12345")
            self.assertEqual(
                items[0].updated_at.isoformat(),
                "2024-01-02T03:04:05+09:00",
            )

            with (
                patch(
                    "kumc_agent.infra.loaders.crafters_colony_impl._collect_article_urls",
                    return_value=["https://crafters.example/12345/"],
                ),
                patch(
                    "kumc_agent.infra.loaders.crafters_colony_impl._http_get_text",
                    return_value=second_html,
                ),
            ):
                downloaded = crafters_impl.download_crafters_colony_articles(
                    author_url="https://crafters.example/member/kumc/",
                    output_dir=output_dir,
                    skip_existing=True,
                    update_existing=True,
                )

            self.assertEqual(downloaded, 1)
            updated_markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("更新後の本文です。", updated_markdown)
            updated_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                updated_metadata["crafters_colony_updated_at"],
                "2024-01-03T03:04:05+09:00",
            )


if __name__ == "__main__":
    unittest.main()
