from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import kumc_agent.infra.loaders.crafters_colony_impl as crafters_impl


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


if __name__ == "__main__":
    unittest.main()
