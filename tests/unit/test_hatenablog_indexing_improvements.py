from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.infra.loaders.hatenablog_impl import _normalize_hatenablog_html


class HatenaBlogIndexingImprovementTests(unittest.TestCase):
    def test_hatenablog_html_is_normalized_to_markdown(self) -> None:
        article = _normalize_hatenablog_html(
            """
            <ul class="table-of-contents"><li>目次</li></ul>
            <h2 style="color:red">活動報告</h2>
            <p style="margin:0">KUMCの<a class="keyword" href="/keyword">例会</a>です。</p>
            <figure>
              <img src="https://example.com/image.png" alt="集合写真" style="width:100%">
              <figcaption>作品の写真</figcaption>
            </figure>
            <iframe class="embed-card" src="https://example.com/card"></iframe>
            """
        )

        self.assertIn("## 活動報告", article.markdown)
        self.assertIn("KUMCの例会です。", article.markdown)
        self.assertIn("![集合写真](https://example.com/image.png)", article.markdown)
        self.assertIn("作品の写真", article.markdown)
        self.assertIn("## 関連リンク", article.markdown)
        self.assertIn("https://example.com/card", article.markdown)
        self.assertNotIn("<p", article.markdown)
        self.assertNotIn("<iframe", article.markdown)
        self.assertNotIn("style=", article.markdown)
        self.assertNotIn("table-of-contents", article.markdown)
        self.assertEqual(article.metadata["hatenablog_image_count"], 1)
        self.assertEqual(article.metadata["hatenablog_related_link_count"], 1)


if __name__ == "__main__":
    unittest.main()
