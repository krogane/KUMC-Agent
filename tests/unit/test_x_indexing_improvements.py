from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.source import NormalizedDocument
from kumc_agent.features.image_search.service import _scan_x_images
from kumc_agent.features.ingestion.chunking import IngestionChunker
from kumc_agent.features.rag.components.generation import GenerationComponent
from kumc_agent.infra.connectors.file_scanner import iter_x_posts
from kumc_agent.infra.loaders.x_impl import convert_x_tweets_js_to_jsonl
from kumc_agent.utils.hashing import stable_hash


class _RecordingLLM:
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        return '{"answer":"参照しました","sources":[1]}'


class _PromptRepo:
    def get(self, name: str) -> str:
        if name == "system_rules":
            return "system"
        if name == "answer_rag":
            return '{"answer":"...", "sources":[1]}'
        raise FileNotFoundError(name)


class XIndexingImprovementTests(unittest.TestCase):
    def test_x_conversion_normalizes_urls_author_and_local_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "x"
            archive = root / "data"
            media_dir = archive / "tweets_media"
            media_dir.mkdir(parents=True)
            media_path = media_dir / "1730181931873243263-GALU7dAagAAtU_B.jpg"
            media_path.write_bytes(b"image-bytes")
            archive.joinpath("account.js").write_text(
                """
window.YTD.account.part0 = [
  {"account": {
    "email": "secret@example.com",
    "username": "KUMC_X",
    "accountId": "1728692402859458560",
    "accountDisplayName": "京大マイクラ同好会"
  }}
]
""",
                encoding="utf-8",
            )
            archive.joinpath("tweets.js").write_text(
                """
window.YTD.tweets.part0 = [
  {"tweet": {
    "id_str": "1730181931873243263",
    "created_at": "Thu Nov 30 11:08:12 +0000 2023",
    "full_text": "企画詳細 https://t.co/page 写真 https://t.co/media",
    "entities": {
      "urls": [
        {
          "url": "https://t.co/page",
          "expanded_url": "https://example.com/event",
          "display_url": "example.com/event"
        }
      ],
      "media": [
        {
          "url": "https://t.co/media",
          "expanded_url": "https://x.com/KUMC_X/status/1730181931873243263/photo/1",
          "media_url_https": "https://pbs.twimg.com/media/GALU7dAagAAtU_B.jpg",
          "type": "photo"
        }
      ]
    },
    "extended_entities": {
      "media": [
        {
          "url": "https://t.co/media",
          "expanded_url": "https://x.com/KUMC_X/status/1730181931873243263/photo/1",
          "media_url_https": "https://pbs.twimg.com/media/GALU7dAagAAtU_B.jpg",
          "type": "photo"
        }
      ]
    }
  }}
]
""",
                encoding="utf-8",
            )

            stats = convert_x_tweets_js_to_jsonl(
                raw_x_dir=root,
                output_path=root / "posts.jsonl",
                skip_existing=False,
                update_existing=True,
                sync_deleted=True,
            )
            record = json.loads(root.joinpath("posts.jsonl").read_text(encoding="utf-8"))

        metadata = record["metadata"]
        self.assertEqual(stats.posts, 1)
        self.assertEqual(record["text"], "企画詳細 https://example.com/event 写真")
        self.assertNotIn("https://t.co/", record["text"])
        self.assertEqual(metadata["x_author_handle"], "KUMC_X")
        self.assertEqual(
            metadata["x_post_url"],
            "https://x.com/KUMC_X/status/1730181931873243263",
        )
        self.assertEqual(metadata["x_expanded_urls"], ["https://example.com/event"])
        self.assertNotIn("email", metadata)
        self.assertEqual(
            metadata["x_media_urls"],
            ["https://pbs.twimg.com/media/GALU7dAagAAtU_B.jpg"],
        )
        self.assertEqual(
            metadata["x_media"][0]["local_relative_path"],
            "data/tweets_media/1730181931873243263-GALU7dAagAAtU_B.jpg",
        )
        self.assertEqual(
            metadata["x_media"][0]["content_hash"],
            hashlib.sha256(b"image-bytes").hexdigest(),
        )

    def test_x_posts_scanner_and_chunker_preserve_post_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "x"
            root.mkdir()
            root.joinpath("posts.jsonl").write_text(
                json.dumps(
                    {
                        "text": "投稿本文",
                        "metadata": {
                            "source_type": "x_posts",
                            "message_timestamp": "2023-11-30T11:08:12+00:00",
                            "x_post_id": "1730181931873243263",
                            "x_post_url": "https://x.com/KUMC_X/status/1730181931873243263",
                            "x_author_handle": "KUMC_X",
                            "email": "must-not-leak@example.com",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            items = iter_x_posts(source_kind="x", root_dir=root)

        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.external_id, "1730181931873243263")
        self.assertEqual(
            item.canonical_url,
            "https://x.com/KUMC_X/status/1730181931873243263",
        )
        self.assertEqual(item.metadata["source_type"], "x_posts")
        self.assertNotIn("email", item.metadata)

        document = NormalizedDocument(
            id=stable_hash(f"document:{item.external_id}"),
            source_item_id=stable_hash(f"x:{item.external_id}"),
            source_kind=item.source_kind,
            external_id=item.external_id,
            version=1,
            title=item.title,
            normalized_text=item.text,
            normalized_format="plain",
            language="ja",
            access_scope=item.access_scope,
            checksum=item.checksum,
            metadata=dict(item.metadata),
        )
        chunks = IngestionChunker().chunk(document)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["chunk_kind"], "x_post")
        self.assertEqual(chunks[0].metadata["source_type"], "x_posts")
        self.assertEqual(chunks[0].metadata["x_post_id"], "1730181931873243263")
        self.assertNotEqual(chunks[0].metadata["external_id"], "x:posts.jsonl")

    def test_x_citation_uses_post_url(self) -> None:
        component = GenerationComponent(
            llm=_RecordingLLM(),
            prompts=_PromptRepo(),
            source_max_count=3,
        )
        answer = component.generate_rag_answer(
            query="質問",
            chunks=[
                Chunk(
                    id="chunk-1",
                    document_id="doc-1",
                    text="投稿本文",
                    index=0,
                    metadata={
                        "source_type": "x_posts",
                        "x_post_id": "1730181931873243263",
                        "x_author_handle": "KUMC_X",
                    },
                )
            ],
            history=None,
            temperature=0.0,
            max_output_tokens=128,
            append_sources_to_response=False,
        )
        self.assertEqual(
            answer.sources[0].uri,
            "https://x.com/KUMC_X/status/1730181931873243263",
        )

    def test_x_image_scan_prefers_local_archive_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ingestion = Path(tmp)
            media_path = ingestion / "x" / "data" / "tweets_media" / "1-photo.jpg"
            media_path.parent.mkdir(parents=True)
            media_path.write_bytes(b"local")
            (ingestion / "x" / "posts.jsonl").write_text(
                json.dumps(
                    {
                        "text": "画像投稿",
                        "metadata": {
                            "source_type": "x_posts",
                            "x_post_id": "1",
                            "x_post_url": "https://x.com/KUMC_X/status/1",
                            "message_timestamp": "2023-11-30T11:08:12+00:00",
                            "x_media": [
                                {
                                    "type": "photo",
                                    "remote_url": "https://pbs.twimg.com/media/photo.jpg",
                                    "local_relative_path": "data/tweets_media/1-photo.jpg",
                                    "content_hash": "hash",
                                    "thumbnail_remote_url": "https://pbs.twimg.com/media/photo.jpg",
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = _scan_x_images(ingestion)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].image_ref, str(media_path))
        self.assertEqual(
            candidates[0].metadata["x_media_local_relative_path"],
            "data/tweets_media/1-photo.jpg",
        )


if __name__ == "__main__":
    unittest.main()
