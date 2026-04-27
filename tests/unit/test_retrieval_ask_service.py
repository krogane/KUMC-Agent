from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.retrieval import AccessContext, RetrievalQuery
from kumc_agent.features.retrieval.answering import ExtractiveAnswerBuilder
from kumc_agent.features.retrieval.ask import AskService
from kumc_agent.features.retrieval.citation import CitationValidator
from kumc_agent.features.retrieval.context import ContextPacker
from kumc_agent.features.retrieval.hybrid import HybridRetrievalConfig, HybridRetrievalService
from kumc_agent.infra.embeddings.local import LocalEmbedder
from kumc_agent.infra.retrieval_wave3 import FileRetrievalRepository


class RetrievalAskServiceTests(unittest.TestCase):
    def _write_chunk(
        self,
        root: Path,
        *,
        chunk_id: str,
        text: str,
        visibility: str = "public",
        guild_id: str = "",
        redaction_policy: str = "quote_allowed",
        index_status: str = "active",
        source_kind: str = "google_drive",
        source_item_id: str = "source-1",
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": chunk_id,
            "document_id": f"doc-{source_item_id}",
            "source_item_id": source_item_id,
            "chunk_index": 0,
            "chunk_kind": "body",
            "text": text,
            "metadata": {
                "source_item_id": source_item_id,
                "source_kind": source_kind,
                "source_title": f"title-{source_item_id}",
                "canonical_url": f"https://example.test/{source_item_id}",
                "access_scope": {
                    "visibility": visibility,
                    "guild_id": guild_id or None,
                    "role_ids": [],
                    "user_ids": [],
                    "source_acl_hash": "acl",
                },
                "redaction_policy": redaction_policy,
                "index_status": index_status,
                "checksum": f"checksum-{chunk_id}",
            },
        }
        with (root / "chunks.jsonl").open("a", encoding="utf-8") as fw:
            fw.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _service(self, root: Path) -> AskService:
        retrieval = HybridRetrievalService(
            repository=FileRetrievalRepository(root_dir=root),
            embedder=LocalEmbedder(model_name="", dimensions=64),
            reranker=None,
            config=HybridRetrievalConfig(
                dense_top_k=10,
                sparse_top_k=10,
                rerank_pool_size=10,
                top_k=4,
                doc_cap=2,
                embedding_model="test-hash",
                embedding_dimensions=64,
            ),
        )
        return AskService(
            retrieval=retrieval,
            packer=ContextPacker(),
            answer_builder=ExtractiveAnswerBuilder(citation_validator=CitationValidator()),
        )

    def test_ask_returns_citation_and_records_embedding_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_chunk(
                root,
                chunk_id="c1",
                text="新歓 企画 の集合時間は 10時 です",
            )

            response = self._service(root).ask(
                RetrievalQuery(
                    text="新歓 企画 集合時間",
                    access=AccessContext(is_admin=False),
                )
            )

            self.assertIn("主な情報源", response.text)
            self.assertEqual(response.citations[0].chunk_id, "c1")
            self.assertTrue((root / "embeddings.jsonl").exists())
            self.assertTrue((root / "search_runs.jsonl").exists())

    def test_unauthorized_and_deny_chunks_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_chunk(
                root,
                chunk_id="public",
                text="公開 情報 活動日は 土曜日 です",
                visibility="public",
                source_item_id="public-source",
            )
            self._write_chunk(
                root,
                chunk_id="guild-private",
                text="内部 情報 活動日は 日曜日 です",
                visibility="guild",
                guild_id="guild-a",
                source_item_id="guild-source",
            )
            self._write_chunk(
                root,
                chunk_id="deny",
                text="secret sk-abcdefghijklmnopqrstuvwxyz",
                redaction_policy="deny",
                index_status="quarantined",
                source_item_id="secret-source",
            )

            response = self._service(root).ask(
                RetrievalQuery(
                    text="活動日",
                    access=AccessContext(guild_id="guild-b", is_admin=False),
                )
            )

            ids = {citation.chunk_id for citation in response.citations}
            self.assertIn("public", ids)
            self.assertNotIn("guild-private", ids)
            self.assertNotIn("deny", ids)

if __name__ == "__main__":
    unittest.main()
