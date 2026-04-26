from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.routing import RoutingDecision
from kumc_agent.domain.models.source import BackfillScope
from kumc_agent.features.rag.config import RagConfig, RagGenerationSettings
from kumc_agent.features.rag.service import RagService
from kumc_agent.infra.connectors.minecraft_wiki import MinecraftWikiConnector


def _rag_config() -> RagConfig:
    return RagConfig(
        top_k=8,
        dense_top_k=15,
        sparse_top_k=15,
        sparse_initial_sparse_top_k=15,
        rerank_pool_size=20,
        mmr_lambda=0.75,
        recency_weight_soft=0.20,
        recency_weight_hard=0.60,
        recency_half_life_days=30.0,
        source_max_count=8,
        recency_mode="off",
        rag_generation=RagGenerationSettings(
            provider="gemini",
            temperature=0.0,
            max_output_tokens=512,
            prompt_name="answer_rag",
        ),
        no_rag_generation=RagGenerationSettings(
            provider="gemini",
            temperature=0.0,
            max_output_tokens=512,
            prompt_name="answer_no_rag",
        ),
        minecraft_wiki_top_k=2,
        minecraft_wiki_dense_top_k=5,
        minecraft_wiki_sparse_top_k=6,
        minecraft_wiki_sparse_initial_sparse_top_k=3,
        minecraft_wiki_sparse_normalized_ratio=0.5,
        minecraft_wiki_rerank_pool_size=3,
        minecraft_wiki_rrf_k=17,
        minecraft_wiki_mmr_lambda=0.5,
        minecraft_wiki_parent_doc_enabled=False,
        minecraft_wiki_parent_chunk_cap=1,
    )


def _chunk(chunk_id: str, *, source_type: str = "minecraft_wiki", parent: int = 1) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=f"text {chunk_id}",
        index=0,
        metadata={
            "source_type": source_type,
            "minecraft_wiki_page_id": "42",
            "minecraft_wiki_title": "丸石",
            "parent_chunk_id": parent,
            "chunk_id": int(parent),
            "canonical_url": "https://ja.minecraft.wiki/w/%E4%B8%B8%E7%9F%B3",
        },
    )


class _Router:
    def route(self, query, *, question_author=None, history=None):  # noqa: ANN001, ARG002
        return RoutingDecision(
            target_model="minecraft_wiki",
            use_additional_memory=True,
            fast_mode=False,
            additional_queries=["使わない"],
        )


class _Retrieval:
    def __init__(self, root: Path) -> None:
        self.index_dir = root / "data" / "index"
        self.calls: list[str] = []
        self.retrieve_kwargs: dict[str, object] = {}

    def retrieve(self, query, **kwargs):  # noqa: ANN001
        self.calls.append("retrieve")
        self.retrieve_kwargs = kwargs
        return [
            _chunk("same-parent-1", parent=1),
            _chunk("same-parent-2", parent=1),
            _chunk("other-parent", parent=2),
            _chunk("not-minecraft", source_type="docs", parent=9),
        ]

    def reorder_with_mmr(self, *, query, chunks, mmr_lambda):  # noqa: ANN001, ARG002
        self.calls.append("mmr")
        return list(chunks)


class _Reranker:
    def __init__(self, retrieval: _Retrieval) -> None:
        self._retrieval = retrieval

    def score_documents(self, *, query, chunks):  # noqa: ANN001, ARG002
        self._retrieval.calls.append("rerank")
        return [(1.0 - (idx * 0.1), idx, chunk) for idx, chunk in enumerate(chunks)]


class _Generation:
    def __init__(self, retrieval: _Retrieval) -> None:
        self._retrieval = retrieval
        self.prompt_name = ""
        self.chunks: list[Chunk] = []
        self.history = None

    def generate_rag_answer(self, **kwargs):  # noqa: ANN001
        self._retrieval.calls.append("generate")
        self.prompt_name = kwargs["answer_prompt_name"]
        self.chunks = list(kwargs["chunks"])
        self.history = kwargs["history"]
        return Answer(text="回答", route="rag", sources=[], metadata={})


class _FailingAnswerFilter:
    def evaluate(self, *, answer_text):  # noqa: ANN001, ARG002
        raise AssertionError("Minecraft Wiki RAG must not call AnswerFilterComponent")


class _Connector(MinecraftWikiConnector):
    def __init__(self, *args, payloads: list[dict[str, object]], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "payloads", list(payloads))

    def _request_json(self, params: dict[str, str]) -> dict[str, object]:
        return self.payloads.pop(0)

    def _wait_for_rate_limit(self) -> None:
        return None


class MinecraftWikiRagTests(unittest.TestCase):
    def test_connector_uses_ja_metadata_and_does_not_emit_edition_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            connector = _Connector(
                raw_dir=Path(tmp),
                page_titles=("丸石",),
                api_url="https://ja.minecraft.wiki/api.php",
                page_url_base="https://ja.minecraft.wiki/w/",
                max_pages=20,
                payloads=[
                    {
                        "parse": {
                            "title": "丸石",
                            "pageid": 123,
                            "revid": 456,
                            "wikitext": "== 入手 ==\n'''丸石'''は[[石|石]]から得られる。",
                        }
                    },
                    {
                        "query": {
                            "pages": [
                                {
                                    "pageid": 123,
                                    "canonicalurl": "https://ja.minecraft.wiki/w/%E4%B8%B8%E7%9F%B3",
                                    "revisions": [
                                        {"revid": 456, "timestamp": "2026-01-02T03:04:05Z"}
                                    ],
                                }
                            ]
                        }
                    },
                ],
            )

            raw = asyncio.run(_first_backfill_item(connector))

        self.assertEqual(raw.metadata["source_type"], "minecraft_wiki")
        self.assertEqual(raw.metadata["canonical_url"], "https://ja.minecraft.wiki/w/%E4%B8%B8%E7%9F%B3")
        self.assertEqual(raw.metadata["minecraft_wiki_revision_id"], "456")
        self.assertIn("# 入手", raw.text)
        self.assertIn("丸石は石から得られる。", raw.text)
        self.assertNotIn("minecraft_version", raw.metadata)
        self.assertNotIn("minecraft_edition", raw.metadata)

    def test_minecraft_wiki_service_skips_synthesis_and_answer_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            retrieval = _Retrieval(Path(tmp))
            generation = _Generation(retrieval)
            service = RagService(
                config=_rag_config(),
                router=_Router(),
                retrieval=retrieval,
                generation=generation,
                reranker=_Reranker(retrieval),
                query_synthesizer=object(),
                answer_filter=_FailingAnswerFilter(),
            )

            answer = service.answer(
                query="Minecraft Wikiで丸石の入手方法を教えて",
                generation_history_override=[("u", "a", [])],
                disable_history=True,
            )

        self.assertEqual(answer.route, "minecraft_wiki_rag")
        self.assertEqual(answer.metadata["query_synthesis"]["used"], False)
        self.assertEqual(answer.metadata["query_synthesis"]["synthetic_query"], "Minecraft Wikiで丸石の入手方法を教えて")
        self.assertNotIn("additional_queries", answer.metadata["routing_decision"])
        self.assertEqual(retrieval.retrieve_kwargs["source_type_filter"], {"minecraft_wiki"})
        self.assertEqual(retrieval.retrieve_kwargs["dense_top_k"], 5)
        self.assertEqual(retrieval.retrieve_kwargs["sparse_top_k"], 6)
        self.assertEqual(retrieval.retrieve_kwargs["rrf_k"], 17)
        self.assertEqual(retrieval.calls, ["retrieve", "rerank", "mmr", "generate"])
        self.assertEqual(generation.prompt_name, "answer_minecraft_wiki")
        self.assertEqual([chunk.id for chunk in generation.chunks], ["same-parent-1", "other-parent"])
        self.assertEqual(generation.history, [("u", "a", [])])


async def _first_backfill_item(connector: MinecraftWikiConnector):
    async for item in connector.backfill(BackfillScope()):
        return item
    raise AssertionError("expected one backfill item")


if __name__ == "__main__":
    unittest.main()
