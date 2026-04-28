from __future__ import annotations

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
from kumc_agent.features.rag.components.query_synthesis import QuerySynthesisResult
from kumc_agent.features.rag.config import RagConfig, RagGenerationSettings
from kumc_agent.features.rag.service import RagService


class _Router:
    def route(self, *_args, **_kwargs):
        return RoutingDecision(
            recency_mode="off",
            use_additional_memory=True,
            additional_queries=["新歓 日程"],
        )


class _FastMaterialRouter:
    def route(self, *_args, **_kwargs):
        return RoutingDecision(
            recency_mode="off",
            fast_mode=True,
            material_names=["運営資料"],
        )


class _Synthesizer:
    def synthesize(self, **_kwargs):
        return QuerySynthesisResult("合成された検索クエリ", used=True, fallback=False)


class _Retrieval:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.queries: list[str] = []

    def retrieve(self, query: str, **_kwargs):
        self.queries.append(query)
        return [
            Chunk(
                id="c1",
                document_id="d1",
                text="KUMCの新歓は春に行います。",
                index=0,
                metadata={"source_type": "hatenablog"},
            )
        ]

    def reorder_with_mmr(self, *, query, chunks, mmr_lambda):  # noqa: ANN001, ARG002
        return list(chunks)


class _Generation:
    def generate_rag_answer(self, **_kwargs):
        return Answer(text="回答", route="rag", sources=[], metadata={})


def _config() -> RagConfig:
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
        parent_doc_enabled=False,
        history_enabled=False,
    )


class QuerySynthesisTests(unittest.TestCase):
    def test_search_uses_single_synthetic_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            retrieval = _Retrieval(Path(tmp) / "data" / "index")
            service = RagService(
                config=_config(),
                router=_Router(),
                retrieval=retrieval,
                generation=_Generation(),
                reranker=None,
                query_synthesizer=_Synthesizer(),
            )

            answer = service.answer(query="それはいつ？")

            self.assertEqual(answer.text, "回答")
            self.assertEqual(retrieval.queries, ["合成された検索クエリ"])
            self.assertEqual(
                answer.metadata["query_synthesis"]["synthetic_query"],
                "合成された検索クエリ",
            )

    def test_fast_mode_skips_material_search_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            retrieval = _Retrieval(Path(tmp) / "data" / "index")
            service = RagService(
                config=_config(),
                router=_FastMaterialRouter(),
                retrieval=retrieval,
                generation=_Generation(),
                reranker=None,
                query_synthesizer=_Synthesizer(),
            )

            answer = service.answer(query="運営資料を見て", disable_history=True)

            self.assertEqual(answer.route, "rag")
            self.assertEqual(answer.metadata["routing_decision"]["material_names"], [])
            self.assertFalse(answer.metadata["query_synthesis"]["used"])


if __name__ == "__main__":
    unittest.main()
