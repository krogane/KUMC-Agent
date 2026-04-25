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

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.routing import RoutingDecision
from kumc_agent.features.rag.config import (
    RagConfig,
    RagGenerationSettings,
)
from kumc_agent.features.rag.service import RagService


class _DummyRetrieval:
    def __init__(
        self,
        *,
        index_dir: Path,
        dense_pick_index: int | None = None,
        dense_rank_order: list[int] | None = None,
        fail_on_retrieve: bool = False,
    ) -> None:
        self.index_dir = index_dir
        self.dense_pick_index = dense_pick_index
        self.dense_rank_order = list(dense_rank_order or [])
        self.fail_on_retrieve = fail_on_retrieve
        self.retrieve_call_count = 0

    def retrieve(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.retrieve_call_count += 1
        if self.fail_on_retrieve:
            raise AssertionError("RAG fallback retrieval should not be called.")
        return []

    def reorder_with_mmr(self, *, query, chunks, mmr_lambda):  # noqa: ANN001, ARG002
        return list(chunks)

    def rank_texts_by_dense(self, *, query: str, texts: list[str], top_k: int = 1):  # noqa: ARG002
        if self.dense_rank_order:
            ranked: list[tuple[int, float]] = []
            score = float(len(self.dense_rank_order))
            for index in self.dense_rank_order:
                if index < 0 or index >= len(texts):
                    continue
                ranked.append((index, score))
                score -= 1.0
                if len(ranked) >= max(0, int(top_k)):
                    break
            return ranked
        if self.dense_pick_index is None:
            return []
        if self.dense_pick_index < 0 or self.dense_pick_index >= len(texts):
            return []
        if top_k <= 0:
            return []
        return [(self.dense_pick_index, 1.0)]


def _rag_config(*, material_full_text_char_limit: int = 3000) -> RagConfig:
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
        material_search_max_names=3,
        parent_doc_enabled=False,
        parent_chunk_cap=2,
        material_full_text_char_limit=material_full_text_char_limit,
        answer_json_max_retries=1,
        history_enabled=False,
        history_max_turns=5,
        prompt_default_turns=3,
        prompt_additional_turns=10,
    )


def _write_material(
    *,
    base_dir: Path,
    material_id: str,
    source_key: str,
    canonical_name: str,
    aliases: list[str],
    raw_rel_path: str,
    raw_text: str,
) -> dict[str, object]:
    raw_path = base_dir / raw_rel_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw_text, encoding="utf-8")
    return {
        "material_id": material_id,
        "source_type": "docs",
        "source_key": source_key,
        "canonical_name": canonical_name,
        "aliases": aliases,
        "raw_path": raw_rel_path,
    }


def _new_service(
    *,
    retrieval: _DummyRetrieval,
    material_full_text_char_limit: int = 3000,
) -> RagService:
    return RagService(
        config=_rag_config(material_full_text_char_limit=material_full_text_char_limit),
        router=object(),  # not used in these tests
        retrieval=retrieval,
        generation=object(),  # not used in these tests
        reranker=None,
    )


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=f"chunk:{chunk_id}",
        document_id="doc:1",
        text=text,
        index=0,
        metadata={
            "source_type": "docs",
            "source_file_name": "id-20250614",
            "chunk_stage": "first_recursive",
            "chunk_id": chunk_id,
        },
    )


class MaterialSearchMatchingTests(unittest.TestCase):
    def test_material_name_partial_match_handles_notation_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            index_dir = base / "data" / "index"
            index_dir.mkdir(parents=True)
            material = _write_material(
                base_dir=base,
                material_id="docs:id-20250614",
                source_key="id-20250614",
                canonical_name="20250614 議事録",
                aliases=["議事録/20250614 議事録"],
                raw_rel_path="data/raw/docs/20250614.md",
                raw_text="2025/06/14 議事録本文",
            )
            (index_dir / "material_catalog.json").write_text(
                json.dumps({"materials": [material]}, ensure_ascii=False),
                encoding="utf-8",
            )

            service = _new_service(retrieval=_DummyRetrieval(index_dir=index_dir))
            matched = service._match_material_entries(  # noqa: SLF001
                material_names=["2025/06/14例会議事録"],
                query="",
                excluded_source_types=set(),
            )

            self.assertEqual(len(matched), 1)
            self.assertEqual(matched[0].source_key, "id-20250614")

    def test_dense_name_fallback_runs_before_rag_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            index_dir = base / "data" / "index"
            index_dir.mkdir(parents=True)
            materials = [
                _write_material(
                    base_dir=base,
                    material_id="docs:id-20250614",
                    source_key="id-20250614",
                    canonical_name="20250614 議事録",
                    aliases=["議事録/20250614 議事録"],
                    raw_rel_path="data/raw/docs/20250614.md",
                    raw_text="2025/06/14 議事録本文",
                ),
                _write_material(
                    base_dir=base,
                    material_id="docs:id-20250621",
                    source_key="id-20250621",
                    canonical_name="20250621 議事録",
                    aliases=["議事録/20250621 議事録"],
                    raw_rel_path="data/raw/docs/20250621.md",
                    raw_text="2025/06/21 議事録本文",
                ),
            ]
            (index_dir / "material_catalog.json").write_text(
                json.dumps({"materials": materials}, ensure_ascii=False),
                encoding="utf-8",
            )

            retrieval = _DummyRetrieval(
                index_dir=index_dir,
                dense_pick_index=1,
                fail_on_retrieve=True,
            )
            service = _new_service(retrieval=retrieval)
            decision = RoutingDecision(
                recency_mode="off",
                material_names=["meeting memo for June"],
            )

            chunks = service._retrieve_material_route_chunks(  # noqa: SLF001
                query="資料を探して",
                decision=decision,
                recency_mode="off",
                force_fast_mode=False,
            )

            self.assertEqual(len(chunks), 1)
            self.assertTrue(chunks[0].id.startswith("material:"))
            self.assertEqual(chunks[0].metadata.get("source_file_name"), "id-20250621")

    def test_sparse_failure_uses_matched_material_full_text_before_rag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            index_dir = base / "data" / "index"
            index_dir.mkdir(parents=True)
            material = _write_material(
                base_dir=base,
                material_id="docs:id-20250614",
                source_key="id-20250614",
                canonical_name="20250614 議事録",
                aliases=["議事録/20250614 議事録"],
                raw_rel_path="data/raw/docs/20250614.md",
                raw_text="2025/06/14 議事録本文",
            )
            (index_dir / "material_catalog.json").write_text(
                json.dumps({"materials": [material]}, ensure_ascii=False),
                encoding="utf-8",
            )

            retrieval = _DummyRetrieval(index_dir=index_dir)
            service = _new_service(retrieval=retrieval)
            service._retrieve_chunks = lambda **_: (_ for _ in ()).throw(  # noqa: SLF001
                AssertionError("RAG fallback should not be called.")
            )
            decision = RoutingDecision(
                recency_mode="off",
                material_names=["20250614 議事録"],
            )

            chunks = service._retrieve_material_route_chunks(  # noqa: SLF001
                query="生活サーバーの方針",
                decision=decision,
                recency_mode="off",
                force_fast_mode=False,
            )

            self.assertEqual(len(chunks), 1)
            self.assertTrue(chunks[0].id.startswith("material:"))
            self.assertEqual(chunks[0].metadata.get("source_file_name"), "id-20250614")

    def test_long_material_text_uses_dense_chunks_up_to_char_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            index_dir = base / "data" / "index"
            index_dir.mkdir(parents=True)
            material = _write_material(
                base_dir=base,
                material_id="docs:id-20250614",
                source_key="id-20250614",
                canonical_name="20250614 議事録",
                aliases=["議事録/20250614 議事録"],
                raw_rel_path="data/raw/docs/20250614.md",
                raw_text="x" * 30,
            )
            (index_dir / "material_catalog.json").write_text(
                json.dumps({"materials": [material]}, ensure_ascii=False),
                encoding="utf-8",
            )

            retrieval = _DummyRetrieval(
                index_dir=index_dir,
                dense_rank_order=[1, 2, 0],
            )
            service = _new_service(
                retrieval=retrieval,
                material_full_text_char_limit=9,
            )
            service._first_rec_chunks_for_material_key = lambda _key: [  # noqa: SLF001
                _make_chunk("A0", "AAAAA"),
                _make_chunk("B1", "BBBBB"),
                _make_chunk("C2", "CCCC"),
            ]

            decision = RoutingDecision(
                recency_mode="off",
                material_names=["20250614 議事録"],
            )
            chunks = service._retrieve_material_route_chunks(  # noqa: SLF001
                query="サークル方針",
                decision=decision,
                recency_mode="off",
                force_fast_mode=False,
            )

            self.assertEqual([chunk.text for chunk in chunks], ["BBBBB", "CCCC"])
            self.assertEqual(sum(len(chunk.text) for chunk in chunks), 9)


if __name__ == "__main__":
    unittest.main()
