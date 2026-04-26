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

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - optional local dependency
    Document = None  # type: ignore[assignment]

from kumc_agent.features.rag.config import RagConfig, RagGenerationSettings
from kumc_agent.features.rag.service import RagService

try:
    from kumc_agent.infra.indexing.keyword_inverted_index import (
        KEYWORD_CORPUS_MATERIAL_NAMES,
        build_and_save_keyword_index,
        tokenize_material_name_text,
    )
except ImportError:  # pragma: no cover - optional local dependency
    KEYWORD_CORPUS_MATERIAL_NAMES = ""
    build_and_save_keyword_index = None  # type: ignore[assignment]
    tokenize_material_name_text = None  # type: ignore[assignment]


class _Retrieval:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir

    def retrieve(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return []

    def reorder_with_mmr(self, *, query, chunks, mmr_lambda):  # noqa: ANN001, ARG002
        return list(chunks)


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
    )


class MaterialNameKeywordIndexTests(unittest.TestCase):
    @unittest.skipIf(
        Document is None or build_and_save_keyword_index is None,
        "langchain_core is not installed",
    )
    def test_material_search_prefers_material_name_keyword_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            index_dir = base / "data" / "index"
            index_dir.mkdir(parents=True)
            materials = [
                {
                    "material_id": "docs:id-1",
                    "source_type": "docs",
                    "source_key": "id-1",
                    "canonical_name": "20250614 議事録",
                    "aliases": ["議事録/20250614"],
                    "raw_path": "data/raw/docs/20250614.md",
                },
                {
                    "material_id": "docs:id-2",
                    "source_type": "docs",
                    "source_key": "id-2",
                    "canonical_name": "20250701 新歓企画",
                    "aliases": ["新歓/20250701"],
                    "raw_path": "data/raw/docs/20250701.md",
                },
            ]
            (index_dir / "material_catalog.json").write_text(
                json.dumps({"materials": materials}, ensure_ascii=False),
                encoding="utf-8",
            )
            build_and_save_keyword_index(
                index_dir=index_dir,
                corpus_name=KEYWORD_CORPUS_MATERIAL_NAMES,
                docs=[
                    Document(
                        page_content="\n".join(
                            [item["canonical_name"], *item["aliases"]]
                        ),
                        metadata={
                            "material_id": item["material_id"],
                            "source_type": item["source_type"],
                            "source_key": item["source_key"],
                        },
                    )
                    for item in materials
                ],
                tokenize_doc=lambda doc: tokenize_material_name_text(doc.page_content),
                k1=1.5,
                b=0.75,
            )
            service = RagService(
                config=_config(),
                router=object(),
                retrieval=_Retrieval(index_dir),
                generation=object(),
                reranker=None,
            )

            matched = service._match_material_entries(  # noqa: SLF001
                material_names=["2025/07/01の新歓資料"],
                query="",
                excluded_source_types=set(),
            )

            self.assertEqual([entry.source_key for entry in matched], ["id-2"])


if __name__ == "__main__":
    unittest.main()
