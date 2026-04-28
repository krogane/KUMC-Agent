from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.features.rag.components.retrieval import RetrievalComponent
from kumc_agent.infra.retrieval.faiss import SearchResult


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=f"text {chunk_id}",
        index=0,
        metadata={},
    )


class RagRetrievalRrfTests(unittest.TestCase):
    def test_retrieve_keeps_sparse_hits_when_dense_is_also_enabled(self) -> None:
        sparse_chunk = _chunk("sparse-only")

        class _Embedder:
            def embed_query(self, query):  # noqa: ANN001, ARG002
                return [1.0, 0.0]

            def embed_documents(self, texts):  # noqa: ANN001, ARG002
                return [[1.0, 0.0] for _ in texts]

        class _DenseIndex:
            def __init__(self, index_dir: Path) -> None:
                self._index_dir = index_dir

            def search(self, *, query_vector, top_k):  # noqa: ANN001, ARG002
                return []

        class _SparseIndex:
            def search_with_scores(self, query, top_k):  # noqa: ANN001, ARG002
                return [(sparse_chunk, 1.0)]

            def _tokenize(self, query):  # noqa: ANN001
                return str(query).split()

        with tempfile.TemporaryDirectory() as tmp:
            component = RetrievalComponent(
                embedder=_Embedder(),
                dense_index=_DenseIndex(Path(tmp) / "index"),
                sparse_index=_SparseIndex(),
            )

            results = component.retrieve(
                "sparse",
                dense_top_k=1,
                sparse_top_k=1,
            )

        self.assertEqual([chunk.id for chunk in results], ["sparse-only"])

    def test_minecraft_wiki_route_uses_dedicated_sparse_settings_and_corpora(self) -> None:
        sparse_chunk = Chunk(
            id="wiki-sparse",
            document_id="wiki-doc",
            text="丸石",
            index=0,
            metadata={"source_type": "minecraft_wiki"},
        )

        class _Embedder:
            def embed_query(self, query):  # noqa: ANN001, ARG002
                return [1.0, 0.0]

            def embed_documents(self, texts):  # noqa: ANN001, ARG002
                return [[1.0, 0.0] for _ in texts]

        class _DenseIndex:
            def __init__(self, index_dir: Path) -> None:
                self._index_dir = index_dir

            def search(self, *, query_vector, top_k):  # noqa: ANN001, ARG002
                return []

        class _SparseIndex:
            def search_with_scores(self, query, *, top_k, query_tokens=None):  # noqa: ANN001, ARG002
                return []

            def _tokenize(self, query):  # noqa: ANN001
                return str(query).split()

        class _Component(RetrievalComponent):
            def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
                super().__init__(*args, **kwargs)
                self.token_settings = None
                self.corpus_names: list[str] = []

            def _query_tokens(self, query, *, token_settings=None):  # noqa: ANN001, ARG002
                self.token_settings = token_settings
                return ["丸石"]

            def _search_keyword_index(
                self,
                *,
                tokens,
                top_k,
                corpus_name,
                restore_sparse_stage,
                second_rec_corpus_name,
            ):  # noqa: ANN001, ARG002
                self.corpus_names.append(corpus_name)
                if corpus_name == "minecraft_wiki_sparse_second_rec":
                    return [(sparse_chunk, 1.0)]
                return []

        with tempfile.TemporaryDirectory() as tmp:
            component = _Component(
                embedder=_Embedder(),
                dense_index=_DenseIndex(Path(tmp) / "index"),
                sparse_index=_SparseIndex(),
                sparse_sudachi_mode="B",
                sparse_use_normalized_form=True,
                sparse_remove_symbols=True,
            )

            results = component.retrieve(
                "丸石",
                dense_top_k=0,
                sparse_top_k=2,
                sparse_initial_sparse_top_k=1,
                source_type_filter={"minecraft_wiki"},
                sparse_sudachi_mode="C",
                sparse_use_normalized_form=False,
                sparse_remove_symbols=False,
            )

        self.assertEqual([chunk.id for chunk in results], ["wiki-sparse"])
        self.assertEqual(component.token_settings, ("C", False, False))
        self.assertEqual(
            component.corpus_names,
            [
                "minecraft_wiki_sparse_second_rec",
                "minecraft_wiki_second_rec_sparse",
            ],
        )

    def test_dense_and_sparse_overlap_beats_single_dense_top_hit(self) -> None:
        dense_top = _chunk("dense-top")
        overlap = _chunk("overlap")
        sparse_top = _chunk("sparse-top")

        scored = RetrievalComponent._merge_scores(
            dense_hits=[
                SearchResult(chunk=dense_top, score=100.0),
                SearchResult(chunk=overlap, score=1.0),
            ],
            sparse_hits=[
                (sparse_top, 1000.0),
                (overlap, 1.0),
            ],
            rrf_k=60,
        )

        self.assertEqual(scored[0].chunk.id, "overlap")
        self.assertEqual([item.chunk.id for item in scored].count("overlap"), 1)

    def test_single_source_results_keep_rank_order(self) -> None:
        first = _chunk("first")
        second = _chunk("second")
        third = _chunk("third")

        dense_only = RetrievalComponent._merge_scores(
            dense_hits=[
                SearchResult(chunk=first, score=0.1),
                SearchResult(chunk=second, score=0.9),
                SearchResult(chunk=third, score=0.8),
            ],
            sparse_hits=[],
            rrf_k=60,
        )
        sparse_only = RetrievalComponent._merge_scores(
            dense_hits=[],
            sparse_hits=[(first, 0.1), (second, 0.9), (third, 0.8)],
            rrf_k=60,
        )

        self.assertEqual([item.chunk.id for item in dense_only], ["first", "second", "third"])
        self.assertEqual([item.chunk.id for item in sparse_only], ["first", "second", "third"])

    def test_tied_rrf_scores_use_candidate_order(self) -> None:
        dense = _chunk("dense")
        sparse = _chunk("sparse")

        scored = RetrievalComponent._merge_scores(
            dense_hits=[SearchResult(chunk=dense, score=0.1)],
            sparse_hits=[(sparse, 1000.0)],
            rrf_k=60,
        )

        self.assertEqual([item.chunk.id for item in scored], ["dense", "sparse"])


if __name__ == "__main__":
    unittest.main()
