from __future__ import annotations

import asyncio
from dataclasses import replace
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.routing import RoutingDecision
from kumc_agent.domain.models.source import AccessScope, BackfillScope, NormalizedDocument
from kumc_agent.config.load import load_runtime_config
from kumc_agent.features.ingestion.chunking import IngestionChunker
from kumc_agent.features.rag.config import RagConfig, RagGenerationSettings
from kumc_agent.features.rag.service import RagService
from kumc_agent.features.indexing.service import IndexBuildResult, IndexingService
from kumc_agent.infra.connectors.minecraft_wiki import MinecraftWikiConnector
from kumc_agent.usecases.indexing.build import BuildIndexRequest, BuildIndexUsecase


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
        minecraft_wiki_sparse_sudachi_mode="C",
        minecraft_wiki_sparse_use_normalized_form=False,
        minecraft_wiki_sparse_remove_symbols=False,
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
                ingestion_dir=Path(tmp),
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

    def test_connector_resolves_redirect_alias_to_article_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            connector = _Connector(
                ingestion_dir=Path(tmp),
                page_titles=("Amethyst Cluster",),
                api_url="https://ja.minecraft.wiki/api.php",
                page_url_base="https://ja.minecraft.wiki/w/",
                max_pages=20,
                payloads=[
                    {
                        "parse": {
                            "title": "Amethyst Cluster",
                            "pageid": 10,
                            "revid": 1,
                            "wikitext": "#REDIRECT [[アメジストの塊]]",
                        }
                    },
                    {
                        "parse": {
                            "title": "アメジストの塊",
                            "pageid": 20,
                            "revid": 2,
                            "wikitext": "== 入手 ==\n<code>minecraft:amethyst_cluster</code> は[[アメジストジオード]]で生成される。",
                        }
                    },
                    {
                        "query": {
                            "pages": [
                                {
                                    "pageid": 20,
                                    "title": "アメジストの塊",
                                    "canonicalurl": "https://ja.minecraft.wiki/w/%E3%82%A2%E3%83%A1%E3%82%B8%E3%82%B9%E3%83%88%E3%81%AE%E5%A1%8A",
                                    "revisions": [
                                        {"revid": 2, "timestamp": "2026-01-03T00:00:00Z"}
                                    ],
                                }
                            ]
                        }
                    },
                ],
            )

            raw = asyncio.run(_first_backfill_item(connector))

        self.assertEqual(raw.title, "アメジストの塊")
        self.assertEqual(raw.metadata["minecraft_wiki_is_redirect"], True)
        self.assertEqual(raw.metadata["minecraft_wiki_redirect_from"], "Amethyst Cluster")
        self.assertEqual(raw.metadata["minecraft_wiki_redirect_to"], "アメジストの塊")
        self.assertEqual(raw.metadata["minecraft_wiki_resolved_page_id"], "20")
        self.assertIn("minecraft:amethyst_cluster", raw.text)
        self.assertNotIn("<code>", raw.text)
        self.assertNotIn("#REDIRECT", raw.text)

    def test_ingestion_chunker_skips_redirect_only_minecraft_wiki_documents(self) -> None:
        document = NormalizedDocument(
            id="doc",
            source_item_id="item",
            source_kind="minecraft_wiki",
            external_id="alias",
            version=1,
            title="Amethyst Cluster",
            normalized_text="#転送 [[アメジストの塊]]",
            normalized_format="wiki_markdown",
            language="ja",
            access_scope=AccessScope(visibility="public"),
            checksum="checksum",
            metadata={"minecraft_wiki_is_redirect": True},
        )

        self.assertEqual(IngestionChunker().chunk(document), [])

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
        self.assertEqual(retrieval.retrieve_kwargs["sparse_sudachi_mode"], "C")
        self.assertEqual(retrieval.retrieve_kwargs["sparse_use_normalized_form"], False)
        self.assertEqual(retrieval.retrieve_kwargs["sparse_remove_symbols"], False)
        self.assertEqual(retrieval.calls, ["retrieve", "rerank", "mmr", "generate"])
        self.assertEqual(generation.prompt_name, "answer_minecraft_wiki")
        self.assertEqual([chunk.id for chunk in generation.chunks], ["same-parent-1", "other-parent"])
        self.assertEqual(generation.history, [])

    def test_connector_refreshes_cached_page_when_revision_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ingestion_dir = Path(tmp)
            raw_path = ingestion_dir / "丸石.md"
            meta_path = raw_path.with_suffix(raw_path.suffix + ".meta.json")
            raw_path.write_text("old text", encoding="utf-8")
            meta_path.write_text(
                '{"minecraft_wiki_title":"丸石","minecraft_wiki_page_id":"123","minecraft_wiki_revision_id":"1","canonical_url":"https://ja.minecraft.wiki/w/%E4%B8%B8%E7%9F%B3"}',
                encoding="utf-8",
            )
            connector = _Connector(
                ingestion_dir=ingestion_dir,
                page_titles=("丸石",),
                api_url="https://ja.minecraft.wiki/api.php",
                page_url_base="https://ja.minecraft.wiki/w/",
                max_pages=20,
                payloads=[
                    {
                        "query": {
                            "pages": [
                                {
                                    "pageid": 123,
                                    "canonicalurl": "https://ja.minecraft.wiki/w/%E4%B8%B8%E7%9F%B3",
                                    "revisions": [
                                        {"revid": 2, "timestamp": "2026-01-03T00:00:00Z"}
                                    ],
                                }
                            ]
                        }
                    },
                    {
                        "parse": {
                            "title": "丸石",
                            "pageid": 123,
                            "revid": 2,
                            "wikitext": "== 入手 ==\n更新後の本文",
                        }
                    },
                    {
                        "query": {
                            "pages": [
                                {
                                    "pageid": 123,
                                    "canonicalurl": "https://ja.minecraft.wiki/w/%E4%B8%B8%E7%9F%B3",
                                    "revisions": [
                                        {"revid": 2, "timestamp": "2026-01-03T00:00:00Z"}
                                    ],
                                }
                            ]
                        }
                    },
                ],
            )

            raw = asyncio.run(_first_backfill_item(connector))

        self.assertIn("更新後の本文", raw.text)
        self.assertEqual(raw.metadata["minecraft_wiki_revision_id"], "2")

    def test_connector_rejects_non_japanese_minecraft_wiki_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                MinecraftWikiConnector(
                    ingestion_dir=Path(tmp),
                    page_titles=("Stone",),
                    api_url="https://minecraft.wiki/api.php",
                    page_url_base="https://minecraft.wiki/w/",
                    max_pages=1,
                )

    def test_minecraft_wiki_summary_uses_llm_when_configured(self) -> None:
        class _SummaryLLM:
            def __init__(self) -> None:
                self.user_prompt = ""

            def generate(self, *, system_prompt, user_prompt, temperature, max_output_tokens):  # noqa: ANN001, ARG002
                self.user_prompt = user_prompt
                return "丸石は石を採掘すると得られる。"

        llm = _SummaryLLM()
        service = object.__new__(IndexingService)
        service._minecraft_wiki_summary_llm = llm
        service._runtime = SimpleNamespace(
            minecraft_wiki_rag=SimpleNamespace(
                chunking=SimpleNamespace(
                    summary_llm_provider="gemini",
                    summary_temperature=0.0,
                    summary_max_output_tokens=128,
                )
            )
        )

        summary = service._build_minecraft_wiki_summary_text(
            text="丸石は石を採掘すると得られる。用途はクラフト素材。",
            metadata={
                "minecraft_wiki_title": "丸石",
                "heading_path": ["丸石", "入手"],
            },
            target_chars=80,
        )

        self.assertIn("記事名: 丸石", summary)
        self.assertIn("丸石は石を採掘すると得られる。", summary)
        self.assertIn("表や箇条書きの情報は文章として保持", llm.user_prompt)

    def test_manual_build_refreshes_minecraft_wiki_connector(self) -> None:
        class _Loader:
            def load(self) -> int:
                return 2

        class _Indexing:
            def __init__(self) -> None:
                self.loaded_sources = 0

            def build(self, **kwargs):  # noqa: ANN001
                self.loaded_sources = int(kwargs["loaded_sources"])
                return IndexBuildResult(
                    loaded_sources=self.loaded_sources,
                    documents=0,
                    chunks=0,
                    index_dir=Path("/tmp/index"),
                )

        class _Ingestion:
            def available_sources(self):
                return ("minecraft_wiki",)

            async def backfill_many(self, *, source_kinds, scope):  # noqa: ANN001
                self.source_kinds = source_kinds
                self.force = scope.force
                return (SimpleNamespace(seen=3),)

        indexing = _Indexing()
        ingestion = _Ingestion()
        usecase = BuildIndexUsecase(
            indexing_service=indexing,
            drive_loader=_Loader(),
            discord_loader=None,
            hatenablog_loader=None,
            crafters_colony_loader=None,
            x_loader=None,
            notion_loader=None,
            ingestion_service=ingestion,
        )

        result = usecase.execute(BuildIndexRequest(refresh_sources=True, full_rebuild=True))

        self.assertEqual(result.loaded_sources, 5)
        self.assertEqual(ingestion.source_kinds, ("minecraft_wiki",))
        self.assertTrue(ingestion.force)

    def test_raw_minecraft_wiki_pipeline_builds_indexable_second_and_summary_chunks(self) -> None:
        class _Storage:
            def __init__(self) -> None:
                self.chunks: list[Chunk] = []

            def save_documents(self, documents):  # noqa: ANN001
                self.documents = documents

            def save_chunks(self, chunks):  # noqa: ANN001
                self.chunks = list(chunks)

        class _Embedder:
            def embed_documents(self, texts):  # noqa: ANN001
                return [[1.0, 0.0] for _ in texts]

        class _DenseIndex:
            def __init__(self, index_dir: Path) -> None:
                self._index_dir = index_dir
                self.chunks: list[Chunk] = []

            def build(self, *, chunks, embeddings):  # noqa: ANN001, ARG002
                self.chunks = list(chunks)

        class _SparseIndex:
            def __init__(self) -> None:
                self.chunks: list[Chunk] = []

            def build(self, chunks):  # noqa: ANN001
                self.chunks = list(chunks)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_runtime_config(base_dir=ROOT)
            app = replace(
                config.app,
                data_dir=root / "data",
                ingestion_dir=root / "data" / "ingestion",
                chunks_path=root / "data" / "chunks",
                index_dir=root / "data" / "index",
            )
            config = replace(config, app=app)
            ingestion_dir = config.app.ingestion_dir / "minecraft_wiki"
            ingestion_dir.mkdir(parents=True, exist_ok=True)
            raw_path = ingestion_dir / "丸石.md"
            raw_path.write_text(
                "# 丸石\n\n== 入手 ==\n丸石は石を採掘すると得られる。\n\n== 用途 ==\nクラフト素材。",
                encoding="utf-8",
            )
            raw_path.with_suffix(raw_path.suffix + ".meta.json").write_text(
                '{"minecraft_wiki_title":"丸石","minecraft_wiki_page_id":"123","minecraft_wiki_revision_id":"7","canonical_url":"https://ja.minecraft.wiki/w/%E4%B8%B8%E7%9F%B3","access_scope":{"visibility":"public"},"visibility":"public"}',
                encoding="utf-8",
            )
            storage = _Storage()
            dense = _DenseIndex(config.app.index_dir)
            sparse = _SparseIndex()
            service = IndexingService(
                storage=storage,
                embedder=_Embedder(),
                faiss_index=dense,
                bm25_index=sparse,
                ingestion_dir=config.app.ingestion_dir,
                app_config=config,
                summary_llm=None,
                minecraft_wiki_summary_llm=None,
            )

            result = service.build(loaded_sources=1)
            keyword_index_exists = (
                result.index_dir / "keyword" / "minecraft_wiki_sparse_second_rec.json"
            ).exists()

        wiki_chunks = [
            chunk
            for chunk in dense.chunks
            if chunk.metadata.get("source_type") == "minecraft_wiki"
        ]
        self.assertGreaterEqual(len(wiki_chunks), 2)
        self.assertTrue(
            any(chunk.metadata.get("chunk_stage") == "second_recursive" for chunk in wiki_chunks)
        )
        self.assertTrue(any(chunk.metadata.get("chunk_stage") == "summary" for chunk in wiki_chunks))
        self.assertTrue(keyword_index_exists)


async def _first_backfill_item(connector: MinecraftWikiConnector):
    async for item in connector.backfill(BackfillScope()):
        return item
    raise AssertionError("expected one backfill item")


if __name__ == "__main__":
    unittest.main()
