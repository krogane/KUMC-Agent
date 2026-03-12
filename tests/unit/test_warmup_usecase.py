from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.usecases.warmup.run import WarmupRequest, WarmupUsecase


class _DummyEmbedder:
    def __init__(self) -> None:
        self.query_calls = 0
        self.documents_calls = 0

    def embed_query(self, text: str):
        _ = text
        self.query_calls += 1
        return [0.0]

    def embed_documents(self, texts: list[str]):
        _ = texts
        self.documents_calls += 1
        return [[0.0]]


class _DummyReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, chunks, *, top_k: int):
        _ = query
        _ = chunks
        _ = top_k
        self.calls += 1
        return []


class _DummyRouteUsecase:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request):
        _ = request
        self.calls += 1
        return None


class _DummyLLM:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.calls = 0
        self.should_fail = should_fail

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        _ = system_prompt
        _ = user_prompt
        _ = temperature
        _ = max_output_tokens
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("warmup failure")
        return "ok"


def _build_config(
    *,
    embedding_provider: str = "local",
    embedding_model: str = "e5",
    reranker_enabled: bool = True,
    reranker_model: str = "cross",
    routing_enabled: bool = True,
    routing_provider: str = "llama",
    routing_model_path: str = "model/route.gguf",
    rag_provider: str = "llama",
    rag_model_path: str = "model/rag.gguf",
    no_rag_provider: str = "llama",
    no_rag_model_path: str = "model/no_rag.gguf",
    refusal_provider: str = "llama",
    refusal_model_path: str = "model/refusal.gguf",
):
    generation_defaults = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "thinking_level": "minimal",
    }
    return SimpleNamespace(
        providers=SimpleNamespace(
            embeddings=SimpleNamespace(
                provider=embedding_provider,
                model=embedding_model,
            ),
            reranker=SimpleNamespace(
                enabled=reranker_enabled,
                model=reranker_model,
            ),
        ),
        rag=SimpleNamespace(
            routing=SimpleNamespace(
                enabled=routing_enabled,
                provider=routing_provider,
                llama_model_path=routing_model_path,
            ),
            generation=SimpleNamespace(
                rag=SimpleNamespace(
                    provider=rag_provider,
                    llama_model_path=rag_model_path,
                    **generation_defaults,
                ),
                no_rag=SimpleNamespace(
                    provider=no_rag_provider,
                    llama_model_path=no_rag_model_path,
                    **generation_defaults,
                ),
                refusal=SimpleNamespace(
                    provider=refusal_provider,
                    llama_model_path=refusal_model_path,
                    **generation_defaults,
                ),
            ),
        ),
    )


class WarmupUsecaseTests(unittest.TestCase):
    def test_local_models_are_warmed(self) -> None:
        config = _build_config()
        embedder = _DummyEmbedder()
        reranker = _DummyReranker()
        route = _DummyRouteUsecase()
        rag_llm = _DummyLLM()
        no_rag_llm = _DummyLLM()
        refusal_llm = _DummyLLM()

        usecase = WarmupUsecase(
            config=config,
            embedder=embedder,
            reranker=reranker,
            route_usecase=route,
            rag_llm=rag_llm,
            no_rag_llm=no_rag_llm,
            refusal_llm=refusal_llm,
        )
        result = usecase.execute(WarmupRequest(trigger="startup"))
        status = {step.name: step.status for step in result.steps}

        self.assertEqual(embedder.query_calls, 1)
        self.assertEqual(embedder.documents_calls, 1)
        self.assertEqual(reranker.calls, 1)
        self.assertEqual(route.calls, 1)
        self.assertEqual(rag_llm.calls, 1)
        self.assertEqual(no_rag_llm.calls, 1)
        self.assertEqual(refusal_llm.calls, 1)
        self.assertEqual(status["embedding"], "completed")
        self.assertEqual(status["cross_encoder_reranker"], "completed")
        self.assertEqual(status["routing_function_calling"], "completed")
        self.assertEqual(status["answer_llm"], "completed")
        self.assertEqual(status["no_rag_llm"], "completed")
        self.assertEqual(status["refusal_llm"], "completed")
        self.assertEqual(result.completed, 6)
        self.assertEqual(result.failed, 0)

    def test_gemini_providers_are_skipped(self) -> None:
        config = _build_config(
            embedding_provider="gemini",
            reranker_enabled=False,
            routing_provider="gemini",
            rag_provider="gemini",
            no_rag_provider="gemini",
            refusal_provider="gemini",
        )
        embedder = _DummyEmbedder()
        route = _DummyRouteUsecase()
        rag_llm = _DummyLLM()
        no_rag_llm = _DummyLLM()
        refusal_llm = _DummyLLM()

        usecase = WarmupUsecase(
            config=config,
            embedder=embedder,
            reranker=None,
            route_usecase=route,
            rag_llm=rag_llm,
            no_rag_llm=no_rag_llm,
            refusal_llm=refusal_llm,
        )
        result = usecase.execute(WarmupRequest(trigger="periodic"))

        self.assertEqual(embedder.query_calls, 0)
        self.assertEqual(embedder.documents_calls, 0)
        self.assertEqual(route.calls, 0)
        self.assertEqual(rag_llm.calls, 0)
        self.assertEqual(no_rag_llm.calls, 0)
        self.assertEqual(refusal_llm.calls, 0)
        self.assertEqual(result.completed, 0)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.skipped, 6)

    def test_local_model_without_path_is_skipped(self) -> None:
        config = _build_config(
            routing_model_path="",
            rag_model_path="",
            no_rag_model_path="",
            refusal_model_path="",
        )
        embedder = _DummyEmbedder()
        reranker = _DummyReranker()
        route = _DummyRouteUsecase()
        rag_llm = _DummyLLM()
        no_rag_llm = _DummyLLM()
        refusal_llm = _DummyLLM()

        usecase = WarmupUsecase(
            config=config,
            embedder=embedder,
            reranker=reranker,
            route_usecase=route,
            rag_llm=rag_llm,
            no_rag_llm=no_rag_llm,
            refusal_llm=refusal_llm,
        )
        result = usecase.execute(WarmupRequest(trigger="startup"))
        status = {step.name: step.status for step in result.steps}

        self.assertEqual(embedder.query_calls, 1)
        self.assertEqual(reranker.calls, 1)
        self.assertEqual(route.calls, 0)
        self.assertEqual(rag_llm.calls, 0)
        self.assertEqual(no_rag_llm.calls, 0)
        self.assertEqual(refusal_llm.calls, 0)
        self.assertEqual(status["routing_function_calling"], "skipped")
        self.assertEqual(status["answer_llm"], "skipped")
        self.assertEqual(status["no_rag_llm"], "skipped")
        self.assertEqual(status["refusal_llm"], "skipped")

    def test_failure_does_not_stop_following_steps(self) -> None:
        config = _build_config()
        embedder = _DummyEmbedder()
        reranker = _DummyReranker()
        route = _DummyRouteUsecase()
        rag_llm = _DummyLLM()
        no_rag_llm = _DummyLLM(should_fail=True)
        refusal_llm = _DummyLLM()

        usecase = WarmupUsecase(
            config=config,
            embedder=embedder,
            reranker=reranker,
            route_usecase=route,
            rag_llm=rag_llm,
            no_rag_llm=no_rag_llm,
            refusal_llm=refusal_llm,
        )
        result = usecase.execute(WarmupRequest(trigger="periodic"))
        status = {step.name: step.status for step in result.steps}

        self.assertEqual(status["no_rag_llm"], "failed")
        self.assertEqual(status["refusal_llm"], "completed")
        self.assertEqual(rag_llm.calls, 1)
        self.assertEqual(no_rag_llm.calls, 1)
        self.assertEqual(refusal_llm.calls, 1)
        self.assertEqual(result.failed, 1)


if __name__ == "__main__":
    unittest.main()
