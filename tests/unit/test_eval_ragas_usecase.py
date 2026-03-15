from __future__ import annotations

import json
import asyncio
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.infra.llm.gemini_rate_limit import ragas_embedding_rate_limiter_name
from kumc_agent.domain.models.answer import Answer
from kumc_agent.usecases.eval.ragas import (
    EvaluateRagasRequest,
    EvaluateRagasUsecase,
    _as_legacy_embeddings,
)


class _FakeChatUsecase:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.requests: list[object] = []

    def execute(self, request):  # type: ignore[no-untyped-def]
        self.queries.append(str(request.query))
        self.requests.append(request)
        if request.append_sources_to_response is not False:
            raise AssertionError("eval should call chat with append_sources_to_response=False")
        return Answer(
            text="KUMCはMinecraftサークルです",
            route="rag",
            metadata={"contexts": ["KUMCは京都大学のMinecraftサークルです。"]},
        )


class _FakeRagasResult:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores


class _ConcurrentTrackingChatUsecase:
    def __init__(self, *, delay_seconds: float = 0.05) -> None:
        self._delay_seconds = delay_seconds
        self._active = 0
        self._lock = threading.Lock()
        self.max_active = 0

    def execute(self, request):  # type: ignore[no-untyped-def]
        if request.append_sources_to_response is not False:
            raise AssertionError("eval should call chat with append_sources_to_response=False")
        with self._lock:
            self._active += 1
            if self._active > self.max_active:
                self.max_active = self._active
        try:
            time.sleep(self._delay_seconds)
            return Answer(
                text="KUMCはMinecraftサークルです",
                route="rag",
                metadata={"contexts": ["KUMCは京都大学のMinecraftサークルです。"]},
            )
        finally:
            with self._lock:
                self._active -= 1


class EvaluateRagasUsecaseTests(unittest.TestCase):
    def _write_eval_jsonl(self, base: Path, *, count: int = 1) -> Path:
        path = base / "ragas.jsonl"
        lines = []
        for i in range(count):
            lines.append(
                json.dumps(
                    {
                        "question": f"KUMCは何のサークル？{i}",
                        "ground_truth": "Minecraft",
                    },
                    ensure_ascii=False,
                )
            )
        path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        return path

    def _write_eval_jsonl_with_query_key(self, base: Path, *, count: int = 1) -> Path:
        path = base / "ragas.jsonl"
        lines = []
        for i in range(count):
            lines.append(
                json.dumps(
                    {
                        "query": f"KUMCは何のサークル？{i}",
                        "ground_truth": "Minecraft",
                    },
                    ensure_ascii=False,
                )
            )
        path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        return path

    def test_execute_runs_enabled_ragas_metrics(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(
            chat_usecase=fake_chat,
            eval_answer_relevancy_enabled=True,
            eval_faithfulness_enabled=False,
            eval_context_precision_enabled=True,
            eval_context_recall_enabled=False,
        )

        captured: dict[str, object] = {}
        metric_objects = {
            "answer_relevancy": object(),
            "faithfulness": object(),
            "context_precision": object(),
            "context_recall": object(),
        }
        names_by_metric = {value: key for key, value in metric_objects.items()}

        datasets_module = types.ModuleType("datasets")

        class _Dataset:
            @staticmethod
            def from_list(records):  # type: ignore[no-untyped-def]
                captured["records"] = records
                return records

        datasets_module.Dataset = _Dataset  # type: ignore[attr-defined]

        ragas_module = types.ModuleType("ragas")

        def _evaluate(dataset, metrics):  # type: ignore[no-untyped-def]
            _ = dataset
            captured["metrics"] = metrics
            scores = {
                names_by_metric[metric]: float(index + 1) / 10.0
                for index, metric in enumerate(metrics)
                if metric in names_by_metric
            }
            return _FakeRagasResult(scores=scores)

        ragas_module.evaluate = _evaluate  # type: ignore[attr-defined]

        ragas_metrics_module = types.ModuleType("ragas.metrics")
        ragas_metrics_module.answer_relevancy = metric_objects["answer_relevancy"]  # type: ignore[attr-defined]
        ragas_metrics_module.faithfulness = metric_objects["faithfulness"]  # type: ignore[attr-defined]
        ragas_metrics_module.context_precision = metric_objects["context_precision"]  # type: ignore[attr-defined]
        ragas_metrics_module.context_recall = metric_objects["context_recall"]  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as tmp:
            eval_file = self._write_eval_jsonl(Path(tmp))
            with patch.dict(
                sys.modules,
                {
                    "datasets": datasets_module,
                    "ragas": ragas_module,
                    "ragas.metrics": ragas_metrics_module,
                },
                clear=False,
            ):
                result = usecase.execute(EvaluateRagasRequest(eval_file=eval_file))

        self.assertEqual(result.total, 1)
        self.assertGreaterEqual(result.exact_match, 0.0)
        self.assertGreaterEqual(result.token_overlap, 0.0)
        self.assertEqual(
            set(result.ragas_metrics.keys()),
            {"answer_relevancy", "context_precision"},
        )
        enabled_metric_objects = captured.get("metrics")
        self.assertIsInstance(enabled_metric_objects, list)
        self.assertEqual(len(enabled_metric_objects), 2)
        self.assertEqual(fake_chat.queries, ["KUMCは何のサークル？0"])

    def test_execute_passes_batch_size_to_single_pass_ragas(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(
            chat_usecase=fake_chat,
            eval_answer_relevancy_enabled=True,
            eval_faithfulness_enabled=False,
            eval_context_precision_enabled=False,
            eval_context_recall_enabled=False,
        )

        captured_batch_sizes: list[int] = []
        metric_objects = {
            "answer_relevancy": object(),
            "faithfulness": object(),
            "context_precision": object(),
            "context_recall": object(),
        }

        datasets_module = types.ModuleType("datasets")

        class _Dataset:
            @staticmethod
            def from_list(records):  # type: ignore[no-untyped-def]
                return records

        datasets_module.Dataset = _Dataset  # type: ignore[attr-defined]

        ragas_module = types.ModuleType("ragas")

        def _evaluate(dataset, metrics, batch_size=None):  # type: ignore[no-untyped-def]
            _ = metrics
            captured_batch_sizes.append(int(batch_size or 0))
            return _FakeRagasResult(scores={"answer_relevancy": 0.5})

        ragas_module.evaluate = _evaluate  # type: ignore[attr-defined]

        ragas_metrics_module = types.ModuleType("ragas.metrics")
        ragas_metrics_module.answer_relevancy = metric_objects["answer_relevancy"]  # type: ignore[attr-defined]
        ragas_metrics_module.faithfulness = metric_objects["faithfulness"]  # type: ignore[attr-defined]
        ragas_metrics_module.context_precision = metric_objects["context_precision"]  # type: ignore[attr-defined]
        ragas_metrics_module.context_recall = metric_objects["context_recall"]  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as tmp:
            eval_file = self._write_eval_jsonl(Path(tmp), count=3)
            with patch.dict(
                sys.modules,
                {
                    "datasets": datasets_module,
                    "ragas": ragas_module,
                    "ragas.metrics": ragas_metrics_module,
                },
                clear=False,
            ):
                result = usecase.execute(
                    EvaluateRagasRequest(
                        eval_file=eval_file,
                        ragas_batch_size=2,
                    )
                )

        self.assertEqual(result.total, 3)
        self.assertEqual(captured_batch_sizes, [2])
        self.assertAlmostEqual(result.ragas_metrics["answer_relevancy"], 0.5)
        self.assertEqual(
            fake_chat.queries,
            ["KUMCは何のサークル？0", "KUMCは何のサークル？1", "KUMCは何のサークル？2"],
        )

    def test_execute_skips_ragas_when_dependency_missing(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(chat_usecase=fake_chat)

        with tempfile.TemporaryDirectory() as tmp:
            eval_file = self._write_eval_jsonl(Path(tmp))
            with patch.dict(
                sys.modules,
                {"datasets": None, "ragas": None, "ragas.metrics": None},
                clear=False,
            ):
                result = usecase.execute(EvaluateRagasRequest(eval_file=eval_file))

        self.assertEqual(result.total, 1)
        self.assertEqual(result.ragas_metrics, {})

    def test_execute_accepts_query_field_alias(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(chat_usecase=fake_chat)

        with tempfile.TemporaryDirectory() as tmp:
            eval_file = self._write_eval_jsonl_with_query_key(Path(tmp), count=2)
            with patch.dict(
                sys.modules,
                {"datasets": None, "ragas": None, "ragas.metrics": None},
                clear=False,
            ):
                result = usecase.execute(EvaluateRagasRequest(eval_file=eval_file))

        self.assertEqual(result.total, 2)
        self.assertEqual(
            fake_chat.queries,
            ["KUMCは何のサークル？0", "KUMCは何のサークル？1"],
        )

    def test_execute_uses_answer_cache_on_second_run(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(chat_usecase=fake_chat)

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            eval_file = self._write_eval_jsonl(base, count=2)
            cache_path = base / "cache" / "ragas_answers.jsonl"
            with patch.dict(
                sys.modules,
                {"datasets": None, "ragas": None, "ragas.metrics": None},
                clear=False,
            ):
                first = usecase.execute(
                    EvaluateRagasRequest(
                        eval_file=eval_file,
                        answer_cache_path=cache_path,
                        answer_cache_enabled=True,
                    )
                )
                second = usecase.execute(
                    EvaluateRagasRequest(
                        eval_file=eval_file,
                        answer_cache_path=cache_path,
                        answer_cache_enabled=True,
                    )
                )

        self.assertEqual(first.total, 2)
        self.assertEqual(second.total, 2)
        self.assertEqual(fake_chat.queries, ["KUMCは何のサークル？0", "KUMCは何のサークル？1"])
        self.assertEqual(second.ragas_metadata["answer_cache_hits"], 2)
        self.assertEqual(second.ragas_metadata["answer_cache_misses"], 0)

    def test_execute_flushes_answer_cache_using_answer_generation_batch_size(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(
            chat_usecase=fake_chat,
            default_answer_generation_batch_size=2,
        )
        written_batch_sizes: list[int] = []

        def _capture_cache_append(path, entries):  # type: ignore[no-untyped-def]
            _ = path
            written_batch_sizes.append(len(entries))

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            eval_file = self._write_eval_jsonl(base, count=5)
            cache_path = base / "cache" / "ragas_answers.jsonl"
            with patch.dict(
                sys.modules,
                {"datasets": None, "ragas": None, "ragas.metrics": None},
                clear=False,
            ):
                with patch.object(
                    EvaluateRagasUsecase,
                    "_append_answer_cache_entries",
                    side_effect=_capture_cache_append,
                ):
                    result = usecase.execute(
                        EvaluateRagasRequest(
                            eval_file=eval_file,
                            answer_cache_path=cache_path,
                            answer_cache_enabled=True,
                            ragas_batch_size=5,
                        )
                    )

        self.assertEqual(result.total, 5)
        self.assertEqual(written_batch_sizes, [2, 2, 1])
        self.assertCountEqual(
            fake_chat.queries,
            [f"KUMCは何のサークル？{i}" for i in range(5)],
        )

    def test_execute_parallelizes_answer_generation_within_batch(self) -> None:
        fake_chat = _ConcurrentTrackingChatUsecase(delay_seconds=0.05)
        usecase = EvaluateRagasUsecase(
            chat_usecase=fake_chat,
            default_answer_generation_batch_size=4,
            default_ragas_max_workers=4,
        )

        with tempfile.TemporaryDirectory() as tmp:
            eval_file = self._write_eval_jsonl(Path(tmp), count=4)
            with patch.dict(
                sys.modules,
                {"datasets": None, "ragas": None, "ragas.metrics": None},
                clear=False,
            ):
                result = usecase.execute(
                    EvaluateRagasRequest(
                        eval_file=eval_file,
                        answer_cache_enabled=False,
                    )
                )

        self.assertEqual(result.total, 4)
        self.assertGreaterEqual(fake_chat.max_active, 2)
        self.assertEqual(result.ragas_metadata["answer_generation_max_workers"], 4)

    def test_execute_fallback_batches_continue_after_failure(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(
            chat_usecase=fake_chat,
            eval_answer_relevancy_enabled=True,
            eval_faithfulness_enabled=False,
            eval_context_precision_enabled=False,
            eval_context_recall_enabled=False,
        )
        metric_objects = {
            "answer_relevancy": object(),
            "faithfulness": object(),
            "context_precision": object(),
            "context_recall": object(),
        }
        call_count = {"value": 0}

        datasets_module = types.ModuleType("datasets")

        class _Dataset:
            @staticmethod
            def from_list(records):  # type: ignore[no-untyped-def]
                return records

        datasets_module.Dataset = _Dataset  # type: ignore[attr-defined]
        ragas_module = types.ModuleType("ragas")

        def _evaluate(dataset, metrics, batch_size=None, run_config=None, return_executor=False):  # type: ignore[no-untyped-def]
            _ = metrics
            _ = batch_size
            _ = run_config
            _ = return_executor
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise RuntimeError("single-pass failure")
            if len(dataset) == 2:
                raise RuntimeError("first fallback batch failure")
            return _FakeRagasResult(scores={"answer_relevancy": 1.0})

        ragas_module.evaluate = _evaluate  # type: ignore[attr-defined]
        ragas_metrics_module = types.ModuleType("ragas.metrics")
        ragas_metrics_module.answer_relevancy = metric_objects["answer_relevancy"]  # type: ignore[attr-defined]
        ragas_metrics_module.faithfulness = metric_objects["faithfulness"]  # type: ignore[attr-defined]
        ragas_metrics_module.context_precision = metric_objects["context_precision"]  # type: ignore[attr-defined]
        ragas_metrics_module.context_recall = metric_objects["context_recall"]  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as tmp:
            eval_file = self._write_eval_jsonl(Path(tmp), count=3)
            with patch.dict(
                sys.modules,
                {
                    "datasets": datasets_module,
                    "ragas": ragas_module,
                    "ragas.metrics": ragas_metrics_module,
                },
                clear=False,
            ):
                result = usecase.execute(
                    EvaluateRagasRequest(
                        eval_file=eval_file,
                        ragas_batch_size=2,
                    )
                )

        self.assertEqual(result.total, 3)
        self.assertAlmostEqual(result.ragas_metrics["answer_relevancy"], 1.0)
        self.assertEqual(result.ragas_metadata["mode"], "fallback_batches")
        self.assertEqual(result.ragas_metadata["failed_batches"], 1)
        self.assertEqual(result.ragas_metadata["failed_records"], 2)

    def test_execute_single_pass_executor_results_do_not_trigger_fallback(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(
            chat_usecase=fake_chat,
            eval_answer_relevancy_enabled=True,
            eval_faithfulness_enabled=False,
            eval_context_precision_enabled=False,
            eval_context_recall_enabled=False,
        )
        metric_objects = {
            "answer_relevancy": object(),
            "faithfulness": object(),
            "context_precision": object(),
            "context_recall": object(),
        }
        evaluate_call_count = {"value": 0}

        datasets_module = types.ModuleType("datasets")

        class _Dataset:
            @staticmethod
            def from_list(records):  # type: ignore[no-untyped-def]
                return records

        datasets_module.Dataset = _Dataset  # type: ignore[attr-defined]
        ragas_module = types.ModuleType("ragas")

        class _Executor:
            def __init__(self, values: list[float]) -> None:
                self._values = values

            def results(self) -> list[float]:
                return list(self._values)

        def _evaluate(dataset, metrics, batch_size=None, run_config=None, return_executor=False):  # type: ignore[no-untyped-def]
            _ = batch_size
            _ = run_config
            evaluate_call_count["value"] += 1
            if not return_executor:
                raise AssertionError("evaluate should be called with return_executor=True")
            values = [0.8 for _ in range(len(dataset) * len(metrics))]
            return _Executor(values)

        ragas_module.evaluate = _evaluate  # type: ignore[attr-defined]
        ragas_metrics_module = types.ModuleType("ragas.metrics")
        ragas_metrics_module.answer_relevancy = metric_objects["answer_relevancy"]  # type: ignore[attr-defined]
        ragas_metrics_module.faithfulness = metric_objects["faithfulness"]  # type: ignore[attr-defined]
        ragas_metrics_module.context_precision = metric_objects["context_precision"]  # type: ignore[attr-defined]
        ragas_metrics_module.context_recall = metric_objects["context_recall"]  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as tmp:
            eval_file = self._write_eval_jsonl(Path(tmp), count=3)
            with patch.dict(
                sys.modules,
                {
                    "datasets": datasets_module,
                    "ragas": ragas_module,
                    "ragas.metrics": ragas_metrics_module,
                },
                clear=False,
            ):
                result = usecase.execute(
                    EvaluateRagasRequest(
                        eval_file=eval_file,
                        ragas_batch_size=2,
                    )
                )

        self.assertEqual(result.total, 3)
        self.assertAlmostEqual(result.ragas_metrics["answer_relevancy"], 0.8)
        self.assertEqual(result.ragas_metadata["mode"], "single_pass")
        self.assertEqual(evaluate_call_count["value"], 1)

    def test_execute_forces_answer_relevancy_strictness_for_instructor_llm(self) -> None:
        fake_chat = _FakeChatUsecase()

        class _FakeInstructorBaseRagasLLM:
            pass

        class _UsecaseWithInstructorLLM(EvaluateRagasUsecase):
            def __init__(self, *, llm, **kwargs) -> None:  # type: ignore[no-untyped-def]
                super().__init__(**kwargs)
                self._llm = llm

            def _build_ragas_llm(self):  # type: ignore[override]
                return self._llm

        llm = _FakeInstructorBaseRagasLLM()
        usecase = _UsecaseWithInstructorLLM(
            llm=llm,
            chat_usecase=fake_chat,
            eval_answer_relevancy_enabled=True,
            eval_faithfulness_enabled=False,
            eval_context_precision_enabled=False,
            eval_context_recall_enabled=False,
        )

        captured: dict[str, object] = {}

        class _AnswerRelevancyMetric:
            def __init__(self) -> None:
                self.strictness = 3

        metric_objects = {
            "answer_relevancy": _AnswerRelevancyMetric(),
            "faithfulness": object(),
            "context_precision": object(),
            "context_recall": object(),
        }

        datasets_module = types.ModuleType("datasets")

        class _Dataset:
            @staticmethod
            def from_list(records):  # type: ignore[no-untyped-def]
                return records

        datasets_module.Dataset = _Dataset  # type: ignore[attr-defined]

        ragas_module = types.ModuleType("ragas")

        def _evaluate(dataset, metrics, **kwargs):  # type: ignore[no-untyped-def]
            _ = dataset
            _ = kwargs
            captured["strictness"] = getattr(metrics[0], "strictness", None)
            return _FakeRagasResult(scores={"answer_relevancy": 0.6})

        ragas_module.evaluate = _evaluate  # type: ignore[attr-defined]

        ragas_metrics_module = types.ModuleType("ragas.metrics")
        ragas_metrics_module.answer_relevancy = metric_objects["answer_relevancy"]  # type: ignore[attr-defined]
        ragas_metrics_module.faithfulness = metric_objects["faithfulness"]  # type: ignore[attr-defined]
        ragas_metrics_module.context_precision = metric_objects["context_precision"]  # type: ignore[attr-defined]
        ragas_metrics_module.context_recall = metric_objects["context_recall"]  # type: ignore[attr-defined]

        ragas_llms_base_module = types.ModuleType("ragas.llms.base")
        ragas_llms_base_module.InstructorBaseRagasLLM = _FakeInstructorBaseRagasLLM  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as tmp:
            eval_file = self._write_eval_jsonl(Path(tmp), count=1)
            with patch.dict(
                sys.modules,
                {
                    "datasets": datasets_module,
                    "ragas": ragas_module,
                    "ragas.metrics": ragas_metrics_module,
                    "ragas.llms.base": ragas_llms_base_module,
                },
                clear=False,
            ):
                result = usecase.execute(EvaluateRagasRequest(eval_file=eval_file))

        self.assertEqual(result.total, 1)
        self.assertEqual(captured["strictness"], 1)
        self.assertTrue(result.ragas_metadata["answer_relevancy_strictness_forced"])

    def test_execute_honors_cancel_event_before_processing(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(chat_usecase=fake_chat)
        cancel_event = threading.Event()
        cancel_event.set()

        with tempfile.TemporaryDirectory() as tmp:
            eval_file = self._write_eval_jsonl(Path(tmp), count=2)
            with patch.dict(
                sys.modules,
                {"datasets": None, "ragas": None, "ragas.metrics": None},
                clear=False,
            ):
                result = usecase.execute(
                    EvaluateRagasRequest(eval_file=eval_file, cancel_event=cancel_event)
                )

        self.assertEqual(result.total, 0)
        self.assertEqual(fake_chat.queries, [])
        self.assertTrue(result.ragas_metadata["canceled"])

    def test_execute_disables_history_by_default(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(chat_usecase=fake_chat)

        with tempfile.TemporaryDirectory() as tmp:
            eval_file = self._write_eval_jsonl(Path(tmp), count=1)
            with patch.dict(
                sys.modules,
                {"datasets": None, "ragas": None, "ragas.metrics": None},
                clear=False,
            ):
                usecase.execute(EvaluateRagasRequest(eval_file=eval_file))

        self.assertEqual(len(fake_chat.requests), 1)
        request = fake_chat.requests[0]
        self.assertEqual(request.routing_history_override, [])
        self.assertEqual(request.generation_history_override, [])
        self.assertTrue(request.force_disable_additional_memory)

    def test_build_ragas_llm_skips_when_provider_keyword_is_unsupported(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(
            chat_usecase=fake_chat,
            gemini_api_key="dummy-key",
            ragas_gemini_model="gemini-2.0-flash",
        )

        called = {"count": 0}

        google_module = types.ModuleType("google")
        genai_module = types.ModuleType("google.genai")

        class _Client:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key

        genai_module.Client = _Client  # type: ignore[attr-defined]
        google_module.genai = genai_module  # type: ignore[attr-defined]

        llms_module = types.ModuleType("ragas.llms")

        def _llm_factory(model, client):  # type: ignore[no-untyped-def]
            _ = model
            _ = client
            called["count"] += 1
            return object()

        llms_module.llm_factory = _llm_factory  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {
                "google": google_module,
                "google.genai": genai_module,
                "ragas.llms": llms_module,
            },
            clear=False,
        ):
            llm = usecase._build_ragas_llm()

        self.assertIsNone(llm)
        self.assertEqual(called["count"], 0)

    def test_build_ragas_llm_passes_google_provider_with_genai_client(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(
            chat_usecase=fake_chat,
            gemini_api_key="dummy-key",
            ragas_gemini_model="gemini-2.0-flash",
        )

        captured: dict[str, object] = {}

        google_module = types.ModuleType("google")
        genai_module = types.ModuleType("google.genai")

        class _Client:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key

        genai_module.Client = _Client  # type: ignore[attr-defined]
        google_module.genai = genai_module  # type: ignore[attr-defined]

        llms_module = types.ModuleType("ragas.llms")

        def _llm_factory(model, provider, client):  # type: ignore[no-untyped-def]
            captured["model"] = model
            captured["provider"] = provider
            captured["client"] = client
            return {"ok": True}

        llms_module.llm_factory = _llm_factory  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {
                "google": google_module,
                "google.genai": genai_module,
                "ragas.llms": llms_module,
            },
            clear=False,
        ):
            llm = usecase._build_ragas_llm()

        self.assertEqual(llm, {"ok": True})
        self.assertEqual(captured["model"], "gemini-2.0-flash")
        self.assertEqual(captured["provider"], "google")
        self.assertIsNotNone(captured.get("client"))

    def test_build_ragas_embeddings_applies_rate_limit(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(
            chat_usecase=fake_chat,
            gemini_api_key="dummy-key",
            ragas_gemini_embedding_requests_per_minute=123,
        )

        google_module = types.ModuleType("google")
        genai_module = types.ModuleType("google.genai")

        class _Client:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key

        genai_module.Client = _Client  # type: ignore[attr-defined]
        google_module.genai = genai_module  # type: ignore[attr-defined]

        ragas_google_embeddings_module = types.ModuleType(
            "ragas.embeddings.google_provider"
        )

        class _GoogleEmbeddings:
            def __init__(self, client) -> None:  # type: ignore[no-untyped-def]
                self.client = client

            def embed_query(self, text: str) -> list[float]:
                _ = text
                return [0.1]

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[float(len(text))] for text in texts]

        ragas_google_embeddings_module.GoogleEmbeddings = _GoogleEmbeddings  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {
                "google": google_module,
                "google.genai": genai_module,
                "ragas.embeddings.google_provider": ragas_google_embeddings_module,
            },
            clear=False,
        ):
            with patch(
                "kumc_agent.usecases.eval.ragas.wait_for_gemini_rate_limit"
            ) as wait_mock:
                embeddings = usecase._build_ragas_embeddings()
                self.assertIsNotNone(embeddings)
                self.assertEqual(embeddings.embed_query("q"), [0.1])
                self.assertEqual(embeddings.embed_documents(["a", "bc"]), [[1.0], [2.0]])

        self.assertEqual(wait_mock.call_count, 2)
        for call in wait_mock.call_args_list:
            self.assertEqual(call.kwargs["max_requests_per_minute"], 123)
            self.assertEqual(
                call.kwargs["limiter_name"],
                ragas_embedding_rate_limiter_name(),
            )

    def test_build_ragas_embeddings_falls_back_to_ragas_limit(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(
            chat_usecase=fake_chat,
            gemini_api_key="dummy-key",
            ragas_gemini_requests_per_minute=77,
        )

        google_module = types.ModuleType("google")
        genai_module = types.ModuleType("google.genai")

        class _Client:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key

        genai_module.Client = _Client  # type: ignore[attr-defined]
        google_module.genai = genai_module  # type: ignore[attr-defined]

        ragas_google_embeddings_module = types.ModuleType(
            "ragas.embeddings.google_provider"
        )

        class _GoogleEmbeddings:
            def __init__(self, client) -> None:  # type: ignore[no-untyped-def]
                self.client = client

            def embed_query(self, text: str) -> list[float]:
                _ = text
                return [0.2]

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.2] for _ in texts]

        ragas_google_embeddings_module.GoogleEmbeddings = _GoogleEmbeddings  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {
                "google": google_module,
                "google.genai": genai_module,
                "ragas.embeddings.google_provider": ragas_google_embeddings_module,
            },
            clear=False,
        ):
            with patch(
                "kumc_agent.usecases.eval.ragas.wait_for_gemini_rate_limit"
            ) as wait_mock:
                embeddings = usecase._build_ragas_embeddings()
                self.assertIsNotNone(embeddings)
                self.assertEqual(embeddings.embed_query("q"), [0.2])

        self.assertEqual(wait_mock.call_count, 1)
        self.assertEqual(
            wait_mock.call_args.kwargs["max_requests_per_minute"],
            77,
        )
        self.assertEqual(
            wait_mock.call_args.kwargs["limiter_name"],
            ragas_embedding_rate_limiter_name(),
        )

    def test_as_legacy_embeddings_wraps_modern_interface(self) -> None:
        class _ModernEmbeddings:
            def embed_text(self, text: str):  # type: ignore[no-untyped-def]
                _ = text
                return [0.1, 0.2]

            def embed_texts(self, texts):  # type: ignore[no-untyped-def]
                return [[float(i)] for i, _ in enumerate(texts)]

            async def aembed_text(self, text: str):  # type: ignore[no-untyped-def]
                _ = text
                return [0.3, 0.4]

            async def aembed_texts(self, texts):  # type: ignore[no-untyped-def]
                return [[float(i) + 10.0] for i, _ in enumerate(texts)]

        wrapped = _as_legacy_embeddings(_ModernEmbeddings())

        self.assertEqual(wrapped.embed_query("q"), [0.1, 0.2])
        self.assertEqual(wrapped.embed_documents(["a", "b"]), [[0.0], [1.0]])
        self.assertEqual(asyncio.run(wrapped.aembed_query("q")), [0.3, 0.4])
        self.assertEqual(
            asyncio.run(wrapped.aembed_documents(["a", "b"])),
            [[10.0], [11.0]],
        )


if __name__ == "__main__":
    unittest.main()
