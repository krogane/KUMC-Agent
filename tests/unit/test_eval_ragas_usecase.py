from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.answer import Answer
from kumc_agent.usecases.eval.ragas import EvaluateRagasRequest, EvaluateRagasUsecase


class _FakeChatUsecase:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def execute(self, request):  # type: ignore[no-untyped-def]
        self.queries.append(str(request.query))
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

    def test_execute_runs_ragas_in_batches(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = EvaluateRagasUsecase(
            chat_usecase=fake_chat,
            eval_answer_relevancy_enabled=True,
            eval_faithfulness_enabled=False,
            eval_context_precision_enabled=False,
            eval_context_recall_enabled=False,
        )

        batch_sizes: list[int] = []
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

        def _evaluate(dataset, metrics):  # type: ignore[no-untyped-def]
            _ = metrics
            batch_sizes.append(len(dataset))
            return _FakeRagasResult(scores={"answer_relevancy": float(len(dataset))})

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
        self.assertEqual(batch_sizes, [2, 1])
        self.assertAlmostEqual(result.ragas_metrics["answer_relevancy"], 5.0 / 3.0)
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


if __name__ == "__main__":
    unittest.main()
