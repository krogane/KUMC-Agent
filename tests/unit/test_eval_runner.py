from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.usecases.eval.runner import EvaluateBatchRequest, EvaluateRequest, EvalRunner


class _FakeOperationsRepository:
    def __init__(self) -> None:
        self.eval_runs = []

    def save_eval_run(self, run):  # type: ignore[no-untyped-def]
        self.eval_runs.append(run)
        return run


class EvalRunnerTests(unittest.TestCase):
    def test_runner_saves_artifact_and_eval_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            eval_set_dir = base / "sets"
            result_dir = base / "results"
            target_dir = eval_set_dir / "task_management"
            target_dir.mkdir(parents=True)
            (target_dir / "smoke.jsonl").write_text(
                json.dumps(
                    {
                        "id": "task-1",
                        "target": "task_management",
                        "suite": "smoke",
                        "input": {
                            "adapter_output": {
                                "text": "候補を作成しました。",
                                "candidates": [{"id": "candidate-1"}],
                                "approval_required": True,
                                "status": "proposed",
                                "metadata": {"side_effects": []},
                            }
                        },
                        "expected": {
                            "expected_candidates": ["candidate-1"],
                            "approval_required": True,
                            "side_effects_allowed": False,
                        },
                        "assertions": [],
                        "severity": "critical",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            repo = _FakeOperationsRepository()
            result = EvalRunner(
                eval_sets_dir=eval_set_dir,
                results_dir=result_dir,
                operations_repository=repo,
            ).execute(EvaluateRequest(target="task_management", suite="smoke"))
            artifact_exists = Path(result.metadata["artifact_path"]).exists()

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.total, 1)
        self.assertEqual(len(repo.eval_runs), 1)
        self.assertTrue(artifact_exists)

    def test_empty_eval_set_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            result = EvalRunner(
                eval_sets_dir=base / "sets",
                results_dir=base / "results",
            ).execute(EvaluateRequest(target="missing", suite="smoke"))
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.total, 0)

    def test_missing_eval_set_can_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            result = EvalRunner(
                eval_sets_dir=base / "sets",
                results_dir=base / "results",
            ).execute(
                EvaluateRequest(
                    target="missing",
                    suite="full",
                    mode="full",
                    missing_eval_set_policy="fail",
                    min_cases=1,
                )
            )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.total, 1)
        self.assertEqual(result.failures[0]["case_id"], "__eval_set_missing__")

    def test_batch_fails_on_missing_required_eval_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            result = EvalRunner(
                eval_sets_dir=base / "sets",
                results_dir=base / "results",
            ).execute_batch(
                EvaluateBatchRequest(
                    suite="full",
                    mode="full",
                    targets=("missing",),
                    missing_eval_set_policy="fail",
                    min_cases=1,
                )
            )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failed, 1)


if __name__ == "__main__":
    unittest.main()
