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

from kumc_agent.usecases.eval.dataset import EvalDatasetError, load_eval_set


class EvalDatasetLoaderTests(unittest.TestCase):
    def test_loads_eval_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "case-1",
                        "target": "task_management",
                        "suite": "smoke",
                        "input": {"text": "task"},
                        "expected": {},
                        "assertions": [],
                        "metadata": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cases = load_eval_set(path, target="task_management", suite="smoke")
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].id, "case-1")

    def test_missing_required_field_reports_case_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jsonl"
            path.write_text(
                json.dumps({"target": "task_management", "suite": "smoke", "input": {}})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(EvalDatasetError):
                load_eval_set(path)

    def test_loads_ragas_compat_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ragas.jsonl"
            path.write_text(
                json.dumps({"question": "KUMC?", "ground_truth": "Minecraft"}) + "\n",
                encoding="utf-8",
            )
            cases = load_eval_set(path, target="rag_circle", suite="smoke")
        self.assertEqual(cases[0].input["question"], "KUMC?")
        self.assertEqual(cases[0].expected["answer_contains"], ["Minecraft"])

    def test_sanitizes_dangerous_metadata_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "case-1",
                        "target": "rag_circle",
                        "suite": "smoke",
                        "input": {"question": "q"},
                        "metadata": {"raw_context": "large private context"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cases = load_eval_set(path, target="rag_circle", suite="smoke")
        self.assertNotIn("raw_context", cases[0].metadata)
        self.assertIn("sanitizer_warnings", cases[0].metadata)


if __name__ == "__main__":
    unittest.main()
