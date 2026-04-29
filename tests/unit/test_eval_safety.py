from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.usecases.eval.safety import SafetyAssertionEngine, sanitize_report_payload
from kumc_agent.usecases.eval.schema import EvalCase


class EvalSafetyTests(unittest.TestCase):
    def test_secret_leak_is_zero_tolerance_failure(self) -> None:
        case = EvalCase(id="case-1", target="agentic", suite="safety", input={})
        result = SafetyAssertionEngine(zero_tolerance=True).evaluate(
            case=case,
            actual={"answer": "token=abcdefghi"},
        )
        self.assertEqual(result["secret_leak_count"], 1)
        self.assertTrue(result["zero_tolerance_failed"])

    def test_side_effect_violation_is_counted(self) -> None:
        case = EvalCase(id="case-1", target="server_management", suite="smoke", input={})
        result = SafetyAssertionEngine(zero_tolerance=True).evaluate(
            case=case,
            actual={"metadata": {"side_effects": [{"type": "execute"}]}},
        )
        self.assertEqual(result["side_effect_violation_count"], 1)

    def test_sanitizer_drops_raw_context(self) -> None:
        payload = sanitize_report_payload({"metadata": {"raw_context": "private", "ok": "x"}})
        self.assertEqual(payload, {"metadata": {"ok": "x"}})

    def test_sanitizer_keeps_run_id_like_digits(self) -> None:
        payload = sanitize_report_payload({"artifact_path": "data/eval/results/eval-83846773.json"})
        self.assertEqual(payload["artifact_path"], "data/eval/results/eval-83846773.json")

    def test_sanitizer_masks_labeled_student_id(self) -> None:
        payload = sanitize_report_payload({"text": "学籍番号: 12345678"})
        self.assertEqual(payload["text"], "[MASKED]")


if __name__ == "__main__":
    unittest.main()
