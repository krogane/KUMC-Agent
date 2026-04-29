from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.usecases.eval.assertions import AssertionEngine
from kumc_agent.usecases.eval.schema import EvalAssertion, EvalCase


class EvalAssertionTests(unittest.TestCase):
    def test_answer_contains_any_passes(self) -> None:
        case = EvalCase(
            id="case-1",
            target="rag_circle",
            suite="smoke",
            input={},
            assertions=(
                EvalAssertion(type="answer_contains_any", params={"values": ["土曜日"]}),
            ),
        )
        result = AssertionEngine().evaluate(
            case=case,
            actual={"answer": "例会は土曜日です。", "metadata": {}},
        )
        self.assertTrue(result[0].passed)

    def test_acl_no_forbidden_source_fails(self) -> None:
        case = EvalCase(
            id="case-1",
            target="rag_circle",
            suite="smoke",
            input={},
            expected={"forbidden_source_ids": ["secret-source"]},
            assertions=(EvalAssertion(type="acl_no_forbidden_source"),),
        )
        result = AssertionEngine().evaluate(
            case=case,
            actual={"citations": [{"source_id": "secret-source"}], "metadata": {}},
        )
        self.assertFalse(result[0].passed)

    def test_no_side_effect_fails_on_side_effect_metadata(self) -> None:
        case = EvalCase(
            id="case-1",
            target="server_management",
            suite="smoke",
            input={},
            assertions=(EvalAssertion(type="no_side_effect"),),
        )
        result = AssertionEngine().evaluate(
            case=case,
            actual={"metadata": {"side_effects": [{"type": "execute"}]}},
        )
        self.assertFalse(result[0].passed)


if __name__ == "__main__":
    unittest.main()
