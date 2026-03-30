from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.features.rag.components.entry_routing import EntryQueryRouter


def _build_router(*, retries: int = 1) -> EntryQueryRouter:
    return EntryQueryRouter(
        provider="gemini",
        gemini_model="gemini-2.5-flash-lite",
        llama_model_path="",
        temperature=0.0,
        max_new_tokens=128,
        max_retries=retries,
        llm_threads=4,
        llm_gpu_layers=0,
        llm_ctx_size=4096,
        gemini_api_key="dummy",
        gemini_requests_per_minute=60,
        prompt_name="routing_openclaw_gate",
        log_enabled=False,
    )


class EntryQueryRouterTests(unittest.TestCase):
    def test_direct_rag_payload_is_parsed(self) -> None:
        router = _build_router(retries=0)
        with patch.object(
            router,
            "_generate_payload",
            return_value='{"route":"direct_rag","reason":"事実照会"}',
        ):
            decision = router.decide("次回の例会は？")

        self.assertEqual(decision.route, "direct_rag")
        self.assertEqual(decision.reason, "事実照会")
        self.assertEqual(decision.payload.get("route"), "direct_rag")

    def test_invalid_payload_falls_back_to_openclaw(self) -> None:
        router = _build_router(retries=0)
        with patch.object(
            router,
            "_generate_payload",
            return_value='{"route":"unknown","reason":"x"}',
        ):
            decision = router.decide("質問")

        self.assertEqual(decision.route, "openclaw")
        self.assertEqual(decision.reason, "fallback:classification_failed")
        self.assertIsInstance(decision.payload.get("raw"), str)

    def test_generation_error_falls_back_to_openclaw(self) -> None:
        router = _build_router(retries=0)
        with patch.object(
            router,
            "_generate_payload",
            side_effect=RuntimeError("generation failed"),
        ):
            decision = router.decide("質問")

        self.assertEqual(decision.route, "openclaw")
        self.assertEqual(decision.reason, "fallback:classification_failed")

    def test_retryable_generation_error_retries_once(self) -> None:
        router = _build_router(retries=1)
        with (
            patch.object(
                router,
                "_generate_payload",
                side_effect=[
                    RuntimeError("503 UNAVAILABLE: high demand"),
                    '{"route":"openclaw","reason":"複雑質問"}',
                ],
            ),
            patch(
                "kumc_agent.features.rag.components.entry_routing.time.sleep"
            ) as sleep_mock,
        ):
            decision = router.decide("計画を作って")

        self.assertEqual(decision.route, "openclaw")
        self.assertEqual(decision.reason, "複雑質問")
        sleep_mock.assert_called_once_with(1.0)


if __name__ == "__main__":
    unittest.main()
