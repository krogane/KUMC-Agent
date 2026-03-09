from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.features.rag.components.routing import QueryRouter


def _build_router(*, refusal_keywords: list[str] | None = None) -> QueryRouter:
    return QueryRouter(
        refusal_keywords=list(refusal_keywords or []),
        routing_enabled=True,
        provider="gemini",
        gemini_model="gemini-2.5-flash-lite",
        llama_model_path="",
        temperature=0.0,
        max_new_tokens=128,
        max_retries=1,
        log_enabled=False,
        material_search_max_names=3,
        llm_thinking_level="minimal",
        llm_threads=4,
        llm_gpu_layers=0,
        llm_ctx_size=4096,
        gemini_api_key="dummy",
        gemini_requests_per_minute=60,
    )


class QueryRouterTests(unittest.TestCase):
    def test_parse_material_search_payload(self) -> None:
        router = _build_router()
        payload = (
            "```json\n"
            "{\n"
            '  "target_model": "material_search",\n'
            '  "material_names": ["運営資料", "運営資料", "議事録"],\n'
            '  "idea_generation": true,\n'
            '  "include_capabilities_info": true,\n'
            '  "recency_mode": "soft",\n'
            '  "use_additional_memory": true,\n'
            '  "needs_additional_query": true,\n'
            '  "additional_queries": ["A", "B"]\n'
            "}\n"
            "```"
        )
        with patch.object(router, "_generate_routing_payload", return_value=payload):
            decision = router.route("資料「運営資料」を探して")

        self.assertEqual(decision.target_model, "material_search")
        self.assertEqual(decision.material_names, ["運営資料", "議事録"])
        self.assertEqual(decision.recency_mode, "soft")
        self.assertTrue(decision.include_capabilities_info)
        self.assertTrue(decision.use_additional_memory)
        self.assertFalse(decision.idea_generation)
        self.assertFalse(decision.needs_additional_query)
        self.assertEqual(decision.additional_queries, [])

    def test_invalid_payload_falls_back_to_safe_default(self) -> None:
        router = _build_router()
        with patch.object(router, "_generate_routing_payload", return_value="not-json"):
            decision = router.route("今日の天気は？")

        self.assertEqual(decision.target_model, "rag")
        self.assertEqual(decision.recency_mode, "off")

    def test_generation_error_falls_back_to_safe_default(self) -> None:
        router = _build_router(refusal_keywords=["住所"])
        with patch.object(
            router,
            "_generate_routing_payload",
            side_effect=RuntimeError("routing llm error"),
        ):
            decision = router.route("今日の天気は？")

        self.assertEqual(decision.target_model, "rag")
        self.assertEqual(decision.recency_mode, "off")

    def test_routing_disabled_returns_safe_default(self) -> None:
        router = _build_router()
        router._routing_enabled = False  # test setup only
        decision = router.route("今日の天気は？")

        self.assertEqual(decision.target_model, "rag")
        self.assertEqual(decision.recency_mode, "off")


if __name__ == "__main__":
    unittest.main()
