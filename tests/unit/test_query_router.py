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


def _build_router() -> QueryRouter:
    return QueryRouter(
        routing_enabled=True,
        provider="gemini",
        gemini_model="gemini-2.5-flash-lite",
        temperature=0.0,
        max_new_tokens=128,
        max_retries=1,
        log_enabled=False,
        material_search_max_names=3,
        gemini_api_key="dummy",
        gemini_requests_per_minute=60,
    )


class QueryRouterTests(unittest.TestCase):
    def test_retryable_generation_error_uses_backoff_and_recovers(self) -> None:
        router = _build_router()
        with (
            patch.object(
                router,
                "_generate_task_payload",
                side_effect=[
                    RuntimeError("503 UNAVAILABLE: high demand"),
                    '{"use_additional_memory": true}',
                ],
            ),
            patch("kumc_agent.features.rag.components.routing.time.sleep") as sleep_mock,
        ):
            result = router._run_task_with_retries(
                task_name="use_additional_memory",
                query="企画案を出して",
                question_author=None,
                history=None,
                context={},
            )

        self.assertTrue(result)
        sleep_mock.assert_called_once_with(1.0)

    def test_non_retryable_generation_error_does_not_sleep(self) -> None:
        router = _build_router()
        with (
            patch.object(
                router,
                "_generate_task_payload",
                side_effect=RuntimeError("schema validation failed"),
            ),
            patch("kumc_agent.features.rag.components.routing.time.sleep") as sleep_mock,
        ):
            result = router._run_task_with_retries(
                task_name="use_additional_memory",
                query="企画案を出して",
                question_author=None,
                history=None,
                context={},
            )

        self.assertFalse(result)
        sleep_mock.assert_not_called()

    def test_route_collects_supported_fields(self) -> None:
        router = _build_router()
        values = {
            "use_additional_memory": True,
            "material_names": ["運営資料", "議事録"],
            "recency_mode": "hard",
            "needs_additional_query": True,
            "additional_queries": ["追加クエリ"],
        }

        with patch.object(router, "_run_task_with_retries") as mocked:
            mocked.side_effect = (
                lambda *, task_name, **_: values.get(task_name, router._default_task_value(task_name))
            )
            decision = router.route("資料「運営資料」を探して")

        self.assertEqual(decision.recency_mode, "hard")
        self.assertEqual(decision.material_names, ["運営資料", "議事録"])
        self.assertFalse(decision.include_capabilities_info)
        self.assertTrue(decision.use_additional_memory)
        self.assertEqual(decision.additional_queries, ["追加クエリ"])

        called_tasks = [
            call.kwargs["task_name"] for call in mocked.call_args_list if "task_name" in call.kwargs
        ]
        self.assertIn("material_names", called_tasks)
        self.assertIn("recency_mode", called_tasks)
        self.assertIn("needs_additional_query", called_tasks)
        self.assertIn("additional_queries", called_tasks)

    def test_minecraft_wiki_route_suppresses_additional_query_fields(self) -> None:
        router = _build_router()
        values = {
            "target_model": "minecraft_wiki",
            "include_capabilities_info": True,
            "material_names": ["運営資料"],
            "recency_mode": "hard",
            "needs_additional_query": True,
            "additional_queries": ["丸石 レシピ"],
        }

        with patch.object(router, "_run_task_with_retries") as mocked:
            mocked.side_effect = (
                lambda *, task_name, **_: values.get(task_name, router._default_task_value(task_name))
            )
            decision = router.route("丸石のレシピを教えて")

        self.assertEqual(decision.target_model, "minecraft_wiki")
        self.assertEqual(decision.recency_mode, "off")
        self.assertEqual(decision.material_names, [])
        self.assertEqual(decision.additional_queries, [])
        self.assertFalse(decision.include_capabilities_info)

    def test_routing_disabled_returns_safe_default(self) -> None:
        router = _build_router()
        router._routing_enabled = False  # test setup only

        decision = router.route("今日の天気は？")

        self.assertEqual(decision.recency_mode, "off")
        self.assertEqual(decision.material_names, [])
        self.assertFalse(decision.use_additional_memory)


if __name__ == "__main__":
    unittest.main()
