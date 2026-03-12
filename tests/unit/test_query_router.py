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
        llm_threads=4,
        llm_gpu_layers=0,
        llm_ctx_size=4096,
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
                    '{"idea_generation": true}',
                ],
            ),
            patch("kumc_agent.features.rag.components.routing.time.sleep") as sleep_mock,
        ):
            result = router._run_task_with_retries(
                task_name="idea_generation",
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
                task_name="idea_generation",
                query="企画案を出して",
                question_author=None,
                history=None,
                context={},
            )

        self.assertFalse(result)
        sleep_mock.assert_not_called()

    def test_material_search_flow_ignores_rag_only_fields(self) -> None:
        router = _build_router()
        values = {
            "target_model": "material_search",
            "use_additional_memory": True,
            "include_capabilities_info": True,
            "idea_generation": True,
            "needs_additional_query": True,
            "material_names": ["運営資料", "議事録"],
            "recency_mode": "hard",
            "additional_queries": ["追加クエリ"],
        }

        with patch.object(router, "_run_task_with_retries") as mocked:
            mocked.side_effect = (
                lambda *, task_name, **_: values.get(task_name, router._default_task_value(task_name))
            )
            decision = router.route("資料「運営資料」を探して")

        self.assertEqual(decision.target_model, "material_search")
        self.assertEqual(decision.recency_mode, "hard")
        self.assertEqual(decision.material_names, ["運営資料", "議事録"])
        self.assertTrue(decision.include_capabilities_info)
        self.assertTrue(decision.use_additional_memory)
        self.assertFalse(decision.idea_generation)
        self.assertFalse(decision.needs_additional_query)
        self.assertEqual(decision.additional_queries, [])

        called_tasks = [
            call.kwargs["task_name"] for call in mocked.call_args_list if "task_name" in call.kwargs
        ]
        self.assertIn("material_names", called_tasks)
        self.assertIn("recency_mode", called_tasks)
        self.assertIn("additional_queries", called_tasks)

    def test_rag_flow_runs_additional_queries_after_needs_flag(self) -> None:
        router = _build_router()
        values = {
            "target_model": "rag",
            "use_additional_memory": False,
            "include_capabilities_info": False,
            "idea_generation": True,
            "needs_additional_query": True,
            "recency_mode": "soft",
            "additional_queries": [],
        }

        with patch.object(router, "_run_task_with_retries") as mocked:
            mocked.side_effect = (
                lambda *, task_name, **_: values.get(task_name, router._default_task_value(task_name))
            )
            decision = router.route("最近の活動を踏まえた企画案を出して")

        self.assertEqual(decision.target_model, "rag")
        self.assertEqual(decision.recency_mode, "soft")
        self.assertTrue(decision.idea_generation)
        self.assertFalse(decision.needs_additional_query)
        self.assertEqual(decision.additional_queries, [])

        called_tasks = [
            call.kwargs["task_name"] for call in mocked.call_args_list if "task_name" in call.kwargs
        ]
        self.assertIn("recency_mode", called_tasks)
        self.assertIn("additional_queries", called_tasks)
        self.assertNotIn("material_names", called_tasks)

    def test_refusal_short_circuits_followup_tasks(self) -> None:
        router = _build_router()
        values = {
            "target_model": "refusal",
            "use_additional_memory": True,
        }

        with patch.object(router, "_run_task_with_retries") as mocked:
            mocked.side_effect = (
                lambda *, task_name, **_: values.get(task_name, router._default_task_value(task_name))
            )
            decision = router.route("契約の内容を教えて")

        self.assertEqual(decision.target_model, "refusal")
        self.assertEqual(decision.recency_mode, "off")
        self.assertTrue(decision.use_additional_memory)
        self.assertFalse(decision.include_capabilities_info)

        called_tasks = [
            call.kwargs["task_name"] for call in mocked.call_args_list if "task_name" in call.kwargs
        ]
        self.assertEqual(sorted(set(called_tasks)), ["target_model", "use_additional_memory"])

    def test_routing_disabled_returns_safe_default(self) -> None:
        router = _build_router()
        router._routing_enabled = False  # test setup only

        decision = router.route("今日の天気は？")

        self.assertEqual(decision.target_model, "rag")
        self.assertEqual(decision.recency_mode, "off")


if __name__ == "__main__":
    unittest.main()
