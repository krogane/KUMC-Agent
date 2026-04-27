from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.cli import _workflow_response_payload, main
from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.source import Source
from kumc_agent.domain.models.workflow import WorkResponse


class _FakeChatAnswer:
    def __init__(self, answers: list[Answer]) -> None:
        self._answers = list(answers)
        self.requests: list[object] = []

    def execute(self, request: object) -> Answer:
        self.requests.append(request)
        index = len(self.requests) - 1
        return self._answers[index]


class CliToolRagTests(unittest.TestCase):
    @staticmethod
    def _build_context(chat_answer: _FakeChatAnswer) -> object:
        return SimpleNamespace(
            config=SimpleNamespace(
                app=SimpleNamespace(log_level="INFO"),
                base_dir=ROOT,
            ),
            chat_answer=chat_answer,
        )

    def test_tool_rag_single_query_keeps_diagnostics_in_metadata(self) -> None:
        chat_answer = _FakeChatAnswer(
            [
                Answer(
                    text="single answer",
                    route="rag",
                    sources=[Source(id="s1", label="source-1", uri="https://example.com/1")],
                    metadata={
                        "routing_decision": {"target_model": "rag"},
                        "fast_mode": True,
                        "contexts": ["omit-me"],
                    },
                )
            ]
        )
        context = self._build_context(chat_answer)

        with (
            patch("kumc_agent.cli.build_runtime_context", return_value=context),
            patch("kumc_agent.cli.configure_logging"),
            patch.object(
                sys,
                "argv",
                [
                    "kumc-agent",
                    "tool",
                    "rag",
                    "--query",
                    "first query",
                    "--question-author",
                    "alice",
                    "--history-scope",
                    "global",
                ],
            ),
            patch("sys.stdout", new=io.StringIO()) as stdout,
        ):
            main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["answer"], "single answer")
        self.assertEqual(payload["route"], "rag")
        self.assertNotIn("query_count", payload)
        self.assertNotIn("results", payload)
        self.assertNotIn("routing_decision", payload)
        self.assertNotIn("fast_mode", payload)
        self.assertNotIn("contexts", payload["metadata"])
        self.assertEqual(payload["metadata"]["routing_decision"], {"target_model": "rag"})
        self.assertEqual(payload["metadata"]["fast_mode"], True)
        self.assertEqual(payload["sources"][0]["id"], "s1")

        self.assertEqual(len(chat_answer.requests), 1)
        request = chat_answer.requests[0]
        self.assertEqual(getattr(request, "query"), "first query")
        self.assertEqual(getattr(request, "question_author"), "alice")
        self.assertEqual(getattr(request, "history_scope"), "global")

    def test_tool_rag_multi_query_returns_results_array(self) -> None:
        chat_answer = _FakeChatAnswer(
            [
                Answer(
                    text="answer 1",
                    route="rag",
                    sources=[Source(id="s1", label="source-1", uri="https://example.com/1")],
                    metadata={"fast_mode": False, "contexts": ["omit-1"]},
                ),
                Answer(
                    text="answer 2",
                    route="no_rag",
                    sources=[Source(id="s2", label="source-2", uri="https://example.com/2")],
                    metadata={"fast_mode": True, "contexts": ["omit-2"]},
                ),
            ]
        )
        context = self._build_context(chat_answer)

        with (
            patch("kumc_agent.cli.build_runtime_context", return_value=context),
            patch("kumc_agent.cli.configure_logging"),
            patch.object(
                sys,
                "argv",
                [
                    "kumc-agent",
                    "tool",
                    "rag",
                    "--query",
                    "first query",
                    "--query",
                    "second query",
                ],
            ),
            patch("sys.stdout", new=io.StringIO()) as stdout,
        ):
            main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["query_count"], 2)
        self.assertEqual(len(payload["results"]), 2)

        first = payload["results"][0]
        self.assertEqual(first["query"], "first query")
        self.assertEqual(first["answer"], "answer 1")
        self.assertNotIn("fast_mode", first)
        self.assertNotIn("contexts", first["metadata"])
        self.assertEqual(first["metadata"]["fast_mode"], False)

        second = payload["results"][1]
        self.assertEqual(second["query"], "second query")
        self.assertEqual(second["answer"], "answer 2")
        self.assertNotIn("fast_mode", second)
        self.assertNotIn("contexts", second["metadata"])
        self.assertEqual(second["metadata"]["fast_mode"], True)

        queries = [getattr(request, "query") for request in chat_answer.requests]
        self.assertEqual(queries, ["first query", "second query"])

    def test_workflow_payload_keeps_event_diagnostics_in_metadata(self) -> None:
        payload = _workflow_response_payload(
            WorkResponse(
                text="events",
                metadata={
                    "routing_decision": {"selected_handler": "event_list"},
                    "query_filters": {"status": "planning"},
                    "contexts": ["omit-me"],
                    "secret": "api_key=abc",
                },
            )
        )

        self.assertNotIn("routing_decision", payload)
        self.assertNotIn("query_filters", payload)
        self.assertIn("metadata", payload)
        self.assertEqual(payload["metadata"]["routing_decision"], {"selected_handler": "event_list"})
        self.assertEqual(payload["metadata"]["query_filters"], {"status": "planning"})
        self.assertNotIn("contexts", payload["metadata"])
        self.assertNotIn("secret", payload["metadata"])


if __name__ == "__main__":
    unittest.main()
