from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.source import Source
from kumc_agent.infra.openclaw.client import OpenClawFailure, OpenClawResponse, OpenClawTurnResult
from kumc_agent.usecases.chat.entry import ChatEntryRequest, ChatEntryUsecase


class _FakeChatUsecase:
    def __init__(self) -> None:
        self.requests = []

    def execute(self, request):  # noqa: ANN001
        self.requests.append(request)
        return Answer(text="fallback", route="rag", sources=[], metadata={})


class _OpenClawSuccessClient:
    enabled = True
    last_session_id = ""

    def run_turn(self, *, query: str, session_id: str, user_context):  # noqa: ANN001
        _ = query
        self.last_session_id = session_id
        _ = user_context
        return OpenClawResponse(
            ok=True,
            result=OpenClawTurnResult(
                text="openclaw answer",
                payload={
                    "route": "openclaw",
                    "sources": [{"id": "s1", "label": "source-1", "uri": "file://x"}],
                    "routing_decision": {"target_model": "rag"},
                    "fast_mode": False,
                    "rag_query": "新歓 2025 予算",
                    "rag_iterations": 2,
                },
            ),
        )


class _OpenClawFailureClient:
    enabled = True

    def run_turn(self, *, query: str, session_id: str, user_context):  # noqa: ANN001
        _ = query
        _ = session_id
        _ = user_context
        return OpenClawResponse(
            ok=False,
            failure=OpenClawFailure(reason="command_not_found"),
        )


class _OpenClawDisabledClient:
    enabled = False

    def run_turn(self, *, query: str, session_id: str, user_context):  # noqa: ANN001
        _ = query
        _ = session_id
        _ = user_context
        raise AssertionError("run_turn should not be called when disabled")


class _OpenClawFastmodeAliasClient:
    enabled = True

    def run_turn(self, *, query: str, session_id: str, user_context):  # noqa: ANN001
        _ = query
        _ = session_id
        _ = user_context
        return OpenClawResponse(
            ok=True,
            result=OpenClawTurnResult(
                text="openclaw answer",
                payload={
                    "sources": [{"id": "s1", "label": "source-1", "uri": "file://x"}],
                    "fastmode": False,
                    "metadata": {},
                },
            ),
        )


class _OpenClawWithEmbeddedSourcesClient:
    enabled = True

    def run_turn(self, *, query: str, session_id: str, user_context):  # noqa: ANN001
        _ = query
        _ = session_id
        _ = user_context
        return OpenClawResponse(
            ok=True,
            result=OpenClawTurnResult(
                text="openclaw answer\n\n主な情報源:\n- source-1",
                payload={
                    "sources": [{"id": "s1", "label": "source-1", "uri": "file://x"}],
                },
            ),
        )


class ChatEntryUsecaseTests(unittest.TestCase):
    def test_openclaw_success_is_returned_without_local_fallback(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = ChatEntryUsecase(
            chat_usecase=fake_chat,
            openclaw_client=_OpenClawSuccessClient(),  # type: ignore[arg-type]
        )

        answer = usecase.execute(
            ChatEntryRequest(query="質問", history_scope="guild:1", question_author="alice")
        )

        self.assertIn("openclaw answer", answer.text)
        self.assertIn("主な情報源:", answer.text)
        self.assertIn("- file://x", answer.text)
        self.assertEqual(answer.route, "openclaw")
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0], Source(id="s1", label="source-1", uri="file://x"))
        self.assertNotIn("routing_decision", answer.metadata)
        self.assertEqual(answer.metadata.get("rag_query"), "新歓 2025 予算")
        self.assertEqual(answer.metadata.get("rag_iterations"), 2)
        payload = answer.metadata.get("openclaw_payload")
        self.assertIsInstance(payload, dict)
        if isinstance(payload, dict):
            self.assertNotIn("route", payload)
            self.assertNotIn("routing_decision", payload)
        self.assertEqual(fake_chat.requests, [])

    def test_openclaw_failure_falls_back_and_disables_local_history(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = ChatEntryUsecase(
            chat_usecase=fake_chat,
            openclaw_client=_OpenClawFailureClient(),  # type: ignore[arg-type]
        )

        answer = usecase.execute(ChatEntryRequest(query="質問"))

        self.assertEqual(answer.text, "fallback")
        self.assertEqual(len(fake_chat.requests), 1)
        request = fake_chat.requests[0]
        self.assertTrue(request.disable_history)
        self.assertTrue(answer.metadata.get("openclaw_fallback"))

    def test_openclaw_uses_valid_default_session_id_when_history_scope_missing(self) -> None:
        fake_chat = _FakeChatUsecase()
        openclaw = _OpenClawSuccessClient()
        usecase = ChatEntryUsecase(
            chat_usecase=fake_chat,
            openclaw_client=openclaw,  # type: ignore[arg-type]
        )

        _ = usecase.execute(ChatEntryRequest(query="質問", history_scope=None))

        self.assertEqual(openclaw.last_session_id, "default")
        self.assertEqual(fake_chat.requests, [])

    def test_openclaw_disabled_uses_local_history_behavior(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = ChatEntryUsecase(
            chat_usecase=fake_chat,
            openclaw_client=_OpenClawDisabledClient(),  # type: ignore[arg-type]
        )

        _ = usecase.execute(ChatEntryRequest(query="質問"))
        self.assertEqual(len(fake_chat.requests), 1)
        request = fake_chat.requests[0]
        self.assertFalse(request.disable_history)

    def test_openclaw_fastmode_alias_is_normalized_to_fast_mode_metadata(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = ChatEntryUsecase(
            chat_usecase=fake_chat,
            openclaw_client=_OpenClawFastmodeAliasClient(),  # type: ignore[arg-type]
        )

        answer = usecase.execute(ChatEntryRequest(query="質問"))

        self.assertIn("openclaw answer", answer.text)
        self.assertIn("主な情報源:", answer.text)
        self.assertEqual(answer.metadata.get("fast_mode"), False)
        self.assertEqual(len(answer.sources), 1)

    def test_openclaw_does_not_append_sources_when_disabled(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = ChatEntryUsecase(
            chat_usecase=fake_chat,
            openclaw_client=_OpenClawSuccessClient(),  # type: ignore[arg-type]
        )

        answer = usecase.execute(ChatEntryRequest(query="質問", append_sources_to_response=False))

        self.assertEqual(answer.text, "openclaw answer")
        self.assertEqual(len(answer.sources), 1)

    def test_openclaw_does_not_duplicate_embedded_sources_section(self) -> None:
        fake_chat = _FakeChatUsecase()
        usecase = ChatEntryUsecase(
            chat_usecase=fake_chat,
            openclaw_client=_OpenClawWithEmbeddedSourcesClient(),  # type: ignore[arg-type]
        )

        answer = usecase.execute(ChatEntryRequest(query="質問"))

        self.assertEqual(answer.text.count("主な情報源:"), 1)


if __name__ == "__main__":
    unittest.main()
