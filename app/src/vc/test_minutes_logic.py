from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from config import AppConfig
from vc.google_docs_minutes import GoogleDocsMinutesClient
from vc.manager import (
    VoiceMeetingManager,
    _MeetingArchive,
    _parse_minutes_edit_response,
    _split_into_batches,
)


class _FakeVoiceChannel:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, text: str) -> None:
        self.messages.append(text)


class MinutesLogicTests(unittest.TestCase):
    def test_parse_minutes_edit_response_success(self) -> None:
        payload = (
            '{"summary":"要約","edits":[{"line":1,"op":"replace","text":"x"}],'
            '"revised_markdown":"# title"}'
        )
        result = _parse_minutes_edit_response(payload)
        self.assertEqual("要約", result.summary)
        self.assertEqual("# title", result.revised_markdown)
        self.assertEqual(1, len(result.edits))
        self.assertEqual("replace", result.edits[0]["op"])

    def test_parse_minutes_edit_response_invalid_json(self) -> None:
        with self.assertRaises(Exception):
            _parse_minutes_edit_response("not-json")

    def test_parse_minutes_edit_response_missing_fields(self) -> None:
        with self.assertRaises(ValueError):
            _parse_minutes_edit_response('{"edits": [], "revised_markdown": "# x"}')
        with self.assertRaises(ValueError):
            _parse_minutes_edit_response('{"summary": "x", "edits": []}')

    def test_minutes_candidate_priority(self) -> None:
        expected = ("20260220議事録", "20260220 議事録")
        first = GoogleDocsMinutesClient._minutes_doc_from_file(
            {
                "id": "a",
                "name": "20260220議事録",
                "mimeType": "application/vnd.google-apps.document",
                "modifiedTime": "2026-02-20T10:00:00.000Z",
            }
        )
        second = GoogleDocsMinutesClient._minutes_doc_from_file(
            {
                "id": "b",
                "name": "20260220議事録",
                "mimeType": "application/vnd.google-apps.document",
                "modifiedTime": "2026-02-20T11:00:00.000Z",
            }
        )
        third = GoogleDocsMinutesClient._minutes_doc_from_file(
            {
                "id": "c",
                "name": "20260220 議事録",
                "mimeType": "application/vnd.google-apps.document",
                "modifiedTime": "2026-02-20T12:00:00.000Z",
            }
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNotNone(third)

        rows = [first, second, third]  # type: ignore[list-item]
        ordered = sorted(
            rows,
            key=lambda item: GoogleDocsMinutesClient._candidate_sort_key(
                item=item,
                expected_names=expected,
            ),
        )
        self.assertEqual("b", ordered[0].doc_id)
        self.assertEqual("a", ordered[1].doc_id)
        self.assertEqual("c", ordered[2].doc_id)

    def test_split_into_batches(self) -> None:
        batches = _split_into_batches(list(range(23)), batch_size=10)
        self.assertEqual(3, len(batches))
        self.assertEqual([10, 10, 3], [len(batch) for batch in batches])

    def test_end_judge_symbols_removed(self) -> None:
        self.assertFalse(hasattr(VoiceMeetingManager, "_queue_pending_end_judge"))
        self.assertFalse(hasattr(VoiceMeetingManager, "_reset_pending_end_judge"))
        self.assertNotIn(
            "vc_end_judge_transcribe_interval_seconds",
            AppConfig.__dataclass_fields__,
        )


class MinutesFailureNotificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_notification_is_suppressed_until_recovered(self) -> None:
        manager = object.__new__(VoiceMeetingManager)
        fake_channel = _FakeVoiceChannel()
        fake_session = SimpleNamespace(voice_channel=fake_channel)
        manager._find_active_session_by_meeting_key = lambda _meeting_key: fake_session

        archive = _MeetingArchive(
            meeting_key="2026-02-20_01",
            guild_id=1,
            meeting_date="2026/02/20",
            meeting_label="2026/02/20 meeting",
            summary_chunk_path=Path("dummy.jsonl"),
        )

        await VoiceMeetingManager._notify_minutes_failure_once(
            manager,
            meeting_key="2026-02-20_01",
            archive=archive,
            text="minutes failed",
        )
        await VoiceMeetingManager._notify_minutes_failure_once(
            manager,
            meeting_key="2026-02-20_01",
            archive=archive,
            text="minutes failed",
        )
        self.assertEqual(1, len(fake_channel.messages))

        VoiceMeetingManager._clear_minutes_failure_notification(archive)
        await VoiceMeetingManager._notify_minutes_failure_once(
            manager,
            meeting_key="2026-02-20_01",
            archive=archive,
            text="minutes failed again",
        )
        self.assertEqual(2, len(fake_channel.messages))


if __name__ == "__main__":
    unittest.main()
