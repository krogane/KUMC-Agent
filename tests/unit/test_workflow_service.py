from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.workflow import Event, Task, WorkRequest
from kumc_agent.features.event_management import EventNotificationDelivery
from kumc_agent.features.task_management import TaskNotificationDelivery
from kumc_agent.features.workflow import WorkflowService
from kumc_agent.infra.workflow import FileWorkflowRepository


class FakeTaskLLM:
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        return """
        {
          "tasks": [
            {
              "title": "新歓資料を作成",
              "description": "TODO: 新歓資料を作成 担当: alice 期限: 2026-05-01",
              "assignee_user_id": "alice",
              "due_at": "2026-05-01T00:00:00+00:00",
              "related_event_id": null,
              "priority": "high",
              "confidence": "high",
              "evidence": ["input"]
            }
          ]
        }
        """


class FakeEventLLM:
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        payload = json.loads(user_prompt)
        text = str(payload.get("text") or "")
        expected = str(payload.get("expected_operation") or "")
        existing_events = list(payload.get("existing_events") or [])
        target = existing_events[0] if existing_events else {}
        if expected == "update" and target:
            after = dict(target)
            after["place"] = "第2会議室" if "第2会議室" in text else target.get("place")
            return json.dumps(
                {
                    "new_events": [],
                    "event_changes": [
                        {
                            "event_id": target["id"],
                            "operation": "update",
                            "after": after,
                            "reason": text,
                            "confidence": "high",
                            "evidence": ["input"],
                        }
                    ],
                    "ignored_items": [],
                    "degraded": False,
                },
                ensure_ascii=False,
            )
        if expected == "delete" and target:
            return json.dumps(
                {
                    "new_events": [],
                    "event_changes": [
                        {
                            "event_id": target["id"],
                            "operation": "delete",
                            "reason": text,
                            "confidence": "high",
                            "evidence": ["input"],
                        }
                    ],
                    "ignored_items": [],
                    "degraded": False,
                },
                ensure_ascii=False,
            )
        starts_at = "2026-05-05T14:00:00+00:00"
        date_match = re.search(
            r"(\d{4}-\d{1,2}-\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?",
            text,
        )
        if date_match:
            date_part, hour, minute = date_match.groups()
            starts_at = f"{date_part}T{hour or '00'}:{minute or '00'}:00+00:00"
        return json.dumps(
            {
                "new_events": [
                    {
                        "title": "新歓会",
                        "summary": "新入生歓迎イベント",
                        "starts_at": starts_at,
                        "ends_at": None,
                        "place": "第2会議室" if "第2会議室" in text else "部室",
                        "related_source_ids": ["discord:1"],
                        "related_task_query": "新歓",
                        "confidence": "high",
                        "evidence": ["input"],
                    }
                ],
                "event_changes": [],
                "ignored_items": [],
                "degraded": False,
            },
            ensure_ascii=False,
        )


class FakeTaskNotificationSender:
    def __init__(self) -> None:
        self.messages = []

    def send(self, message):
        self.messages.append(message)
        return TaskNotificationDelivery(
            status="sent",
            channel="fake",
            message_id=f"msg-{len(self.messages)}",
        )


class FakeEventNotificationSender:
    def __init__(self) -> None:
        self.messages = []

    def send(self, message):
        self.messages.append(message)
        return EventNotificationDelivery(
            status="sent",
            channel="fake",
            message_id=f"event-msg-{len(self.messages)}",
        )


class WorkflowServiceTests(unittest.TestCase):
    def _service(
        self,
        root: Path,
        *,
        llm: object | None = None,
        task_notification_sender: object | None = None,
        event_notification_sender: object | None = None,
    ) -> WorkflowService:
        return WorkflowService(
            repository=FileWorkflowRepository(root_dir=root / "workflow"),
            llm=llm,
            task_notification_sender=task_notification_sender,
            task_notification_channel_id="tasks",
            event_notification_sender=event_notification_sender,
            event_notification_channel_id="events",
            event_notification_before_days=1,
        )

    def test_task_extract_creates_candidate_not_task_until_approved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp), llm=FakeTaskLLM())

            response = service.run(
                WorkRequest(
                    work_type="task_extract",
                    instruction="TODO: 新歓資料を作成 担当: @alice 期限: 2026-05-01",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )

            self.assertEqual(len(response.task_candidates), 1)
            self.assertEqual(response.task_candidates[0].status, "proposed")
            self.assertEqual(service.repository.list_tasks(), [])

            approved = service.approval(
                action="approve",
                target_type="task",
                target_id=response.task_candidates[0].id,
                access=AccessContext(user_id="admin", is_admin=True),
            )

            self.assertEqual(len(approved.tasks), 1)
            self.assertEqual(approved.tasks[0].title, response.task_candidates[0].title)
            self.assertEqual(
                service.repository.get_task_candidate(response.task_candidates[0].id).status,
                "merged",
            )

    def test_meeting_minutes_registers_task_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp), llm=FakeTaskLLM())

            response = service.run(
                WorkRequest(
                    work_type="meeting_minutes_draft",
                    instruction=(
                        "決定: NF 企画を進める\n"
                        "未決: 予算確認\n"
                        "タスク: 告知文を作成 担当: bob 期限: 2026-06-10"
                    ),
                    access=AccessContext(user_id="user-1"),
                )
            )

            self.assertEqual(len(response.task_candidates), 1)
            self.assertIn("決定事項", response.detail_markdown)
            self.assertIn("未決事項", response.detail_markdown)

    def test_task_add_list_done_and_approval_show_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            added = service.run(
                WorkRequest(
                    work_type="task_add",
                    instruction="タスク: 会場予約 担当: alice 期限: 2026-05-01",
                    access=AccessContext(user_id="organizer", is_admin=True),
                )
            )
            candidate = added.task_candidates[0]
            shown = service.approval(
                action="show",
                target_type="task",
                target_id=candidate.id,
                access=AccessContext(user_id="organizer", is_admin=True),
            )
            edited = service.approval(
                action="edit",
                target_type="task",
                target_id=candidate.id,
                comment="title: 会場予約を確定 担当: bob 2026-05-02",
                access=AccessContext(user_id="organizer", is_admin=True),
            )
            approved = service.approval(
                action="approve",
                target_type="task",
                target_id=candidate.id,
                access=AccessContext(user_id="organizer", is_admin=True),
            )
            listed = service.run(
                WorkRequest(
                    work_type="task_list",
                    access=AccessContext(user_id="organizer", is_admin=True),
                )
            )
            done = service.run(
                WorkRequest(
                    work_type="task_done",
                    target=approved.tasks[0].id,
                    access=AccessContext(user_id="bob"),
                )
            )

            self.assertIn("会場予約", shown.detail_markdown)
            self.assertEqual(edited.task_candidates[0].proposed_assignee_user_id, "bob")
            self.assertEqual(len(listed.tasks), 1)
            self.assertEqual(done.tasks[0].status, "done")

    def test_task_extract_degrades_without_llm_and_does_not_create_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))

            response = service.run(
                WorkRequest(
                    work_type="task_extract",
                    instruction="TODO: 新歓資料を作成 担当: @alice 期限: 2026-05-01",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )

            self.assertEqual(response.task_candidates, tuple())
            self.assertTrue(response.metadata["extraction"]["degraded"])
            self.assertEqual(service.repository.list_tasks(), [])

    def test_task_update_and_delete_require_approval_before_changing_master_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            task = service.repository.save_task(
                Task(
                    id="task-1",
                    title="会場予約",
                    assignee_user_id="alice",
                    status="todo",
                )
            )

            update = service.run(
                WorkRequest(
                    work_type="task_update",
                    target=task.id,
                    instruction="status: doing priority: high",
                    access=AccessContext(user_id="alice"),
                )
            )
            self.assertEqual(len(update.task_change_candidates), 1)
            self.assertEqual(service.repository.get_task(task.id).status, "todo")

            approved = service.approval(
                action="approve",
                target_type="task",
                target_id=update.task_change_candidates[0].id,
                access=AccessContext(user_id="admin", is_admin=True),
            )
            self.assertEqual(approved.tasks[0].status, "doing")

            delete = service.run(
                WorkRequest(
                    work_type="task_delete",
                    target=task.id,
                    instruction="不要になった",
                    access=AccessContext(user_id="alice"),
                )
            )
            self.assertEqual(service.repository.get_task(task.id).status, "doing")
            approved_delete = service.approval(
                action="approve",
                target_type="task",
                target_id=delete.task_change_candidates[0].id,
                access=AccessContext(user_id="admin", is_admin=True),
            )
            self.assertEqual(approved_delete.tasks[0].status, "deleted")

    def test_task_access_denial_does_not_leak_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            service.repository.save_task(Task(id="task-1", title="秘匿タスク"))

            response = service.run(
                WorkRequest(
                    work_type="task_list",
                    access=AccessContext(user_id="outsider"),
                )
            )

            self.assertEqual(response.tasks, tuple())
            self.assertNotIn("1", response.text)
            self.assertFalse(response.metadata["authorized"])

    def test_task_notify_due_marks_notifications_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            approved = service.run(
                WorkRequest(
                    work_type="task_add",
                    instruction="タスク: 会場予約 担当: alice 期限: 2026-05-01",
                    access=AccessContext(user_id="organizer", is_admin=True),
                )
            )
            task = service.approval(
                action="approve",
                target_type="task",
                target_id=approved.task_candidates[0].id,
                access=AccessContext(user_id="admin", is_admin=True),
            ).tasks[0]

            notified = service.run(
                WorkRequest(
                    work_type="task_notify_due",
                    instruction="days: 999",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )
            notified_again = service.run(
                WorkRequest(
                    work_type="task_notify_due",
                    instruction="days: 999",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )

            self.assertEqual(len(notified.tasks), 1)
            self.assertEqual(len(notified_again.tasks), 0)
            self.assertIn("notifications", service.repository.get_task(task.id).metadata)

    def test_task_notify_due_sends_discord_delivery_and_done_records_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sender = FakeTaskNotificationSender()
            service = self._service(Path(tmp), task_notification_sender=sender)
            task = service.repository.save_task(
                Task(
                    id="task-1",
                    title="期限確認",
                    due_at=datetime(2026, 5, 1, tzinfo=UTC),
                    status="todo",
                )
            )

            notified = service.run(
                WorkRequest(
                    work_type="task_notify_due",
                    instruction="days: 999",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )
            done = service.run(
                WorkRequest(
                    work_type="task_done",
                    target=task.id,
                    instruction="discord_component:done",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )

            self.assertEqual(len(sender.messages), 2)  # unassigned + due_soon
            self.assertEqual(notified.metadata["deliveries"][0]["delivery"]["status"], "sent")
            self.assertEqual(done.tasks[0].status, "done")
            self.assertEqual(done.approvals[0].action, "done")

    def test_task_batch_approval_records_period_and_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sender = FakeTaskNotificationSender()
            service = self._service(Path(tmp), llm=FakeTaskLLM(), task_notification_sender=sender)
            extracted = service.run(
                WorkRequest(
                    work_type="task_extract",
                    instruction="TODO: 新歓資料を作成 担当: @alice 期限: 2026-05-01",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )

            batch = service.run(
                WorkRequest(
                    work_type="task_batch_approval",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            ).task_approval_batches[0]

            self.assertEqual(len(extracted.task_candidates), 1)
            self.assertIsNotNone(batch.period_start)
            self.assertIsNotNone(batch.period_end)
            self.assertEqual(batch.notification_message_id, "msg-1")
            self.assertEqual(sender.messages[0].metadata["components"][0]["type"], 1)

    def test_event_brief_includes_related_open_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp), llm=FakeEventLLM())
            event = service.run(
                WorkRequest(
                    work_type="event_add",
                    instruction="イベント: 新歓会 日時: 2026-05-05 場所: 部室",
                    access=AccessContext(user_id="organizer", is_admin=True),
                )
            ).event_candidates[0]
            self.assertEqual(event.status, "proposed")
            approved = service.approval(
                action="approve",
                target_type="event",
                target_id=event.id,
                access=AccessContext(user_id="organizer", is_admin=True),
            )
            stored_event = approved.events[0]
            self.assertEqual(
                service.repository.get_event_candidate(event.id).status,
                "merged",
            )
            service.repository.save_task(
                Task(
                    id="task-1",
                    title="新歓会の受付表を作成",
                    related_event_id=stored_event.id,
                    status="todo",
                )
            )

            brief = service.run(
                WorkRequest(
                    work_type="event_brief",
                    target=stored_event.id,
                    access=AccessContext(is_admin=True),
                )
            )

            self.assertIn("新歓会", brief.detail_markdown)
            self.assertIn("受付表", brief.detail_markdown)

    def test_event_add_requires_admin_and_required_fields_before_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))

            denied = service.run(
                WorkRequest(
                    work_type="event_add",
                    instruction="イベント: 新歓会 日時: 2026-05-05",
                    access=AccessContext(user_id="outsider"),
                )
            )
            missing = service.run(
                WorkRequest(
                    work_type="event_add",
                    instruction="新歓会をやります",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )

            self.assertFalse(denied.metadata["authorized"])
            self.assertEqual(denied.event_candidates, tuple())
            self.assertEqual(missing.event_candidates, tuple())
            self.assertIn("starts_at", missing.metadata["missing_fields"])
            self.assertEqual(service.repository.list_event_candidates(), [])

    def test_event_extract_uses_llm_and_degrades_without_candidate_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp), llm=FakeEventLLM())
            extracted = service.run(
                WorkRequest(
                    work_type="event_extract",
                    instruction="5/5 14:00 新歓会を部室で開催します。",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )
            self.assertEqual(len(extracted.event_candidates), 1)
            self.assertEqual(service.repository.list_events(), [])

        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            degraded = service.run(
                WorkRequest(
                    work_type="event_extract",
                    instruction="5/5 14:00 新歓会を部室で開催します。",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )
            self.assertEqual(degraded.event_candidates, tuple())
            self.assertTrue(degraded.metadata["extraction"]["degraded"])

    def test_event_update_without_llm_does_not_create_change_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            event = service.repository.save_event(
                Event(id="event-1", title="新歓会", starts_at=datetime(2026, 5, 5, tzinfo=UTC))
            )

            response = service.run(
                WorkRequest(
                    work_type="event_update",
                    target=event.id,
                    instruction="場所: 第2会議室",
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )

            self.assertEqual(response.event_change_candidates, tuple())
            self.assertFalse(response.metadata["candidate_created"])
            self.assertEqual(service.repository.list_event_change_candidates(), [])

    def test_event_list_filters_duplicates_update_delete_and_notify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp), llm=FakeEventLLM())
            access = AccessContext(user_id="admin", is_admin=True)
            event_start = datetime.now(UTC) + timedelta(days=1)
            starts_text = event_start.strftime("%Y-%m-%d %H:%M")
            first = service.run(
                WorkRequest(
                    work_type="event_add",
                    instruction=f"イベント: 新歓会 日時: {starts_text} 場所: 部室",
                    access=access,
                )
            ).event_candidates[0]
            second = service.run(
                WorkRequest(
                    work_type="event_add",
                    instruction=f"イベント: 新歓会 日時: {starts_text} 場所: 部室",
                    access=access,
                )
            ).event_candidates[0]
            self.assertIn("duplicate_candidates", second.metadata)
            event = service.approval(
                action="approve",
                target_type="event",
                target_id=first.id,
                access=access,
            ).events[0]
            with self.assertRaises(ValueError):
                service.approval(
                    action="approve",
                    target_type="event",
                    target_id=first.id,
                    access=access,
                )

            listed = service.run(
                WorkRequest(
                    work_type="event_list",
                    instruction=(
                        "状態: planning 場所: 部室 "
                        f"{(event_start - timedelta(days=1)).strftime('%Y-%m-%d')}から"
                        f"{(event_start + timedelta(days=1)).strftime('%Y-%m-%d')}まで"
                    ),
                    access=access,
                )
            )
            self.assertEqual(len(listed.events), 1)
            self.assertIn("query_filters", listed.metadata)

            update = service.run(
                WorkRequest(
                    work_type="event_update",
                    target=event.id,
                    instruction="場所: 第2会議室",
                    access=access,
                )
            )
            self.assertEqual(service.repository.get_event(event.id).place, "部室")
            approved_update = service.approval(
                action="approve",
                target_type="event",
                target_id=update.event_change_candidates[0].id,
                access=access,
            )
            self.assertEqual(approved_update.events[0].place, "第2会議室")

            notified = service.run(
                WorkRequest(
                    work_type="event_notify",
                    instruction="days: 1",
                    access=access,
                )
            )
            notified_again = service.run(
                WorkRequest(
                    work_type="event_notify",
                    instruction="days: 1",
                    access=access,
                )
            )
            self.assertEqual(len(notified.events), 1)
            self.assertEqual(len(notified_again.events), 0)

            delete = service.run(
                WorkRequest(
                    work_type="event_delete",
                    target=event.id,
                    instruction="中止になった",
                    access=access,
                )
            )
            self.assertNotEqual(service.repository.get_event(event.id).status, "canceled")
            approved_delete = service.approval(
                action="approve",
                target_type="event",
                target_id=delete.event_change_candidates[0].id,
                access=access,
            )
            self.assertEqual(approved_delete.events[0].status, "canceled")

    def test_event_batch_and_completion_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sender = FakeEventNotificationSender()
            service = self._service(Path(tmp), llm=FakeEventLLM(), event_notification_sender=sender)
            access = AccessContext(user_id="admin", is_admin=True)
            extracted = service.run(
                WorkRequest(
                    work_type="event_extract",
                    instruction="新歓会を開催します。",
                    access=access,
                )
            )
            batch = service.run(
                WorkRequest(
                    work_type="event_batch_approval",
                    instruction="channel: events",
                    access=access,
                )
            )
            self.assertEqual(len(batch.event_approval_batches), 1)
            self.assertEqual(batch.event_approval_batches[0].notification_message_id, "event-msg-1")
            self.assertEqual(sender.messages[0].metadata["buttons"][0]["label"], "Approve")
            event = service.approval(
                action="approve",
                target_type="event",
                target_id=extracted.event_candidates[0].id,
                access=access,
            ).events[0]
            completed = service.run(
                WorkRequest(
                    work_type="event_complete",
                    target=event.id,
                    instruction="完了確認済み",
                    access=access,
                )
            )
            self.assertEqual(completed.events[0].status, "done")
            self.assertEqual(len(completed.approvals), 1)

    def test_schedule_add_and_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            added = service.run(
                WorkRequest(
                    work_type="schedule_add",
                    instruction="予定: 定例会 2026-05-12 19:00 場所: Discord",
                )
            )
            self.assertEqual(len(added.schedule_candidates), 1)
            self.assertEqual(added.schedule_candidates[0].status, "proposed")
            approved = service.approval(
                action="approve",
                target_type="schedule",
                target_id=added.schedule_candidates[0].id,
                access=AccessContext(user_id="organizer", is_admin=True),
            )
            listed = service.run(WorkRequest(work_type="schedule_list"))

            self.assertEqual(len(approved.schedules), 1)
            self.assertIn("定例会", listed.detail_markdown)

if __name__ == "__main__":
    unittest.main()
