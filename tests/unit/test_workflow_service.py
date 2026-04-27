from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.workflow import Task, WorkRequest
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


class WorkflowServiceTests(unittest.TestCase):
    def _service(self, root: Path, *, llm: object | None = None) -> WorkflowService:
        return WorkflowService(
            repository=FileWorkflowRepository(root_dir=root / "workflow"),
            llm=llm,
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

    def test_event_brief_includes_related_open_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            event = service.run(
                WorkRequest(
                    work_type="event_add",
                    instruction="イベント: 新歓会 日時: 2026-05-05 場所: 部室",
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
