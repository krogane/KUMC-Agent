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


class Wave4WorkflowTests(unittest.TestCase):
    def _service(self, root: Path) -> WorkflowService:
        return WorkflowService(repository=FileWorkflowRepository(root_dir=root / "workflow"))

    def test_task_extract_creates_candidate_not_task_until_approved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))

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
            service = self._service(Path(tmp))

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
            listed = service.run(WorkRequest(work_type="task_list"))
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

    def test_event_brief_includes_related_open_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            event = service.run(
                WorkRequest(
                    work_type="event_add",
                    instruction="イベント: 新歓会 日時: 2026-05-05 場所: 部室",
                )
            ).events[0]
            service.repository.save_task(
                Task(
                    id="task-1",
                    title="新歓会の受付表を作成",
                    related_event_id=event.id,
                    status="todo",
                )
            )

            brief = service.run(
                WorkRequest(
                    work_type="event_brief",
                    target=event.id,
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
            listed = service.run(WorkRequest(work_type="schedule_list"))

            self.assertEqual(len(added.schedules), 1)
            self.assertIn("定例会", listed.detail_markdown)

    def test_migration_contains_wave4_tables(self) -> None:
        sql = (
            ROOT / "infrastructure" / "migrations" / "004_wave4_workflow.sql"
        ).read_text(encoding="utf-8")
        for table in (
            "events",
            "meetings",
            "task_candidates",
            "tasks",
            "schedule_events",
            "approval_records",
        ):
            self.assertIn(f"create table if not exists {table}", sql)


if __name__ == "__main__":
    unittest.main()
