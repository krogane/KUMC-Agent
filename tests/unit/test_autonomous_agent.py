from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.autonomous_agent import AutonomousAgentRequest
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.workflow import Event, Task
from kumc_agent.features.autonomous_agent.idempotency import (
    AutonomousIdempotencyInput,
    build_autonomous_idempotency_key,
)
from kumc_agent.features.autonomous_agent.integrated_input import AutonomousIntegratedInputAdapter
from kumc_agent.features.autonomous_agent.planner import AutonomousPlanner, PlannerConfig
from kumc_agent.features.autonomous_agent.service import (
    AutonomousAgentService,
    AutonomousAgentServiceConfig,
)
from kumc_agent.features.autonomous_agent.snapshot import (
    AutonomousSnapshotCollector,
    SnapshotCollectorConfig,
)
from kumc_agent.features.autonomous_agent.verifier import AutonomousVerifier, VerifierConfig
from kumc_agent.features.workflow import WorkflowService
from kumc_agent.infra.agentic import FileAgentTraceRepository
from kumc_agent.infra.automation import FileAutomationRepository
from kumc_agent.infra.workflow import FileWorkflowRepository


class AutonomousAgentTests(unittest.TestCase):
    def _service(self, root: Path, *, workflow_repository: FileWorkflowRepository) -> AutonomousAgentService:
        automation_repository = FileAutomationRepository(root_dir=root / "automation")
        trace_repository = FileAgentTraceRepository(root_dir=root / "agentic")
        workflow = WorkflowService(repository=workflow_repository)
        return AutonomousAgentService(
            config=AutonomousAgentServiceConfig(
                enabled=True,
                timezone="Asia/Tokyo",
                scopes=("tasks", "events"),
                notification_channel_id="channel-1",
                dry_run=True,
                lookahead_days={"tasks": 2, "events": 7},
            ),
            trace_repository=trace_repository,
            automation_repository=automation_repository,
            snapshot_collector=AutonomousSnapshotCollector(
                workflow_repository=workflow_repository,
                automation_repository=automation_repository,
                agent_trace_repository=trace_repository,
                config=SnapshotCollectorConfig(task_lookahead_days=2, event_lookahead_days=7),
            ),
            planner=AutonomousPlanner(config=PlannerConfig(notification_channel_id="channel-1")),
            adapter=AutonomousIntegratedInputAdapter(workflow_service=workflow),
            verifier=AutonomousVerifier(config=VerifierConfig(notification_channel_id="channel-1")),
        )

    def test_idempotency_key_is_stable_and_scope_sensitive(self) -> None:
        now = datetime(2026, 4, 28, 0, 30, tzinfo=UTC)

        first = build_autonomous_idempotency_key(
            AutonomousIdempotencyInput(
                slot="08:00",
                scopes=("tasks", "events"),
                timezone="Asia/Tokyo",
                channel_id="c1",
                lookahead_days={"tasks": 2, "events": 7},
                now=now,
            )
        )
        second = build_autonomous_idempotency_key(
            AutonomousIdempotencyInput(
                slot="08:00",
                scopes=("events", "tasks"),
                timezone="Asia/Tokyo",
                channel_id="c1",
                lookahead_days={"tasks": 2, "events": 7},
                now=now,
            )
        )
        changed = build_autonomous_idempotency_key(
            AutonomousIdempotencyInput(
                slot="08:00",
                scopes=("events",),
                timezone="Asia/Tokyo",
                channel_id="c1",
                lookahead_days={"tasks": 2, "events": 7},
                now=now,
            )
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertTrue(first.startswith("autonomous-agent:2026-04-28:08:00:"))

    def test_due_task_creates_notification_and_duplicate_skips_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow_repository = FileWorkflowRepository(root_dir=root / "workflow")
            service = self._service(root, workflow_repository=workflow_repository)
            task = workflow_repository.save_task(
                Task(
                    id="task-1",
                    title="会場予約",
                    due_at=datetime.now(UTC) + timedelta(days=1),
                    status="todo",
                )
            )

            request = AutonomousAgentRequest(
                trigger="manual",
                slot="morning",
                scopes=("tasks",),
                dry_run=True,
                access=AccessContext(user_id="system"),
            )
            first = service.run(request)
            second = service.run(request)

            self.assertEqual(first.status, "succeeded")
            self.assertEqual(first.notification_proposals[0].target_refs, (f"task:{task.id}",))
            self.assertEqual(second.status, "duplicate")
            self.assertEqual(len(service.trace_repository.list_steps(first.run.id)), 2)
            payload = first.to_payload()
            self.assertNotIn("idempotency_key", payload)
            self.assertIn("idempotency_key", payload["metadata"])

    def test_event_without_tasks_dry_run_skips_candidate_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow_repository = FileWorkflowRepository(root_dir=root / "workflow")
            service = self._service(root, workflow_repository=workflow_repository)
            workflow_repository.save_event(
                Event(
                    id="event-1",
                    title="新歓会",
                    starts_at=datetime.now(UTC) + timedelta(days=3),
                    place="部室",
                    status="planning",
                )
            )

            response = service.run(
                AutonomousAgentRequest(
                    trigger="manual",
                    slot="event-check",
                    scopes=("events",),
                    dry_run=True,
                    access=AccessContext(user_id="system"),
                )
            )

            self.assertEqual(response.status, "noop")
            self.assertIn("dry_run_skipped_candidate_creation", response.warnings)
            self.assertEqual(workflow_repository.list_task_candidates(), [])
            steps = service.trace_repository.list_steps(response.run.id)
            self.assertEqual([step.state for step in steps], ["PLAN", "TOOL", "VERIFY"])


if __name__ == "__main__":
    unittest.main()
