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

from kumc_agent.apps.automation import _autonomous_agent_rules_from_config
from kumc_agent.domain.models.autonomous_agent import (
    AutonomousAgentRequest,
    AutonomousAgentSnapshot,
    AutonomousCheck,
    AutonomousPlan,
    AutonomousToolResult,
)
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.workflow import Event, Task
from kumc_agent.features.autonomous_agent.llm import AutonomousLLMConfig
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


class FakeLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)

    def generate(self, **_: object) -> str:
        return self.responses.pop(0)


class FakeIngestionRepository:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def load_active_chunks(self, *, source_kinds: tuple[str, ...] = tuple()) -> list[Chunk]:
        return list(self.chunks)


class AutonomousConfigStub:
    enabled = True
    schedule_times = ["08:00", "13:30"]
    timezone = "Asia/Tokyo"
    scopes = ["tasks", "events"]
    dry_run = True


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

    def test_request_dry_run_none_uses_config_and_allows_candidate_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow_repository = FileWorkflowRepository(root_dir=root / "workflow")
            service = self._service(root, workflow_repository=workflow_repository)
            service = AutonomousAgentService(
                config=AutonomousAgentServiceConfig(
                    enabled=True,
                    timezone="Asia/Tokyo",
                    scopes=("events",),
                    notification_channel_id="channel-1",
                    dry_run=False,
                    lookahead_days={"tasks": 2, "events": 7},
                ),
                trace_repository=service.trace_repository,
                automation_repository=service.automation_repository,
                snapshot_collector=service.snapshot_collector,
                planner=service.planner,
                adapter=service.adapter,
                verifier=service.verifier,
            )
            workflow_repository.save_event(
                Event(
                    id="event-candidate",
                    title="合宿",
                    starts_at=datetime.now(UTC) + timedelta(days=3),
                    place="研修施設",
                    status="planning",
                )
            )

            response = service.run(
                AutonomousAgentRequest(
                    trigger="manual",
                    slot="event-candidate",
                    scopes=("events",),
                    access=AccessContext(user_id="system"),
                )
            )

            self.assertNotEqual(workflow_repository.list_task_candidates(), [])
            self.assertIn("candidate_citations_missing", response.warnings)

    def test_rag_delta_collector_uses_recent_ingestion_chunks(self) -> None:
        now = datetime(2026, 4, 28, 12, 0, tzinfo=UTC)
        collector = AutonomousSnapshotCollector(
            ingestion_repository=FakeIngestionRepository(
                [
                    Chunk(
                        id="chunk-1",
                        document_id="doc-1",
                        text="新しいイベント準備の資料です。",
                        index=0,
                        metadata={
                            "source_item_id": "source-1",
                            "source_kind": "drive",
                            "external_id": "drive-1",
                            "source_title": "準備資料",
                            "updated_at": now.isoformat(),
                        },
                    )
                ]
            ),
            config=SnapshotCollectorConfig(rag_delta_lookback_hours=24),
        )

        snapshot = collector.collect(scopes=("rag_delta",), now=now)

        self.assertEqual(len(snapshot.rag_delta), 1)
        self.assertEqual(snapshot.rag_delta[0].id, "source-1")
        self.assertEqual(snapshot.rag_delta[0].citations[0].chunk_id, "chunk-1")

    def test_llm_planner_and_verifier_use_deterministic_guard(self) -> None:
        planner = AutonomousPlanner(
            config=PlannerConfig(notification_channel_id="channel-1"),
            llm=FakeLLM(
                """
                {
                  "checks": [
                    {
                      "kind": "task_due_soon",
                      "target_ref": "task:llm",
                      "reason": "LLM check",
                      "risk": "low",
                      "side_effect_boundary": "candidate_only"
                    }
                  ],
                  "required_queries": [],
                  "target_refs": ["task:llm"],
                  "success_criteria": ["notification_or_approval_proposals_created"]
                }
                """
            ),
            llm_config=AutonomousLLMConfig(enabled=True, prompt_name="missing"),
        )
        plan = planner.plan(AutonomousAgentSnapshot())
        verifier = AutonomousVerifier(
            config=VerifierConfig(notification_channel_id="channel-1"),
            llm=FakeLLM('{"decision":"notify","satisfied":["llm_verified"],"metadata":{"reason":"ok"}}'),
            llm_config=AutonomousLLMConfig(enabled=True, prompt_name="missing"),
        )

        decision = verifier.verify(plan=plan, tool_results=tuple())

        self.assertEqual(plan.metadata["planner"], "llm_with_deterministic_guard")
        self.assertEqual(decision.decision, "notify")
        self.assertEqual(decision.notification_proposals[0].target_refs, ("task:llm",))
        self.assertEqual(decision.metadata["verifier"], "llm_with_deterministic_guard")

    def test_verifier_blocks_structured_forbidden_side_effects(self) -> None:
        verifier = AutonomousVerifier(config=VerifierConfig(notification_channel_id="channel-1"))
        decision = verifier.verify(
            plan=AutonomousPlan(
                checks=(
                    AutonomousCheck(
                        id="check-1",
                        kind="task_due_soon",
                        target_ref="task:1",
                        reason="check",
                    ),
                )
            ),
            tool_results=(
                AutonomousToolResult(
                    id="result-1",
                    tool_name="workflow_query",
                    status="succeeded",
                    metadata={
                        "side_effects": "master_write",
                        "master_write_count": 1,
                    },
                ),
            ),
        )

        self.assertEqual(decision.decision, "noop")
        self.assertIn("forbidden_side_effect_detected", decision.conflicts)

    def test_autonomous_schedule_rules_follow_auto_index_pattern(self) -> None:
        rules = _autonomous_agent_rules_from_config(AutonomousConfigStub())

        self.assertEqual([rule.id for rule in rules], ["autonomous_agent_0800", "autonomous_agent_1330"])
        self.assertEqual(rules[0].trigger.kind, "schedule_cron")
        self.assertEqual(rules[0].trigger.params["cron"], "0 8 * * *")
        self.assertEqual(rules[0].actions[0].action_type, "autonomous_agent_run")


if __name__ == "__main__":
    unittest.main()
