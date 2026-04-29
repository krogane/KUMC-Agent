from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.agentic import ComprehensiveAgentResponse
from kumc_agent.domain.models.answer import Answer
from kumc_agent.domain.models.integrated_input import (
    IntegratedInputDecision,
    IntegratedInputRequest,
)
from kumc_agent.domain.models.retrieval import AccessContext, AskResponse, Citation, RetrievalQuery
from kumc_agent.domain.models.source import Source
from kumc_agent.domain.models.workflow import Event, EventChangeCandidate, Task, WorkRequest, WorkResponse
from kumc_agent.features.event_management import EventExtractionResult
from kumc_agent.features.foundation.payload_sanitizer import sanitize_payload
from kumc_agent.features.rag.components.integrated_input_routing import (
    IntegratedInputRouter,
    IntegratedRoutingPolicy,
)
from kumc_agent.features.workflow import WorkflowService
from kumc_agent.infra.workflow import FileWorkflowRepository
from kumc_agent.usecases.integrated_input import IntegratedInputUsecase


@dataclass
class FakeAskService:
    queries: list[RetrievalQuery]

    def ask(self, query: RetrievalQuery) -> AskResponse:
        self.queries.append(query)
        return AskResponse(
            text=f"answer:{query.source_filter}:{query.text}",
            detail_markdown="detail",
            citations=(
                Citation(
                    source_item_id="source",
                    chunk_id="chunk",
                    label="label",
                ),
            ),
            confidence="medium",
            metadata={"contexts": "hidden", "trace": "ok"},
        )


@dataclass
class FakeChatAnswerService:
    requests: list[object]

    def execute(self, request) -> Answer:
        self.requests.append(request)
        return Answer(
            text=f"rag:{request.query}",
            route="rag",
            sources=[Source(id="s1", label="source-label", uri="https://example.com/s1")],
            metadata={"contexts": ["hidden"], "routing_decision": {"target_model": "rag"}},
        )


@dataclass
class FakeWorkflowService:
    requests: list[WorkRequest]

    def run(self, request: WorkRequest) -> WorkResponse:
        self.requests.append(request)
        return WorkResponse(
            text=f"work:{request.work_type}:{request.instruction}",
            metadata={"downloaded_image_path": "/tmp/secret.png", "trace": "ok"},
        )


@dataclass
class FakeAgent:
    requests: list[object]

    def run(self, request):
        self.requests.append(request)
        return ComprehensiveAgentResponse(
            text="agent answer",
            detail_markdown="agent detail",
            citations=tuple(),
            confidence="medium",
            metadata={"agent_run_id": "run-1", "raw": "hidden"},
        )


class FakeEventChangeExtractor:
    def extract(
        self,
        *,
        text: str,
        evidence: tuple[Citation, ...],
        access: AccessContext,
        metadata: dict[str, object],
        existing_events: tuple[Event, ...] = tuple(),
    ) -> EventExtractionResult:
        event = existing_events[0]
        return EventExtractionResult(
            candidates=tuple(),
            change_candidates=(
                EventChangeCandidate(
                    id=f"event-change-{event.id}",
                    event_id=event.id,
                    operation="update",
                    before={"status": event.status},
                    after={"status": "done"},
                    reason=text,
                    evidence=evidence,
                    confidence="high",
                ),
            ),
            metadata={**metadata, "degraded": False},
        )


class StaticRouter:
    def __init__(self, decision: IntegratedInputDecision) -> None:
        self.decision = decision

    def decide(self, *args, **kwargs) -> IntegratedInputDecision:
        return self.decision


class IntegratedInputTests(unittest.TestCase):
    def test_router_parses_extended_schema(self) -> None:
        router = IntegratedInputRouter()
        decision = router._parse_payload(
            """```json
            {"route":"task_management","intent":"create_candidate","required_features":["task_management"],"source_filters":[],"attribute_filters":{"due":"today"},"risk":"candidate_only","freshness_required":true,"needs_clarification":false,"reason":"task"}
            ```"""
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.route, "task_management")
        self.assertEqual(decision.intent, "create_candidate")
        self.assertEqual(decision.attribute_filters["due"], "today")

    def test_policy_prefers_explicit_source_and_escalates_multiple_features(self) -> None:
        policy = IntegratedRoutingPolicy()
        explicit = policy.apply(
            IntegratedInputDecision(route="circle_rag", required_features=("circle_rag",)),
            text="丸石の作り方",
            source="minecraft_wiki",
        )
        self.assertEqual(explicit.route, "minecraft_wiki_rag")

        escalated = policy.apply(
            IntegratedInputDecision(route="task_management", required_features=("task_management", "member_search")),
            text="担当候補を探してタスク候補を作って",
            source="all",
        )
        self.assertEqual(escalated.route, "comprehensive_agent")

        deep_rag = policy.apply(
            IntegratedInputDecision(route="circle_rag", required_features=("circle_rag",)),
            text="KUMCの活動時間を詳しく調べて",
            source="all",
            depth="deep",
        )
        self.assertEqual(deep_rag.route, "comprehensive_agent")

        explicit_composite = policy.apply(
            IntegratedInputDecision(route="member_search", required_features=("member_search",)),
            text="担当候補を探してタスク候補を作って",
            source="member",
        )
        self.assertEqual(explicit_composite.route, "comprehensive_agent")
        self.assertEqual(explicit_composite.required_features, ("member_search", "task_management"))

    def test_classifier_fallback_clarifies_side_effect_requests(self) -> None:
        router = IntegratedInputRouter(provider="none")

        side_effect = router.decide("タスクを追加して", source="task")
        self.assertEqual(side_effect.route, "clarify")
        self.assertEqual(side_effect.risk, "read_only")
        self.assertTrue(side_effect.needs_clarification)

        read_only = router.decide("タスク一覧", source="task")
        self.assertEqual(read_only.route, "task_management")
        self.assertEqual(read_only.risk, "read_only")

    def test_usecase_routes_rag_with_same_access_context(self) -> None:
        ask = FakeAskService([])
        workflow = FakeWorkflowService([])
        usecase = IntegratedInputUsecase(
            ask_service=ask,
            workflow_service=workflow,
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(route="minecraft_wiki_rag", required_features=("minecraft_wiki",))
            ),  # type: ignore[arg-type]
        )
        access = AccessContext(user_id="u1", guild_id="g1", role_ids=("r1",))

        response = usecase.execute(
            IntegratedInputRequest(text="丸石", source="minecraft_wiki", access=access)
        )

        self.assertEqual(response.metadata["route"], "minecraft_wiki_rag")
        self.assertEqual(ask.queries[0].source_filter, "minecraft_wiki")
        self.assertEqual(ask.queries[0].access, access)

    def test_usecase_routes_circle_rag_to_chat_answer_service(self) -> None:
        ask = FakeAskService([])
        chat = FakeChatAnswerService([])
        usecase = IntegratedInputUsecase(
            ask_service=ask,
            chat_answer_service=chat,
            workflow_service=FakeWorkflowService([]),
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(route="circle_rag", required_features=("circle_rag",))
            ),  # type: ignore[arg-type]
        )
        access = AccessContext(user_id="u1", guild_id="g1", role_ids=("r1",))

        response = usecase.execute(
            IntegratedInputRequest(text="例会はいつ", source="all", access=access)
        )

        self.assertEqual(response.text, "rag:例会はいつ")
        self.assertEqual(response.metadata["route"], "circle_rag")
        self.assertEqual(response.metadata["handler"], "circle_rag")
        self.assertEqual(chat.requests[0].access_context, access)
        self.assertEqual(ask.queries, [])
        self.assertNotIn("contexts", response.metadata)
        self.assertEqual(response.citations[0]["url"], "https://example.com/s1")

    def test_usecase_routes_minecraft_wiki_rag_to_chat_answer_service(self) -> None:
        ask = FakeAskService([])
        chat = FakeChatAnswerService([])
        usecase = IntegratedInputUsecase(
            ask_service=ask,
            chat_answer_service=chat,
            workflow_service=FakeWorkflowService([]),
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(route="minecraft_wiki_rag", required_features=("minecraft_wiki",))
            ),  # type: ignore[arg-type]
        )
        access = AccessContext(user_id="u1", guild_id="g1", role_ids=("r1",))

        response = usecase.execute(
            IntegratedInputRequest(text="丸石の入手方法", source="minecraft_wiki", access=access)
        )

        self.assertEqual(response.text, "rag:丸石の入手方法")
        self.assertEqual(response.metadata["route"], "minecraft_wiki_rag")
        self.assertEqual(response.metadata["handler"], "minecraft_wiki_rag")
        self.assertEqual(chat.requests[0].access_context, access)
        self.assertEqual(chat.requests[0].route_override, "minecraft_wiki")
        self.assertTrue(chat.requests[0].force_disable_additional_memory)
        self.assertEqual(ask.queries, [])

    def test_usecase_routes_workflow_and_sanitizes_metadata(self) -> None:
        workflow = FakeWorkflowService([])
        usecase = IntegratedInputUsecase(
            ask_service=FakeAskService([]),
            workflow_service=workflow,
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(
                    route="image_search",
                    intent="search",
                    required_features=("image_search",),
                )
            ),  # type: ignore[arg-type]
        )

        response = usecase.execute(IntegratedInputRequest(text="写真を探して", source="image"))

        self.assertEqual(workflow.requests[0].work_type, "image_search")
        self.assertNotIn("downloaded_image_path", response.metadata)

    def test_usecase_passes_image_source_filters_to_workflow(self) -> None:
        workflow = FakeWorkflowService([])
        usecase = IntegratedInputUsecase(
            ask_service=FakeAskService([]),
            workflow_service=workflow,
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(
                    route="image_search",
                    intent="search",
                    required_features=("image_search",),
                    source_filters=("drive",),
                )
            ),  # type: ignore[arg-type]
        )

        usecase.execute(IntegratedInputRequest(text="Driveの画像", source="all"))

        self.assertEqual(("drive",), workflow.requests[0].source_filter)

    def test_usecase_passes_history_scope_to_workflow_metadata(self) -> None:
        workflow = FakeWorkflowService([])
        usecase = IntegratedInputUsecase(
            ask_service=FakeAskService([]),
            workflow_service=workflow,
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(
                    route="image_search",
                    intent="search",
                    required_features=("image_search",),
                )
            ),  # type: ignore[arg-type]
        )

        response = usecase.execute(
            IntegratedInputRequest(text="写真を探して", source="image", history_scope="discord:g:c:t")
        )

        self.assertEqual(workflow.requests[0].metadata["history_scope"], "discord:g:c:t")
        self.assertEqual(response.metadata["history_scope"], "discord:g:c:t")

    def test_usecase_maps_task_done_to_change_candidate_work_type(self) -> None:
        workflow = FakeWorkflowService([])
        usecase = IntegratedInputUsecase(
            ask_service=FakeAskService([]),
            workflow_service=workflow,
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(
                    route="task_management",
                    intent="complete",
                    required_features=("task_management",),
                    risk="candidate_only",
                )
            ),  # type: ignore[arg-type]
        )

        usecase.execute(IntegratedInputRequest(text="task_id: t1 を完了候補にして"))

        self.assertEqual(workflow.requests[0].work_type, "task_update")
        self.assertIn("status: done", workflow.requests[0].instruction)

    def test_usecase_clarifies_missing_mutation_target_before_workflow(self) -> None:
        workflow = FakeWorkflowService([])
        usecase = IntegratedInputUsecase(
            ask_service=FakeAskService([]),
            workflow_service=workflow,
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(
                    route="task_management",
                    intent="complete",
                    required_features=("task_management",),
                    risk="candidate_only",
                )
            ),  # type: ignore[arg-type]
        )

        response = usecase.execute(IntegratedInputRequest(text="このタスクを完了にして"))

        self.assertEqual(response.metadata["route"], "clarify")
        self.assertEqual(workflow.requests, [])
        self.assertIn("Task ID", response.text)

    def test_usecase_clarifies_event_add_missing_title_before_workflow(self) -> None:
        workflow = FakeWorkflowService([])
        usecase = IntegratedInputUsecase(
            ask_service=FakeAskService([]),
            workflow_service=workflow,
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(
                    route="event_management",
                    intent="create_candidate",
                    required_features=("event_management",),
                    risk="candidate_only",
                )
            ),  # type: ignore[arg-type]
        )

        response = usecase.execute(IntegratedInputRequest(text="イベント追加 日時: 2026-05-05 14:00", is_admin=True))

        self.assertEqual(response.metadata["route"], "clarify")
        self.assertEqual(workflow.requests, [])
        self.assertIn("タイトル", response.text)

    def test_usecase_maps_event_complete_to_change_candidate_work_type(self) -> None:
        workflow = FakeWorkflowService([])
        usecase = IntegratedInputUsecase(
            ask_service=FakeAskService([]),
            workflow_service=workflow,
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(
                    route="event_management",
                    intent="complete",
                    required_features=("event_management",),
                    risk="candidate_only",
                )
            ),  # type: ignore[arg-type]
        )

        usecase.execute(IntegratedInputRequest(text="event_id: e1 を完了候補にして", is_admin=True))

        self.assertEqual(workflow.requests[0].work_type, "event_update")
        self.assertIn("status: done", workflow.requests[0].instruction)

    def test_usecase_returns_notification_candidate_without_workflow_mutation(self) -> None:
        workflow = FakeWorkflowService([])
        usecase = IntegratedInputUsecase(
            ask_service=FakeAskService([]),
            workflow_service=workflow,
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(
                    route="event_management",
                    intent="notify",
                    required_features=("event_management",),
                    risk="candidate_only",
                )
            ),  # type: ignore[arg-type]
        )

        response = usecase.execute(IntegratedInputRequest(text="イベント通知を作って", is_admin=True))

        self.assertEqual(workflow.requests, [])
        self.assertEqual(response.workflow_candidates[0]["candidate_type"], "event_notification")
        self.assertEqual(response.candidates[0]["approval_target_type"], "other")

    def test_read_only_notification_route_clarifies_without_candidate(self) -> None:
        workflow = FakeWorkflowService([])
        usecase = IntegratedInputUsecase(
            ask_service=FakeAskService([]),
            workflow_service=workflow,
            comprehensive_agent=None,
            router=StaticRouter(
                IntegratedInputDecision(
                    route="event_management",
                    intent="notify",
                    required_features=("event_management",),
                    risk="read_only",
                )
            ),  # type: ignore[arg-type]
        )

        response = usecase.execute(IntegratedInputRequest(text="イベント通知を作って", is_admin=True))

        self.assertEqual(workflow.requests, [])
        self.assertEqual(response.workflow_candidates, tuple())
        self.assertIn("read_only route blocked candidate creation", response.warnings)

    def test_task_done_keeps_master_task_unchanged_in_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = FileWorkflowRepository(root_dir=Path(tmp) / "workflow")
            repository.save_task(
                Task(
                    id="task-1",
                    title="会場予約",
                    assignee_user_id="alice",
                    status="todo",
                )
            )
            usecase = IntegratedInputUsecase(
                ask_service=FakeAskService([]),
                workflow_service=WorkflowService(repository=repository),
                comprehensive_agent=None,
                router=StaticRouter(
                    IntegratedInputDecision(
                        route="task_management",
                        intent="complete",
                        required_features=("task_management",),
                        risk="candidate_only",
                    )
                ),  # type: ignore[arg-type]
            )

            response = usecase.execute(
                IntegratedInputRequest(
                    text="task_id: task-1 を完了候補にして",
                    access=AccessContext(user_id="alice"),
                )
            )

            stored = repository.get_task("task-1")
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(stored.status, "todo")
            self.assertEqual(response.task_change_candidates[0]["after"]["status"], "done")
            self.assertEqual(repository.list_task_change_candidates()[0].after["status"], "done")

    def test_event_complete_keeps_master_event_unchanged_in_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = FileWorkflowRepository(root_dir=Path(tmp) / "workflow")
            repository.save_event(
                Event(
                    id="event-1",
                    title="新歓会",
                    starts_at=datetime(2026, 5, 5, tzinfo=UTC),
                    status="planning",
                )
            )
            usecase = IntegratedInputUsecase(
                ask_service=FakeAskService([]),
                workflow_service=WorkflowService(
                    repository=repository,
                    event_extractor=FakeEventChangeExtractor(),
                ),
                comprehensive_agent=None,
                router=StaticRouter(
                    IntegratedInputDecision(
                        route="event_management",
                        intent="complete",
                        required_features=("event_management",),
                        risk="candidate_only",
                    )
                ),  # type: ignore[arg-type]
            )

            response = usecase.execute(
                IntegratedInputRequest(
                    text="event_id: event-1 を完了候補にして",
                    is_admin=True,
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )

            stored = repository.get_event("event-1")
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(stored.status, "planning")
            self.assertEqual(response.event_change_candidates[0]["after"]["status"], "done")
            self.assertEqual(repository.list_event_change_candidates()[0].after["status"], "done")

    def test_notification_requests_keep_master_metadata_unchanged_in_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = FileWorkflowRepository(root_dir=Path(tmp) / "workflow")
            due_at = datetime.now(UTC) + timedelta(days=1)
            repository.save_task(
                Task(
                    id="task-1",
                    title="期限確認",
                    due_at=due_at,
                    status="todo",
                    metadata={},
                )
            )
            usecase = IntegratedInputUsecase(
                ask_service=FakeAskService([]),
                workflow_service=WorkflowService(repository=repository),
                comprehensive_agent=None,
                router=StaticRouter(
                    IntegratedInputDecision(
                        route="task_management",
                        intent="notify",
                        required_features=("task_management",),
                        risk="candidate_only",
                    )
                ),  # type: ignore[arg-type]
            )

            response = usecase.execute(
                IntegratedInputRequest(
                    text="task_id: task-1 の期限通知候補を作って",
                    is_admin=True,
                    access=AccessContext(user_id="admin", is_admin=True),
                )
            )

            stored = repository.get_task("task-1")
            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(stored.metadata, {})
            self.assertEqual(response.workflow_candidates[0]["candidate_type"], "task_notification")

    def test_usecase_escalates_to_comprehensive_agent(self) -> None:
        agent = FakeAgent([])
        usecase = IntegratedInputUsecase(
            ask_service=FakeAskService([]),
            workflow_service=FakeWorkflowService([]),
            comprehensive_agent=agent,
            router=StaticRouter(
                IntegratedInputDecision(
                    route="task_management",
                    required_features=("task_management", "member_search"),
                    risk="candidate_only",
                )
            ),  # type: ignore[arg-type]
        )

        response = usecase.execute(
            IntegratedInputRequest(text="担当候補を探してタスク候補を作って", is_admin=True)
        )

        self.assertEqual(response.metadata["route"], "comprehensive_agent")
        self.assertEqual(agent.requests[0].required_features, ("task_management", "member_search"))
        self.assertEqual(response.metadata["agent_run_id"], "run-1")

    def test_usecase_denies_non_admin_comprehensive_candidate_creation(self) -> None:
        agent = FakeAgent([])
        usecase = IntegratedInputUsecase(
            ask_service=FakeAskService([]),
            workflow_service=FakeWorkflowService([]),
            comprehensive_agent=agent,
            router=StaticRouter(
                IntegratedInputDecision(
                    route="comprehensive_agent",
                    required_features=("task_management", "member_search"),
                    risk="candidate_only",
                )
            ),  # type: ignore[arg-type]
        )

        response = usecase.execute(
            IntegratedInputRequest(text="担当候補を探してタスク候補を作って", is_admin=False)
        )

        self.assertEqual(response.metadata["route"], "deny")
        self.assertEqual(agent.requests, [])

    def test_sanitizer_removes_secrets_context_and_image_paths(self) -> None:
        payload = sanitize_payload(
            {
                "metadata": {
                    "context": "hidden",
                    "downloaded_image_path": "/tmp/image.png",
                    "message": "api_key=abc123 token:xyz",
                }
            }
        )

        self.assertEqual(payload["metadata"]["message"], "api_key=[REDACTED] token=[REDACTED]")
        self.assertNotIn("context", payload["metadata"])
        self.assertNotIn("downloaded_image_path", payload["metadata"])

    def test_http_access_resolves_admin_from_allowlist_only(self) -> None:
        from kumc_agent.frontends.http.app import _access

        spoofed = _access({"user_id": "user", "is_admin": True}, admin_user_ids=("admin",))
        allowed = _access({"user_id": "admin"}, admin_user_ids=("admin",))

        self.assertFalse(spoofed.is_admin)
        self.assertTrue(allowed.is_admin)

    def test_removed_legacy_entrypoints_are_not_importable(self) -> None:
        self.assertIsNone(importlib.util.find_spec("kumc_agent.usecases.chat.entry"))
        self.assertIsNone(importlib.util.find_spec("kumc_agent.features.rag.components.entry_routing"))
        self.assertIsNone(importlib.util.find_spec("kumc_agent.domain.models.entry_routing"))


if __name__ == "__main__":
    unittest.main()
