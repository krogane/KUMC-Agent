from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import sys
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
from kumc_agent.domain.models.workflow import WorkRequest, WorkResponse
from kumc_agent.features.foundation.payload_sanitizer import sanitize_payload
from kumc_agent.features.rag.components.integrated_input_routing import (
    IntegratedInputRouter,
    IntegratedRoutingPolicy,
)
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

    def test_removed_legacy_entrypoints_are_not_importable(self) -> None:
        self.assertIsNone(importlib.util.find_spec("kumc_agent.usecases.chat.entry"))
        self.assertIsNone(importlib.util.find_spec("kumc_agent.features.rag.components.entry_routing"))
        self.assertIsNone(importlib.util.find_spec("kumc_agent.domain.models.entry_routing"))


if __name__ == "__main__":
    unittest.main()
