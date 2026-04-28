from __future__ import annotations

from dataclasses import dataclass
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kumc_agent.domain.models.agentic import AgentBudget, ComprehensiveAgentRequest
from kumc_agent.domain.models.docgen import DocGenRequest
from kumc_agent.domain.models.retrieval import AccessContext, AskResponse, Citation
from kumc_agent.domain.models.workflow import WorkRequest
from kumc_agent.features.agentic import ComprehensiveAgentService, ToolSchemaRegistry
from kumc_agent.features.agentic.comprehensive import ComprehensiveLLMConfig
from kumc_agent.features.announcement import AnnouncementDraftService
from kumc_agent.features.docgen.service import DocGenService
from kumc_agent.features.workflow import WorkflowService
from kumc_agent.infra.agentic import FileAgentTraceRepository
from kumc_agent.infra.announcement import FileAnnouncementRepository
from kumc_agent.infra.workflow import FileWorkflowRepository


@dataclass(frozen=True)
class DummyAskService:
    with_citation: bool = True

    def ask(self, query) -> AskResponse:
        citations = (
            (
                Citation(
                    source_item_id="source-1",
                    chunk_id=f"chunk-{query.text[:8]}",
                    label="source",
                    url="https://example.test/source",
                    quote="根拠",
                ),
            )
            if self.with_citation
            else tuple()
        )
        return AskResponse(
            text=f"回答: {query.text}",
            detail_markdown=f"詳細: {query.text}",
            citations=citations,
            confidence="medium" if citations else "low",
        )


@dataclass
class FakeLLM:
    payloads: list[dict[str, object]]
    calls: int = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return json.dumps(payload, ensure_ascii=False)


class AgenticDocgenAnnouncementTests(unittest.TestCase):
    def test_comprehensive_agent_runs_state_machine_with_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = FileAgentTraceRepository(root_dir=Path(tmp) / "agentic")
            service = ComprehensiveAgentService(
                ask_service=DummyAskService(),
                repository=repository,
            )

            response = service.run(
                ComprehensiveAgentRequest(
                    query="新歓企画の要点は?",
                    access=AccessContext(user_id="user-1"),
                    required_features=("circle_rag",),
                    metadata={"depth": "deep"},
                )
            )

            states = [
                step.state
                for step in repository.list_steps(str(response.metadata["agent_run_id"]))
            ]
            self.assertIn("PLAN", states)
            self.assertIn("TOOL", states)
            self.assertIn("VERIFY", states)
            self.assertIn("ANSWER", states)
            self.assertEqual(response.confidence, "high")
            self.assertGreaterEqual(len(response.citations), 1)

    def test_comprehensive_agent_stops_when_evidence_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ComprehensiveAgentService(
                ask_service=DummyAskService(with_citation=False),
                repository=FileAgentTraceRepository(root_dir=Path(tmp) / "agentic"),
            )

            response = service.run(
                ComprehensiveAgentRequest(query="不明な質問", metadata={"depth": "deep"})
            )

            self.assertEqual(response.confidence, "low")
            self.assertIn("十分な根拠", response.text)

    def test_comprehensive_agent_creates_approval_record_after_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = WorkflowService(
                repository=FileWorkflowRepository(root_dir=root / "workflow"),
                ask_service=DummyAskService(),
            )
            service = ComprehensiveAgentService(
                ask_service=DummyAskService(),
                repository=FileAgentTraceRepository(root_dir=root / "agentic"),
                workflow_service=workflow,
            )

            response = service.run(
                ComprehensiveAgentRequest(
                    query="タスク: 会場予約 を作成",
                    access=AccessContext(user_id="admin", is_admin=True),
                    required_features=("task_management",),
                    metadata={"depth": "deep"},
                    budget=AgentBudget(allow_write_tools=True),
                )
            )

            self.assertEqual(response.confidence, "medium")
            self.assertGreaterEqual(len(response.task_candidates), 1)
            self.assertGreaterEqual(len(response.approvals), 1)

    def test_comprehensive_agent_uses_dedicated_llm_planner_and_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planner = FakeLLM(
                [
                    {
                        "tasks": [
                            {
                                "id": "task-1",
                                "description": "根拠を検索する",
                                "tool_name": "circle_rag_search",
                                "input": {"query": "LLM計画 query"},
                                "success_criteria": ["citation"],
                            }
                        ],
                        "required_tools": ["circle_rag_search"],
                        "tool_sequence": [
                            {
                                "tool_name": "circle_rag_search",
                                "input": {"query": "LLM計画 query"},
                                "reason": "専用LLM計画",
                                "side_effect_boundary": "read_only",
                            }
                        ],
                        "success_criteria": ["citation"],
                        "side_effect_boundary": "read_only",
                    }
                ]
            )
            verifier = FakeLLM(
                [
                    {
                        "status": "succeeded",
                        "satisfied": ["citation"],
                        "missing": [],
                        "conflicts": [],
                        "warnings": [],
                        "metadata": {"checked_by": "llm"},
                    }
                ]
            )
            service = ComprehensiveAgentService(
                ask_service=DummyAskService(),
                repository=FileAgentTraceRepository(root_dir=Path(tmp) / "agentic"),
                planner_llm=planner,
                verifier_llm=verifier,
                planner_config=ComprehensiveLLMConfig(enabled=True, max_retries=1),
                verifier_config=ComprehensiveLLMConfig(enabled=True, max_retries=1),
            )

            response = service.run(
                ComprehensiveAgentRequest(
                    query="専用LLMで計画して",
                    metadata={"depth": "deep"},
                )
            )

            self.assertEqual(planner.calls, 1)
            self.assertEqual(verifier.calls, 1)
            self.assertIn("LLM計画 query", response.text)

    def test_docgen_renders_markdown_and_redacts_public_secrets(self) -> None:
        draft = DocGenService().run(
            DocGenRequest(
                title="告知下書き",
                instruction="イベント告知を作る",
                source_text="会場: 部室\nPIN: 1234\n連絡先: test@example.com",
                doc_type="announcement",
                public=True,
            )
        )

        self.assertIn("# 告知下書き", draft.markdown)
        self.assertNotIn("1234", draft.markdown)
        self.assertNotIn("test@example.com", draft.markdown)
        self.assertGreaterEqual(len(draft.plan.fact_checks), 1)

    def test_workflow_doc_x_and_announcement_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docgen = DocGenService()
            service = WorkflowService(
                repository=FileWorkflowRepository(root_dir=root / "workflow"),
                ask_service=DummyAskService(),
                docgen=docgen,
                announcement=AnnouncementDraftService(
                    repository=FileAnnouncementRepository(root_dir=root / "announcement"),
                    docgen=docgen,
                ),
            )

            doc = service.run(
                WorkRequest(
                    work_type="doc_draft",
                    instruction="週報 title: KUMC 週報",
                )
            )
            x = service.run(
                WorkRequest(work_type="x_draft", instruction="新歓会を告知する")
            )
            announcement = service.run(
                WorkRequest(
                    work_type="announcement_draft",
                    instruction="title: 新歓告知 会場: 部室 PIN: 1234",
                )
            )

            self.assertIn("#", doc.detail_markdown)
            self.assertLessEqual(x.metadata["selected_length"], 280)
            self.assertNotIn("1234", announcement.detail_markdown)
            self.assertNotIn("1234", announcement.text)
            self.assertEqual(announcement.metadata["status"], "needs_review")

    def test_tool_registry(self) -> None:
        names = {schema.name for schema in ToolSchemaRegistry().list()}
        self.assertIn("circle_rag_search", names)
        self.assertIn("task_candidate_create", names)
        self.assertFalse(ToolSchemaRegistry().get("task_candidate_create").read_only)


if __name__ == "__main__":
    unittest.main()
