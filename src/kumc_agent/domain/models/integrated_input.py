from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

from kumc_agent.domain.models.minecraft import ServerOperation
from kumc_agent.domain.models.operations import Asset, MemberProfile, WorkflowCandidate
from kumc_agent.domain.models.retrieval import AccessContext, Citation
from kumc_agent.domain.models.workflow import (
    ApprovalRecord,
    Event,
    EventApprovalBatch,
    EventCandidate,
    EventChangeCandidate,
    ScheduleCandidate,
    ScheduleEvent,
    Task,
    TaskApprovalBatch,
    TaskCandidate,
    TaskChangeCandidate,
)

IntegratedRoute = Literal[
    "no_rag",
    "circle_rag",
    "minecraft_wiki_rag",
    "member_search",
    "image_search",
    "task_management",
    "event_management",
    "server_management",
    "comprehensive_agent",
    "clarify",
    "deny",
]

RequiredFeature = Literal[
    "circle_rag",
    "minecraft_wiki",
    "minecraft_wiki_rag",
    "member_search",
    "image_search",
    "task_management",
    "event_management",
    "server_management",
]

RiskLevel = Literal[
    "read_only",
    "candidate_only",
    "approval_required",
    "admin_only",
]

InputIntent = Literal[
    "question",
    "search",
    "create_candidate",
    "update_candidate",
    "delete_candidate",
    "approval",
    "admin_operation",
    "compose",
    "extract",
    "list",
    "notify",
    "complete",
    "unknown",
]

ConfidenceLevel = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class IntegratedInputRequest:
    text: str
    source: str = "all"
    mode: str = "answer"
    depth: str = "normal"
    history_scope: str | None = None
    user_id: str = ""
    guild_id: str = ""
    role_ids: tuple[str, ...] = tuple()
    is_admin: bool = False
    frontend: str = "cli"
    access: AccessContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_access(self) -> AccessContext:
        if self.access is not None:
            return self.access
        return AccessContext(
            user_id=str(self.user_id or ""),
            guild_id=str(self.guild_id or ""),
            role_ids=tuple(str(role) for role in self.role_ids),
            is_admin=bool(self.is_admin),
        )


@dataclass(frozen=True)
class IntegratedInputDecision:
    route: IntegratedRoute = "circle_rag"
    intent: InputIntent = "question"
    required_features: tuple[str, ...] = tuple()
    source_filters: tuple[str, ...] = tuple()
    attribute_filters: dict[str, Any] = field(default_factory=dict)
    risk: RiskLevel = "read_only"
    freshness_required: bool = False
    needs_clarification: bool = False
    clarification_question: str = ""
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegratedInputResponse:
    text: str = ""
    detail_markdown: str = ""
    citations: tuple[Citation, ...] = tuple()
    confidence: ConfidenceLevel = "low"
    candidates: tuple[dict[str, Any], ...] = tuple()
    task_candidates: tuple[TaskCandidate | dict[str, Any], ...] = tuple()
    task_change_candidates: tuple[TaskChangeCandidate | dict[str, Any], ...] = tuple()
    task_approval_batches: tuple[TaskApprovalBatch | dict[str, Any], ...] = tuple()
    event_candidates: tuple[EventCandidate | dict[str, Any], ...] = tuple()
    event_change_candidates: tuple[EventChangeCandidate | dict[str, Any], ...] = tuple()
    event_approval_batches: tuple[EventApprovalBatch | dict[str, Any], ...] = tuple()
    schedule_candidates: tuple[ScheduleCandidate | dict[str, Any], ...] = tuple()
    workflow_candidates: tuple[WorkflowCandidate | dict[str, Any], ...] = tuple()
    assets: tuple[Asset | dict[str, Any], ...] = tuple()
    member_profiles: tuple[MemberProfile | dict[str, Any], ...] = tuple()
    tasks: tuple[Task | dict[str, Any], ...] = tuple()
    events: tuple[Event | dict[str, Any], ...] = tuple()
    schedules: tuple[ScheduleEvent | dict[str, Any], ...] = tuple()
    approvals: tuple[ApprovalRecord | dict[str, Any], ...] = tuple()
    server_operations: tuple[ServerOperation | dict[str, Any], ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "detail_markdown": self.detail_markdown,
            "citations": _dump_items(self.citations),
            "confidence": self.confidence,
            "candidates": _dump_items(self.candidates),
            "task_candidates": _dump_items(self.task_candidates),
            "task_change_candidates": _dump_items(self.task_change_candidates),
            "task_approval_batches": _dump_items(self.task_approval_batches),
            "event_candidates": _dump_items(self.event_candidates),
            "event_change_candidates": _dump_items(self.event_change_candidates),
            "event_approval_batches": _dump_items(self.event_approval_batches),
            "schedule_candidates": _dump_items(self.schedule_candidates),
            "workflow_candidates": _dump_items(self.workflow_candidates),
            "assets": _dump_items(self.assets),
            "member_profiles": _dump_items(self.member_profiles),
            "tasks": _dump_items(self.tasks),
            "events": _dump_items(self.events),
            "schedules": _dump_items(self.schedules),
            "approvals": _dump_items(self.approvals),
            "server_operations": _dump_items(self.server_operations),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def _dump_items(items: tuple[object, ...]) -> list[object]:
    return [_dump_item(item) for item in items or tuple()]


def _dump_item(item: object) -> object:
    if is_dataclass(item):
        return {
            key: _dump_value(value)
            for key, value in asdict(item).items()
        }
    if isinstance(item, dict):
        return {
            str(key): _dump_value(value)
            for key, value in item.items()
        }
    return _dump_value(item)


def _dump_value(value: object) -> object:
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[union-attr]
    if isinstance(value, tuple):
        return [_dump_value(item) for item in value]
    if isinstance(value, list):
        return [_dump_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _dump_value(item) for key, item in value.items()}
    return value
