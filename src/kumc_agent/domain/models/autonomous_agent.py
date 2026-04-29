from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any

from kumc_agent.domain.models.agentic import AgentBudget, AgentRun
from kumc_agent.domain.models.retrieval import AccessContext, Citation


@dataclass(frozen=True)
class AutonomousAgentRequest:
    trigger: str = "manual"
    slot: str = "manual"
    scopes: tuple[str, ...] = tuple()
    dry_run: bool | None = None
    idempotency_key: str = ""
    access: AccessContext = field(default_factory=AccessContext)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotItem:
    id: str
    kind: str
    title: str
    summary: str = ""
    status: str = ""
    due_at: datetime | None = None
    starts_at: datetime | None = None
    risk: str = "low"
    citations: tuple[Citation, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutonomousAgentSnapshot:
    tasks_due_soon: tuple[SnapshotItem, ...] = tuple()
    tasks_overdue: tuple[SnapshotItem, ...] = tuple()
    tasks_stale: tuple[SnapshotItem, ...] = tuple()
    task_candidates: tuple[SnapshotItem, ...] = tuple()
    task_approval_batches: tuple[SnapshotItem, ...] = tuple()
    events_upcoming: tuple[SnapshotItem, ...] = tuple()
    events_missing_details: tuple[SnapshotItem, ...] = tuple()
    events_without_tasks: tuple[SnapshotItem, ...] = tuple()
    event_candidates: tuple[SnapshotItem, ...] = tuple()
    event_approval_batches: tuple[SnapshotItem, ...] = tuple()
    rag_delta: tuple[SnapshotItem, ...] = tuple()
    server_ops: tuple[SnapshotItem, ...] = tuple()
    automation_runs: tuple[SnapshotItem, ...] = tuple()
    recent_runs: tuple[SnapshotItem, ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutonomousCheck:
    id: str
    kind: str
    target_ref: str
    reason: str
    risk: str = "low"
    side_effect_boundary: str = "candidate_only"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutonomousQuery:
    id: str
    query: str
    source: str = "all"
    mode: str = "careful"
    depth: str = "normal"
    target_refs: tuple[str, ...] = tuple()
    work_type: str = ""
    risk: str = "candidate_only"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutonomousPlan:
    checks: tuple[AutonomousCheck, ...] = tuple()
    required_queries: tuple[AutonomousQuery, ...] = tuple()
    target_refs: tuple[str, ...] = tuple()
    success_criteria: tuple[str, ...] = tuple()
    risk: str = "low"
    side_effect_boundary: str = "candidate_only"
    notification_policy: dict[str, Any] = field(default_factory=dict)
    retry_policy: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutonomousToolResult:
    id: str
    tool_name: str
    status: str
    query_id: str = ""
    target_refs: tuple[str, ...] = tuple()
    candidate_ids: tuple[str, ...] = tuple()
    approval_ids: tuple[str, ...] = tuple()
    citations: tuple[Citation, ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NotificationProposal:
    id: str
    target_channel_id: str
    body: str
    target_refs: tuple[str, ...] = tuple()
    risk: str = "low"
    status: str = "proposed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovalRequestProposal:
    id: str
    target_type: str
    target_id: str
    reason: str
    risk: str = "medium"
    status: str = "proposed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutonomousDecision:
    decision: str
    satisfied: tuple[str, ...] = tuple()
    missing: tuple[str, ...] = tuple()
    conflicts: tuple[str, ...] = tuple()
    notification_proposals: tuple[NotificationProposal, ...] = tuple()
    approval_requests: tuple[ApprovalRequestProposal, ...] = tuple()
    candidate_refs: tuple[str, ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutonomousAgentResponse:
    status: str
    text: str
    detail_markdown: str = ""
    proposals: tuple[dict[str, Any], ...] = tuple()
    notification_proposals: tuple[NotificationProposal, ...] = tuple()
    approval_requests: tuple[ApprovalRequestProposal, ...] = tuple()
    candidate_refs: tuple[str, ...] = tuple()
    task_candidates: tuple[dict[str, Any], ...] = tuple()
    event_candidates: tuple[dict[str, Any], ...] = tuple()
    automation_runs: tuple[dict[str, Any], ...] = tuple()
    server_operations: tuple[dict[str, Any], ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    run: AgentRun | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "text": self.text,
            "detail_markdown": self.detail_markdown,
            "proposals": _dump_items(self.proposals),
            "notification_proposals": _dump_items(self.notification_proposals),
            "approval_requests": _dump_items(self.approval_requests),
            "candidate_refs": list(self.candidate_refs),
            "task_candidates": _dump_items(self.task_candidates),
            "event_candidates": _dump_items(self.event_candidates),
            "automation_runs": _dump_items(self.automation_runs),
            "server_operations": _dump_items(self.server_operations),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def autonomous_budget_from_config(value: object) -> AgentBudget:
    return AgentBudget(
        max_steps=int(getattr(value, "max_steps", 10)),
        max_search_calls=int(getattr(value, "max_search_calls", 6)),
        max_replans=int(getattr(value, "max_replans", 1)),
        max_cost_usd=float(getattr(value, "max_cost_usd", 0.50)),
        max_latency_seconds=float(getattr(value, "max_latency_seconds", 120.0)),
        allow_write_tools=False,
        require_citations=True,
    )


def _dump_items(items: tuple[object, ...]) -> list[object]:
    return [_dump_value(item) for item in items]


def _dump_value(value: object) -> object:
    if is_dataclass(value):
        return {
            key: _dump_value(item)
            for key, item in asdict(value).items()
            if key != "run"
        }
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[union-attr]
    if isinstance(value, tuple):
        return [_dump_value(item) for item in value]
    if isinstance(value, list):
        return [_dump_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _dump_value(item) for key, item in value.items()}
    return value
