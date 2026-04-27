from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kumc_agent.domain.models.minecraft import ServerOperation
from kumc_agent.domain.models.operations import (
    Asset,
    MemberProfile,
    WorkflowCandidate,
)
from kumc_agent.domain.models.retrieval import AccessContext, Citation


@dataclass(frozen=True)
class TaskCandidate:
    id: str
    title: str
    description: str | None = None
    proposed_assignee_user_id: str | None = None
    proposed_due_at: datetime | None = None
    related_event_id: str | None = None
    evidence: tuple[Citation, ...] = tuple()
    confidence: str = "low"
    status: str = "proposed"
    created_by: str = "agent"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    description: str | None = None
    assignee_user_id: str | None = None
    due_at: datetime | None = None
    related_event_id: str | None = None
    source_candidate_id: str | None = None
    status: str = "todo"
    priority: str = "normal"
    evidence: tuple[Citation, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class TaskChangeCandidate:
    id: str
    task_id: str
    operation: str
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    evidence: tuple[Citation, ...] = tuple()
    confidence: str = "medium"
    status: str = "proposed"
    created_by: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class TaskApprovalBatch:
    id: str
    candidate_ids: tuple[str, ...] = tuple()
    change_candidate_ids: tuple[str, ...] = tuple()
    period_start: datetime | None = None
    period_end: datetime | None = None
    notification_channel_id: str | None = None
    notification_message_id: str | None = None
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class Event:
    id: str
    title: str
    summary: str | None = None
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    place: str | None = None
    status: str = "planning"
    related_source_ids: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class EventCandidate:
    id: str
    title: str
    summary: str | None = None
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    place: str | None = None
    related_source_ids: tuple[str, ...] = tuple()
    evidence: tuple[Citation, ...] = tuple()
    confidence: str = "low"
    status: str = "proposed"
    created_by: str = "agent"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class Meeting:
    id: str
    title: str
    scheduled_at: datetime | None = None
    related_event_id: str | None = None
    agenda_markdown: str = ""
    minutes_markdown: str = ""
    decisions: tuple[str, ...] = tuple()
    open_questions: tuple[str, ...] = tuple()
    task_candidate_ids: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class ScheduleEvent:
    id: str
    title: str
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    place: str | None = None
    related_event_id: str | None = None
    status: str = "planned"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class ScheduleCandidate:
    id: str
    title: str
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    place: str | None = None
    related_event_id: str | None = None
    evidence: tuple[Citation, ...] = tuple()
    confidence: str = "low"
    status: str = "proposed"
    created_by: str = "agent"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class ApprovalRecord:
    id: str
    target_type: str
    target_id: str
    action: str
    actor_id: str
    comment: str = ""
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[Citation, ...] = tuple()
    created_at: datetime | None = None


@dataclass(frozen=True)
class WorkRequest:
    work_type: str
    instruction: str = ""
    target: str = ""
    output_format: str = "markdown"
    access: AccessContext = field(default_factory=AccessContext)


@dataclass(frozen=True)
class WorkResponse:
    text: str
    detail_markdown: str = ""
    task_candidates: tuple[TaskCandidate, ...] = tuple()
    task_change_candidates: tuple[TaskChangeCandidate, ...] = tuple()
    task_approval_batches: tuple[TaskApprovalBatch, ...] = tuple()
    event_candidates: tuple[EventCandidate, ...] = tuple()
    schedule_candidates: tuple[ScheduleCandidate, ...] = tuple()
    workflow_candidates: tuple[WorkflowCandidate, ...] = tuple()
    assets: tuple[Asset, ...] = tuple()
    member_profiles: tuple[MemberProfile, ...] = tuple()
    tasks: tuple[Task, ...] = tuple()
    events: tuple[Event, ...] = tuple()
    schedules: tuple[ScheduleEvent, ...] = tuple()
    meetings: tuple[Meeting, ...] = tuple()
    approvals: tuple[ApprovalRecord, ...] = tuple()
    server_operations: tuple[ServerOperation, ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)
