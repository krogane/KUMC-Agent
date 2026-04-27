from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class WorkflowCandidate:
    id: str
    candidate_type: str
    title: str
    payload: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[dict[str, Any], ...] = tuple()
    confidence: str = "low"
    status: str = "proposed"
    created_by: str = "agent"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class WorkflowRun:
    workflow_id: str
    trigger: str = "manual"
    actor_user_id: str = ""
    guild_id: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    candidates: tuple[str, ...] = tuple()
    drafts: tuple[str, ...] = tuple()
    validation_result: dict[str, Any] = field(default_factory=dict)
    approval_required: bool = False
    status: str = "running"
    error: str = ""
    audit_log_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class Asset:
    id: str
    source_kind: str = ""
    source_item_id: str = ""
    title: str = ""
    description: str = ""
    uri: str = ""
    media_type: str = "image"
    captured_at: datetime | None = None
    access_scope: dict[str, Any] = field(default_factory=dict)
    rights_status: str = "unknown"
    contains_people: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class AssetUsageRequest:
    id: str
    asset_id: str = ""
    purpose: str = ""
    medium: str = ""
    requested_by: str = ""
    status: str = "proposed"
    needs_owner_check: bool = True
    needs_people_check: bool = True
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class MemberProfile:
    id: str
    display_name: str = ""
    discord_user_id: str = ""
    roles: tuple[str, ...] = tuple()
    skills: tuple[str, ...] = tuple()
    interests: tuple[str, ...] = tuple()
    past_assignments: tuple[str, ...] = tuple()
    evidence: tuple[dict[str, Any], ...] = tuple()
    access_scope: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class ActionRun:
    id: str
    action_type: str
    target: str = ""
    actor_user_id: str = ""
    status: str = "planned"
    risk_level: str = "low"
    idempotency_key: str = ""
    request_payload: dict[str, Any] = field(default_factory=dict)
    result_payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    trace_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class IndexingRun:
    id: str
    source_kind: str = ""
    status: str = "running"
    seen: int = 0
    changed: int = 0
    skipped: int = 0
    deleted: int = 0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class EvalRun:
    id: str
    eval_set_id: str = ""
    status: str = "running"
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
