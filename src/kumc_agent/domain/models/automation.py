from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class TriggerSpec:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionSpec:
    field: str
    operator: str
    value: Any


@dataclass(frozen=True)
class ActionSpecRef:
    action_type: str
    target: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    approval_required: bool = False


@dataclass(frozen=True)
class AutomationRule:
    id: str
    name: str
    enabled: bool
    trigger: TriggerSpec
    conditions: tuple[ConditionSpec, ...] = tuple()
    actions: tuple[ActionSpecRef, ...] = tuple()
    mode: str = "dry_run"
    risk_level: str = "low"
    created_by_user_id: str = ""
    approved_by_user_id: str = ""
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class AutomationRun:
    rule_id: str
    trigger_key: str
    mode: str
    status: str
    idempotency_key: str
    action_plan: tuple[dict[str, Any], ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class AutomationResponse:
    text: str
    detail_markdown: str = ""
    rules: tuple[AutomationRule, ...] = tuple()
    runs: tuple[AutomationRun, ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)
