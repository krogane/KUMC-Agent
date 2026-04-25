from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ActionSpec:
    operation: str
    description: str
    risk_level: str
    approval_policy: str
    required_args: tuple[str, ...] = tuple()
    optional_args: tuple[str, ...] = tuple()
    read_only: bool = False


@dataclass(frozen=True)
class MinecraftDryRun:
    operation: str
    server_name: str
    args: dict[str, str] = field(default_factory=dict)
    risk_level: str = "low"
    approval_policy: str = "self"
    impact: str = ""
    expected_downtime: str = "none"
    rollback: str = ""
    command_preview: tuple[str, ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    execution_allowed: bool = False


@dataclass(frozen=True)
class ServerOperation:
    id: str
    server_name: str
    operation: str
    requested_by_user_id: str
    approved_by_user_ids: tuple[str, ...] = tuple()
    status: str = "waiting_approval"
    risk_level: str = "medium"
    action_run_id: str | None = None
    dry_run: MinecraftDryRun | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
