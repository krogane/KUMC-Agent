from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class AuditEvent:
    action: str
    actor_id: str
    actor_type: str
    outcome: str
    target: str = ""
    risk_level: str = "low"
    trace_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    event_id: str = field(default_factory=lambda: str(uuid4()))

    def with_created_at(self, created_at: datetime) -> AuditEvent:
        return AuditEvent(
            event_id=self.event_id,
            action=self.action,
            actor_id=self.actor_id,
            actor_type=self.actor_type,
            target=self.target,
            outcome=self.outcome,
            risk_level=self.risk_level,
            trace_id=self.trace_id,
            metadata=dict(self.metadata),
            created_at=created_at,
        )
