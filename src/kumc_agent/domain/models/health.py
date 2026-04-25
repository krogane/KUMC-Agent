from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ComponentHealth:
    name: str
    status: str
    detail: str = ""
    latency_ms: float | None = None


@dataclass(frozen=True)
class HealthReport:
    status: str
    checked_at: datetime
    components: tuple[ComponentHealth, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checked_at": self.checked_at.isoformat(timespec="seconds"),
            "components": [
                {
                    "name": component.name,
                    "status": component.status,
                    "detail": component.detail,
                    "latency_ms": component.latency_ms,
                }
                for component in self.components
            ],
        }
