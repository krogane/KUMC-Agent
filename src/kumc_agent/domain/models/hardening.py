from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class PromptInjectionFinding:
    source_label: str
    pattern: str
    severity: str
    excerpt: str


@dataclass(frozen=True)
class ProductionReadinessCheck:
    id: str
    title: str
    status: str
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductionReadinessReport:
    status: str
    checked_at: datetime
    checks: tuple[ProductionReadinessCheck, ...]
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checked_at": self.checked_at.isoformat(),
            "summary": self.summary,
            "checks": [
                {
                    "id": check.id,
                    "title": check.title,
                    "status": check.status,
                    "detail": check.detail,
                    "metadata": dict(check.metadata),
                }
                for check in self.checks
            ],
            "metadata": dict(self.metadata),
        }
