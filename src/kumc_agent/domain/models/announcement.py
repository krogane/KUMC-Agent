from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kumc_agent.domain.models.docgen import FactCheckFinding
from kumc_agent.domain.models.retrieval import Citation


@dataclass(frozen=True)
class AnnouncementDraft:
    id: str
    title: str
    body_markdown: str
    medium: str = "discord"
    audience: str = ""
    status: str = "draft"
    fact_checks: tuple[FactCheckFinding, ...] = tuple()
    citations: tuple[Citation, ...] = tuple()
    created_by: str = "agent"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
