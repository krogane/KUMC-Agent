from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kumc_agent.domain.models.retrieval import Citation


@dataclass(frozen=True)
class FactCheckFinding:
    kind: str
    message: str
    severity: str = "medium"
    evidence: tuple[Citation, ...] = tuple()


@dataclass(frozen=True)
class SectionDraft:
    title: str
    body: str
    citations: tuple[Citation, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentPlan:
    id: str
    title: str
    doc_type: str
    audience: str = ""
    purpose: str = ""
    sections: tuple[SectionDraft, ...] = tuple()
    fact_checks: tuple[FactCheckFinding, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class DocumentDraft:
    plan: DocumentPlan
    markdown: str
    warnings: tuple[str, ...] = tuple()


@dataclass(frozen=True)
class DocGenRequest:
    title: str
    instruction: str = ""
    source_text: str = ""
    doc_type: str = "generic"
    audience: str = ""
    purpose: str = ""
    citations: tuple[Citation, ...] = tuple()
    public: bool = False
