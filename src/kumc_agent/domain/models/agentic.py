from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kumc_agent.domain.models.retrieval import AccessContext, Citation


@dataclass(frozen=True)
class AgentBudget:
    max_steps: int = 6
    max_search_calls: int = 4
    max_read_chunks: int = 20
    max_cost_usd: float = 0.50
    max_latency_seconds: float = 60.0
    allow_write_tools: bool = False
    require_citations: bool = True


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    read_only: bool = True


@dataclass(frozen=True)
class AgentStep:
    id: str
    run_id: str
    state: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    status: str = "succeeded"
    cost_usd: float = 0.0
    created_at: datetime | None = None


@dataclass(frozen=True)
class AgentRun:
    id: str
    query: str
    status: str
    access: AccessContext = field(default_factory=AccessContext)
    budget: AgentBudget = field(default_factory=AgentBudget)
    steps: tuple[AgentStep, ...] = tuple()
    citations: tuple[Citation, ...] = tuple()
    answer: str = ""
    confidence: str = "low"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class AgenticSearchRequest:
    query: str
    source_filter: str = "all"
    access: AccessContext = field(default_factory=AccessContext)
    budget: AgentBudget = field(default_factory=AgentBudget)


@dataclass(frozen=True)
class AgenticSearchResponse:
    text: str
    detail_markdown: str
    citations: tuple[Citation, ...]
    confidence: str
    run: AgentRun
    warnings: tuple[str, ...] = tuple()
