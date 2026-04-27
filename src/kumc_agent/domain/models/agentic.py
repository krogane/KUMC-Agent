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
    max_replans: int = 1
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
class AgentTask:
    id: str
    description: str
    tool_name: str
    input: dict[str, Any] = field(default_factory=dict)
    success_criteria: tuple[str, ...] = tuple()


@dataclass(frozen=True)
class ToolCallPlan:
    tool_name: str
    input: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    read_only: bool = True
    side_effect_boundary: str = "read_only"


@dataclass(frozen=True)
class AgentPlan:
    tasks: tuple[AgentTask, ...] = tuple()
    required_tools: tuple[str, ...] = tuple()
    tool_sequence: tuple[ToolCallPlan, ...] = tuple()
    success_criteria: tuple[str, ...] = tuple()
    side_effect_boundary: str = "read_only"
    retry_policy: dict[str, Any] = field(default_factory=dict)
    answer_requirements: tuple[str, ...] = tuple()
    needs_clarification: bool = False
    clarification_question: str = ""
    direct_route: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentToolResult:
    tool_name: str
    status: str
    text: str = ""
    citations: tuple[Citation, ...] = tuple()
    candidates: tuple[dict[str, Any], ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    status: str
    satisfied: tuple[str, ...] = tuple()
    missing: tuple[str, ...] = tuple()
    conflicts: tuple[str, ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComprehensiveAgentRequest:
    query: str
    source_filter: str = "all"
    access: AccessContext = field(default_factory=AccessContext)
    budget: AgentBudget = field(default_factory=AgentBudget)
    required_features: tuple[str, ...] = tuple()
    source_filters: dict[str, Any] = field(default_factory=dict)
    attribute_filters: dict[str, Any] = field(default_factory=dict)
    risk: str = "low"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComprehensiveAgentResponse:
    text: str
    detail_markdown: str
    citations: tuple[Citation, ...]
    confidence: str
    run: AgentRun
    task_candidates: tuple[dict[str, Any], ...] = tuple()
    event_candidates: tuple[dict[str, Any], ...] = tuple()
    server_operations: tuple[dict[str, Any], ...] = tuple()
    assets: tuple[dict[str, Any], ...] = tuple()
    member_profiles: tuple[dict[str, Any], ...] = tuple()
    warnings: tuple[str, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)
