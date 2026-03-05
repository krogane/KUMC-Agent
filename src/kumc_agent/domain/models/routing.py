from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoutingDecision:
    target_model: str
    recency_mode: str = "off"
    material_names: list[str] = field(default_factory=list)
    idea_generation: bool = False
    include_capabilities_info: bool = False
    use_additional_memory: bool = False
    needs_additional_query: bool = False
    additional_queries: list[str] = field(default_factory=list)
