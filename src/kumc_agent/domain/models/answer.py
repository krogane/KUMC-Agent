from __future__ import annotations

from dataclasses import dataclass, field

from kumc_agent.domain.models.source import Source


@dataclass(frozen=True)
class Answer:
    text: str
    route: str
    sources: list[Source] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
