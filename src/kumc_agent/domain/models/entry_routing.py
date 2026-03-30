from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EntryRoute = Literal["direct_rag", "openclaw"]


@dataclass(frozen=True)
class EntryRoutingDecision:
    route: EntryRoute
    reason: str
    payload: dict[str, object] = field(default_factory=dict)
