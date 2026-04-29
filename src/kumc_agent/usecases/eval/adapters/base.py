from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from kumc_agent.usecases.eval.schema import EvalCase


@dataclass(frozen=True)
class AdapterRunResult:
    actual: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"


class EvalAdapter(Protocol):
    target: str

    def run_case(self, *, case: EvalCase, request: Any) -> AdapterRunResult:
        ...
