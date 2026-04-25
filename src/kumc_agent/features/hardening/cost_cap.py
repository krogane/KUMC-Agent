from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostCapPolicy:
    daily_usd_cap: float = 5.0
    per_run_usd_cap: float = 1.0

    def check(self, *, projected_daily_usd: float, projected_run_usd: float) -> tuple[str, ...]:
        warnings: list[str] = []
        if projected_run_usd > self.per_run_usd_cap:
            warnings.append("per_run_cost_cap_exceeded")
        if projected_daily_usd > self.daily_usd_cap:
            warnings.append("daily_cost_cap_exceeded")
        return tuple(warnings)
