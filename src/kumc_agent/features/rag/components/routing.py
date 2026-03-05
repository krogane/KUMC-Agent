from __future__ import annotations

from kumc_agent.domain.models.routing import RoutingDecision
from kumc_agent.domain.policies.refusal import should_refuse


class QueryRouter:
    def __init__(self, *, refusal_keywords: list[str]) -> None:
        self._refusal_keywords = refusal_keywords

    def route(self, query: str) -> RoutingDecision:
        if should_refuse(query=query, keywords=self._refusal_keywords):
            return RoutingDecision(target_model="refusal", recency_mode="off")
        return RoutingDecision(target_model="rag", recency_mode="soft")
