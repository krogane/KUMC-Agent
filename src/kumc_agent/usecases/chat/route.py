from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.domain.models.routing import RoutingDecision
from kumc_agent.features.rag.components.routing import QueryRouter


@dataclass(frozen=True)
class RouteRequest:
    query: str


class ChatRouteUsecase:
    def __init__(self, *, router: QueryRouter) -> None:
        self._router = router

    def execute(self, request: RouteRequest) -> RoutingDecision:
        return self._router.route(request.query)
