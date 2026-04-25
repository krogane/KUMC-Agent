from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter

from kumc_agent.domain.models.health import ComponentHealth


@dataclass(frozen=True)
class ElasticsearchAdapter:
    url: str = ""

    def resolved_url(self) -> str:
        return (self.url or os.getenv("KUMC_ELASTICSEARCH_URL", "")).strip()

    def check(self) -> ComponentHealth:
        url = self.resolved_url()
        if not url:
            return ComponentHealth(
                name="elasticsearch",
                status="not_configured",
                detail="KUMC_ELASTICSEARCH_URL is empty.",
            )
        started = perf_counter()
        try:
            from urllib.request import urlopen

            with urlopen(url, timeout=3) as response:  # noqa: S310 - operator-configured URL health check
                status_code = getattr(response, "status", 200)
        except Exception as exc:
            return ComponentHealth(
                name="elasticsearch",
                status="unhealthy",
                detail=str(exc),
                latency_ms=round((perf_counter() - started) * 1000, 2),
            )
        return ComponentHealth(
            name="elasticsearch",
            status="healthy" if int(status_code) < 500 else "unhealthy",
            detail=f"status={status_code}",
            latency_ms=round((perf_counter() - started) * 1000, 2),
        )
