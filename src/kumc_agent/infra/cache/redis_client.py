from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from kumc_agent.config.schema import RedisSection
from kumc_agent.domain.models.health import ComponentHealth


@dataclass(frozen=True)
class RedisClient:
    config: RedisSection

    def is_configured(self) -> bool:
        return bool((self.config.url or "").strip())

    def client(self) -> Any:
        if not self.is_configured():
            raise RuntimeError("Redis URL is not configured.")
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - depends on deployment env
            raise RuntimeError("redis is not installed.") from exc

        return redis.Redis.from_url(
            self.config.url,
            socket_timeout=max(0.1, float(self.config.socket_timeout_seconds)),
            socket_connect_timeout=max(0.1, float(self.config.socket_timeout_seconds)),
        )

    def check(self) -> ComponentHealth:
        if not self.is_configured():
            return ComponentHealth(
                name="redis",
                status="not_configured",
                detail="KUMC_REDIS_URL is empty.",
            )
        started = perf_counter()
        try:
            self.client().ping()
        except Exception as exc:
            return ComponentHealth(
                name="redis",
                status="unhealthy",
                detail=str(exc),
                latency_ms=round((perf_counter() - started) * 1000, 2),
            )
        return ComponentHealth(
            name="redis",
            status="healthy",
            latency_ms=round((perf_counter() - started) * 1000, 2),
        )
