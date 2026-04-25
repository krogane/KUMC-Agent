from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from kumc_agent.config.schema import DatabaseSection
from kumc_agent.domain.models.health import ComponentHealth


class PostgresUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class PostgresClient:
    config: DatabaseSection

    def is_configured(self) -> bool:
        return bool((self.config.url or "").strip())

    def connect(self) -> Any:
        if not self.is_configured():
            raise PostgresUnavailable("PostgreSQL URL is not configured.")
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - depends on deployment env
            raise PostgresUnavailable("psycopg is not installed.") from exc

        timeout = max(1, int(self.config.connect_timeout_seconds))
        return psycopg.connect(
            self.config.url,
            connect_timeout=timeout,
            application_name=self.config.application_name,
        )

    def check(self) -> ComponentHealth:
        if not self.is_configured():
            return ComponentHealth(
                name="postgres",
                status="not_configured",
                detail="KUMC_DATABASE_URL is empty.",
            )
        started = perf_counter()
        try:
            with self.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("select 1")
                    cur.fetchone()
        except Exception as exc:
            return ComponentHealth(
                name="postgres",
                status="unhealthy",
                detail=str(exc),
                latency_ms=round((perf_counter() - started) * 1000, 2),
            )
        return ComponentHealth(
            name="postgres",
            status="healthy",
            latency_ms=round((perf_counter() - started) * 1000, 2),
        )
