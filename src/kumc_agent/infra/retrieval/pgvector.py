from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.domain.models.health import ComponentHealth
from kumc_agent.infra.database.postgres import PostgresClient


@dataclass(frozen=True)
class PgVectorAdapter:
    postgres: PostgresClient

    def check(self) -> ComponentHealth:
        if not self.postgres.is_configured():
            return ComponentHealth(
                name="pgvector",
                status="not_configured",
                detail="KUMC_DATABASE_URL is empty.",
            )
        try:
            with self.postgres.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("select 1 from pg_extension where extname = 'vector'")
                    installed = cur.fetchone() is not None
        except Exception as exc:
            return ComponentHealth(name="pgvector", status="unhealthy", detail=str(exc))
        if not installed:
            return ComponentHealth(
                name="pgvector",
                status="not_configured",
                detail="PostgreSQL extension 'vector' is not installed.",
            )
        return ComponentHealth(name="pgvector", status="healthy")
