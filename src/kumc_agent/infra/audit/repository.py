from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

from kumc_agent.domain.models.audit import AuditEvent
from kumc_agent.infra.database.postgres import PostgresClient


class AuditLogRepository(Protocol):
    def append(self, event: AuditEvent) -> AuditEvent:
        ...


@dataclass(frozen=True)
class FileAuditLogRepository:
    path: Path

    def append(self, event: AuditEvent) -> AuditEvent:
        stored = event.with_created_at(event.created_at or datetime.now(UTC))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fw:
            fw.write(json.dumps(asdict(stored), ensure_ascii=False, default=str) + "\n")
        return stored


@dataclass(frozen=True)
class PostgresAuditLogRepository:
    client: PostgresClient

    def append(self, event: AuditEvent) -> AuditEvent:
        stored = event.with_created_at(event.created_at or datetime.now(UTC))
        sql = """
            insert into audit_logs (
              event_id,
              action,
              actor_id,
              actor_type,
              target,
              outcome,
              risk_level,
              trace_id,
              metadata,
              created_at
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
        """
        with self.client.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        stored.event_id,
                        stored.action,
                        stored.actor_id,
                        stored.actor_type,
                        stored.target,
                        stored.outcome,
                        stored.risk_level,
                        stored.trace_id,
                        json.dumps(stored.metadata, ensure_ascii=False),
                        stored.created_at,
                    ),
                )
            conn.commit()
        return stored


def build_audit_repository(
    *,
    postgres: PostgresClient,
    fallback_path: Path,
) -> AuditLogRepository:
    if postgres.is_configured():
        return PostgresAuditLogRepository(client=postgres)
    return FileAuditLogRepository(path=fallback_path)
