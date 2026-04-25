from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

from kumc_agent.domain.models.job import JobRecord
from kumc_agent.infra.cache.redis_client import RedisClient
from kumc_agent.infra.database.postgres import PostgresClient


class JobLifecycleRepository(Protocol):
    def start(self, job_type: str, *, metadata: dict[str, object] | None = None) -> JobRecord:
        ...

    def complete(self, job: JobRecord, *, metadata: dict[str, object] | None = None) -> JobRecord:
        ...

    def fail(self, job: JobRecord, error: str, *, metadata: dict[str, object] | None = None) -> JobRecord:
        ...


@dataclass(frozen=True)
class FileJobLifecycleRepository:
    path: Path

    def _append(self, job: JobRecord) -> JobRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fw:
            fw.write(json.dumps(asdict(job), ensure_ascii=False, default=str) + "\n")
        return job

    def start(self, job_type: str, *, metadata: dict[str, object] | None = None) -> JobRecord:
        job = JobRecord(
            job_type=job_type,
            status="running",
            started_at=datetime.now(UTC),
            metadata=dict(metadata or {}),
        )
        return self._append(job)

    def complete(self, job: JobRecord, *, metadata: dict[str, object] | None = None) -> JobRecord:
        updated = replace(
            job,
            status="succeeded",
            finished_at=datetime.now(UTC),
            metadata={**job.metadata, **dict(metadata or {})},
        )
        return self._append(updated)

    def fail(self, job: JobRecord, error: str, *, metadata: dict[str, object] | None = None) -> JobRecord:
        updated = replace(
            job,
            status="failed",
            finished_at=datetime.now(UTC),
            error=error,
            metadata={**job.metadata, **dict(metadata or {})},
        )
        return self._append(updated)


@dataclass(frozen=True)
class RedisJobLifecycleRepository(FileJobLifecycleRepository):
    redis_client: RedisClient

    def _append(self, job: JobRecord) -> JobRecord:
        payload = json.dumps(asdict(job), ensure_ascii=False, default=str)
        client = self.redis_client.client()
        client.hset(f"kumc-agent:job:{job.job_id}", mapping={"payload": payload})
        client.rpush("kumc-agent:jobs", payload)
        return job


@dataclass(frozen=True)
class PostgresJobLifecycleRepository(FileJobLifecycleRepository):
    postgres: PostgresClient

    def _append(self, job: JobRecord) -> JobRecord:
        sql = """
            insert into job_runs (
              job_id,
              job_type,
              status,
              started_at,
              finished_at,
              error,
              metadata
            )
            values (%s, %s, %s, %s, %s, %s, %s::jsonb)
            on conflict (job_id) do update set
              status = excluded.status,
              finished_at = excluded.finished_at,
              error = excluded.error,
              metadata = excluded.metadata
        """
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        job.job_id,
                        job.job_type,
                        job.status,
                        job.started_at,
                        job.finished_at,
                        job.error,
                        json.dumps(job.metadata, ensure_ascii=False),
                    ),
                )
            conn.commit()
        return job


def build_job_lifecycle_repository(
    *,
    postgres: PostgresClient | None = None,
    redis_client: RedisClient,
    fallback_path: Path,
) -> JobLifecycleRepository:
    if postgres is not None and postgres.is_configured():
        return PostgresJobLifecycleRepository(path=fallback_path, postgres=postgres)
    if redis_client.is_configured():
        return RedisJobLifecycleRepository(path=fallback_path, redis_client=redis_client)
    return FileJobLifecycleRepository(path=fallback_path)
