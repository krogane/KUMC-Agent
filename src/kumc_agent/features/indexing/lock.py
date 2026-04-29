from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
from types import TracebackType
from typing import Protocol

from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.infra.cache.redis_client import RedisClient
from kumc_agent.infra.database.postgres import PostgresClient


@dataclass(frozen=True)
class IndexingLockResult:
    acquired: bool
    reason: str = ""
    metadata: dict[str, object] | None = None


class IndexingLock(Protocol):
    def acquire(self, *, run_id: str) -> IndexingLockResult:
        ...

    def release(self) -> None:
        ...

    def refresh(self) -> None:
        ...


class FallbackIndexingLock:
    def __init__(self, locks: tuple[IndexingLock, ...]) -> None:
        self._locks = locks
        self._acquired: IndexingLock | None = None

    def acquire(self, *, run_id: str) -> IndexingLockResult:
        unavailable: list[dict[str, object]] = []
        for lock in self._locks:
            result = lock.acquire(run_id=run_id)
            if result.acquired:
                self._acquired = lock
                return result
            if result.reason.endswith("_unavailable"):
                unavailable.append({"reason": result.reason, "metadata": result.metadata or {}})
                continue
            return result
        return IndexingLockResult(
            acquired=False,
            reason="lock_backend_unavailable",
            metadata={"attempts": unavailable},
        )

    def release(self) -> None:
        if self._acquired is None:
            return
        self._acquired.release()
        self._acquired = None

    def refresh(self) -> None:
        if self._acquired is None:
            return
        self._acquired.refresh()


class RedisIndexingLock:
    def __init__(self, *, redis: RedisClient, ttl_minutes: int) -> None:
        self._redis = redis
        self._ttl_seconds = max(60, int(ttl_minutes) * 60)
        self._key = "kumc-agent:auto-index:lock"
        self._token = ""

    def acquire(self, *, run_id: str) -> IndexingLockResult:
        self._token = f"{run_id}:{os.getpid()}"
        try:
            client = self._redis.client()
            acquired = client.set(self._key, self._token, nx=True, ex=self._ttl_seconds)
        except Exception as exc:
            return IndexingLockResult(
                acquired=False,
                reason="redis_unavailable",
                metadata={"error": str(exc)},
            )
        if not acquired:
            return IndexingLockResult(
                acquired=False,
                reason="lock_already_held",
                metadata={"backend": "redis"},
            )
        return IndexingLockResult(acquired=True, metadata={"backend": "redis"})

    def release(self) -> None:
        if not self._token:
            return
        try:
            client = self._redis.client()
            if client.get(self._key) in {self._token, self._token.encode("utf-8")}:
                client.delete(self._key)
        except Exception:
            return

    def refresh(self) -> None:
        if not self._token:
            return
        try:
            client = self._redis.client()
            if client.get(self._key) in {self._token, self._token.encode("utf-8")}:
                client.expire(self._key, self._ttl_seconds)
        except Exception:
            return


class PostgresIndexingLock:
    def __init__(self, *, postgres: PostgresClient) -> None:
        self._postgres = postgres
        self._conn = None
        self._lock_key = _advisory_lock_key("kumc-agent:auto-index")

    def acquire(self, *, run_id: str) -> IndexingLockResult:
        try:
            self._conn = self._postgres.connect()
            with self._conn.cursor() as cur:
                cur.execute("select pg_try_advisory_lock(%s)", (self._lock_key,))
                row = cur.fetchone()
            acquired = bool(row and row[0])
        except Exception as exc:
            self._close()
            return IndexingLockResult(
                acquired=False,
                reason="postgres_unavailable",
                metadata={"error": str(exc), "run_id": run_id},
            )
        if not acquired:
            self._close()
            return IndexingLockResult(
                acquired=False,
                reason="lock_already_held",
                metadata={"backend": "postgres"},
            )
        return IndexingLockResult(acquired=True, metadata={"backend": "postgres"})

    def release(self) -> None:
        if self._conn is None:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute("select pg_advisory_unlock(%s)", (self._lock_key,))
            self._conn.commit()
        except Exception:
            pass
        finally:
            self._close()

    def refresh(self) -> None:
        return

    def _close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = None


class FileIndexingLock:
    def __init__(self, *, path: Path, ttl_minutes: int) -> None:
        self._path = path
        self._ttl = timedelta(minutes=max(1, int(ttl_minutes)))
        self._acquired = False

    def acquire(self, *, run_id: str) -> IndexingLockResult:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._clear_stale_lock()
        payload = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
        }
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(self._path), flags)
        except FileExistsError:
            return IndexingLockResult(
                acquired=False,
                reason="lock_already_held",
                metadata=self._read_payload(),
            )
        with os.fdopen(fd, "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False)
        self._acquired = True
        return IndexingLockResult(acquired=True)

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass
        self._acquired = False

    def refresh(self) -> None:
        if not self._acquired:
            return
        try:
            payload = self._read_payload()
            payload["refreshed_at"] = datetime.now(UTC).isoformat()
            self._path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.utime(self._path, None)
        except Exception:
            return

    def __enter__(self) -> FileIndexingLock:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def _clear_stale_lock(self) -> None:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        if datetime.now(UTC) - modified_at <= self._ttl:
            return
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass

    def _read_payload(self) -> dict[str, object]:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}


def build_indexing_lock(config: RuntimeConfig) -> FallbackIndexingLock:
    locks: list[IndexingLock] = []
    postgres = PostgresClient(config.infrastructure.database)
    redis = RedisClient(config.infrastructure.redis)
    if postgres.is_configured():
        locks.append(PostgresIndexingLock(postgres=postgres))
    if redis.is_configured():
        locks.append(
            RedisIndexingLock(
                redis=redis,
                ttl_minutes=config.scheduler.auto_index_lock_ttl_minutes,
            )
        )
    locks.append(
        FileIndexingLock(
            path=config.app.data_dir / "locks" / "auto_index.lock",
            ttl_minutes=config.scheduler.auto_index_lock_ttl_minutes,
        )
    )
    return FallbackIndexingLock(tuple(locks))


def _advisory_lock_key(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=True)
