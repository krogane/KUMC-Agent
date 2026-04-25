from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

from kumc_agent.domain.models.minecraft import MinecraftDryRun, ServerOperation
from kumc_agent.infra.database.postgres import PostgresClient


class ServerOperationRepository(Protocol):
    def save(self, operation: ServerOperation) -> ServerOperation:
        ...

    def get(self, operation_id: str) -> ServerOperation | None:
        ...

    def list(self, *, status: str | None = None) -> list[ServerOperation]:
        ...


@dataclass(frozen=True)
class FileServerOperationRepository:
    root_dir: Path

    def save(self, operation: ServerOperation) -> ServerOperation:
        stored = _touch(operation)
        _append_jsonl(self.root_dir / "server_operations.jsonl", _operation_payload(stored))
        return stored

    def get(self, operation_id: str) -> ServerOperation | None:
        return self._latest().get(operation_id)

    def list(self, *, status: str | None = None) -> list[ServerOperation]:
        operations = list(self._latest().values())
        if status:
            operations = [operation for operation in operations if operation.status == status]
        return sorted(
            operations,
            key=lambda operation: operation.created_at or datetime.min.replace(tzinfo=UTC),
        )

    def _latest(self) -> dict[str, ServerOperation]:
        latest: dict[str, ServerOperation] = {}
        path = self.root_dir / "server_operations.jsonl"
        if not path.exists():
            return latest
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                if not line.strip():
                    continue
                operation = _operation_from_payload(json.loads(line))
                latest[operation.id] = operation
        return latest


@dataclass(frozen=True)
class PostgresServerOperationRepository:
    postgres: PostgresClient

    def save(self, operation: ServerOperation) -> ServerOperation:
        stored = _touch(operation)
        payload = _operation_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into server_operations (
                      id, server_name, operation, requested_by_user_id,
                      approved_by_user_ids, status, risk_level, action_run_id,
                      dry_run, metadata, created_at, updated_at
                    )
                    values (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                    on conflict (id) do update set
                      approved_by_user_ids = excluded.approved_by_user_ids,
                      status = excluded.status,
                      action_run_id = excluded.action_run_id,
                      dry_run = excluded.dry_run,
                      metadata = excluded.metadata,
                      updated_at = excluded.updated_at
                    """,
                    (
                        payload["id"],
                        payload["server_name"],
                        payload["operation"],
                        payload["requested_by_user_id"],
                        json.dumps(payload["approved_by_user_ids"], ensure_ascii=False),
                        payload["status"],
                        payload["risk_level"],
                        payload["action_run_id"],
                        json.dumps(payload["dry_run"], ensure_ascii=False, default=str),
                        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                        payload["created_at"],
                        payload["updated_at"],
                    ),
                )
            conn.commit()
        return stored

    def get(self, operation_id: str) -> ServerOperation | None:
        rows = self._fetch("select * from server_operations where id = %s", (operation_id,))
        return _operation_from_row(rows[0]) if rows else None

    def list(self, *, status: str | None = None) -> list[ServerOperation]:
        if status:
            rows = self._fetch(
                "select * from server_operations where status = %s order by created_at asc",
                (status,),
            )
        else:
            rows = self._fetch("select * from server_operations order by created_at asc", ())
        return [_operation_from_row(row) for row in rows]

    def _fetch(self, sql: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())


def build_server_operation_repository(
    *,
    postgres: PostgresClient,
    fallback_dir: Path,
) -> ServerOperationRepository:
    if postgres.is_configured():
        return PostgresServerOperationRepository(postgres=postgres)
    return FileServerOperationRepository(root_dir=fallback_dir)


def _touch(operation: ServerOperation) -> ServerOperation:
    now = datetime.now(UTC)
    return replace(
        operation,
        created_at=operation.created_at or now,
        updated_at=now,
    )


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fw:
        fw.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _operation_payload(operation: ServerOperation) -> dict[str, object]:
    return {
        "id": operation.id,
        "server_name": operation.server_name,
        "operation": operation.operation,
        "requested_by_user_id": operation.requested_by_user_id,
        "approved_by_user_ids": list(operation.approved_by_user_ids),
        "status": operation.status,
        "risk_level": operation.risk_level,
        "action_run_id": operation.action_run_id,
        "dry_run": asdict(operation.dry_run) if operation.dry_run else None,
        "metadata": dict(operation.metadata),
        "created_at": operation.created_at,
        "updated_at": operation.updated_at,
    }


def _operation_from_payload(payload: dict[str, object]) -> ServerOperation:
    dry_run_payload = _json(payload.get("dry_run"))
    return ServerOperation(
        id=str(payload["id"]),
        server_name=str(payload["server_name"]),
        operation=str(payload["operation"]),
        requested_by_user_id=str(payload.get("requested_by_user_id") or ""),
        approved_by_user_ids=tuple(
            str(item) for item in _json(payload.get("approved_by_user_ids") or [])
        ),
        status=str(payload.get("status") or "waiting_approval"),
        risk_level=str(payload.get("risk_level") or "medium"),
        action_run_id=payload.get("action_run_id") and str(payload["action_run_id"]),
        dry_run=_dry_run_from_payload(dry_run_payload) if dry_run_payload else None,
        metadata=dict(_json(payload.get("metadata") or {})),
        created_at=_dt(payload.get("created_at")),
        updated_at=_dt(payload.get("updated_at")),
    )


def _operation_from_row(row: tuple[object, ...]) -> ServerOperation:
    return _operation_from_payload(
        {
            "id": row[0],
            "server_name": row[1],
            "operation": row[2],
            "requested_by_user_id": row[3],
            "approved_by_user_ids": row[4],
            "status": row[5],
            "risk_level": row[6],
            "action_run_id": row[7],
            "dry_run": row[8],
            "metadata": row[9],
            "created_at": row[10],
            "updated_at": row[11],
        }
    )


def _dry_run_from_payload(payload: object) -> MinecraftDryRun:
    data = dict(_json(payload) or {})
    return MinecraftDryRun(
        operation=str(data.get("operation") or ""),
        server_name=str(data.get("server_name") or ""),
        args={str(key): str(value) for key, value in dict(data.get("args") or {}).items()},
        risk_level=str(data.get("risk_level") or "low"),
        approval_policy=str(data.get("approval_policy") or "self"),
        impact=str(data.get("impact") or ""),
        expected_downtime=str(data.get("expected_downtime") or "none"),
        rollback=str(data.get("rollback") or ""),
        command_preview=tuple(str(item) for item in data.get("command_preview") or []),
        warnings=tuple(str(item) for item in data.get("warnings") or []),
        execution_allowed=bool(data.get("execution_allowed", False)),
    )


def _json(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _dt(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    return datetime.fromisoformat(str(value))
