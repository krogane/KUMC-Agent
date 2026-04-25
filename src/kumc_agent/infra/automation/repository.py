from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Protocol

from kumc_agent.domain.models.automation import (
    ActionSpecRef,
    AutomationRule,
    AutomationRun,
    ConditionSpec,
    TriggerSpec,
)
from kumc_agent.infra.database.postgres import PostgresClient

_MIN_DT = datetime.min.replace(tzinfo=UTC)


class AutomationRepository(Protocol):
    def save_rule(self, rule: AutomationRule) -> AutomationRule:
        ...

    def get_rule(self, rule_id: str) -> AutomationRule | None:
        ...

    def list_rules(self) -> list[AutomationRule]:
        ...

    def save_run(self, run: AutomationRun) -> AutomationRun:
        ...

    def get_run_by_idempotency_key(self, idempotency_key: str) -> AutomationRun | None:
        ...

    def list_runs(self, *, rule_id: str | None = None) -> list[AutomationRun]:
        ...


@dataclass(frozen=True)
class FileAutomationRepository:
    root_dir: Path

    def save_rule(self, rule: AutomationRule) -> AutomationRule:
        now = datetime.now(UTC)
        stored = replace(
            rule,
            created_at=rule.created_at or now,
            updated_at=now,
        )
        _append_jsonl(self.root_dir / "automation_rules.jsonl", _rule_payload(stored))
        return stored

    def get_rule(self, rule_id: str) -> AutomationRule | None:
        return _latest_by_id(self.root_dir / "automation_rules.jsonl", _rule_from_payload).get(rule_id)

    def list_rules(self) -> list[AutomationRule]:
        rules = list(
            _latest_by_id(
                self.root_dir / "automation_rules.jsonl",
                _rule_from_payload,
            ).values()
        )
        return sorted(rules, key=lambda rule: (rule.name.lower(), rule.id))

    def save_run(self, run: AutomationRun) -> AutomationRun:
        stored = replace(run, created_at=run.created_at or datetime.now(UTC))
        _append_jsonl(self.root_dir / "automation_runs.jsonl", _run_payload(stored))
        return stored

    def get_run_by_idempotency_key(self, idempotency_key: str) -> AutomationRun | None:
        for run in reversed(
            _read_jsonl(self.root_dir / "automation_runs.jsonl", _run_from_payload)
        ):
            if run.idempotency_key == idempotency_key:
                return run
        return None

    def list_runs(self, *, rule_id: str | None = None) -> list[AutomationRun]:
        runs = _read_jsonl(self.root_dir / "automation_runs.jsonl", _run_from_payload)
        if rule_id:
            runs = [run for run in runs if run.rule_id == rule_id]
        return sorted(runs, key=lambda run: run.created_at or _MIN_DT, reverse=True)


@dataclass(frozen=True)
class PostgresAutomationRepository:
    postgres: PostgresClient

    def save_rule(self, rule: AutomationRule) -> AutomationRule:
        now = datetime.now(UTC)
        stored = replace(
            rule,
            created_at=rule.created_at or now,
            updated_at=now,
        )
        payload = _rule_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into automation_rules (
                      id, name, enabled, trigger, conditions, actions, mode,
                      risk_level, created_by_user_id, approved_by_user_id,
                      last_run_at, next_run_at, metadata, created_at, updated_at
                    )
                    values (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                      %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    on conflict (id) do update set
                      name = excluded.name,
                      enabled = excluded.enabled,
                      trigger = excluded.trigger,
                      conditions = excluded.conditions,
                      actions = excluded.actions,
                      mode = excluded.mode,
                      risk_level = excluded.risk_level,
                      approved_by_user_id = excluded.approved_by_user_id,
                      last_run_at = excluded.last_run_at,
                      next_run_at = excluded.next_run_at,
                      metadata = excluded.metadata,
                      updated_at = excluded.updated_at
                    """,
                    (
                        payload["id"],
                        payload["name"],
                        payload["enabled"],
                        json.dumps(payload["trigger"], ensure_ascii=False, default=str),
                        json.dumps(payload["conditions"], ensure_ascii=False, default=str),
                        json.dumps(payload["actions"], ensure_ascii=False, default=str),
                        payload["mode"],
                        payload["risk_level"],
                        payload["created_by_user_id"],
                        payload["approved_by_user_id"],
                        _parse_datetime(payload["last_run_at"]),
                        _parse_datetime(payload["next_run_at"]),
                        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                        _parse_datetime(payload["created_at"]),
                        _parse_datetime(payload["updated_at"]),
                    ),
                )
            conn.commit()
        return stored

    def get_rule(self, rule_id: str) -> AutomationRule | None:
        rows = self._fetch("select * from automation_rules where id = %s", (rule_id,))
        return _rule_from_row(rows[0]) if rows else None

    def list_rules(self) -> list[AutomationRule]:
        rows = self._fetch("select * from automation_rules order by name asc, id asc", ())
        return [_rule_from_row(row) for row in rows]

    def save_run(self, run: AutomationRun) -> AutomationRun:
        stored = replace(run, created_at=run.created_at or datetime.now(UTC))
        payload = _run_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into automation_runs (
                      id, rule_id, trigger_key, mode, status, idempotency_key,
                      action_plan, warnings, metadata, created_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                    on conflict (idempotency_key) do nothing
                    """,
                    (
                        payload["id"],
                        payload["rule_id"],
                        payload["trigger_key"],
                        payload["mode"],
                        payload["status"],
                        payload["idempotency_key"],
                        json.dumps(payload["action_plan"], ensure_ascii=False, default=str),
                        json.dumps(payload["warnings"], ensure_ascii=False, default=str),
                        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                        _parse_datetime(payload["created_at"]),
                    ),
                )
            conn.commit()
        return self.get_run_by_idempotency_key(stored.idempotency_key) or stored

    def get_run_by_idempotency_key(self, idempotency_key: str) -> AutomationRun | None:
        rows = self._fetch(
            "select * from automation_runs where idempotency_key = %s",
            (idempotency_key,),
        )
        return _run_from_row(rows[0]) if rows else None

    def list_runs(self, *, rule_id: str | None = None) -> list[AutomationRun]:
        if rule_id:
            rows = self._fetch(
                "select * from automation_runs where rule_id = %s order by created_at desc",
                (rule_id,),
            )
        else:
            rows = self._fetch("select * from automation_runs order by created_at desc", ())
        return [_run_from_row(row) for row in rows]

    def _fetch(self, sql: str, params: tuple[object, ...]) -> list[dict[str, Any]]:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                columns = [column.name for column in cur.description or []]
                return [dict(zip(columns, row, strict=False)) for row in cur.fetchall()]


def build_automation_repository(
    *,
    postgres: PostgresClient,
    fallback_dir: Path,
) -> AutomationRepository:
    if postgres.is_configured():
        return PostgresAutomationRepository(postgres=postgres)
    return FileAutomationRepository(root_dir=fallback_dir)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fw:
        fw.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path, factory) -> list[Any]:
    if not path.exists():
        return []
    records: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(factory(json.loads(line)))
    return records


def _latest_by_id(path: Path, factory) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for item in _read_jsonl(path, factory):
        latest[item.id] = item
    return latest


def _rule_payload(rule: AutomationRule) -> dict[str, object]:
    return {
        "id": rule.id,
        "name": rule.name,
        "enabled": rule.enabled,
        "trigger": asdict(rule.trigger),
        "conditions": [asdict(condition) for condition in rule.conditions],
        "actions": [asdict(action) for action in rule.actions],
        "mode": rule.mode,
        "risk_level": rule.risk_level,
        "created_by_user_id": rule.created_by_user_id,
        "approved_by_user_id": rule.approved_by_user_id,
        "last_run_at": rule.last_run_at,
        "next_run_at": rule.next_run_at,
        "metadata": dict(rule.metadata),
        "created_at": rule.created_at,
        "updated_at": rule.updated_at,
    }


def _run_payload(run: AutomationRun) -> dict[str, object]:
    return {
        "id": run.id,
        "rule_id": run.rule_id,
        "trigger_key": run.trigger_key,
        "mode": run.mode,
        "status": run.status,
        "idempotency_key": run.idempotency_key,
        "action_plan": [dict(item) for item in run.action_plan],
        "warnings": list(run.warnings),
        "metadata": dict(run.metadata),
        "created_at": run.created_at,
    }


def _rule_from_payload(payload: dict[str, Any]) -> AutomationRule:
    return AutomationRule(
        id=str(payload["id"]),
        name=str(payload["name"]),
        enabled=bool(payload["enabled"]),
        trigger=_trigger_from_payload(payload.get("trigger", {})),
        conditions=tuple(
            ConditionSpec(**condition)
            for condition in payload.get("conditions", [])
            if isinstance(condition, dict)
        ),
        actions=tuple(
            _action_from_payload(action)
            for action in payload.get("actions", [])
            if isinstance(action, dict)
        ),
        mode=str(payload.get("mode", "dry_run")),
        risk_level=str(payload.get("risk_level", "low")),
        created_by_user_id=str(payload.get("created_by_user_id", "")),
        approved_by_user_id=str(payload.get("approved_by_user_id", "")),
        last_run_at=_parse_datetime(payload.get("last_run_at")),
        next_run_at=_parse_datetime(payload.get("next_run_at")),
        metadata=dict(payload.get("metadata", {}) or {}),
        created_at=_parse_datetime(payload.get("created_at")),
        updated_at=_parse_datetime(payload.get("updated_at")),
    )


def _run_from_payload(payload: dict[str, Any]) -> AutomationRun:
    return AutomationRun(
        id=str(payload["id"]),
        rule_id=str(payload["rule_id"]),
        trigger_key=str(payload.get("trigger_key", "")),
        mode=str(payload.get("mode", "dry_run")),
        status=str(payload.get("status", "")),
        idempotency_key=str(payload.get("idempotency_key", "")),
        action_plan=tuple(
            dict(item)
            for item in payload.get("action_plan", [])
            if isinstance(item, dict)
        ),
        warnings=tuple(str(item) for item in payload.get("warnings", [])),
        metadata=dict(payload.get("metadata", {}) or {}),
        created_at=_parse_datetime(payload.get("created_at")),
    )


def _rule_from_row(row: dict[str, Any]) -> AutomationRule:
    return _rule_from_payload(dict(row))


def _run_from_row(row: dict[str, Any]) -> AutomationRun:
    return _run_from_payload(dict(row))


def _trigger_from_payload(payload: dict[str, Any]) -> TriggerSpec:
    return TriggerSpec(
        kind=str(payload.get("kind", "manual")),
        params=dict(payload.get("params", {}) or {}),
    )


def _action_from_payload(payload: dict[str, Any]) -> ActionSpecRef:
    return ActionSpecRef(
        action_type=str(payload.get("action_type", "")),
        target=str(payload.get("target", "")),
        payload=dict(payload.get("payload", {}) or {}),
        risk_level=str(payload.get("risk_level", "low")),
        approval_required=bool(payload.get("approval_required", False)),
    )


def _parse_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
