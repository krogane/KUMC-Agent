from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Protocol

from kumc_agent.domain.models.operations import (
    ActionRun,
    Asset,
    EvalRun,
    IndexingRun,
    MemberProfile,
    WorkflowCandidate,
    WorkflowRun,
)
from kumc_agent.infra.database.postgres import PostgresClient


class OperationsRepository(Protocol):
    def save_workflow_run(self, run: WorkflowRun) -> WorkflowRun:
        ...

    def save_workflow_candidate(self, candidate: WorkflowCandidate) -> WorkflowCandidate:
        ...

    def list_workflow_candidates(
        self,
        *,
        candidate_type: str | None = None,
        status: str | None = None,
    ) -> list[WorkflowCandidate]:
        ...

    def save_asset(self, asset: Asset) -> Asset:
        ...

    def list_assets(self, *, query: str = "") -> list[Asset]:
        ...

    def get_asset(self, asset_id: str) -> Asset | None:
        ...

    def save_member_profile(self, profile: MemberProfile) -> MemberProfile:
        ...

    def search_member_profiles(self, *, query: str) -> list[MemberProfile]:
        ...

    def list_member_profiles(self) -> list[MemberProfile]:
        ...

    def save_action_run(self, run: ActionRun) -> ActionRun:
        ...

    def save_indexing_run(self, run: IndexingRun) -> IndexingRun:
        ...

    def save_eval_run(self, run: EvalRun) -> EvalRun:
        ...


@dataclass(frozen=True)
class FileOperationsRepository:
    root_dir: Path

    def save_workflow_run(self, run: WorkflowRun) -> WorkflowRun:
        stored = _touch(run)
        _append_jsonl(self.root_dir / "workflow_runs.jsonl", _payload(stored))
        return stored

    def save_workflow_candidate(self, candidate: WorkflowCandidate) -> WorkflowCandidate:
        stored = _touch(candidate)
        _append_jsonl(self.root_dir / "workflow_candidates.jsonl", _payload(stored))
        return stored

    def list_workflow_candidates(
        self,
        *,
        candidate_type: str | None = None,
        status: str | None = None,
    ) -> list[WorkflowCandidate]:
        items = _latest_by_id(self.root_dir / "workflow_candidates.jsonl", _workflow_candidate_from_payload)
        candidates = list(items.values())
        if candidate_type:
            candidates = [item for item in candidates if item.candidate_type == candidate_type]
        if status:
            candidates = [item for item in candidates if item.status == status]
        return sorted(candidates, key=lambda item: item.created_at or _MIN_DT)

    def save_asset(self, asset: Asset) -> Asset:
        stored = _touch(asset)
        _append_jsonl(self.root_dir / "assets.jsonl", _payload(stored))
        return stored

    def list_assets(self, *, query: str = "") -> list[Asset]:
        assets = list(_latest_by_id(self.root_dir / "assets.jsonl", _asset_from_payload).values())
        needle = query.strip().lower()
        if needle:
            assets = [
                asset
                for asset in assets
                if needle in _asset_search_text(asset).lower()
            ]
        return sorted(assets, key=lambda item: item.created_at or _MIN_DT)

    def get_asset(self, asset_id: str) -> Asset | None:
        return _latest_by_id(self.root_dir / "assets.jsonl", _asset_from_payload).get(asset_id)

    def save_member_profile(self, profile: MemberProfile) -> MemberProfile:
        stored = _touch(profile)
        _append_jsonl(self.root_dir / "member_profiles.jsonl", _payload(stored))
        return stored

    def search_member_profiles(self, *, query: str) -> list[MemberProfile]:
        profiles = list(
            _latest_by_id(
                self.root_dir / "member_profiles.jsonl",
                _member_profile_from_payload,
            ).values()
        )
        needle = query.strip().lower()
        if needle:
            profiles = [
                profile
                for profile in profiles
                if needle
                in " ".join(
                    (
                        profile.display_name,
                        profile.discord_user_id,
                        " ".join(profile.roles),
                        " ".join(profile.skills),
                        " ".join(profile.interests),
                        " ".join(profile.past_assignments),
                        " ".join(_evidence_search_text(item) for item in profile.evidence),
                    )
                ).lower()
            ]
        return sorted(profiles, key=lambda item: item.display_name or item.id)

    def list_member_profiles(self) -> list[MemberProfile]:
        profiles = list(
            _latest_by_id(
                self.root_dir / "member_profiles.jsonl",
                _member_profile_from_payload,
            ).values()
        )
        return sorted(profiles, key=lambda item: item.display_name or item.id)

    def save_action_run(self, run: ActionRun) -> ActionRun:
        stored = _touch(run)
        _append_jsonl(self.root_dir / "action_runs.jsonl", _payload(stored))
        return stored

    def save_indexing_run(self, run: IndexingRun) -> IndexingRun:
        stored = _touch(run)
        _append_jsonl(self.root_dir / "indexing_runs.jsonl", _payload(stored))
        return stored

    def save_eval_run(self, run: EvalRun) -> EvalRun:
        stored = _touch(run)
        _append_jsonl(self.root_dir / "eval_runs.jsonl", _payload(stored))
        return stored


@dataclass(frozen=True)
class PostgresOperationsRepository(FileOperationsRepository):
    postgres: PostgresClient | None = None

    def _insert_payload(self, table: str, payload: dict[str, object]) -> None:
        if self.postgres is None:
            return
        columns = tuple(payload)
        placeholders = ", ".join(
            "%s::jsonb" if column in _JSON_COLUMNS else "%s"
            for column in columns
        )
        assignments = ", ".join(
            f"{column} = excluded.{column}"
            for column in columns
            if column not in {"id", "created_at"}
        )
        sql = (
            f"insert into {table} ({', '.join(columns)}) values ({placeholders}) "
            f"on conflict (id) do update set {assignments}"
        )
        values = tuple(_sql_value(payload[column]) for column in columns)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
            conn.commit()

    def save_workflow_run(self, run: WorkflowRun) -> WorkflowRun:
        stored = _touch(run)
        self._insert_payload("workflow_runs", _workflow_run_payload(stored))
        return stored

    def save_workflow_candidate(self, candidate: WorkflowCandidate) -> WorkflowCandidate:
        stored = _touch(candidate)
        self._insert_payload("workflow_candidates", _workflow_candidate_payload(stored))
        return stored

    def save_asset(self, asset: Asset) -> Asset:
        stored = _touch(asset)
        self._insert_payload("assets", _asset_payload(stored))
        return stored

    def save_member_profile(self, profile: MemberProfile) -> MemberProfile:
        stored = _touch(profile)
        self._insert_payload("member_profiles", _member_profile_payload(stored))
        return stored

    def save_action_run(self, run: ActionRun) -> ActionRun:
        stored = _touch(run)
        self._insert_payload("action_runs", _action_run_payload(stored))
        return stored

    def save_indexing_run(self, run: IndexingRun) -> IndexingRun:
        stored = _touch(run)
        self._insert_payload("indexing_runs", _indexing_run_payload(stored))
        return stored

    def save_eval_run(self, run: EvalRun) -> EvalRun:
        stored = _touch(run)
        self._insert_payload("eval_runs", _eval_run_payload(stored))
        return stored


_MIN_DT = datetime.min.replace(tzinfo=UTC)
_JSON_COLUMNS = {
    "access_scope",
    "candidates",
    "drafts",
    "evidence",
    "input",
    "interests",
    "metadata",
    "metrics",
    "output",
    "past_assignments",
    "payload",
    "request_payload",
    "result_payload",
    "roles",
    "skills",
    "validation_result",
}


def build_operations_repository(
    *,
    postgres: PostgresClient,
    fallback_dir: Path,
) -> OperationsRepository:
    if postgres.is_configured():
        return PostgresOperationsRepository(root_dir=fallback_dir, postgres=postgres)
    return FileOperationsRepository(root_dir=fallback_dir)


def _touch(item: Any) -> Any:
    now = datetime.now(UTC)
    return replace(
        item,
        created_at=getattr(item, "created_at", None) or now,
        updated_at=now,
    )


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fw:
        fw.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path, loader: Any) -> list[Any]:
    if not path.exists():
        return []
    out: list[Any] = []
    with path.open("r", encoding="utf-8") as fr:
        for line in fr:
            if line.strip():
                out.append(loader(json.loads(line)))
    return out


def _latest_by_id(path: Path, loader: Any) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for item in _read_jsonl(path, loader):
        latest[item.id] = item
    return latest


def _payload(item: Any) -> dict[str, object]:
    if isinstance(item, WorkflowRun):
        return _workflow_run_payload(item)
    if isinstance(item, WorkflowCandidate):
        return _workflow_candidate_payload(item)
    if isinstance(item, Asset):
        return _asset_payload(item)
    if isinstance(item, MemberProfile):
        return _member_profile_payload(item)
    if isinstance(item, ActionRun):
        return _action_run_payload(item)
    if isinstance(item, IndexingRun):
        return _indexing_run_payload(item)
    if isinstance(item, EvalRun):
        return _eval_run_payload(item)
    return asdict(item)


def _workflow_run_payload(item: WorkflowRun) -> dict[str, object]:
    return _base_payload(item) | {
        "workflow_id": item.workflow_id,
        "trigger": item.trigger,
        "actor_user_id": item.actor_user_id,
        "guild_id": item.guild_id,
        "input": item.input,
        "candidates": list(item.candidates),
        "drafts": list(item.drafts),
        "validation_result": item.validation_result,
        "approval_required": item.approval_required,
        "status": item.status,
        "error": item.error,
        "audit_log_id": item.audit_log_id,
        "metadata": item.metadata,
    }


def _workflow_run_from_payload(payload: dict[str, object]) -> WorkflowRun:
    return WorkflowRun(
        id=str(payload["id"]),
        workflow_id=str(payload.get("workflow_id") or ""),
        trigger=str(payload.get("trigger") or "manual"),
        actor_user_id=str(payload.get("actor_user_id") or ""),
        guild_id=str(payload.get("guild_id") or ""),
        input=dict(_json(payload.get("input") or {})),
        candidates=tuple(str(item) for item in _json(payload.get("candidates") or [])),
        drafts=tuple(str(item) for item in _json(payload.get("drafts") or [])),
        validation_result=dict(_json(payload.get("validation_result") or {})),
        approval_required=bool(payload.get("approval_required", False)),
        status=str(payload.get("status") or "running"),
        error=str(payload.get("error") or ""),
        audit_log_id=str(payload.get("audit_log_id") or ""),
        metadata=dict(_json(payload.get("metadata") or {})),
        created_at=_dt_from(payload.get("created_at")),
        updated_at=_dt_from(payload.get("updated_at")),
    )


def _workflow_candidate_payload(item: WorkflowCandidate) -> dict[str, object]:
    return _base_payload(item) | {
        "candidate_type": item.candidate_type,
        "title": item.title,
        "payload": item.payload,
        "evidence": list(item.evidence),
        "confidence": item.confidence,
        "status": item.status,
        "created_by": item.created_by,
        "metadata": item.metadata,
    }


def _workflow_candidate_from_payload(payload: dict[str, object]) -> WorkflowCandidate:
    return WorkflowCandidate(
        id=str(payload["id"]),
        candidate_type=str(payload.get("candidate_type") or ""),
        title=str(payload.get("title") or ""),
        payload=dict(_json(payload.get("payload") or {})),
        evidence=tuple(dict(item) for item in _json(payload.get("evidence") or [])),
        confidence=str(payload.get("confidence") or "low"),
        status=str(payload.get("status") or "proposed"),
        created_by=str(payload.get("created_by") or "agent"),
        metadata=dict(_json(payload.get("metadata") or {})),
        created_at=_dt_from(payload.get("created_at")),
        updated_at=_dt_from(payload.get("updated_at")),
    )


def _asset_payload(item: Asset) -> dict[str, object]:
    return _base_payload(item) | {
        "source_kind": item.source_kind,
        "source_item_id": item.source_item_id,
        "title": item.title,
        "description": item.description,
        "uri": item.uri,
        "media_type": item.media_type,
        "captured_at": _dt(item.captured_at),
        "access_scope": item.access_scope,
        "rights_status": item.rights_status,
        "contains_people": item.contains_people,
        "metadata": item.metadata,
    }


def _asset_from_payload(payload: dict[str, object]) -> Asset:
    return Asset(
        id=str(payload["id"]),
        source_kind=str(payload.get("source_kind") or ""),
        source_item_id=str(payload.get("source_item_id") or ""),
        title=str(payload.get("title") or ""),
        description=str(payload.get("description") or ""),
        uri=str(payload.get("uri") or ""),
        media_type=str(payload.get("media_type") or "image"),
        captured_at=_dt_from(payload.get("captured_at")),
        access_scope=dict(_json(payload.get("access_scope") or {})),
        rights_status=str(payload.get("rights_status") or "unknown"),
        contains_people=bool(payload.get("contains_people", False)),
        metadata=dict(_json(payload.get("metadata") or {})),
        created_at=_dt_from(payload.get("created_at")),
        updated_at=_dt_from(payload.get("updated_at")),
    )


def _asset_search_text(asset: Asset) -> str:
    metadata = dict(asset.metadata or {})
    metadata_text = " ".join(
        str(metadata.get(key) or "")
        for key in (
            "caption",
            "ocr_text",
            "surrounding_text",
            "source_url",
            "source_label",
            "source_kind",
            "content_hash",
        )
    )
    return " ".join(
        (
            asset.title,
            asset.description,
            asset.source_kind,
            asset.source_item_id,
            asset.uri,
            metadata_text,
        )
    )


def _member_profile_payload(item: MemberProfile) -> dict[str, object]:
    return _base_payload(item) | {
        "display_name": item.display_name,
        "discord_user_id": item.discord_user_id,
        "roles": list(item.roles),
        "skills": list(item.skills),
        "interests": list(item.interests),
        "past_assignments": list(item.past_assignments),
        "evidence": list(item.evidence),
        "access_scope": item.access_scope,
        "metadata": item.metadata,
    }


def _member_profile_from_payload(payload: dict[str, object]) -> MemberProfile:
    return MemberProfile(
        id=str(payload["id"]),
        display_name=str(payload.get("display_name") or ""),
        discord_user_id=str(payload.get("discord_user_id") or ""),
        roles=tuple(str(item) for item in _json(payload.get("roles") or [])),
        skills=tuple(str(item) for item in _json(payload.get("skills") or [])),
        interests=tuple(str(item) for item in _json(payload.get("interests") or [])),
        past_assignments=tuple(str(item) for item in _json(payload.get("past_assignments") or [])),
        evidence=tuple(dict(item) for item in _json(payload.get("evidence") or [])),
        access_scope=dict(_json(payload.get("access_scope") or {})),
        metadata=dict(_json(payload.get("metadata") or {})),
        created_at=_dt_from(payload.get("created_at")),
        updated_at=_dt_from(payload.get("updated_at")),
    )


def _evidence_search_text(evidence: dict[str, Any]) -> str:
    return " ".join(
        str(evidence.get(key) or "")
        for key in ("source_type", "source_item_id", "chunk_id", "label", "quote")
    )


def _action_run_payload(item: ActionRun) -> dict[str, object]:
    return _base_payload(item) | {
        "action_type": item.action_type,
        "target": item.target,
        "actor_user_id": item.actor_user_id,
        "status": item.status,
        "risk_level": item.risk_level,
        "idempotency_key": item.idempotency_key,
        "request_payload": item.request_payload,
        "result_payload": item.result_payload,
        "error": item.error,
        "trace_id": item.trace_id,
        "metadata": item.metadata,
    }


def _indexing_run_payload(item: IndexingRun) -> dict[str, object]:
    return _base_payload(item) | {
        "source_kind": item.source_kind,
        "status": item.status,
        "seen": item.seen,
        "changed": item.changed,
        "skipped": item.skipped,
        "deleted": item.deleted,
        "error": item.error,
        "metadata": item.metadata,
    }


def _eval_run_payload(item: EvalRun) -> dict[str, object]:
    return _base_payload(item) | {
        "eval_set_id": item.eval_set_id,
        "status": item.status,
        "metrics": item.metrics,
        "metadata": item.metadata,
    }


def _base_payload(item: Any) -> dict[str, object]:
    return {
        "id": item.id,
        "created_at": _dt(getattr(item, "created_at", None)),
        "updated_at": _dt(getattr(item, "updated_at", None)),
    }


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dt_from(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _json(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _sql_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value
