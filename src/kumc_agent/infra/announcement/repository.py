from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

from kumc_agent.domain.models.announcement import AnnouncementDraft
from kumc_agent.domain.models.docgen import FactCheckFinding
from kumc_agent.domain.models.retrieval import Citation
from kumc_agent.infra.database.postgres import PostgresClient


class AnnouncementRepository(Protocol):
    def save(self, draft: AnnouncementDraft) -> AnnouncementDraft:
        ...

    def get(self, draft_id: str) -> AnnouncementDraft | None:
        ...

    def list(self, *, status: str | None = None) -> list[AnnouncementDraft]:
        ...


@dataclass(frozen=True)
class FileAnnouncementRepository:
    root_dir: Path

    def save(self, draft: AnnouncementDraft) -> AnnouncementDraft:
        stored = _touch(draft)
        _append_jsonl(self.root_dir / "announcements.jsonl", _draft_payload(stored))
        return stored

    def get(self, draft_id: str) -> AnnouncementDraft | None:
        return self._latest().get(draft_id)

    def list(self, *, status: str | None = None) -> list[AnnouncementDraft]:
        drafts = list(self._latest().values())
        if status:
            drafts = [draft for draft in drafts if draft.status == status]
        return sorted(drafts, key=lambda draft: draft.created_at or datetime.min.replace(tzinfo=UTC))

    def _latest(self) -> dict[str, AnnouncementDraft]:
        latest: dict[str, AnnouncementDraft] = {}
        path = self.root_dir / "announcements.jsonl"
        if not path.exists():
            return latest
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                if line.strip():
                    draft = _draft_from_payload(json.loads(line))
                    latest[draft.id] = draft
        return latest


@dataclass(frozen=True)
class PostgresAnnouncementRepository:
    postgres: PostgresClient

    def save(self, draft: AnnouncementDraft) -> AnnouncementDraft:
        stored = _touch(draft)
        payload = _draft_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into announcements (
                      id, title, body_markdown, medium, audience, status,
                      fact_checks, citations, created_by, metadata,
                      created_at, updated_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s, %s)
                    on conflict (id) do update set
                      title = excluded.title,
                      body_markdown = excluded.body_markdown,
                      medium = excluded.medium,
                      audience = excluded.audience,
                      status = excluded.status,
                      fact_checks = excluded.fact_checks,
                      citations = excluded.citations,
                      metadata = excluded.metadata,
                      updated_at = excluded.updated_at
                    """,
                    (
                        payload["id"],
                        payload["title"],
                        payload["body_markdown"],
                        payload["medium"],
                        payload["audience"],
                        payload["status"],
                        json.dumps(payload["fact_checks"], ensure_ascii=False, default=str),
                        json.dumps(payload["citations"], ensure_ascii=False, default=str),
                        payload["created_by"],
                        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                        payload["created_at"],
                        payload["updated_at"],
                    ),
                )
            conn.commit()
        return stored

    def get(self, draft_id: str) -> AnnouncementDraft | None:
        rows = self._fetch("select * from announcements where id = %s", (draft_id,))
        return _draft_from_row(rows[0]) if rows else None

    def list(self, *, status: str | None = None) -> list[AnnouncementDraft]:
        if status:
            rows = self._fetch(
                "select * from announcements where status = %s order by created_at asc",
                (status,),
            )
        else:
            rows = self._fetch("select * from announcements order by created_at asc", ())
        return [_draft_from_row(row) for row in rows]

    def _fetch(self, sql: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())


def build_announcement_repository(
    *,
    postgres: PostgresClient,
    fallback_dir: Path,
) -> AnnouncementRepository:
    if postgres.is_configured():
        return PostgresAnnouncementRepository(postgres=postgres)
    return FileAnnouncementRepository(root_dir=fallback_dir)


def _touch(draft: AnnouncementDraft) -> AnnouncementDraft:
    now = datetime.now(UTC)
    return replace(
        draft,
        created_at=draft.created_at or now,
        updated_at=now,
    )


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fw:
        fw.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _draft_payload(draft: AnnouncementDraft) -> dict[str, object]:
    return {
        "id": draft.id,
        "title": draft.title,
        "body_markdown": draft.body_markdown,
        "medium": draft.medium,
        "audience": draft.audience,
        "status": draft.status,
        "fact_checks": [finding.__dict__ for finding in draft.fact_checks],
        "citations": [citation.__dict__ for citation in draft.citations],
        "created_by": draft.created_by,
        "metadata": dict(draft.metadata),
        "created_at": draft.created_at,
        "updated_at": draft.updated_at,
    }


def _draft_from_payload(payload: dict[str, object]) -> AnnouncementDraft:
    fact_checks_payload = _json(payload.get("fact_checks") or [])
    citations_payload = _json(payload.get("citations") or [])
    return AnnouncementDraft(
        id=str(payload["id"]),
        title=str(payload["title"]),
        body_markdown=str(payload.get("body_markdown") or ""),
        medium=str(payload.get("medium") or "discord"),
        audience=str(payload.get("audience") or ""),
        status=str(payload.get("status") or "draft"),
        fact_checks=tuple(
            FactCheckFinding(
                kind=str(item.get("kind") or "unknown"),
                message=str(item.get("message") or ""),
                severity=str(item.get("severity") or "medium"),
            )
            for item in fact_checks_payload
        ),
        citations=tuple(
            Citation(
                source_item_id=str(item.get("source_item_id") or ""),
                chunk_id=str(item.get("chunk_id") or ""),
                label=str(item.get("label") or ""),
                url=str(item.get("url") or ""),
                quote=str(item.get("quote") or ""),
                score=float(item["score"]) if item.get("score") is not None else None,
                access_scope=dict(item.get("access_scope") or {}),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in citations_payload
        ),
        created_by=str(payload.get("created_by") or "agent"),
        metadata=dict(_json(payload.get("metadata") or {})),
        created_at=_dt(payload.get("created_at")),
        updated_at=_dt(payload.get("updated_at")),
    )


def _draft_from_row(row: tuple[object, ...]) -> AnnouncementDraft:
    return _draft_from_payload(
        {
            "id": row[0],
            "title": row[1],
            "body_markdown": row[2],
            "medium": row[3],
            "audience": row[4],
            "status": row[5],
            "fact_checks": row[6],
            "citations": row[7],
            "created_by": row[8],
            "metadata": row[9],
            "created_at": row[10],
            "updated_at": row[11],
        }
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
