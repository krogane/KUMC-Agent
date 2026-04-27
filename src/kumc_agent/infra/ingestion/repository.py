from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.secret import SecretFinding
from kumc_agent.domain.models.source import NormalizedDocument, SourceRawItem
from kumc_agent.features.indexing.change_detection import (
    SourceItemState,
    state_from_source_item_payload,
)
from kumc_agent.infra.database.postgres import PostgresClient
from kumc_agent.utils.hashing import stable_hash


class IngestionRepository(Protocol):
    def load_checksums(self, source_kind: str) -> dict[str, str]:
        ...

    def load_item_states(self, source_kind: str) -> dict[str, SourceItemState]:
        ...

    def save_item(
        self,
        *,
        raw: SourceRawItem,
        document: NormalizedDocument,
        chunks: list[Chunk],
        findings: list[SecretFinding],
        raw_object_key: str,
    ) -> None:
        ...

    def mark_deleted(
        self,
        *,
        source_kind: str,
        external_id: str,
        status: str = "deleted",
    ) -> None:
        ...


@dataclass(frozen=True)
class FileIngestionRepository:
    root_dir: Path

    def load_checksums(self, source_kind: str) -> dict[str, str]:
        return {
            external_id: state.checksum
            for external_id, state in self.load_item_states(source_kind).items()
        }

    def load_item_states(self, source_kind: str) -> dict[str, SourceItemState]:
        path = self.root_dir / "source_items.jsonl"
        states: dict[str, SourceItemState] = {}
        if not path.exists():
            return states
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                payload = json.loads(line)
                if payload.get("source_kind") != source_kind:
                    continue
                state = state_from_source_item_payload(payload)
                states[state.external_id] = state
        return states

    def save_item(
        self,
        *,
        raw: SourceRawItem,
        document: NormalizedDocument,
        chunks: list[Chunk],
        findings: list[SecretFinding],
        raw_object_key: str,
    ) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        _append_jsonl(
            self.root_dir / "source_items.jsonl",
            _source_item_payload(raw=raw, raw_object_key=raw_object_key),
        )
        _append_jsonl(self.root_dir / "documents.jsonl", _document_payload(document))
        for chunk in chunks:
            _append_jsonl(self.root_dir / "chunks.jsonl", _chunk_payload(chunk))
            for entry in _chunk_acl_entries(chunk):
                _append_jsonl(self.root_dir / "chunk_acl_entries.jsonl", entry)
        for finding in findings:
            _append_jsonl(self.root_dir / "secret_findings.jsonl", asdict(finding))

    def mark_deleted(
        self,
        *,
        source_kind: str,
        external_id: str,
        status: str = "deleted",
    ) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        _append_jsonl(
            self.root_dir / "source_deletes.jsonl",
            {
                "source_kind": source_kind,
                "external_id": external_id,
                "index_status": status,
                "deleted_at": datetime.now(UTC).isoformat(),
            },
        )


@dataclass(frozen=True)
class PostgresIngestionRepository:
    postgres: PostgresClient

    def load_checksums(self, source_kind: str) -> dict[str, str]:
        return {
            external_id: state.checksum
            for external_id, state in self.load_item_states(source_kind).items()
        }

    def load_item_states(self, source_kind: str) -> dict[str, SourceItemState]:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select external_id, checksum, metadata, access_scope, index_status
                    from source_items
                    where source_kind = %s
                    """,
                    (source_kind,),
                )
                states: dict[str, SourceItemState] = {}
                for row in cur.fetchall():
                    metadata = dict(row[2] or {})
                    access_scope = dict(row[3] or {})
                    payload = {
                        "source_kind": source_kind,
                        "external_id": str(row[0]),
                        "checksum": str(row[1] or ""),
                        "metadata": metadata,
                        "access_scope": access_scope,
                        "index_status": str(row[4] or "active"),
                    }
                    state = state_from_source_item_payload(payload)
                    states[state.external_id] = state
                return states

    def save_item(
        self,
        *,
        raw: SourceRawItem,
        document: NormalizedDocument,
        chunks: list[Chunk],
        findings: list[SecretFinding],
        raw_object_key: str,
    ) -> None:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                source_item = _source_item_payload(raw=raw, raw_object_key=raw_object_key)
                cur.execute(
                    """
                    insert into source_accounts (id, kind, display_name, enabled, metadata)
                    values (%s, %s, %s, true, '{}'::jsonb)
                    on conflict (id) do update set updated_at = now()
                    """,
                    (raw.source_kind, raw.source_kind, raw.source_kind),
                )
                cur.execute(
                    """
                    insert into source_items (
                      id, source_account_id, source_kind, external_id,
                      canonical_url, title, author_id, created_at, updated_at,
                      deleted_at, index_status, access_scope, raw_object_key,
                      checksum, metadata
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, null, %s, %s::jsonb, %s, %s, %s::jsonb)
                    on conflict (source_kind, external_id) do update set
                      canonical_url = excluded.canonical_url,
                      title = excluded.title,
                      author_id = excluded.author_id,
                      updated_at = excluded.updated_at,
                      deleted_at = null,
                      index_status = excluded.index_status,
                      access_scope = excluded.access_scope,
                      raw_object_key = excluded.raw_object_key,
                      checksum = excluded.checksum,
                      metadata = excluded.metadata,
                      ingested_at = now()
                    """,
                    (
                        source_item["id"],
                        source_item["source_account_id"],
                        source_item["source_kind"],
                        source_item["external_id"],
                        source_item["canonical_url"],
                        source_item["title"],
                        source_item["author_id"],
                        raw.created_at,
                        raw.updated_at,
                        source_item["index_status"],
                        json.dumps(source_item["access_scope"], ensure_ascii=False),
                        raw_object_key,
                        source_item["checksum"],
                        json.dumps(source_item["metadata"], ensure_ascii=False, default=str),
                    ),
                )
                cur.execute(
                    """
                    insert into documents (
                      id, source_item_id, version, title, normalized_text,
                      normalized_format, language, access_scope, checksum, metadata
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
                    on conflict (id) do update set
                      title = excluded.title,
                      normalized_text = excluded.normalized_text,
                      normalized_format = excluded.normalized_format,
                      language = excluded.language,
                      access_scope = excluded.access_scope,
                      checksum = excluded.checksum,
                      metadata = excluded.metadata,
                      updated_at = now()
                    """,
                    _document_sql_values(document),
                )
                for chunk in chunks:
                    payload = _chunk_payload(chunk)
                    cur.execute(
                        """
                        insert into chunks (
                          id, document_id, source_item_id, chunk_index, chunk_kind,
                          text, token_count, parent_chunk_id, access_scope,
                          index_status, redaction_policy, checksum, metadata
                        )
                        values (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb)
                        on conflict (id) do update set
                          text = excluded.text,
                          token_count = excluded.token_count,
                          access_scope = excluded.access_scope,
                          index_status = excluded.index_status,
                          redaction_policy = excluded.redaction_policy,
                          checksum = excluded.checksum,
                          metadata = excluded.metadata,
                          updated_at = now()
                        """,
                        _chunk_sql_values(payload),
                    )
                    cur.execute("delete from chunk_acl_entries where chunk_id = %s", (chunk.id,))
                    for entry in _chunk_acl_entries(chunk):
                        cur.execute(
                            """
                            insert into chunk_acl_entries (chunk_id, acl_type, acl_value)
                            values (%s, %s, %s)
                            on conflict do nothing
                            """,
                            (
                                entry["chunk_id"],
                                entry["acl_type"],
                                entry["acl_value"],
                            ),
                        )
                for finding in findings:
                    cur.execute(
                        """
                        insert into secret_findings (
                          id, source_item_id, chunk_id, secret_type, severity,
                          redaction_policy, detected_span_hash, status, metadata
                        )
                        values (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        on conflict do nothing
                        """,
                        (
                            finding.id,
                            finding.source_item_id,
                            finding.chunk_id,
                            finding.secret_type,
                            finding.severity,
                            finding.redaction_policy,
                            finding.detected_span_hash,
                            finding.status,
                            json.dumps(finding.metadata, ensure_ascii=False, default=str),
                        ),
                    )
            conn.commit()

    def mark_deleted(
        self,
        *,
        source_kind: str,
        external_id: str,
        status: str = "deleted",
    ) -> None:
        index_status = status if status in {"deleted", "permission_lost"} else "deleted"
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update source_items
                    set deleted_at = now(), index_status = %s
                    where source_kind = %s and external_id = %s
                    """,
                    (index_status, source_kind, external_id),
                )
                cur.execute(
                    """
                    update chunks
                    set index_status = %s
                    where source_item_id in (
                      select id from source_items
                      where source_kind = %s and external_id = %s
                    )
                    """,
                    (index_status, source_kind, external_id),
                )
            conn.commit()


def build_ingestion_repository(
    *,
    postgres: PostgresClient,
    fallback_dir: Path,
) -> IngestionRepository:
    if postgres.is_configured():
        return PostgresIngestionRepository(postgres=postgres)
    return FileIngestionRepository(root_dir=fallback_dir)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fw:
        fw.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _source_item_payload(
    *,
    raw: SourceRawItem,
    raw_object_key: str,
) -> dict[str, object]:
    metadata = _metadata_with_terms(raw.source_kind, raw.metadata)
    return {
        "id": stable_hash(f"{raw.source_kind}:{raw.external_id}"),
        "source_account_id": raw.source_kind,
        "source_kind": raw.source_kind,
        "external_id": raw.external_id,
        "canonical_url": raw.canonical_url,
        "title": raw.title,
        "author_id": raw.author_id,
        "created_at": raw.created_at.isoformat() if raw.created_at else None,
        "updated_at": raw.updated_at.isoformat() if raw.updated_at else None,
        "index_status": "active",
        "access_scope": raw.access_scope.as_dict(),
        "raw_object_key": raw_object_key,
        "checksum": raw.checksum,
        "metadata": metadata,
    }


def _document_payload(document: NormalizedDocument) -> dict[str, object]:
    metadata = _metadata_with_terms(document.source_kind, document.metadata)
    return {
        "id": document.id,
        "source_item_id": document.source_item_id,
        "version": document.version,
        "title": document.title,
        "normalized_text": document.normalized_text,
        "normalized_format": document.normalized_format,
        "language": document.language,
        "access_scope": document.access_scope.as_dict(),
        "checksum": document.checksum,
        "metadata": metadata,
    }


def _chunk_payload(chunk: Chunk) -> dict[str, object]:
    metadata = dict(chunk.metadata)
    source_kind = str(metadata.get("source_kind") or "")
    if source_kind:
        metadata = _metadata_with_terms(source_kind, metadata)
    redaction_policy = str(metadata.get("redaction_policy") or "quote_allowed")
    index_status = str(metadata.get("index_status") or "active")
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "source_item_id": str(metadata.get("source_item_id") or ""),
        "chunk_index": chunk.index,
        "chunk_kind": str(metadata.get("chunk_kind") or "body"),
        "text": chunk.text,
        "token_count": int(metadata.get("token_count") or 0),
        "parent_chunk_id": metadata.get("parent_chunk_id"),
        "access_scope": metadata.get("access_scope") or {},
        "index_status": index_status,
        "redaction_policy": redaction_policy,
        "checksum": str(metadata.get("checksum") or stable_hash(chunk.text)),
        "metadata": metadata,
    }


def _document_sql_values(document: NormalizedDocument) -> tuple[object, ...]:
    payload = _document_payload(document)
    return (
        payload["id"],
        payload["source_item_id"],
        payload["version"],
        payload["title"],
        payload["normalized_text"],
        payload["normalized_format"],
        payload["language"],
        json.dumps(payload["access_scope"], ensure_ascii=False),
        payload["checksum"],
        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
    )


def _chunk_sql_values(payload: dict[str, object]) -> tuple[object, ...]:
    return (
        payload["id"],
        payload["document_id"],
        payload["source_item_id"],
        payload["chunk_index"],
        payload["chunk_kind"],
        payload["text"],
        payload["token_count"],
        payload["parent_chunk_id"],
        json.dumps(payload["access_scope"], ensure_ascii=False),
        payload["index_status"],
        payload["redaction_policy"],
        payload["checksum"],
        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
    )


def _chunk_acl_entries(chunk: Chunk) -> list[dict[str, str]]:
    scope = dict(chunk.metadata.get("access_scope") or {})
    entries: list[dict[str, str]] = []
    visibility = str(scope.get("visibility") or "admin")
    entries.append({"chunk_id": chunk.id, "acl_type": "visibility", "acl_value": visibility})
    guild_id = str(scope.get("guild_id") or "").strip()
    if guild_id:
        entries.append({"chunk_id": chunk.id, "acl_type": "guild", "acl_value": guild_id})
    for role_id in scope.get("role_ids") or []:
        entries.append({"chunk_id": chunk.id, "acl_type": "role", "acl_value": str(role_id)})
    for user_id in scope.get("user_ids") or []:
        entries.append({"chunk_id": chunk.id, "acl_type": "user", "acl_value": str(user_id)})
    return entries


def _metadata_with_terms(
    source_kind: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    merged = dict(metadata)
    status = str(merged.get("terms_review_status") or "").strip()
    if not status:
        status = _default_terms_review_status(source_kind)
        merged["terms_review_status"] = status
    merged.setdefault("external_reuse_allowed", status == "approved")
    return merged


def _default_terms_review_status(source_kind: str) -> str:
    if source_kind in {"google_drive", "discord", "notion"}:
        return "internal_only"
    return "pending"
