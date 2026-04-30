from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.secret import SecretFinding
from kumc_agent.domain.models.source import NormalizedDocument, SourceRawItem, SyncCursor
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

    def load_sync_cursor(self, source_kind: str) -> SyncCursor | None:
        ...

    def save_sync_cursor(self, cursor: SyncCursor) -> SyncCursor:
        ...

    def load_active_chunks(self, *, source_kinds: tuple[str, ...] = tuple()) -> list[Chunk]:
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
    auto_compact: bool = True

    def load_checksums(self, source_kind: str) -> dict[str, str]:
        return {
            external_id: state.checksum
            for external_id, state in self.load_item_states(source_kind).items()
        }

    def load_item_states(self, source_kind: str) -> dict[str, SourceItemState]:
        path = self.root_dir / "source_items.jsonl"
        states: dict[str, SourceItemState] = {}
        if not path.exists():
            return _apply_file_deletes(root_dir=self.root_dir, source_kind=source_kind, states=states)
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                payload = json.loads(line)
                if payload.get("source_kind") != source_kind:
                    continue
                state = state_from_source_item_payload(payload)
                states[state.external_id] = state
        return _apply_file_deletes(root_dir=self.root_dir, source_kind=source_kind, states=states)

    def load_sync_cursor(self, source_kind: str) -> SyncCursor | None:
        latest: SyncCursor | None = None
        path = self.root_dir / "sync_cursors.jsonl"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                payload = json.loads(line)
                if payload.get("source_kind") != source_kind:
                    continue
                latest = SyncCursor(
                    source_kind=source_kind,
                    cursor=str(payload.get("cursor") or ""),
                    metadata=dict(payload.get("metadata") or {}),
                )
        return latest

    def save_sync_cursor(self, cursor: SyncCursor) -> SyncCursor:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        _append_jsonl(
            self.root_dir / "sync_cursors.jsonl",
            {
                "source_kind": cursor.source_kind,
                "cursor": cursor.cursor,
                "metadata": cursor.metadata,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        self._compact_after_write(source_kind=cursor.source_kind)
        return cursor

    def load_active_chunks(self, *, source_kinds: tuple[str, ...] = tuple()) -> list[Chunk]:
        return _load_file_active_chunks(root_dir=self.root_dir, source_kinds=source_kinds)

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
        _append_jsonl(
            self.root_dir / "source_deletes.jsonl",
            {
                "source_kind": raw.source_kind,
                "external_id": raw.external_id,
                "index_status": "active",
                "deleted_at": "",
            },
        )
        _append_jsonl(self.root_dir / "documents.jsonl", _document_payload(document))
        for chunk in chunks:
            _append_jsonl(self.root_dir / "chunks.jsonl", _chunk_payload(chunk))
            for entry in _chunk_acl_entries(chunk):
                _append_jsonl(self.root_dir / "chunk_acl_entries.jsonl", entry)
        for finding in findings:
            _append_jsonl(self.root_dir / "secret_findings.jsonl", asdict(finding))
        self._compact_after_write(source_kind=raw.source_kind)

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
        self._compact_after_write(source_kind=source_kind)

    def compact_history(
        self,
        *,
        source_kinds: tuple[str, ...] = tuple(),
    ) -> dict[str, object]:
        return compact_file_ingestion_history(
            root_dir=self.root_dir,
            source_kinds=source_kinds,
        )

    def _compact_after_write(self, *, source_kind: str) -> None:
        if not self.auto_compact:
            return
        compact_file_ingestion_history(
            root_dir=self.root_dir,
            source_kinds=(source_kind,),
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

    def load_sync_cursor(self, source_kind: str) -> SyncCursor | None:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select cursor, metadata
                    from sync_cursors
                    where source_kind = %s
                    """,
                    (source_kind,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return SyncCursor(
            source_kind=source_kind,
            cursor=str(row[0] or ""),
            metadata=dict(row[1] or {}),
        )

    def save_sync_cursor(self, cursor: SyncCursor) -> SyncCursor:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into sync_cursors (source_kind, cursor, metadata)
                    values (%s, %s, %s::jsonb)
                    on conflict (source_kind) do update set
                      cursor = excluded.cursor,
                      metadata = excluded.metadata,
                      updated_at = now()
                    """,
                    (
                        cursor.source_kind,
                        cursor.cursor,
                        json.dumps(cursor.metadata, ensure_ascii=False, default=str),
                    ),
                )
            conn.commit()
        return cursor

    def load_active_chunks(self, *, source_kinds: tuple[str, ...] = tuple()) -> list[Chunk]:
        filters = ["c.index_status = 'active'", "si.index_status = 'active'"]
        params: list[object] = []
        if source_kinds:
            filters.append("si.source_kind = any(%s)")
            params.append(list(source_kinds))
        sql = f"""
            select c.id, c.document_id, c.text, c.chunk_index, c.metadata,
                   c.index_status, c.access_scope, si.source_kind, si.external_id, si.title
            from chunks c
            join source_items si on si.id = c.source_item_id
            where {' and '.join(filters)}
            order by si.source_kind, si.external_id, c.chunk_index
        """
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        chunks: list[Chunk] = []
        for row in rows:
            metadata = dict(row[4] or {})
            metadata.setdefault("index_status", str(row[5] or "active"))
            metadata.setdefault("access_scope", dict(row[6] or {}))
            metadata.setdefault("source_kind", str(row[7] or ""))
            metadata.setdefault("source_type", str(row[7] or ""))
            metadata.setdefault("external_id", str(row[8] or ""))
            metadata.setdefault("source_title", str(row[9] or ""))
            chunks.append(
                Chunk(
                    id=str(row[0]),
                    document_id=str(row[1]),
                    text=str(row[2] or ""),
                    index=int(row[3] or 0),
                    metadata=metadata,
                )
            )
        return chunks

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
                    update chunks
                    set index_status = 'deleted'
                    where source_item_id = %s
                    """,
                    (source_item["id"],),
                )
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


def compact_file_ingestion_history(
    *,
    root_dir: Path,
    source_kinds: tuple[str, ...] = tuple(),
) -> dict[str, object]:
    root_dir.mkdir(parents=True, exist_ok=True)
    source_filter = {str(value) for value in source_kinds if str(value).strip()}
    before_quality = build_file_ingestion_quality_report(
        root_dir=root_dir,
        source_kinds=source_kinds,
    )
    files = (
        "source_items.jsonl",
        "source_deletes.jsonl",
        "documents.jsonl",
        "chunks.jsonl",
        "chunk_acl_entries.jsonl",
        "secret_findings.jsonl",
        "sync_cursors.jsonl",
    )
    before_counts = {
        name: len(_read_jsonl(root_dir / name))
        for name in files
        if (root_dir / name).exists()
    }

    source_item_rows = _read_jsonl(root_dir / "source_items.jsonl")
    delete_rows = _read_jsonl(root_dir / "source_deletes.jsonl")
    document_rows = _read_jsonl(root_dir / "documents.jsonl")
    chunk_rows = _read_jsonl(root_dir / "chunks.jsonl")
    acl_rows = _read_jsonl(root_dir / "chunk_acl_entries.jsonl")
    finding_rows = _read_jsonl(root_dir / "secret_findings.jsonl")
    cursor_rows = _read_jsonl(root_dir / "sync_cursors.jsonl")

    latest_deletes = _latest_payloads(
        delete_rows,
        key_fn=_source_item_key,
        source_filter=source_filter,
    )

    passthrough_source_items: list[dict[str, object]] = []
    selected_source_items: list[dict[str, object]] = []
    selected_source_item_ids: set[str] = set()
    for payload in source_item_rows:
        if not _source_matches(payload, source_filter):
            passthrough_source_items.append(payload)
            continue
        selected_source_items.append(payload)
        source_item_id = str(payload.get("id") or "")
        if source_item_id:
            selected_source_item_ids.add(source_item_id)

    latest_source_items = _latest_payloads(
        selected_source_items,
        key_fn=_source_item_key,
        source_filter=set(),
    )
    active_source_items: list[dict[str, object]] = []
    active_source_item_ids: set[str] = set()
    for key, payload in latest_source_items.items():
        compacted = dict(payload)
        latest_delete = latest_deletes.get(key)
        index_status = str(
            (latest_delete or {}).get("index_status")
            or compacted.get("index_status")
            or "active"
        )
        compacted["index_status"] = index_status
        if index_status != "active":
            continue
        source_item_id = str(compacted.get("id") or "")
        if source_item_id:
            active_source_item_ids.add(source_item_id)
        active_source_items.append(compacted)
    source_items_by_id = {
        str(payload.get("id") or ""): payload
        for payload in active_source_items
        if str(payload.get("id") or "")
    }

    source_items_out = passthrough_source_items + active_source_items

    selected_delete_rows: list[dict[str, object]] = []
    passthrough_delete_rows: list[dict[str, object]] = []
    for payload in delete_rows:
        if _source_matches(payload, source_filter):
            continue
        passthrough_delete_rows.append(payload)
    selected_delete_rows = list(latest_deletes.values())
    delete_out = passthrough_delete_rows + selected_delete_rows

    passthrough_documents: list[dict[str, object]] = []
    selected_documents: list[dict[str, object]] = []
    selected_document_ids: set[str] = set()
    for payload in document_rows:
        source_item_id = str(payload.get("source_item_id") or "")
        if source_item_id not in selected_source_item_ids:
            passthrough_documents.append(payload)
            continue
        selected_documents.append(payload)
        document_id = str(payload.get("id") or "")
        if document_id:
            selected_document_ids.add(document_id)

    latest_documents = _latest_payloads(
        selected_documents,
        key_fn=lambda payload: str(payload.get("source_item_id") or ""),
        source_filter=set(),
    )
    active_documents = []
    for payload in latest_documents.values():
        source_item_id = str(payload.get("source_item_id") or "")
        if source_item_id not in active_source_item_ids:
            continue
        active_documents.append(
            _enrich_document_payload(
                payload,
                source_item=source_items_by_id.get(source_item_id),
            )
        )
    active_document_ids = {
        str(payload.get("id") or "")
        for payload in active_documents
        if str(payload.get("id") or "")
    }
    documents_out = passthrough_documents + active_documents

    passthrough_chunks: list[dict[str, object]] = []
    selected_chunks: list[dict[str, object]] = []
    selected_chunk_ids: set[str] = set()
    for payload in chunk_rows:
        document_id = str(payload.get("document_id") or "")
        if document_id not in selected_document_ids:
            passthrough_chunks.append(payload)
            continue
        selected_chunks.append(payload)
        chunk_id = str(payload.get("id") or "")
        if chunk_id:
            selected_chunk_ids.add(chunk_id)

    latest_chunks = _latest_payloads(
        selected_chunks,
        key_fn=lambda payload: str(payload.get("id") or ""),
        source_filter=set(),
    )
    active_chunks = [
        payload
        for payload in latest_chunks.values()
        if str(payload.get("document_id") or "") in active_document_ids
        and str(payload.get("index_status") or "active") == "active"
    ]
    active_chunk_ids = {
        str(payload.get("id") or "")
        for payload in active_chunks
        if str(payload.get("id") or "")
    }
    chunks_out = passthrough_chunks + active_chunks

    acl_out = _compact_chunk_acl_entries(
        rows=acl_rows,
        selected_chunk_ids=selected_chunk_ids,
        active_chunk_ids=active_chunk_ids,
        compact_all=not source_filter,
    )
    finding_out = _compact_secret_findings(
        rows=finding_rows,
        selected_source_item_ids=selected_source_item_ids,
        active_source_item_ids=active_source_item_ids,
        selected_chunk_ids=selected_chunk_ids,
        active_chunk_ids=active_chunk_ids,
        compact_all=not source_filter,
    )
    cursor_out = _compact_sync_cursors(rows=cursor_rows, source_filter=source_filter)

    outputs = {
        "source_items.jsonl": source_items_out,
        "source_deletes.jsonl": delete_out,
        "documents.jsonl": documents_out,
        "chunks.jsonl": chunks_out,
        "chunk_acl_entries.jsonl": acl_out,
        "secret_findings.jsonl": finding_out,
        "sync_cursors.jsonl": cursor_out,
    }
    for name, rows in outputs.items():
        path = root_dir / name
        if path.exists() or rows:
            _write_jsonl_atomic(path, rows)

    current_views = write_file_ingestion_current_views(root_dir=root_dir)
    after_quality = build_file_ingestion_quality_report(
        root_dir=root_dir,
        source_kinds=source_kinds,
    )
    quality_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_kinds": sorted(source_filter),
        "before_compaction": before_quality,
        "after_compaction": after_quality,
        "current_views": current_views,
    }
    _write_json_atomic(root_dir / "ingestion_quality_report.json", quality_payload)

    after_counts = {
        name: len(_read_jsonl(root_dir / name))
        for name in files
        if (root_dir / name).exists()
    }
    return {
        "status": "succeeded",
        "source_kinds": sorted(source_filter),
        "files": {
            name: {
                "before": before_counts.get(name, 0),
                "after": after_counts.get(name, 0),
            }
            for name in files
            if name in before_counts or name in after_counts
        },
        "active_source_items": len(active_source_item_ids),
        "active_documents": len(active_document_ids),
        "active_chunks": len(active_chunk_ids),
        "current_views": current_views,
        "quality_report": quality_payload,
    }


def write_file_ingestion_current_views(*, root_dir: Path) -> dict[str, object]:
    current = _current_file_ingestion_rows(root_dir=root_dir)
    outputs = {
        "current_source_items.jsonl": current["source_items"],
        "current_documents.jsonl": current["documents"],
        "current_chunks.jsonl": current["chunks"],
        "current_chunk_acl_entries.jsonl": current["chunk_acl_entries"],
    }
    for name, rows in outputs.items():
        _write_jsonl_atomic(root_dir / name, list(rows))
    return {
        name: len(rows)
        for name, rows in outputs.items()
    }


def build_file_ingestion_quality_report(
    *,
    root_dir: Path,
    source_kinds: tuple[str, ...] = tuple(),
) -> dict[str, object]:
    source_filter = {str(value) for value in source_kinds if str(value).strip()}
    source_items = _read_jsonl(root_dir / "source_items.jsonl")
    deletes = _read_jsonl(root_dir / "source_deletes.jsonl")
    documents = _read_jsonl(root_dir / "documents.jsonl")
    chunks = _read_jsonl(root_dir / "chunks.jsonl")
    acl_entries = _read_jsonl(root_dir / "chunk_acl_entries.jsonl")
    source_by_id = {
        str(payload.get("id") or ""): str(payload.get("source_kind") or "")
        for payload in source_items
        if str(payload.get("id") or "")
    }
    document_source_by_id: dict[str, str] = {}
    for payload in documents:
        source_kind = _source_kind_for_document(payload, source_by_id=source_by_id)
        document_id = str(payload.get("id") or "")
        if document_id:
            document_source_by_id[document_id] = source_kind

    return {
        "source_kinds": sorted(source_filter),
        "files": {
            "source_items.jsonl": _duplicate_report(
                rows=source_items,
                source_filter=source_filter,
                source_fn=lambda payload: str(payload.get("source_kind") or ""),
                key_fn=_source_item_key,
            ),
            "source_deletes.jsonl": _duplicate_report(
                rows=deletes,
                source_filter=source_filter,
                source_fn=lambda payload: str(payload.get("source_kind") or ""),
                key_fn=_source_item_key,
            ),
            "source_deletes_active.jsonl": _duplicate_report(
                rows=[
                    payload
                    for payload in deletes
                    if str(payload.get("index_status") or "") == "active"
                ],
                source_filter=source_filter,
                source_fn=lambda payload: str(payload.get("source_kind") or ""),
                key_fn=_source_item_key,
            ),
            "documents.jsonl": _duplicate_report(
                rows=documents,
                source_filter=source_filter,
                source_fn=lambda payload: _source_kind_for_document(
                    payload,
                    source_by_id=source_by_id,
                ),
                key_fn=lambda payload: str(payload.get("source_item_id") or payload.get("id") or ""),
            ),
            "chunks.jsonl": _duplicate_report(
                rows=chunks,
                source_filter=source_filter,
                source_fn=lambda payload: _source_kind_for_chunk(
                    payload,
                    document_source_by_id=document_source_by_id,
                ),
                key_fn=lambda payload: str(payload.get("id") or ""),
            ),
            "chunk_acl_entries.jsonl": _duplicate_report(
                rows=acl_entries,
                source_filter=set(),
                source_fn=lambda _payload: "",
                key_fn=lambda payload: "|".join(
                    (
                        str(payload.get("chunk_id") or ""),
                        str(payload.get("acl_type") or ""),
                        str(payload.get("acl_value") or ""),
                    )
                ),
            ),
        },
    }


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fw:
        fw.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _write_jsonl_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fw:
        for payload in rows:
            fw.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    tmp_path.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _apply_file_deletes(
    *,
    root_dir: Path,
    source_kind: str,
    states: dict[str, SourceItemState],
) -> dict[str, SourceItemState]:
    for payload in _read_jsonl(root_dir / "source_deletes.jsonl"):
        if payload.get("source_kind") != source_kind:
            continue
        external_id = str(payload.get("external_id") or "")
        status = str(payload.get("index_status") or "deleted")
        previous = states.get(external_id)
        if previous is None:
            states[external_id] = SourceItemState(
                source_kind=source_kind,
                external_id=external_id,
                index_status=status,
            )
        else:
            states[external_id] = SourceItemState(
                source_kind=previous.source_kind,
                external_id=previous.external_id,
                checksum=previous.checksum,
                revision=previous.revision,
                acl_hash=previous.acl_hash,
                index_status=status,
                metadata=previous.metadata,
            )
    return states


def _load_file_active_chunks(
    *,
    root_dir: Path,
    source_kinds: tuple[str, ...],
) -> list[Chunk]:
    source_filter = {value for value in source_kinds if value}
    source_items = _latest_payloads_by_key(
        root_dir / "source_items.jsonl",
        lambda payload: f"{payload.get('source_kind')}:{payload.get('external_id')}",
    )
    for payload in _read_jsonl(root_dir / "source_deletes.jsonl"):
        key = f"{payload.get('source_kind')}:{payload.get('external_id')}"
        existing = source_items.get(key)
        if existing is not None:
            existing = dict(existing)
            existing["index_status"] = str(payload.get("index_status") or "deleted")
            source_items[key] = existing
    active_source_item_ids = {
        str(payload.get("id") or "")
        for payload in source_items.values()
        if str(payload.get("index_status") or "active") == "active"
        and (not source_filter or str(payload.get("source_kind") or "") in source_filter)
    }
    latest_documents: dict[str, str] = {}
    for payload in _read_jsonl(root_dir / "documents.jsonl"):
        source_item_id = str(payload.get("source_item_id") or "")
        if source_item_id in active_source_item_ids:
            latest_documents[source_item_id] = str(payload.get("id") or "")
    active_document_ids = {value for value in latest_documents.values() if value}
    latest_chunks = _latest_payloads_by_key(
        root_dir / "chunks.jsonl",
        lambda payload: str(payload.get("id") or ""),
    )
    chunks: list[Chunk] = []
    for payload in latest_chunks.values():
        document_id = str(payload.get("document_id") or "")
        if document_id not in active_document_ids:
            continue
        index_status = str(payload.get("index_status") or "active")
        if index_status != "active":
            continue
        metadata = dict(payload.get("metadata") or {})
        metadata.setdefault("index_status", index_status)
        metadata.setdefault("access_scope", payload.get("access_scope") or {})
        chunks.append(
            Chunk(
                id=str(payload.get("id") or ""),
                document_id=document_id,
                text=str(payload.get("text") or ""),
                index=int(payload.get("chunk_index") or 0),
                metadata=metadata,
            )
        )
    return sorted(chunks, key=lambda chunk: (str(chunk.metadata.get("source_kind") or ""), chunk.document_id, chunk.index))


def _latest_payloads_by_key(path: Path, key_fn) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for payload in _read_jsonl(path):
        key = key_fn(payload)
        if key:
            latest[str(key)] = payload
    return latest


def _latest_payloads(
    rows: list[dict[str, object]],
    *,
    key_fn,
    source_filter: set[str],
) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for payload in rows:
        if not _source_matches(payload, source_filter):
            continue
        key = key_fn(payload)
        if key:
            latest[str(key)] = payload
    return latest


def _current_file_ingestion_rows(*, root_dir: Path) -> dict[str, list[dict[str, object]]]:
    source_item_rows = _read_jsonl(root_dir / "source_items.jsonl")
    delete_rows = _read_jsonl(root_dir / "source_deletes.jsonl")
    document_rows = _read_jsonl(root_dir / "documents.jsonl")
    chunk_rows = _read_jsonl(root_dir / "chunks.jsonl")
    acl_rows = _read_jsonl(root_dir / "chunk_acl_entries.jsonl")

    latest_source_items = _latest_payloads(
        source_item_rows,
        key_fn=_source_item_key,
        source_filter=set(),
    )
    latest_deletes = _latest_payloads(
        delete_rows,
        key_fn=_source_item_key,
        source_filter=set(),
    )
    active_source_items: list[dict[str, object]] = []
    active_source_item_ids: set[str] = set()
    for key, payload in latest_source_items.items():
        current = dict(payload)
        latest_delete = latest_deletes.get(key)
        index_status = str(
            (latest_delete or {}).get("index_status")
            or current.get("index_status")
            or "active"
        )
        current["index_status"] = index_status
        if index_status != "active":
            continue
        source_item_id = str(current.get("id") or "")
        if source_item_id:
            active_source_item_ids.add(source_item_id)
        active_source_items.append(current)

    source_items_by_id = {
        str(payload.get("id") or ""): payload
        for payload in active_source_items
        if str(payload.get("id") or "")
    }
    latest_documents = _latest_payloads(
        [
            payload
            for payload in document_rows
            if str(payload.get("source_item_id") or "") in active_source_item_ids
        ],
        key_fn=lambda payload: str(payload.get("source_item_id") or ""),
        source_filter=set(),
    )
    active_documents: list[dict[str, object]] = []
    active_document_ids: set[str] = set()
    for payload in latest_documents.values():
        source_item_id = str(payload.get("source_item_id") or "")
        enriched = _enrich_document_payload(
            payload,
            source_item=source_items_by_id.get(source_item_id),
        )
        active_documents.append(enriched)
        document_id = str(enriched.get("id") or "")
        if document_id:
            active_document_ids.add(document_id)

    latest_chunks = _latest_payloads(
        chunk_rows,
        key_fn=lambda payload: str(payload.get("id") or ""),
        source_filter=set(),
    )
    active_chunks = [
        payload
        for payload in latest_chunks.values()
        if str(payload.get("document_id") or "") in active_document_ids
        and str(payload.get("index_status") or "active") == "active"
    ]
    active_chunk_ids = {
        str(payload.get("id") or "")
        for payload in active_chunks
        if str(payload.get("id") or "")
    }
    active_acl_entries = _current_chunk_acl_entries(
        rows=acl_rows,
        active_chunk_ids=active_chunk_ids,
    )
    return {
        "source_items": active_source_items,
        "documents": active_documents,
        "chunks": active_chunks,
        "chunk_acl_entries": active_acl_entries,
    }


def _source_matches(payload: dict[str, object], source_filter: set[str]) -> bool:
    if not source_filter:
        return True
    return str(payload.get("source_kind") or "") in source_filter


def _source_item_key(payload: dict[str, object]) -> str:
    source_kind = str(payload.get("source_kind") or "")
    external_id = str(payload.get("external_id") or "")
    if not source_kind or not external_id:
        return ""
    return f"{source_kind}:{external_id}"


def _enrich_document_payload(
    payload: dict[str, object],
    *,
    source_item: dict[str, object] | None,
) -> dict[str, object]:
    enriched = dict(payload)
    source_kind = str(enriched.get("source_kind") or "").strip()
    external_id = str(enriched.get("external_id") or "").strip()
    if source_item is not None:
        source_kind = source_kind or str(source_item.get("source_kind") or "").strip()
        external_id = external_id or str(source_item.get("external_id") or "").strip()
    if source_kind:
        enriched["source_kind"] = source_kind
        enriched.setdefault("source_type", source_kind)
        metadata = dict(enriched.get("metadata") or {})
        metadata.setdefault("source_kind", source_kind)
        metadata.setdefault("source_type", source_kind)
        if external_id:
            metadata.setdefault("external_id", external_id)
        enriched["metadata"] = metadata
    if external_id:
        enriched["external_id"] = external_id
    return enriched


def _source_kind_for_document(
    payload: dict[str, object],
    *,
    source_by_id: dict[str, str],
) -> str:
    source_kind = str(payload.get("source_kind") or payload.get("source_type") or "")
    if source_kind:
        return source_kind
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        source_kind = str(metadata.get("source_kind") or metadata.get("source_type") or "")
        if source_kind:
            return source_kind
    return source_by_id.get(str(payload.get("source_item_id") or ""), "")


def _source_kind_for_chunk(
    payload: dict[str, object],
    *,
    document_source_by_id: dict[str, str],
) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        source_kind = str(metadata.get("source_kind") or metadata.get("source_type") or "")
        if source_kind:
            return source_kind
    return document_source_by_id.get(str(payload.get("document_id") or ""), "")


def _duplicate_report(
    *,
    rows: list[dict[str, object]],
    source_filter: set[str],
    source_fn,
    key_fn,
) -> dict[str, object]:
    counts_by_source: dict[str, dict[str, object]] = {}
    key_counts_by_source: dict[str, dict[str, int]] = {}
    total_rows = 0
    for payload in rows:
        source_kind = source_fn(payload)
        if source_filter and source_kind not in source_filter:
            continue
        key = str(key_fn(payload) or "")
        if not key:
            continue
        total_rows += 1
        source_bucket = source_kind or "unknown"
        key_counts = key_counts_by_source.setdefault(source_bucket, {})
        key_counts[key] = key_counts.get(key, 0) + 1
    for source_kind, key_counts in key_counts_by_source.items():
        row_count = sum(key_counts.values())
        unique_count = len(key_counts)
        duplicate_keys = {
            key: count
            for key, count in sorted(key_counts.items())
            if count > 1
        }
        counts_by_source[source_kind] = {
            "rows": row_count,
            "unique_keys": unique_count,
            "duplicate_rows": row_count - unique_count,
            "duplicate_keys": len(duplicate_keys),
            "duplicate_key_counts": duplicate_keys,
        }
    unique_total = sum(len(value) for value in key_counts_by_source.values())
    return {
        "rows": total_rows,
        "unique_keys": unique_total,
        "duplicate_rows": total_rows - unique_total,
        "by_source": counts_by_source,
    }


def _compact_chunk_acl_entries(
    *,
    rows: list[dict[str, object]],
    selected_chunk_ids: set[str],
    active_chunk_ids: set[str],
    compact_all: bool,
) -> list[dict[str, object]]:
    if not selected_chunk_ids and not compact_all:
        return rows
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for payload in rows:
        chunk_id = str(payload.get("chunk_id") or "")
        if chunk_id in selected_chunk_ids and chunk_id not in active_chunk_ids:
            continue
        if selected_chunk_ids and chunk_id not in selected_chunk_ids:
            out.append(payload)
            continue
        key = (
            chunk_id,
            str(payload.get("acl_type") or ""),
            str(payload.get("acl_value") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        if chunk_id in active_chunk_ids:
            out.append(payload)
    return out


def _current_chunk_acl_entries(
    *,
    rows: list[dict[str, object]],
    active_chunk_ids: set[str],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for payload in rows:
        chunk_id = str(payload.get("chunk_id") or "")
        if chunk_id not in active_chunk_ids:
            continue
        key = (
            chunk_id,
            str(payload.get("acl_type") or ""),
            str(payload.get("acl_value") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(payload)
    return out


def _compact_secret_findings(
    *,
    rows: list[dict[str, object]],
    selected_source_item_ids: set[str],
    active_source_item_ids: set[str],
    selected_chunk_ids: set[str],
    active_chunk_ids: set[str],
    compact_all: bool,
) -> list[dict[str, object]]:
    if not selected_source_item_ids and not selected_chunk_ids and not compact_all:
        return rows
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for payload in rows:
        source_item_id = str(payload.get("source_item_id") or "")
        chunk_id = str(payload.get("chunk_id") or "")
        belongs_to_selected = (
            source_item_id in selected_source_item_ids
            or chunk_id in selected_chunk_ids
        )
        if not belongs_to_selected:
            if compact_all:
                continue
            out.append(payload)
            continue
        if (
            source_item_id not in active_source_item_ids
            and chunk_id not in active_chunk_ids
        ):
            continue
        finding_id = str(payload.get("id") or "")
        if finding_id and finding_id in seen:
            continue
        if finding_id:
            seen.add(finding_id)
        out.append(payload)
    return out


def _compact_sync_cursors(
    *,
    rows: list[dict[str, object]],
    source_filter: set[str],
) -> list[dict[str, object]]:
    passthrough: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for payload in rows:
        if _source_matches(payload, source_filter):
            selected.append(payload)
        else:
            passthrough.append(payload)
    latest = _latest_payloads(
        selected,
        key_fn=lambda payload: str(payload.get("source_kind") or ""),
        source_filter=set(),
    )
    return passthrough + list(latest.values())


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    out: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                out.append(payload)
    return out


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
    metadata.setdefault("source_kind", document.source_kind)
    metadata.setdefault("source_type", document.source_kind)
    metadata.setdefault("external_id", document.external_id)
    return {
        "id": document.id,
        "source_item_id": document.source_item_id,
        "source_kind": document.source_kind,
        "source_type": document.source_kind,
        "external_id": document.external_id,
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
