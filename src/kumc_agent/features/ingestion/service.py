from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging

from kumc_agent.domain.models.audit import AuditEvent
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.secret import SecretFinding
from kumc_agent.domain.models.source import BackfillScope, SourceDeleteItem, SourceRawItem, SyncCursor
from kumc_agent.domain.ports.connectors import SourceConnector
from kumc_agent.features.foundation.tracing import current_trace_id
from kumc_agent.features.ingestion.chunking import IngestionChunker
from kumc_agent.features.indexing.change_detection import detect_source_change
from kumc_agent.infra.audit.repository import AuditLogRepository
from kumc_agent.infra.ingestion.repository import IngestionRepository
from kumc_agent.infra.object_storage.raw_snapshot import RawSnapshotStore
from kumc_agent.infra.secret_finding.detector import (
    SecretFindingDetector,
    strictest_redaction_policy,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    source_kind: str
    seen: int
    changed: int
    skipped: int
    deleted: int
    documents: int
    chunks: int
    secret_findings: int
    status: str = "succeeded"
    error: str = ""


class IngestionService:
    def __init__(
        self,
        *,
        connectors: dict[str, SourceConnector],
        repository: IngestionRepository,
        raw_snapshots: RawSnapshotStore,
        chunker: IngestionChunker,
        secret_detector: SecretFindingDetector,
        audit_log: AuditLogRepository,
    ) -> None:
        self._connectors = connectors
        self._repository = repository
        self._raw_snapshots = raw_snapshots
        self._chunker = chunker
        self._secret_detector = secret_detector
        self._audit_log = audit_log

    def available_sources(self) -> tuple[str, ...]:
        return tuple(sorted(self._connectors))

    async def backfill(
        self,
        *,
        source_kind: str,
        scope: BackfillScope | None = None,
    ) -> IngestionResult:
        if source_kind not in self._connectors:
            raise KeyError(f"Unknown source connector: {source_kind}")
        connector = self._connectors[source_kind]
        resolved_scope = scope or BackfillScope()
        item_states = self._repository.load_item_states(source_kind)
        cursor = self._repository.load_sync_cursor(source_kind)
        supports_incremental = bool(getattr(connector, "supports_incremental", False))
        use_poll_changes = bool(
            supports_incremental
            and cursor is not None
            and cursor.cursor
            and not resolved_scope.force
        )
        seen = changed = skipped = deleted = documents = chunks = findings_count = 0

        stream = (
            connector.poll_changes(cursor)
            if use_poll_changes and cursor is not None
            else connector.backfill(resolved_scope)
        )
        async for item in stream:
            if isinstance(item, SourceDeleteItem):
                self._repository.mark_deleted(
                    source_kind=item.source_kind,
                    external_id=item.external_id,
                    status=(
                        "permission_lost"
                        if item.reason == "permission_lost"
                        else "deleted"
                    ),
                )
                deleted += 1
                continue
            seen += 1
            decision = detect_source_change(
                item=item,
                previous=item_states.get(item.external_id),
                force=resolved_scope.force,
            )
            if decision.change_kind == "skipped":
                skipped += 1
                continue
            outcome = await self._ingest_raw_item(connector=connector, raw=item)
            changed += 1
            documents += 1
            chunks += outcome.chunks
            findings_count += outcome.secret_findings
        connector_sync_metadata: dict[str, object] = {}
        sync_metadata_fn = getattr(connector, "sync_metadata", None)
        if callable(sync_metadata_fn):
            value = sync_metadata_fn()
            if isinstance(value, dict):
                connector_sync_metadata = {str(key): item for key, item in value.items()}
        cursor_metadata = {
            "mode": (
                "poll_changes"
                if use_poll_changes
                else "full_scan_cursor_unsupported"
                if cursor is not None and cursor.cursor and not resolved_scope.force
                else "backfill"
            ),
            "cursor_supported": supports_incremental,
            "previous_cursor_present": bool(cursor is not None and cursor.cursor),
            "seen": seen,
            "changed": changed,
            "skipped": skipped,
            "deleted": deleted,
        }
        if connector_sync_metadata:
            cursor_metadata["source_sync"] = connector_sync_metadata
        self._repository.save_sync_cursor(
            SyncCursor(
                source_kind=source_kind,
                cursor=datetime.now(UTC).isoformat(),
                metadata=cursor_metadata,
            )
        )

        result = IngestionResult(
            source_kind=source_kind,
            seen=seen,
            changed=changed,
            skipped=skipped,
            deleted=deleted,
            documents=documents,
            chunks=chunks,
            secret_findings=findings_count,
        )
        self._audit_log.append(
            AuditEvent(
                action="ingestion.backfill",
                actor_id="worker",
                actor_type="service",
                target=source_kind,
                outcome="succeeded",
                risk_level="low",
                trace_id=current_trace_id(),
                metadata=result.__dict__,
            )
        )
        return result

    async def backfill_many(
        self,
        *,
        source_kinds: tuple[str, ...],
        scope: BackfillScope | None = None,
    ) -> tuple[IngestionResult, ...]:
        targets = source_kinds or self.available_sources()
        results: list[IngestionResult] = []
        for source_kind in targets:
            try:
                results.append(await self.backfill(source_kind=source_kind, scope=scope))
            except Exception as exc:
                logger.exception("Ingestion source failed: %s", source_kind)
                results.append(
                    IngestionResult(
                        source_kind=source_kind,
                        seen=0,
                        changed=0,
                        skipped=0,
                        deleted=0,
                        documents=0,
                        chunks=0,
                        secret_findings=0,
                        status="failed",
                        error=str(exc),
                    )
                )
        return tuple(results)

    async def _ingest_raw_item(
        self,
        *,
        connector: SourceConnector,
        raw: SourceRawItem,
    ) -> IngestionResult:
        raw_object_key = self._raw_snapshots.put(raw)
        document = await connector.normalize(raw)
        source_findings = self._secret_detector.detect(
            source_item_id=document.source_item_id,
            text=document.normalized_text,
            chunk_id=None,
        )
        chunk_items = self._chunker.chunk(document)
        all_findings: list[SecretFinding] = list(source_findings)
        processed_chunks: list[Chunk] = []
        for chunk in chunk_items:
            chunk_findings = self._secret_detector.detect(
                source_item_id=document.source_item_id,
                chunk_id=chunk.id,
                text=chunk.text,
            )
            all_findings.extend(chunk_findings)
            processed_chunks.append(
                _with_secret_metadata(
                    chunk=chunk,
                    findings=chunk_findings,
                )
            )
        self._repository.save_item(
            raw=raw,
            document=document,
            chunks=processed_chunks,
            findings=all_findings,
            raw_object_key=raw_object_key,
        )
        return IngestionResult(
            source_kind=raw.source_kind,
            seen=1,
            changed=1,
            skipped=0,
            deleted=0,
            documents=1,
            chunks=len(processed_chunks),
            secret_findings=len(all_findings),
        )


def _with_secret_metadata(*, chunk: Chunk, findings: list[SecretFinding]) -> Chunk:
    policy = strictest_redaction_policy(findings)
    existing_status = str(chunk.metadata.get("index_status") or "active").strip().lower()
    if existing_status in {"deleted", "permission_lost", "quarantined"}:
        index_status = existing_status
    else:
        index_status = "quarantined" if policy == "deny" else "active"
    metadata = {
        **chunk.metadata,
        "redaction_policy": policy,
        "index_status": index_status,
        "secret_finding_ids": [finding.id for finding in findings],
    }
    return Chunk(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        index=chunk.index,
        metadata=metadata,
    )
