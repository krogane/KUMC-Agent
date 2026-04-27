from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.domain.models.operations import IndexingRun
from kumc_agent.domain.models.source import BackfillScope
from kumc_agent.features.indexing.lock import build_indexing_lock
from kumc_agent.features.indexing.quality import IndexQualitySmokeChecker
from kumc_agent.features.indexing.snapshot import IndexSnapshotPublisher
from kumc_agent.features.ingestion.service import IngestionService
from kumc_agent.infra.operations import OperationsRepository
from kumc_agent.usecases.indexing.build import BuildIndexRequest, BuildIndexUsecase


@dataclass(frozen=True)
class AutoIndexUpdateRequest:
    trigger: str = "manual"
    source_filter: tuple[str, ...] = tuple()
    force: bool = False
    full_rebuild: bool = False
    quality_check_enabled: bool = True
    refresh_sources: bool = True
    stage_selection: tuple[str, ...] | None = None
    scheduled_at: datetime | None = None


@dataclass(frozen=True)
class AutoIndexUpdateResult:
    status: str
    run_id: str
    seen: int = 0
    changed: int = 0
    skipped: int = 0
    deleted: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "seen": self.seen,
            "changed": self.changed,
            "skipped": self.skipped,
            "deleted": self.deleted,
            "metadata": _sanitize_metadata(self.metadata),
        }


class AutoIndexUpdateUsecase:
    def __init__(
        self,
        *,
        config: RuntimeConfig,
        build_usecase: BuildIndexUsecase,
        operations: OperationsRepository,
        ingestion_service: IngestionService | None = None,
    ) -> None:
        self._config = config
        self._build_usecase = build_usecase
        self._operations = operations
        self._ingestion_service = ingestion_service
        self._publisher = IndexSnapshotPublisher(
            index_dir=config.app.index_dir,
            keep_snapshots=config.scheduler.rollback_keep_snapshots,
        )

    def execute(self, request: AutoIndexUpdateRequest) -> AutoIndexUpdateResult:
        run_id = _run_id(request.trigger)
        metadata: dict[str, Any] = {
            "trigger": request.trigger,
            "source_filter": list(request.source_filter),
            "force": request.force,
            "full_rebuild": request.full_rebuild,
            "quality_check_enabled": request.quality_check_enabled,
        }
        schedule_skip = _schedule_skip_reason(self._config, request)
        if schedule_skip:
            return self._save_result(
                IndexingRun(
                    id=run_id,
                    source_kind="all",
                    status="skipped",
                    metadata=metadata | {"reason": schedule_skip},
                )
            )

        lock = build_indexing_lock(self._config)
        lock_result = lock.acquire(run_id=run_id)
        if not lock_result.acquired:
            return self._save_result(
                IndexingRun(
                    id=run_id,
                    source_kind="all",
                    status="skipped",
                    metadata=metadata
                    | {
                        "reason": lock_result.reason,
                        "lock": dict(lock_result.metadata or {}),
                    },
                )
            )

        run = self._operations.save_indexing_run(
            IndexingRun(id=run_id, source_kind="all", status="running", metadata=metadata)
        )
        try:
            ingestion_results = self._ingest_sources(request)
            source_results = [result.__dict__ for result in ingestion_results]
            seen = sum(result.seen for result in ingestion_results)
            changed = sum(result.changed for result in ingestion_results)
            skipped = sum(result.skipped for result in ingestion_results)
            deleted = sum(result.deleted for result in ingestion_results)
            metadata["source_results"] = source_results

            has_changes = bool(changed or deleted or request.force or request.full_rebuild)
            if request.refresh_sources and self._ingestion_service is not None and not has_changes:
                return self._save_result(
                    replace(
                        run,
                        status="skipped",
                        seen=seen,
                        changed=changed,
                        skipped=skipped,
                        deleted=deleted,
                        metadata=metadata | {"reason": "no_source_changes"},
                    )
                )

            staging_dir = self._publisher.staging_dir(run_id)
            build_result = self._build_usecase.execute(
                BuildIndexRequest(
                    refresh_sources=request.refresh_sources and self._ingestion_service is None,
                    full_rebuild=request.full_rebuild,
                    stage_selection=request.stage_selection,
                    index_dir=staging_dir,
                )
            )
            metadata["stage_results"] = {
                "index": {
                    "loaded_sources": build_result.loaded_sources,
                    "documents": build_result.documents,
                    "chunks": build_result.chunks,
                    "staging_dir": str(build_result.index_dir),
                }
            }
            quality = None
            if request.quality_check_enabled:
                checker = IndexQualitySmokeChecker(
                    min_chunk_ratio=self._config.scheduler.quality_min_chunk_ratio,
                    smoke_queries=tuple(self._config.scheduler.quality_smoke_queries),
                )
                quality = checker.check(
                    staging_dir=staging_dir,
                    current_dir=self._config.app.index_dir,
                )
                metadata["quality_check"] = quality.metadata | {
                    "passed": quality.passed,
                    "critical_failures": list(quality.critical_failures),
                }
                if not quality.passed:
                    return self._save_result(
                        replace(
                            run,
                            status="failed",
                            seen=seen,
                            changed=changed,
                            skipped=skipped,
                            deleted=deleted,
                            error="; ".join(quality.critical_failures),
                            metadata=metadata
                            | {
                                "notification": _notification_payload(
                                    run_id=run_id,
                                    status="failed",
                                    reason="quality_check_failed",
                                )
                            },
                        )
                    )

            publish = self._publisher.publish(run_id=run_id, staging_dir=staging_dir)
            metadata["index_snapshot_id"] = publish.snapshot_id
            metadata["previous_snapshot_id"] = publish.previous_snapshot_id
            metadata["publish"] = {
                "current_pointer": str(publish.current_pointer),
                "previous_pointer": str(publish.previous_pointer),
            }
            return self._save_result(
                replace(
                    run,
                    status="succeeded",
                    seen=seen,
                    changed=changed,
                    skipped=skipped,
                    deleted=deleted,
                    metadata=metadata,
                )
            )
        except Exception as exc:
            failed = replace(
                run,
                status="failed",
                error=str(exc),
                metadata=metadata
                | {
                    "notification": _notification_payload(
                        run_id=run_id,
                        status="failed",
                        reason=str(exc),
                    )
                },
            )
            return self._save_result(failed)
        finally:
            lock.release()

    def _ingest_sources(self, request: AutoIndexUpdateRequest):
        if not request.refresh_sources or self._ingestion_service is None:
            return tuple()
        return asyncio.run(
            self._ingestion_service.backfill_many(
                source_kinds=request.source_filter,
                scope=BackfillScope(force=bool(request.force or request.full_rebuild)),
            )
        )

    def _save_result(self, run: IndexingRun) -> AutoIndexUpdateResult:
        stored = self._operations.save_indexing_run(run)
        return AutoIndexUpdateResult(
            status=stored.status,
            run_id=stored.id,
            seen=stored.seen,
            changed=stored.changed,
            skipped=stored.skipped,
            deleted=stored.deleted,
            metadata=dict(stored.metadata),
        )


def _schedule_skip_reason(config: RuntimeConfig, request: AutoIndexUpdateRequest) -> str:
    if request.trigger not in {"schedule", "automation"}:
        return ""
    if not config.scheduler.auto_index_enabled:
        return "auto_index_disabled"
    now = request.scheduled_at or datetime.now(UTC)
    if now.weekday() not in set(config.scheduler.auto_index_weekdays):
        return "weekday_not_allowed"
    hour, minute = _parse_hhmm(config.scheduler.auto_index_time)
    if now.hour != hour or now.minute != minute:
        return "outside_scheduled_time"
    return ""


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = (value or "00:00").split(":", 1)
    return int(hour), int(minute)


def _run_id(trigger: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in trigger)
    return f"auto-index-{stamp}-{normalized or 'manual'}"


def _notification_payload(*, run_id: str, status: str, reason: str) -> dict[str, object]:
    return {
        "type": "indexing_run",
        "run_id": run_id,
        "status": status,
        "reason": reason[:500],
    }


def _sanitize_metadata(value: dict[str, Any]) -> dict[str, object]:
    blocked = {"context", "contexts", "raw", "raw_text", "normalized_text", "secret", "llm_prompt"}
    sanitized: dict[str, object] = {}
    for key, item in value.items():
        if key in blocked:
            continue
        if isinstance(item, str):
            sanitized[key] = _mask_secret(item[:1200])
        elif isinstance(item, dict):
            sanitized[key] = _sanitize_metadata(item)
        elif isinstance(item, list):
            sanitized[key] = [
                _sanitize_metadata(entry) if isinstance(entry, dict) else entry
                for entry in item[:200]
            ]
        else:
            sanitized[key] = item
    return sanitized


def _mask_secret(value: str) -> str:
    lowered = value.lower()
    if any(token in lowered for token in ("api_key", "apikey", "token", "secret", "password")):
        return "[REDACTED]"
    return value
