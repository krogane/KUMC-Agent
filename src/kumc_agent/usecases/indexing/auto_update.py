from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import threading
import time
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.operations import ActionRun, IndexingRun
from kumc_agent.domain.models.retrieval import AccessContext, Citation
from kumc_agent.domain.models.source import BackfillScope
from kumc_agent.features.indexing.lock import build_indexing_lock
from kumc_agent.features.indexing.quality import IndexQualitySmokeChecker
from kumc_agent.features.indexing.snapshot import IndexSnapshotPublisher
from kumc_agent.features.foundation.payload_sanitizer import sanitize_payload_metadata
from kumc_agent.features.ingestion.service import IngestionResult, IngestionService
from kumc_agent.features.workflow.extraction_window import (
    build_extraction_window,
    normalize_lookback_days,
    select_recent_chunks,
)
from kumc_agent.infra.operations import OperationsRepository
from kumc_agent.usecases.indexing.build import BuildIndexRequest, BuildIndexUsecase
from kumc_agent.utils.hashing import stable_hash

_MEMBER_PROFILES_SOURCE = "member_profiles"
_TASK_EVENT_SOURCES = {"task_event", "tasks", "events", "task", "event"}


class MemberProfileRebuildPort(Protocol):
    def rebuild_guild(self, *, guild_id: str, index_dir: Path | None = None) -> IndexingRun:
        ...


class TaskEventIndexPort(Protocol):
    def rebuild(self, *, index_dir: Path) -> IndexingRun:
        ...


class EventDeltaExtractionPort(Protocol):
    def event_extract_from_delta(
        self,
        *,
        text: str,
        evidence: tuple[Citation, ...],
        access: AccessContext,
        metadata: dict[str, Any],
    ) -> Any:
        ...


class TaskDeltaExtractionPort(Protocol):
    def task_extract_from_delta(
        self,
        *,
        text: str,
        evidence: tuple[Citation, ...],
        access: AccessContext,
        metadata: dict[str, Any],
    ) -> Any:
        ...


class EventDeltaChunkSourcePort(Protocol):
    def load_active_chunks(self, *, source_kinds: tuple[str, ...] = tuple()) -> list[Chunk]:
        ...


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
        member_profile_builder: MemberProfileRebuildPort | None = None,
        member_profile_guild_ids: tuple[str, ...] = tuple(),
        task_event_indexer: TaskEventIndexPort | None = None,
        task_delta_extractor: TaskDeltaExtractionPort | None = None,
        event_delta_extractor: EventDeltaExtractionPort | None = None,
        event_delta_chunk_source: EventDeltaChunkSourcePort | None = None,
    ) -> None:
        self._config = config
        self._build_usecase = build_usecase
        self._operations = operations
        self._ingestion_service = ingestion_service
        self._member_profile_builder = member_profile_builder
        self._member_profile_guild_ids = tuple(
            str(value) for value in member_profile_guild_ids if str(value)
        )
        self._task_event_indexer = task_event_indexer
        self._task_delta_extractor = task_delta_extractor
        self._event_delta_extractor = event_delta_extractor
        self._event_delta_chunk_source = event_delta_chunk_source
        self._publisher = IndexSnapshotPublisher(
            index_dir=config.app.index_dir,
            keep_snapshots=config.scheduler.rollback_keep_snapshots,
        )

    def execute(self, request: AutoIndexUpdateRequest) -> AutoIndexUpdateResult:
        run_id = _run_id(request.trigger)
        max_runtime_minutes = int(
            getattr(self._config.scheduler, "auto_index_max_runtime_minutes", 120)
        )
        metadata: dict[str, Any] = {
            "trigger": request.trigger,
            "source_filter": list(request.source_filter),
            "force": request.force,
            "full_rebuild": request.full_rebuild,
            "quality_check_enabled": request.quality_check_enabled,
            "max_runtime_minutes": max_runtime_minutes,
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

        heartbeat = _LockHeartbeat(
            lock=lock,
            interval_seconds=_heartbeat_interval_seconds(
                self._config.scheduler.auto_index_lock_ttl_minutes
            ),
        )
        heartbeat.start()
        deadline = datetime.now(UTC) + timedelta(minutes=max(1, max_runtime_minutes))
        run = IndexingRun(id=run_id, source_kind="all", status="running", metadata=metadata)
        try:
            _ensure_before_deadline(deadline)
            ingestion_results = self._ingest_sources(request)
            _ensure_before_deadline(deadline)
            source_results = [result.__dict__ for result in ingestion_results]
            seen = sum(result.seen for result in ingestion_results)
            changed = sum(result.changed for result in ingestion_results)
            skipped = sum(result.skipped for result in ingestion_results)
            deleted = sum(result.deleted for result in ingestion_results)
            metadata["source_results"] = source_results
            failed_sources = [
                result
                for result in ingestion_results
                if getattr(result, "status", "succeeded") != "succeeded"
            ]
            if failed_sources:
                metadata["degraded"] = True
                metadata["failed_sources"] = [
                    {
                        "source_kind": result.source_kind,
                        "status": getattr(result, "status", "failed"),
                        "error": getattr(result, "error", ""),
                    }
                    for result in failed_sources
                ]

            member_profile_refresh_planned = self._should_refresh_member_profiles(request)
            task_event_refresh_planned = self._should_refresh_task_event(request)
            has_changes = bool(
                changed
                or deleted
                or request.force
                or request.full_rebuild
                or member_profile_refresh_planned
                or task_event_refresh_planned
            )
            if failed_sources and len(failed_sources) == len(ingestion_results):
                return self._save_result(
                    replace(
                        run,
                        status="failed",
                        seen=seen,
                        changed=changed,
                        skipped=skipped,
                        deleted=deleted,
                        error="all_sources_failed",
                        metadata=metadata
                        | {
                            "notification": _notification_payload(
                                run_id=run_id,
                                status="failed",
                                reason="all_sources_failed",
                            )
                        },
                    )
                )

            if request.refresh_sources and self._ingestion_service is not None and not has_changes:
                status = "failed" if failed_sources else "skipped"
                return self._save_result(
                    replace(
                        run,
                        status=status,
                        seen=seen,
                        changed=changed,
                        skipped=skipped,
                        deleted=deleted,
                        error="source_failed_without_publish" if failed_sources else "",
                        metadata=metadata
                        | {
                            "reason": (
                                "source_failed_without_publish"
                                if failed_sources
                                else "no_source_changes"
                            ),
                            **(
                                {
                                    "notification": _notification_payload(
                                        run_id=run_id,
                                        status="failed",
                                        reason="source_failed_without_publish",
                                    )
                                }
                                if failed_sources
                                else {}
                            ),
                        },
                    )
                )

            staging_dir = self._publisher.staging_dir(run_id)
            build_result = self._build_usecase.execute(
                BuildIndexRequest(
                    refresh_sources=request.refresh_sources and self._ingestion_service is None,
                    full_rebuild=request.full_rebuild,
                    stage_selection=request.stage_selection,
                    index_dir=staging_dir,
                    prefer_ingestion_repository=self._ingestion_service is not None,
                )
            )
            _ensure_before_deadline(deadline)
            metadata["stage_results"] = {
                "index": {
                    "loaded_sources": build_result.loaded_sources,
                    "documents": build_result.documents,
                    "chunks": build_result.chunks,
                    "staging_dir": str(build_result.index_dir),
                }
            }
            if getattr(build_result, "stage_results", None):
                metadata["stage_results"].update(dict(build_result.stage_results or {}))
            _lift_notion_coverage_metadata(metadata)
            if member_profile_refresh_planned:
                member_profile_runs = self._rebuild_member_profiles(staging_dir=staging_dir)
                _ensure_before_deadline(deadline)
                member_profile_results = [profile_run.__dict__ for profile_run in member_profile_runs]
                metadata["source_results"] = source_results + member_profile_results
                metadata["stage_results"]["member_profiles"] = {
                    "runs": member_profile_results,
                    "guild_ids": list(self._member_profile_guild_ids),
                }
                seen += sum(profile_run.seen for profile_run in member_profile_runs)
                changed += sum(profile_run.changed for profile_run in member_profile_runs)
                skipped += sum(profile_run.skipped for profile_run in member_profile_runs)
                deleted += sum(profile_run.deleted for profile_run in member_profile_runs)
                failed = [profile_run for profile_run in member_profile_runs if profile_run.status != "succeeded"]
                if failed:
                    reasons = [
                        profile_run.error or f"{profile_run.source_kind}:{profile_run.status}"
                        for profile_run in failed
                    ]
                    return self._save_result(
                        replace(
                            run,
                            status="failed",
                            seen=seen,
                            changed=changed,
                            skipped=skipped,
                            deleted=deleted,
                            error="; ".join(reasons),
                            metadata=metadata
                            | {
                                "notification": _notification_payload(
                                    run_id=run_id,
                                    status="failed",
                                    reason="member_profiles_failed",
                                )
                            },
                        )
                    )
            if task_event_refresh_planned and self._task_event_indexer is not None:
                task_event_run = self._task_event_indexer.rebuild(index_dir=staging_dir)
                _ensure_before_deadline(deadline)
                metadata["source_results"] = metadata.get("source_results", []) + [
                    task_event_run.__dict__
                ]
                metadata["stage_results"]["task_event"] = task_event_run.__dict__
                seen += task_event_run.seen
                changed += task_event_run.changed
                skipped += task_event_run.skipped
                deleted += task_event_run.deleted
                if task_event_run.status != "succeeded":
                    return self._save_result(
                        replace(
                            run,
                            status="failed",
                            seen=seen,
                            changed=changed,
                            skipped=skipped,
                            deleted=deleted,
                            error=task_event_run.error or "task_event_index_failed",
                            metadata=metadata
                            | {
                                "notification": _notification_payload(
                                    run_id=run_id,
                                    status="failed",
                                    reason="task_event_index_failed",
                                )
                            },
                        )
                    )
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

            try:
                _ensure_before_deadline(deadline)
                publish = self._publisher.publish(run_id=run_id, staging_dir=staging_dir)
                _ensure_before_deadline(deadline)
            except Exception as exc:
                rollback = self._publisher.rollback_to_latest_previous()
                return self._save_result(
                    replace(
                        run,
                        status="rolled_back" if rollback.get("status") == "succeeded" else "failed",
                        seen=seen,
                        changed=changed,
                        skipped=skipped,
                        deleted=deleted,
                        error=str(exc),
                        metadata=metadata
                        | {
                            "rollback": rollback,
                            "notification": _notification_payload(
                                run_id=run_id,
                                status="failed",
                                reason=f"publish_failed:{exc}",
                            ),
                        },
                    )
                )
            metadata["index_snapshot_id"] = publish.snapshot_id
            metadata["previous_snapshot_id"] = publish.previous_snapshot_id
            metadata["publish"] = {
                "current_pointer": str(publish.current_pointer),
                "previous_pointer": str(publish.previous_pointer),
                "release_dir": str(publish.release_dir),
            }
            self._commit_staged_side_effects(
                release_dir=publish.release_dir,
                metadata=metadata,
            )
            self._compact_embedding_cache_after_publish(
                build_result=build_result,
                metadata=metadata,
            )
            self._run_workflow_delta_extraction(
                run_id=run_id,
                ingestion_results=ingestion_results,
                metadata=metadata,
            )
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
            status = "cancelled" if "cancel" in str(exc).lower() else "failed"
            failed = replace(
                run,
                status=status,
                error=str(exc),
                metadata=metadata
                | {
                    "notification": _notification_payload(
                        run_id=run_id,
                        status=status,
                        reason=str(exc),
                    )
                },
            )
            return self._save_result(failed)
        finally:
            heartbeat.stop()
            lock.release()

    def _commit_staged_side_effects(
        self,
        *,
        release_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        commit = getattr(self._build_usecase, "commit_staged_side_effects", None)
        if not callable(commit):
            return
        stage_results = metadata.setdefault("stage_results", {})
        if not isinstance(stage_results, dict):
            return
        try:
            result = commit(release_dir)
        except Exception as exc:
            metadata["degraded"] = True
            stage_results["staged_side_effect_commit"] = {
                "status": "failed",
                "error": str(exc)[:500],
            }
            return
        if not result:
            return
        stage_results["staged_side_effect_commit"] = result

    def _ingest_sources(self, request: AutoIndexUpdateRequest):
        if not request.refresh_sources or self._ingestion_service is None:
            return tuple()
        source_kinds = tuple(
            source
            for source in request.source_filter
            if source != _MEMBER_PROFILES_SOURCE and source not in _TASK_EVENT_SOURCES
        )
        if request.source_filter and not source_kinds:
            return tuple()
        return asyncio.run(
            self._ingestion_service.backfill_many(
                source_kinds=source_kinds,
                scope=BackfillScope(force=bool(request.force or request.full_rebuild)),
            )
        )

    def _should_refresh_member_profiles(self, request: AutoIndexUpdateRequest) -> bool:
        if not request.refresh_sources:
            return False
        if self._member_profile_builder is None or not self._member_profile_guild_ids:
            return False
        filters = {source for source in request.source_filter if source}
        return not filters or _MEMBER_PROFILES_SOURCE in filters

    def _should_refresh_task_event(self, request: AutoIndexUpdateRequest) -> bool:
        if self._task_event_indexer is None:
            return False
        filters = {source for source in request.source_filter if source}
        return not filters or bool(filters & _TASK_EVENT_SOURCES)

    def _rebuild_member_profiles(self, *, staging_dir: Path) -> tuple[IndexingRun, ...]:
        if self._member_profile_builder is None:
            return tuple()
        return tuple(
            self._member_profile_builder.rebuild_guild(
                guild_id=guild_id,
                index_dir=staging_dir,
            )
            for guild_id in self._member_profile_guild_ids
        )

    def _compact_embedding_cache_after_publish(
        self,
        *,
        build_result: Any,
        metadata: dict[str, Any],
    ) -> None:
        cache_config = getattr(getattr(self._config, "indexing", None), "embedding_cache", None)
        if not bool(getattr(cache_config, "compact_after_publish", False)):
            return
        active_keys = tuple(getattr(build_result, "embedding_cache_keys", tuple()) or tuple())
        stage_results = metadata.setdefault("stage_results", {})
        if not isinstance(stage_results, dict):
            return
        embedding_stage = stage_results.setdefault("embedding", {})
        if not isinstance(embedding_stage, dict):
            embedding_stage = {}
            stage_results["embedding"] = embedding_stage
        compact = getattr(self._build_usecase, "compact_embedding_cache", None)
        if not callable(compact):
            embedding_stage["cache_compaction"] = {
                "status": "skipped",
                "reason": "not_supported",
            }
            return
        try:
            embedding_stage["cache_compaction"] = compact(active_keys)
        except Exception as exc:
            embedding_stage["cache_compaction"] = {
                "status": "failed",
                "degraded": True,
                "error": str(exc)[:500],
            }

    def _run_workflow_delta_extraction(
        self,
        *,
        run_id: str,
        ingestion_results: tuple[IngestionResult, ...],
        metadata: dict[str, Any],
    ) -> None:
        task_config = getattr(self._config, "task_management", None)
        event_config = getattr(self._config, "event_management", None)
        task_enabled = bool(getattr(task_config, "auto_extract_after_index_update", False))
        event_enabled = bool(getattr(event_config, "auto_extract_after_index_update", False))
        if not task_enabled and not event_enabled:
            return
        changed_results = [
            result
            for result in ingestion_results
            if result.status == "succeeded" and (result.changed or result.deleted)
        ]
        if not changed_results:
            return
        workflow_metadata = metadata.setdefault("workflow_extraction", {})
        if not isinstance(workflow_metadata, dict):
            workflow_metadata = {}
            metadata["workflow_extraction"] = workflow_metadata
        if self._event_delta_chunk_source is None:
            source_kinds = [result.source_kind for result in changed_results]
            if task_enabled:
                task_payload = {"status": "not_configured", "source_kinds": source_kinds}
                workflow_metadata["task"] = task_payload
                metadata["task_delta_extraction"] = task_payload
            if event_enabled:
                event_payload = {"status": "not_configured", "source_kinds": source_kinds}
                workflow_metadata["event"] = event_payload
                metadata["event_extraction"] = event_payload
            return
        source_kinds = tuple(
            dict.fromkeys(result.source_kind for result in changed_results if result.source_kind)
        )
        extraction_at = datetime.now(UTC)
        lookback_days = _workflow_extraction_lookback_days(self._config)
        base_window = build_extraction_window(
            lookback_days=lookback_days,
            extraction_at=extraction_at,
        )
        window_metadata: dict[str, object] = {
            **base_window.as_metadata(),
            "selected_chunks": 0,
            "excluded_older_chunks": 0,
            "excluded_missing_timestamp_chunks": 0,
        }
        try:
            chunks = self._event_delta_chunk_source.load_active_chunks(source_kinds=source_kinds)
            recent_selection = select_recent_chunks(
                chunks,
                lookback_days=lookback_days,
                extraction_at=extraction_at,
            )
            selected_chunks = _event_delta_chunks(recent_selection.chunks)
            window_metadata = recent_selection.as_metadata(selected_chunks=len(selected_chunks))
            evidence = tuple(_event_delta_citation(chunk) for chunk in selected_chunks[:12])
            if task_enabled:
                task_payload = self._run_single_delta_extraction(
                    kind="task",
                    run_id=run_id,
                    source_kinds=source_kinds,
                    selected_chunks=selected_chunks,
                    changed_results=changed_results,
                    evidence=evidence,
                    window_metadata=window_metadata,
                )
                workflow_metadata["task"] = task_payload
                metadata["task_delta_extraction"] = task_payload
            if event_enabled:
                event_payload = self._run_single_delta_extraction(
                    kind="event",
                    run_id=run_id,
                    source_kinds=source_kinds,
                    selected_chunks=selected_chunks,
                    changed_results=changed_results,
                    evidence=evidence,
                    window_metadata=window_metadata,
                )
                workflow_metadata["event"] = event_payload
                metadata["event_extraction"] = event_payload
        except Exception as exc:
            failure = {
                "status": "failed",
                "degraded": True,
                "source_kinds": list(source_kinds),
                **window_metadata,
                "error": str(exc)[:500],
            }
            if task_enabled:
                workflow_metadata["task"] = failure
                metadata["task_delta_extraction"] = failure
            if event_enabled:
                workflow_metadata["event"] = failure
                metadata["event_extraction"] = failure

    def _run_single_delta_extraction(
        self,
        *,
        kind: str,
        run_id: str,
        source_kinds: tuple[str, ...],
        selected_chunks: list[Chunk],
        changed_results: list[IngestionResult],
        evidence: tuple[Citation, ...],
        window_metadata: dict[str, object],
    ) -> dict[str, Any]:
        extractor = self._task_delta_extractor if kind == "task" else self._event_delta_extractor
        if extractor is None:
            return {
                "status": "not_configured",
                "source_kinds": list(source_kinds),
                **window_metadata,
            }
        if not selected_chunks:
            skipped = {
                "status": "skipped",
                "reason": "no_recent_chunks",
                "source_kinds": list(source_kinds),
                "chunks": 0,
                "candidate_count": 0,
                "change_candidate_count": 0,
                **window_metadata,
            }
            skipped["metadata"] = dict(window_metadata) | {
                "degraded": False,
                "skipped_reason": "no_recent_chunks",
            }
            return skipped
        text = _delta_text(
            kind=kind,
            chunks=selected_chunks,
            source_results=changed_results,
        )
        try:
            access = AccessContext(
                user_id="auto_index_update",
                role_ids=("admin",),
                is_admin=True,
            )
            request_metadata = {
                "source": "auto_index_update",
                "run_id": run_id,
                "source_kinds": list(source_kinds),
                "changed": sum(result.changed for result in changed_results),
                "deleted": sum(result.deleted for result in changed_results),
                **window_metadata,
            }
            if kind == "task":
                response = extractor.task_extract_from_delta(
                    text=text,
                    evidence=evidence,
                    access=access,
                    metadata=request_metadata,
                )
                candidate_count = len(getattr(response, "task_candidates", tuple()))
                change_candidate_count = len(getattr(response, "task_change_candidates", tuple()))
            else:
                response = extractor.event_extract_from_delta(
                    text=text,
                    evidence=evidence,
                    access=access,
                    metadata=request_metadata,
                )
                candidate_count = len(getattr(response, "event_candidates", tuple()))
                change_candidate_count = len(getattr(response, "event_change_candidates", tuple()))
            extraction_metadata = dict(getattr(response, "metadata", {}) or {})
            extraction_detail = extraction_metadata.get("extraction", extraction_metadata)
            if not isinstance(extraction_detail, dict):
                extraction_detail = {}
            return {
                "status": "succeeded",
                "source_kinds": list(source_kinds),
                "chunks": len(selected_chunks),
                "candidate_count": candidate_count,
                "change_candidate_count": change_candidate_count,
                **window_metadata,
                "metadata": dict(window_metadata) | extraction_detail,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "degraded": True,
                "source_kinds": list(source_kinds),
                **window_metadata,
                "error": str(exc)[:500],
            }

    def _save_result(self, run: IndexingRun) -> AutoIndexUpdateResult:
        run = self._with_notification_delivery(run)
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

    def _with_notification_delivery(self, run: IndexingRun) -> IndexingRun:
        notification = run.metadata.get("notification")
        if not isinstance(notification, dict) or notification.get("delivery"):
            return run
        delivery = self._record_notification(run=run, payload=notification)
        return replace(
            run,
            metadata={
                **run.metadata,
                "notification": {
                    **notification,
                    "delivery": delivery,
                },
            },
        )

    def _record_notification(
        self,
        *,
        run: IndexingRun,
        payload: dict[str, object],
    ) -> dict[str, object]:
        action_run_id = stable_hash(
            f"indexing-notification:{run.id}:{payload.get('reason') or ''}"
        )[:32]
        try:
            self._operations.save_action_run(
                ActionRun(
                    id=action_run_id,
                    action_type="indexing_notification",
                    target="admin",
                    status="succeeded",
                    risk_level="low",
                    request_payload=payload,
                    result_payload={"channel": "operations_repository", "sent": True},
                    metadata={"run_id": run.id},
                )
            )
            return {
                "status": "recorded",
                "action_run_id": action_run_id,
                "channel": "operations_repository",
            }
        except Exception as exc:
            return {"status": "failed", "error": str(exc)[:500]}


def _event_delta_chunks(chunks: list[Chunk]) -> list[Chunk]:
    selected: list[Chunk] = []
    seen_documents: set[str] = set()
    for chunk in chunks:
        text = str(chunk.text or "").strip()
        if not text:
            continue
        document_id = chunk.document_id or chunk.id
        if document_id in seen_documents and chunk.index > 1:
            continue
        seen_documents.add(document_id)
        selected.append(chunk)
        if len(selected) >= 24:
            break
    return selected


def _delta_text(
    *,
    kind: str,
    chunks: list[Chunk],
    source_results: list[IngestionResult],
) -> str:
    if kind == "task":
        instruction = (
            "auto_index_update で変更または削除が検出されたソースから、"
            "タスクの新規登録・変更・削除だけを抽出してください。"
        )
        empty_instruction = (
            "アクティブchunkはありません。削除差分だけの場合は既存Taskとの対応が明確な場合のみ削除候補を返してください。"
        )
    else:
        instruction = (
            "auto_index_update で変更または削除が検出されたソースから、"
            "イベントの新規登録・変更・削除だけを抽出してください。"
        )
        empty_instruction = (
            "アクティブchunkはありません。削除差分だけの場合は既存Eventとの対応が明確な場合のみ削除候補を返してください。"
        )
    lines = [
        instruction,
        "変更サマリ:",
        *[
            (
                f"- {result.source_kind}: changed={result.changed}, "
                f"deleted={result.deleted}, seen={result.seen}"
            )
            for result in source_results
        ],
    ]
    if not chunks:
        lines.append(empty_instruction)
        return "\n".join(lines)
    for index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.metadata or {})
        source_kind = str(metadata.get("source_kind") or metadata.get("source_type") or "")
        external_id = str(metadata.get("external_id") or metadata.get("source_item_id") or "")
        title = str(metadata.get("source_title") or metadata.get("title") or "")
        lines.extend(
            [
                "",
                f"## chunk {index}: {source_kind}:{external_id}",
                f"title: {title}",
                _limit_text(chunk.text, 1600),
            ]
        )
    return "\n".join(lines)


def _event_delta_text(
    *,
    chunks: list[Chunk],
    source_results: list[IngestionResult],
) -> str:
    return _delta_text(kind="event", chunks=chunks, source_results=source_results)


def _lift_notion_coverage_metadata(metadata: dict[str, Any]) -> None:
    stage_results = metadata.get("stage_results")
    if not isinstance(stage_results, dict):
        return
    notion_quality = stage_results.get("notion_quality")
    if not isinstance(notion_quality, dict):
        return
    quality_metadata = notion_quality.get("metadata")
    if not isinstance(quality_metadata, dict):
        return
    for key in (
        "repository_coverage_ratio",
        "index_coverage_ratio",
        "repository_unique_page_ids",
        "index_unique_page_ids",
        "unique_page_ids",
    ):
        if key in quality_metadata:
            metadata[f"notion_{key}"] = quality_metadata[key]


def _event_delta_citation(chunk: Chunk) -> Citation:
    metadata = dict(chunk.metadata or {})
    source_item_id = str(
        metadata.get("source_item_id")
        or metadata.get("external_id")
        or chunk.document_id
        or chunk.id
    )
    source_kind = str(metadata.get("source_kind") or metadata.get("source_type") or "")
    title = str(metadata.get("source_title") or metadata.get("title") or source_kind or "source")
    label = f"{source_kind}:{title}" if source_kind and title else title
    return Citation(
        source_item_id=source_item_id,
        chunk_id=chunk.id,
        label=label[:160],
        quote=_limit_text(chunk.text, 360),
        access_scope=dict(metadata.get("access_scope") or {}),
        metadata={
            "source": "auto_index_update",
            "source_kind": source_kind,
            "external_id": str(metadata.get("external_id") or ""),
            "document_id": chunk.document_id,
        },
    )


def _limit_text(value: str, limit: int) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _schedule_skip_reason(config: RuntimeConfig, request: AutoIndexUpdateRequest) -> str:
    if request.trigger not in {"schedule", "automation"}:
        return ""
    if not config.scheduler.auto_index_enabled:
        return "auto_index_disabled"
    timezone = _schedule_timezone(config)
    now = request.scheduled_at or datetime.now(timezone)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone)
    else:
        now = now.astimezone(timezone)
    if now.weekday() not in set(config.scheduler.auto_index_weekdays):
        return "weekday_not_allowed"
    hour, minute = _parse_hhmm(config.scheduler.auto_index_time)
    if now.hour != hour or now.minute != minute:
        return "outside_scheduled_time"
    return ""


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = (value or "00:00").split(":", 1)
    return int(hour), int(minute)


def _schedule_timezone(config: RuntimeConfig):
    name = str(getattr(config.scheduler, "auto_index_timezone", "") or "Asia/Tokyo")
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return UTC


def _workflow_extraction_lookback_days(config: RuntimeConfig) -> int:
    workflow_extraction = getattr(config, "workflow_extraction", None)
    return normalize_lookback_days(getattr(workflow_extraction, "lookback_days", 1))


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
    return sanitize_payload_metadata(value)


def _ensure_before_deadline(deadline: datetime) -> None:
    if datetime.now(UTC) > deadline:
        raise TimeoutError("auto_index_max_runtime_exceeded")


def _heartbeat_interval_seconds(ttl_minutes: int) -> float:
    ttl_seconds = max(60, int(ttl_minutes) * 60)
    return float(max(10, min(60, ttl_seconds // 3)))


class _LockHeartbeat:
    def __init__(self, *, lock: Any, interval_seconds: float) -> None:
        self._lock = lock
        self._interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="auto-index-lock-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._lock.refresh()
            except Exception:
                time.sleep(0)
