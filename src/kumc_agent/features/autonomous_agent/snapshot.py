from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from kumc_agent.domain.models.autonomous_agent import (
    AutonomousAgentSnapshot,
    SnapshotItem,
)
from kumc_agent.domain.models.retrieval import Citation
from kumc_agent.infra.agentic import AgentTraceRepository
from kumc_agent.infra.automation import AutomationRepository
from kumc_agent.infra.ingestion.repository import IngestionRepository
from kumc_agent.infra.minecraft import ServerOperationRepository
from kumc_agent.infra.workflow import WorkflowRepository
from kumc_agent.features.workflow.extraction_window import (
    build_extraction_window,
    changed_at_from_metadata,
)

_OPEN_TASK_STATUSES = {"todo", "doing", "blocked"}


@dataclass(frozen=True)
class SnapshotCollectorConfig:
    task_lookahead_days: int = 2
    event_lookahead_days: int = 7
    stale_task_hours: int = 72
    recent_run_limit: int = 20
    rag_delta_lookback_days: int = 1
    rag_delta_lookback_hours: int = 24


class AutonomousSnapshotCollector:
    def __init__(
        self,
        *,
        workflow_repository: WorkflowRepository | None = None,
        automation_repository: AutomationRepository | None = None,
        agent_trace_repository: AgentTraceRepository | None = None,
        server_operation_repository: ServerOperationRepository | None = None,
        ingestion_repository: IngestionRepository | None = None,
        config: SnapshotCollectorConfig | None = None,
    ) -> None:
        self.workflow_repository = workflow_repository
        self.automation_repository = automation_repository
        self.agent_trace_repository = agent_trace_repository
        self.server_operation_repository = server_operation_repository
        self.ingestion_repository = ingestion_repository
        self.config = config or SnapshotCollectorConfig()

    def collect(self, *, scopes: tuple[str, ...], now: datetime | None = None) -> AutonomousAgentSnapshot:
        current = _aware(now or datetime.now(UTC))
        scope_set = set(scopes or ("tasks", "events", "rag_delta", "server_ops", "automation"))
        warnings: list[str] = []
        tasks_due_soon: list[SnapshotItem] = []
        tasks_overdue: list[SnapshotItem] = []
        tasks_stale: list[SnapshotItem] = []
        task_candidates: list[SnapshotItem] = []
        task_approval_batches: list[SnapshotItem] = []
        events_upcoming: list[SnapshotItem] = []
        events_missing_details: list[SnapshotItem] = []
        events_without_tasks: list[SnapshotItem] = []
        event_candidates: list[SnapshotItem] = []
        event_approval_batches: list[SnapshotItem] = []
        server_ops: list[SnapshotItem] = []
        automation_runs: list[SnapshotItem] = []
        recent_runs: list[SnapshotItem] = []

        if {"tasks", "events"} & scope_set and self.workflow_repository is None:
            warnings.append("workflow_repository_unconfigured")
        if "tasks" in scope_set and self.workflow_repository is not None:
            try:
                tasks_due_soon, tasks_overdue, tasks_stale = self._collect_tasks(current)
                task_candidates = [
                    _item("task_candidate", candidate.id, candidate.title, status=candidate.status)
                    for candidate in self.workflow_repository.list_task_candidates(status="proposed")
                ]
                task_approval_batches = [
                    _item(
                        "task_approval_batch",
                        batch.id,
                        f"Task approval batch {batch.id}",
                        status=batch.status,
                        metadata={"candidate_ids": list(batch.candidate_ids)},
                    )
                    for batch in self.workflow_repository.list_task_approval_batches(status="pending")
                ]
            except Exception as exc:
                warnings.append(f"tasks_collector_failed:{type(exc).__name__}")

        if "events" in scope_set and self.workflow_repository is not None:
            try:
                (
                    events_upcoming,
                    events_missing_details,
                    events_without_tasks,
                ) = self._collect_events(current)
                event_candidates = [
                    _item("event_candidate", candidate.id, candidate.title, status=candidate.status)
                    for candidate in self.workflow_repository.list_event_candidates(status="proposed")
                ]
                event_approval_batches = [
                    _item(
                        "event_approval_batch",
                        batch.id,
                        f"Event approval batch {batch.id}",
                        status=batch.status,
                        metadata={"candidate_ids": list(batch.candidate_ids)},
                    )
                    for batch in self.workflow_repository.list_event_approval_batches(status="pending")
                ]
            except Exception as exc:
                warnings.append(f"events_collector_failed:{type(exc).__name__}")

        rag_delta: tuple[SnapshotItem, ...] = tuple()
        if "rag_delta" in scope_set:
            if self.ingestion_repository is None:
                warnings.append("rag_delta_repository_unconfigured")
            else:
                try:
                    rag_delta = tuple(self._collect_rag_delta(current))
                except Exception as exc:
                    warnings.append(f"rag_delta_collector_failed:{type(exc).__name__}")

        if "server_ops" in scope_set:
            if self.server_operation_repository is None:
                warnings.append("server_operation_repository_unconfigured")
            else:
                try:
                    server_ops = [
                        _item(
                            "server_operation",
                            operation.id,
                            f"{operation.server_name}:{operation.operation}",
                            status=operation.status,
                            risk=operation.risk_level,
                        )
                        for operation in self.server_operation_repository.list_pending_for_approval()
                    ]
                except Exception as exc:
                    warnings.append(f"server_ops_collector_failed:{type(exc).__name__}")

        if "automation" in scope_set:
            if self.automation_repository is None:
                warnings.append("automation_repository_unconfigured")
            else:
                try:
                    automation_runs = [
                        _item(
                            "automation_run",
                            run.id,
                            f"{run.rule_id}:{run.trigger_key}",
                            status=run.status,
                            metadata={"rule_id": run.rule_id},
                        )
                        for run in self.automation_repository.list_runs()
                        if run.status in {"waiting_approval", "blocked"}
                    ][:20]
                except Exception as exc:
                    warnings.append(f"automation_collector_failed:{type(exc).__name__}")

        if self.agent_trace_repository is not None:
            try:
                recent_runs = [
                    _item(
                        "agent_run",
                        run.id,
                        run.query,
                        status=run.status,
                        metadata={
                            "notification_target_refs": list(
                                run.metadata.get("notification_target_refs") or []
                            ),
                            "updated_at": (run.updated_at or run.created_at).isoformat()
                            if (run.updated_at or run.created_at)
                            else "",
                        },
                    )
                    for run in self.agent_trace_repository.latest_runs(limit=self.config.recent_run_limit)
                    if run.metadata.get("agent") == "autonomous_agent"
                ]
            except Exception as exc:
                warnings.append(f"recent_runs_collector_failed:{type(exc).__name__}")

        return AutonomousAgentSnapshot(
            tasks_due_soon=tuple(tasks_due_soon),
            tasks_overdue=tuple(tasks_overdue),
            tasks_stale=tuple(tasks_stale),
            task_candidates=tuple(task_candidates),
            task_approval_batches=tuple(task_approval_batches),
            events_upcoming=tuple(events_upcoming),
            events_missing_details=tuple(events_missing_details),
            events_without_tasks=tuple(events_without_tasks),
            event_candidates=tuple(event_candidates),
            event_approval_batches=tuple(event_approval_batches),
            rag_delta=rag_delta,
            server_ops=tuple(server_ops),
            automation_runs=tuple(automation_runs),
            recent_runs=tuple(recent_runs),
            warnings=tuple(warnings),
            metadata={
                "collected_at": current.isoformat(),
                "scopes": list(scopes),
            },
        )

    def _collect_tasks(self, now: datetime) -> tuple[list[SnapshotItem], list[SnapshotItem], list[SnapshotItem]]:
        assert self.workflow_repository is not None
        due_to = now + timedelta(days=self.config.task_lookahead_days)
        tasks = [
            task
            for task in self.workflow_repository.list_tasks()
            if task.status in _OPEN_TASK_STATUSES
        ]
        due_soon: list[SnapshotItem] = []
        overdue: list[SnapshotItem] = []
        stale: list[SnapshotItem] = []
        stale_before = now - timedelta(hours=self.config.stale_task_hours)
        for task in tasks:
            due_at = _aware(task.due_at) if task.due_at else None
            item = _item(
                "task",
                task.id,
                task.title,
                status=task.status,
                due_at=due_at,
                metadata={
                    "assignee_user_id": task.assignee_user_id or "",
                    "related_event_id": task.related_event_id or "",
                },
            )
            if due_at is not None and due_at < now:
                overdue.append(item)
            elif due_at is not None and due_at <= due_to:
                due_soon.append(item)
            updated = _aware(task.updated_at or task.created_at) if (task.updated_at or task.created_at) else None
            if task.status in {"doing", "blocked"} and updated is not None and updated < stale_before:
                stale.append(item)
        return due_soon, overdue, stale

    def _collect_events(self, now: datetime) -> tuple[list[SnapshotItem], list[SnapshotItem], list[SnapshotItem]]:
        assert self.workflow_repository is not None
        starts_to = now + timedelta(days=self.config.event_lookahead_days)
        events = self.workflow_repository.list_events(starts_from=now, starts_to=starts_to)
        upcoming: list[SnapshotItem] = []
        missing: list[SnapshotItem] = []
        without_tasks: list[SnapshotItem] = []
        for event in events:
            item = _item(
                "event",
                event.id,
                event.title,
                summary=event.summary or "",
                status=event.status,
                starts_at=_aware(event.starts_at) if event.starts_at else None,
                metadata={"place": event.place or ""},
            )
            upcoming.append(item)
            if event.starts_at is None or not (event.place or "").strip():
                missing.append(item)
            related_tasks = [
                task
                for task in self.workflow_repository.list_tasks(related_event_id=event.id)
                if task.status in _OPEN_TASK_STATUSES
            ]
            if not related_tasks:
                without_tasks.append(item)
        undated = [
            event
            for event in self.workflow_repository.list_events()
            if event.starts_at is None or not (event.place or "").strip()
        ]
        for event in undated:
            if event.id not in {item.id for item in missing}:
                missing.append(
                    _item(
                        "event",
                        event.id,
                        event.title,
                        summary=event.summary or "",
                        status=event.status,
                        metadata={"place": event.place or ""},
                    )
                )
        return upcoming, missing, without_tasks

    def _collect_rag_delta(self, now: datetime) -> list[SnapshotItem]:
        assert self.ingestion_repository is not None
        window = build_extraction_window(
            lookback_days=self.config.rag_delta_lookback_days,
            extraction_at=now,
        )
        by_source: dict[str, dict[str, Any]] = {}
        for chunk in self.ingestion_repository.load_active_chunks():
            metadata = dict(chunk.metadata)
            changed_at = changed_at_from_metadata(metadata)
            if changed_at is None or changed_at < window.since:
                continue
            source_item_id = str(
                metadata.get("source_item_id")
                or metadata.get("source_id")
                or chunk.document_id
            )
            if not source_item_id:
                continue
            source_kind = str(metadata.get("source_kind") or metadata.get("source_type") or "")
            external_id = str(metadata.get("external_id") or source_item_id)
            title = str(metadata.get("source_title") or metadata.get("title") or external_id)
            url = str(metadata.get("canonical_url") or metadata.get("url") or "")
            current = by_source.setdefault(
                source_item_id,
                {
                    "id": source_item_id,
                    "title": title,
                    "source_kind": source_kind,
                    "external_id": external_id,
                    "url": url,
                    "changed_at": changed_at,
                    "chunks": [],
                    "citations": [],
                },
            )
            if changed_at > current["changed_at"]:
                current["changed_at"] = changed_at
            current["chunks"].append(chunk.text)
            current["citations"].append(
                Citation(
                    source_item_id=source_item_id,
                    chunk_id=chunk.id,
                    label=title,
                    url=url,
                    quote=_compact(chunk.text, 220),
                    access_scope=dict(metadata.get("access_scope") or {}),
                    metadata={
                        "source_kind": source_kind,
                        "external_id": external_id,
                    },
                )
            )
        items: list[SnapshotItem] = []
        for payload in sorted(by_source.values(), key=lambda item: item["changed_at"], reverse=True)[:20]:
            chunks = [str(text) for text in payload["chunks"][:3]]
            items.append(
                _item(
                    "rag_delta",
                    str(payload["id"]),
                    str(payload["title"]),
                    summary=_compact(" / ".join(chunks), 500),
                    status="changed",
                    metadata={
                        "source_kind": payload["source_kind"],
                        "external_id": payload["external_id"],
                        "changed_at": payload["changed_at"].isoformat(),
                        "chunk_count": len(payload["chunks"]),
                        **window.as_metadata(),
                    },
                    citations=tuple(payload["citations"][:3]),
                )
            )
        return items


def _item(
    kind: str,
    item_id: str,
    title: str,
    *,
    summary: str = "",
    status: str = "",
    due_at: datetime | None = None,
    starts_at: datetime | None = None,
    risk: str = "low",
    metadata: dict[str, Any] | None = None,
    citations: tuple[Citation, ...] = tuple(),
) -> SnapshotItem:
    return SnapshotItem(
        id=str(item_id),
        kind=kind,
        title=str(title),
        summary=str(summary or ""),
        status=str(status or ""),
        due_at=due_at,
        starts_at=starts_at,
        risk=risk,
        citations=citations,
        metadata=dict(metadata or {}),
    )


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _compact(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."
