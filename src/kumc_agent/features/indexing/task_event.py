from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.operations import IndexingRun
from kumc_agent.domain.models.workflow import Event, Task
from kumc_agent.domain.ports.embedders import EmbedderPort
from kumc_agent.infra.retrieval.faiss import FaissLikeIndex
from kumc_agent.infra.retrieval.sudachi_bm25 import SudachiBM25Retriever
from kumc_agent.utils.hashing import stable_hash


class WorkflowIndexRepository(Protocol):
    def list_tasks(
        self,
        *,
        status: str | None = None,
        related_event_id: str | None = None,
        assignee_user_id: str | None = None,
        due_from: datetime | None = None,
        due_to: datetime | None = None,
        priority: str | None = None,
        include_deleted: bool = False,
    ) -> list[Task]:
        ...

    def list_events(
        self,
        *,
        status: str | None = None,
        starts_from: datetime | None = None,
        starts_to: datetime | None = None,
        place: str | None = None,
        include_canceled: bool = False,
    ) -> list[Event]:
        ...


@dataclass(frozen=True)
class TaskEventIndexBuildService:
    repository: WorkflowIndexRepository
    embedder: EmbedderPort

    def rebuild(self, *, index_dir: Path) -> IndexingRun:
        target_dir = index_dir / "task_event"
        target_dir.mkdir(parents=True, exist_ok=True)
        tasks = self.repository.list_tasks(include_deleted=True)
        events = self.repository.list_events(include_canceled=True)
        active_tasks = [task for task in tasks if _task_index_status(task) == "active"]
        active_events = [event for event in events if _event_index_status(event) == "active"]
        docs = [
            *[_task_doc(task) for task in active_tasks],
            *[_event_doc(event) for event in active_events],
        ]
        chunks = [
            Chunk(
                id=str(doc["id"]),
                document_id=str(doc["id"]),
                text=str(doc["text"]),
                index=index,
                metadata=dict(doc["metadata"]),
            )
            for index, doc in enumerate(docs)
        ]
        vectors = (
            self.embedder.embed_documents([chunk.text for chunk in chunks])
            if chunks
            else np.empty((0, 1), dtype=np.float32)
        )
        FaissLikeIndex(index_dir=target_dir).build(chunks=chunks, embeddings=vectors)
        SudachiBM25Retriever(index_dir=target_dir).build(chunks)
        with (target_dir / "task_event_documents.jsonl").open("w", encoding="utf-8") as fw:
            for doc in docs:
                fw.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
        return IndexingRun(
            id=stable_hash(f"task-event-index:{datetime.now(UTC).isoformat()}")[:32],
            source_kind="task_event",
            status="succeeded",
            seen=len(tasks) + len(events),
            changed=len(active_tasks) + len(active_events),
            deleted=(len(tasks) - len(active_tasks)) + (len(events) - len(active_events)),
            metadata={
                "tasks": len(active_tasks),
                "events": len(active_events),
                "index_dir": str(target_dir),
            },
        )


def _task_index_status(task: Task) -> str:
    status = str(task.status or "").lower()
    if status == "deleted":
        return "deleted"
    if str(task.metadata.get("index_status") or "").lower() in {"deleted", "permission_lost"}:
        return "deleted"
    return "active"


def _event_index_status(event: Event) -> str:
    status = str(event.status or "").lower()
    if status == "canceled":
        return "deleted"
    if str(event.metadata.get("index_status") or "").lower() in {"deleted", "permission_lost"}:
        return "deleted"
    return "active"


def _task_doc(task: Task) -> dict[str, object]:
    text = "\n".join(
        part
        for part in (
            f"Task: {task.title}",
            f"Description: {task.description or ''}",
            f"Status: {task.status}",
            f"Priority: {task.priority}",
            f"Assignee: {task.assignee_user_id or ''}",
            f"Due: {task.due_at.isoformat() if task.due_at else ''}",
            f"Related event: {task.related_event_id or ''}",
        )
        if part.strip()
    )
    return {
        "id": f"task:{task.id}",
        "kind": "task",
        "text": text,
        "metadata": {
            "source_kind": "task_event",
            "source_type": "task",
            "task_id": task.id,
            "status": task.status,
            "priority": task.priority,
            "index_status": "active",
            "updated_at": task.updated_at.isoformat() if task.updated_at else "",
        },
    }


def _event_doc(event: Event) -> dict[str, object]:
    text = "\n".join(
        part
        for part in (
            f"Event: {event.title}",
            f"Summary: {event.summary or ''}",
            f"Status: {event.status}",
            f"Starts: {event.starts_at.isoformat() if event.starts_at else ''}",
            f"Ends: {event.ends_at.isoformat() if event.ends_at else ''}",
            f"Place: {event.place or ''}",
        )
        if part.strip()
    )
    return {
        "id": f"event:{event.id}",
        "kind": "event",
        "text": text,
        "metadata": {
            "source_kind": "task_event",
            "source_type": "event",
            "event_id": event.id,
            "status": event.status,
            "index_status": "active",
            "updated_at": event.updated_at.isoformat() if event.updated_at else "",
        },
    }
