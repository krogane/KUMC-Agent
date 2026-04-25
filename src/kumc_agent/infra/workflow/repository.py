from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Protocol

from kumc_agent.domain.models.retrieval import Citation
from kumc_agent.domain.models.workflow import (
    ApprovalRecord,
    Event,
    EventCandidate,
    Meeting,
    ScheduleCandidate,
    ScheduleEvent,
    Task,
    TaskCandidate,
)
from kumc_agent.infra.database.postgres import PostgresClient

_MIN_DT = datetime.min.replace(tzinfo=UTC)
_MAX_DT = datetime.max.replace(tzinfo=UTC)


class WorkflowRepository(Protocol):
    def save_task_candidate(self, candidate: TaskCandidate) -> TaskCandidate:
        ...

    def get_task_candidate(self, candidate_id: str) -> TaskCandidate | None:
        ...

    def list_task_candidates(self, *, status: str | None = None) -> list[TaskCandidate]:
        ...

    def update_task_candidate_status(
        self,
        *,
        candidate_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> TaskCandidate:
        ...

    def save_task(self, task: Task) -> Task:
        ...

    def get_task(self, task_id: str) -> Task | None:
        ...

    def list_tasks(
        self,
        *,
        status: str | None = None,
        related_event_id: str | None = None,
    ) -> list[Task]:
        ...

    def save_event(self, event: Event) -> Event:
        ...

    def save_event_candidate(self, candidate: EventCandidate) -> EventCandidate:
        ...

    def get_event_candidate(self, candidate_id: str) -> EventCandidate | None:
        ...

    def list_event_candidates(self, *, status: str | None = None) -> list[EventCandidate]:
        ...

    def update_event_candidate_status(
        self,
        *,
        candidate_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> EventCandidate:
        ...

    def get_event(self, event_id: str) -> Event | None:
        ...

    def list_events(self, *, status: str | None = None) -> list[Event]:
        ...

    def save_meeting(self, meeting: Meeting) -> Meeting:
        ...

    def list_meetings(self, *, related_event_id: str | None = None) -> list[Meeting]:
        ...

    def save_schedule(self, schedule: ScheduleEvent) -> ScheduleEvent:
        ...

    def save_schedule_candidate(self, candidate: ScheduleCandidate) -> ScheduleCandidate:
        ...

    def get_schedule_candidate(self, candidate_id: str) -> ScheduleCandidate | None:
        ...

    def list_schedule_candidates(
        self,
        *,
        status: str | None = None,
    ) -> list[ScheduleCandidate]:
        ...

    def update_schedule_candidate_status(
        self,
        *,
        candidate_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduleCandidate:
        ...

    def get_schedule(self, schedule_id: str) -> ScheduleEvent | None:
        ...

    def list_schedules(
        self,
        *,
        related_event_id: str | None = None,
        status: str | None = None,
    ) -> list[ScheduleEvent]:
        ...

    def save_approval(self, record: ApprovalRecord) -> ApprovalRecord:
        ...

    def list_approvals(
        self,
        *,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> list[ApprovalRecord]:
        ...


@dataclass(frozen=True)
class FileWorkflowRepository:
    root_dir: Path

    def save_task_candidate(self, candidate: TaskCandidate) -> TaskCandidate:
        stored = _touch(candidate)
        _append_jsonl(self.root_dir / "task_candidates.jsonl", _task_candidate_payload(stored))
        return stored

    def get_task_candidate(self, candidate_id: str) -> TaskCandidate | None:
        return _latest_by_id(
            self.root_dir / "task_candidates.jsonl",
            _task_candidate_from_payload,
        ).get(candidate_id)

    def list_task_candidates(self, *, status: str | None = None) -> list[TaskCandidate]:
        candidates = list(
            _latest_by_id(
                self.root_dir / "task_candidates.jsonl",
                _task_candidate_from_payload,
            ).values()
        )
        if status:
            candidates = [candidate for candidate in candidates if candidate.status == status]
        return sorted(candidates, key=lambda candidate: candidate.created_at or _MIN_DT)

    def update_task_candidate_status(
        self,
        *,
        candidate_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> TaskCandidate:
        candidate = self.get_task_candidate(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        next_metadata = dict(candidate.metadata)
        next_metadata.update(metadata or {})
        return self.save_task_candidate(
            replace(candidate, status=status, metadata=next_metadata)
        )

    def save_task(self, task: Task) -> Task:
        stored = _touch(task)
        _append_jsonl(self.root_dir / "tasks.jsonl", _task_payload(stored))
        return stored

    def get_task(self, task_id: str) -> Task | None:
        return _latest_by_id(self.root_dir / "tasks.jsonl", _task_from_payload).get(task_id)

    def list_tasks(
        self,
        *,
        status: str | None = None,
        related_event_id: str | None = None,
    ) -> list[Task]:
        tasks = list(_latest_by_id(self.root_dir / "tasks.jsonl", _task_from_payload).values())
        if status:
            tasks = [task for task in tasks if task.status == status]
        if related_event_id:
            tasks = [task for task in tasks if task.related_event_id == related_event_id]
        return sorted(tasks, key=lambda task: task.created_at or _MIN_DT)

    def save_event(self, event: Event) -> Event:
        stored = _touch(event)
        _append_jsonl(self.root_dir / "events.jsonl", _event_payload(stored))
        return stored

    def save_event_candidate(self, candidate: EventCandidate) -> EventCandidate:
        stored = _touch(candidate)
        _append_jsonl(self.root_dir / "event_candidates.jsonl", _event_candidate_payload(stored))
        return stored

    def get_event_candidate(self, candidate_id: str) -> EventCandidate | None:
        return _latest_by_id(
            self.root_dir / "event_candidates.jsonl",
            _event_candidate_from_payload,
        ).get(candidate_id)

    def list_event_candidates(self, *, status: str | None = None) -> list[EventCandidate]:
        candidates = list(
            _latest_by_id(
                self.root_dir / "event_candidates.jsonl",
                _event_candidate_from_payload,
            ).values()
        )
        if status:
            candidates = [candidate for candidate in candidates if candidate.status == status]
        return sorted(candidates, key=lambda candidate: candidate.created_at or _MIN_DT)

    def update_event_candidate_status(
        self,
        *,
        candidate_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> EventCandidate:
        candidate = self.get_event_candidate(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        next_metadata = dict(candidate.metadata)
        next_metadata.update(metadata or {})
        return self.save_event_candidate(
            replace(candidate, status=status, metadata=next_metadata)
        )

    def get_event(self, event_id: str) -> Event | None:
        return _latest_by_id(self.root_dir / "events.jsonl", _event_from_payload).get(event_id)

    def list_events(self, *, status: str | None = None) -> list[Event]:
        events = list(_latest_by_id(self.root_dir / "events.jsonl", _event_from_payload).values())
        if status:
            events = [event for event in events if event.status == status]
        return sorted(events, key=lambda event: event.starts_at or event.created_at or _MAX_DT)

    def save_meeting(self, meeting: Meeting) -> Meeting:
        stored = _touch(meeting)
        _append_jsonl(self.root_dir / "meetings.jsonl", _meeting_payload(stored))
        return stored

    def list_meetings(self, *, related_event_id: str | None = None) -> list[Meeting]:
        meetings = list(
            _latest_by_id(self.root_dir / "meetings.jsonl", _meeting_from_payload).values()
        )
        if related_event_id:
            meetings = [
                meeting for meeting in meetings if meeting.related_event_id == related_event_id
            ]
        return sorted(meetings, key=lambda meeting: meeting.scheduled_at or meeting.created_at or _MAX_DT)

    def save_schedule(self, schedule: ScheduleEvent) -> ScheduleEvent:
        stored = _touch(schedule)
        _append_jsonl(self.root_dir / "schedule_events.jsonl", _schedule_payload(stored))
        return stored

    def save_schedule_candidate(self, candidate: ScheduleCandidate) -> ScheduleCandidate:
        stored = _touch(candidate)
        _append_jsonl(
            self.root_dir / "schedule_candidates.jsonl",
            _schedule_candidate_payload(stored),
        )
        return stored

    def get_schedule_candidate(self, candidate_id: str) -> ScheduleCandidate | None:
        return _latest_by_id(
            self.root_dir / "schedule_candidates.jsonl",
            _schedule_candidate_from_payload,
        ).get(candidate_id)

    def list_schedule_candidates(
        self,
        *,
        status: str | None = None,
    ) -> list[ScheduleCandidate]:
        candidates = list(
            _latest_by_id(
                self.root_dir / "schedule_candidates.jsonl",
                _schedule_candidate_from_payload,
            ).values()
        )
        if status:
            candidates = [candidate for candidate in candidates if candidate.status == status]
        return sorted(candidates, key=lambda candidate: candidate.created_at or _MIN_DT)

    def update_schedule_candidate_status(
        self,
        *,
        candidate_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduleCandidate:
        candidate = self.get_schedule_candidate(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        next_metadata = dict(candidate.metadata)
        next_metadata.update(metadata or {})
        return self.save_schedule_candidate(
            replace(candidate, status=status, metadata=next_metadata)
        )

    def get_schedule(self, schedule_id: str) -> ScheduleEvent | None:
        return _latest_by_id(
            self.root_dir / "schedule_events.jsonl",
            _schedule_from_payload,
        ).get(schedule_id)

    def list_schedules(
        self,
        *,
        related_event_id: str | None = None,
        status: str | None = None,
    ) -> list[ScheduleEvent]:
        schedules = list(
            _latest_by_id(
                self.root_dir / "schedule_events.jsonl",
                _schedule_from_payload,
            ).values()
        )
        if related_event_id:
            schedules = [
                schedule for schedule in schedules if schedule.related_event_id == related_event_id
            ]
        if status:
            schedules = [schedule for schedule in schedules if schedule.status == status]
        return sorted(schedules, key=lambda schedule: schedule.starts_at or schedule.created_at or _MAX_DT)

    def save_approval(self, record: ApprovalRecord) -> ApprovalRecord:
        stored = replace(record, created_at=record.created_at or datetime.now(UTC))
        _append_jsonl(self.root_dir / "approval_records.jsonl", _approval_payload(stored))
        return stored

    def list_approvals(
        self,
        *,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> list[ApprovalRecord]:
        records = _read_jsonl(
            self.root_dir / "approval_records.jsonl",
            _approval_from_payload,
        )
        if target_type:
            records = [record for record in records if record.target_type == target_type]
        if target_id:
            records = [record for record in records if record.target_id == target_id]
        return sorted(records, key=lambda record: record.created_at or _MIN_DT)


@dataclass(frozen=True)
class PostgresWorkflowRepository:
    postgres: PostgresClient

    def save_task_candidate(self, candidate: TaskCandidate) -> TaskCandidate:
        stored = _touch(candidate)
        payload = _task_candidate_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into task_candidates (
                      id, title, description, proposed_assignee_user_id,
                      proposed_due_at, related_event_id, evidence, confidence,
                      status, created_by, metadata, created_at, updated_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s, %s)
                    on conflict (id) do update set
                      title = excluded.title,
                      description = excluded.description,
                      proposed_assignee_user_id = excluded.proposed_assignee_user_id,
                      proposed_due_at = excluded.proposed_due_at,
                      related_event_id = excluded.related_event_id,
                      evidence = excluded.evidence,
                      confidence = excluded.confidence,
                      status = excluded.status,
                      metadata = excluded.metadata,
                      updated_at = excluded.updated_at
                    """,
                    (
                        payload["id"],
                        payload["title"],
                        payload["description"],
                        payload["proposed_assignee_user_id"],
                        _parse_datetime(payload["proposed_due_at"]),
                        payload["related_event_id"],
                        json.dumps(payload["evidence"], ensure_ascii=False, default=str),
                        payload["confidence"],
                        payload["status"],
                        payload["created_by"],
                        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                        _parse_datetime(payload["created_at"]),
                        _parse_datetime(payload["updated_at"]),
                    ),
                )
            conn.commit()
        return stored

    def get_task_candidate(self, candidate_id: str) -> TaskCandidate | None:
        rows = self._fetch(
            "select * from task_candidates where id = %s",
            (candidate_id,),
        )
        return _task_candidate_from_row(rows[0]) if rows else None

    def list_task_candidates(self, *, status: str | None = None) -> list[TaskCandidate]:
        if status:
            rows = self._fetch(
                "select * from task_candidates where status = %s order by created_at asc",
                (status,),
            )
        else:
            rows = self._fetch("select * from task_candidates order by created_at asc", ())
        return [_task_candidate_from_row(row) for row in rows]

    def update_task_candidate_status(
        self,
        *,
        candidate_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> TaskCandidate:
        candidate = self.get_task_candidate(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        next_metadata = dict(candidate.metadata)
        next_metadata.update(metadata or {})
        return self.save_task_candidate(
            replace(candidate, status=status, metadata=next_metadata)
        )

    def save_task(self, task: Task) -> Task:
        stored = _touch(task)
        payload = _task_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into tasks (
                      id, title, description, assignee_user_id, due_at,
                      related_event_id, source_candidate_id, status, priority,
                      evidence, metadata, created_at, updated_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                    on conflict (id) do update set
                      title = excluded.title,
                      description = excluded.description,
                      assignee_user_id = excluded.assignee_user_id,
                      due_at = excluded.due_at,
                      related_event_id = excluded.related_event_id,
                      status = excluded.status,
                      priority = excluded.priority,
                      evidence = excluded.evidence,
                      metadata = excluded.metadata,
                      updated_at = excluded.updated_at
                    """,
                    (
                        payload["id"],
                        payload["title"],
                        payload["description"],
                        payload["assignee_user_id"],
                        _parse_datetime(payload["due_at"]),
                        payload["related_event_id"],
                        payload["source_candidate_id"],
                        payload["status"],
                        payload["priority"],
                        json.dumps(payload["evidence"], ensure_ascii=False, default=str),
                        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                        _parse_datetime(payload["created_at"]),
                        _parse_datetime(payload["updated_at"]),
                    ),
                )
            conn.commit()
        return stored

    def get_task(self, task_id: str) -> Task | None:
        rows = self._fetch("select * from tasks where id = %s", (task_id,))
        return _task_from_row(rows[0]) if rows else None

    def list_tasks(
        self,
        *,
        status: str | None = None,
        related_event_id: str | None = None,
    ) -> list[Task]:
        where: list[str] = []
        params: list[object] = []
        if status:
            where.append("status = %s")
            params.append(status)
        if related_event_id:
            where.append("related_event_id = %s")
            params.append(related_event_id)
        sql = "select * from tasks"
        if where:
            sql += " where " + " and ".join(where)
        sql += " order by created_at asc"
        return [_task_from_row(row) for row in self._fetch(sql, tuple(params))]

    def save_event(self, event: Event) -> Event:
        stored = _touch(event)
        payload = _event_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into events (
                      id, title, summary, starts_at, ends_at, place, status,
                      related_source_ids, metadata, created_at, updated_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                    on conflict (id) do update set
                      title = excluded.title,
                      summary = excluded.summary,
                      starts_at = excluded.starts_at,
                      ends_at = excluded.ends_at,
                      place = excluded.place,
                      status = excluded.status,
                      related_source_ids = excluded.related_source_ids,
                      metadata = excluded.metadata,
                      updated_at = excluded.updated_at
                    """,
                    (
                        payload["id"],
                        payload["title"],
                        payload["summary"],
                        _parse_datetime(payload["starts_at"]),
                        _parse_datetime(payload["ends_at"]),
                        payload["place"],
                        payload["status"],
                        json.dumps(payload["related_source_ids"], ensure_ascii=False),
                        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                        _parse_datetime(payload["created_at"]),
                        _parse_datetime(payload["updated_at"]),
                    ),
                )
            conn.commit()
        return stored

    def save_event_candidate(self, candidate: EventCandidate) -> EventCandidate:
        stored = _touch(candidate)
        payload = _event_candidate_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into event_candidates (
                      id, title, summary, starts_at, ends_at, place,
                      related_source_ids, evidence, confidence, status,
                      created_by, metadata, created_at, updated_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s::jsonb, %s, %s)
                    on conflict (id) do update set
                      title = excluded.title,
                      summary = excluded.summary,
                      starts_at = excluded.starts_at,
                      ends_at = excluded.ends_at,
                      place = excluded.place,
                      related_source_ids = excluded.related_source_ids,
                      evidence = excluded.evidence,
                      confidence = excluded.confidence,
                      status = excluded.status,
                      metadata = excluded.metadata,
                      updated_at = excluded.updated_at
                    """,
                    (
                        payload["id"],
                        payload["title"],
                        payload["summary"],
                        _parse_datetime(payload["starts_at"]),
                        _parse_datetime(payload["ends_at"]),
                        payload["place"],
                        json.dumps(payload["related_source_ids"], ensure_ascii=False),
                        json.dumps(payload["evidence"], ensure_ascii=False, default=str),
                        payload["confidence"],
                        payload["status"],
                        payload["created_by"],
                        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                        _parse_datetime(payload["created_at"]),
                        _parse_datetime(payload["updated_at"]),
                    ),
                )
            conn.commit()
        return stored

    def get_event_candidate(self, candidate_id: str) -> EventCandidate | None:
        rows = self._fetch("select * from event_candidates where id = %s", (candidate_id,))
        return _event_candidate_from_row(rows[0]) if rows else None

    def list_event_candidates(self, *, status: str | None = None) -> list[EventCandidate]:
        if status:
            rows = self._fetch(
                "select * from event_candidates where status = %s order by created_at asc",
                (status,),
            )
        else:
            rows = self._fetch("select * from event_candidates order by created_at asc", ())
        return [_event_candidate_from_row(row) for row in rows]

    def update_event_candidate_status(
        self,
        *,
        candidate_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> EventCandidate:
        candidate = self.get_event_candidate(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        next_metadata = dict(candidate.metadata)
        next_metadata.update(metadata or {})
        return self.save_event_candidate(
            replace(candidate, status=status, metadata=next_metadata)
        )

    def get_event(self, event_id: str) -> Event | None:
        rows = self._fetch("select * from events where id = %s", (event_id,))
        return _event_from_row(rows[0]) if rows else None

    def list_events(self, *, status: str | None = None) -> list[Event]:
        if status:
            rows = self._fetch(
                "select * from events where status = %s order by starts_at asc nulls last, created_at asc",
                (status,),
            )
        else:
            rows = self._fetch(
                "select * from events order by starts_at asc nulls last, created_at asc",
                (),
            )
        return [_event_from_row(row) for row in rows]

    def save_meeting(self, meeting: Meeting) -> Meeting:
        stored = _touch(meeting)
        payload = _meeting_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into meetings (
                      id, title, scheduled_at, related_event_id, agenda_markdown,
                      minutes_markdown, decisions, open_questions,
                      task_candidate_ids, metadata, created_at, updated_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s)
                    on conflict (id) do update set
                      title = excluded.title,
                      scheduled_at = excluded.scheduled_at,
                      related_event_id = excluded.related_event_id,
                      agenda_markdown = excluded.agenda_markdown,
                      minutes_markdown = excluded.minutes_markdown,
                      decisions = excluded.decisions,
                      open_questions = excluded.open_questions,
                      task_candidate_ids = excluded.task_candidate_ids,
                      metadata = excluded.metadata,
                      updated_at = excluded.updated_at
                    """,
                    _meeting_sql_values(payload),
                )
            conn.commit()
        return stored

    def list_meetings(self, *, related_event_id: str | None = None) -> list[Meeting]:
        if related_event_id:
            rows = self._fetch(
                "select * from meetings where related_event_id = %s order by scheduled_at asc nulls last, created_at asc",
                (related_event_id,),
            )
        else:
            rows = self._fetch(
                "select * from meetings order by scheduled_at asc nulls last, created_at asc",
                (),
            )
        return [_meeting_from_row(row) for row in rows]

    def save_schedule(self, schedule: ScheduleEvent) -> ScheduleEvent:
        stored = _touch(schedule)
        payload = _schedule_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into schedule_events (
                      id, title, starts_at, ends_at, place, related_event_id,
                      status, metadata, created_at, updated_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    on conflict (id) do update set
                      title = excluded.title,
                      starts_at = excluded.starts_at,
                      ends_at = excluded.ends_at,
                      place = excluded.place,
                      related_event_id = excluded.related_event_id,
                      status = excluded.status,
                      metadata = excluded.metadata,
                      updated_at = excluded.updated_at
                    """,
                    (
                        payload["id"],
                        payload["title"],
                        _parse_datetime(payload["starts_at"]),
                        _parse_datetime(payload["ends_at"]),
                        payload["place"],
                        payload["related_event_id"],
                        payload["status"],
                        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                        _parse_datetime(payload["created_at"]),
                        _parse_datetime(payload["updated_at"]),
                    ),
                )
            conn.commit()
        return stored

    def save_schedule_candidate(self, candidate: ScheduleCandidate) -> ScheduleCandidate:
        stored = _touch(candidate)
        payload = _schedule_candidate_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into schedule_candidates (
                      id, title, starts_at, ends_at, place, related_event_id,
                      evidence, confidence, status, created_by, metadata,
                      created_at, updated_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s, %s)
                    on conflict (id) do update set
                      title = excluded.title,
                      starts_at = excluded.starts_at,
                      ends_at = excluded.ends_at,
                      place = excluded.place,
                      related_event_id = excluded.related_event_id,
                      evidence = excluded.evidence,
                      confidence = excluded.confidence,
                      status = excluded.status,
                      metadata = excluded.metadata,
                      updated_at = excluded.updated_at
                    """,
                    (
                        payload["id"],
                        payload["title"],
                        _parse_datetime(payload["starts_at"]),
                        _parse_datetime(payload["ends_at"]),
                        payload["place"],
                        payload["related_event_id"],
                        json.dumps(payload["evidence"], ensure_ascii=False, default=str),
                        payload["confidence"],
                        payload["status"],
                        payload["created_by"],
                        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                        _parse_datetime(payload["created_at"]),
                        _parse_datetime(payload["updated_at"]),
                    ),
                )
            conn.commit()
        return stored

    def get_schedule_candidate(self, candidate_id: str) -> ScheduleCandidate | None:
        rows = self._fetch("select * from schedule_candidates where id = %s", (candidate_id,))
        return _schedule_candidate_from_row(rows[0]) if rows else None

    def list_schedule_candidates(
        self,
        *,
        status: str | None = None,
    ) -> list[ScheduleCandidate]:
        if status:
            rows = self._fetch(
                "select * from schedule_candidates where status = %s order by created_at asc",
                (status,),
            )
        else:
            rows = self._fetch("select * from schedule_candidates order by created_at asc", ())
        return [_schedule_candidate_from_row(row) for row in rows]

    def update_schedule_candidate_status(
        self,
        *,
        candidate_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduleCandidate:
        candidate = self.get_schedule_candidate(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        next_metadata = dict(candidate.metadata)
        next_metadata.update(metadata or {})
        return self.save_schedule_candidate(
            replace(candidate, status=status, metadata=next_metadata)
        )

    def get_schedule(self, schedule_id: str) -> ScheduleEvent | None:
        rows = self._fetch("select * from schedule_events where id = %s", (schedule_id,))
        return _schedule_from_row(rows[0]) if rows else None

    def list_schedules(
        self,
        *,
        related_event_id: str | None = None,
        status: str | None = None,
    ) -> list[ScheduleEvent]:
        where: list[str] = []
        params: list[object] = []
        if related_event_id:
            where.append("related_event_id = %s")
            params.append(related_event_id)
        if status:
            where.append("status = %s")
            params.append(status)
        sql = "select * from schedule_events"
        if where:
            sql += " where " + " and ".join(where)
        sql += " order by starts_at asc nulls last, created_at asc"
        return [_schedule_from_row(row) for row in self._fetch(sql, tuple(params))]

    def save_approval(self, record: ApprovalRecord) -> ApprovalRecord:
        stored = replace(record, created_at=record.created_at or datetime.now(UTC))
        payload = _approval_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into approval_records (
                      id, target_type, target_id, action, actor_id, comment,
                      before_payload, after_payload, evidence, created_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                    on conflict (id) do nothing
                    """,
                    (
                        payload["id"],
                        payload["target_type"],
                        payload["target_id"],
                        payload["action"],
                        payload["actor_id"],
                        payload["comment"],
                        json.dumps(payload["before"], ensure_ascii=False, default=str),
                        json.dumps(payload["after"], ensure_ascii=False, default=str),
                        json.dumps(payload["evidence"], ensure_ascii=False, default=str),
                        _parse_datetime(payload["created_at"]),
                    ),
                )
            conn.commit()
        return stored

    def list_approvals(
        self,
        *,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> list[ApprovalRecord]:
        where: list[str] = []
        params: list[object] = []
        if target_type:
            where.append("target_type = %s")
            params.append(target_type)
        if target_id:
            where.append("target_id = %s")
            params.append(target_id)
        sql = "select * from approval_records"
        if where:
            sql += " where " + " and ".join(where)
        sql += " order by created_at asc"
        return [_approval_from_row(row) for row in self._fetch(sql, tuple(params))]

    def _fetch(self, sql: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())


def build_workflow_repository(
    *,
    postgres: PostgresClient,
    fallback_dir: Path,
) -> WorkflowRepository:
    if postgres.is_configured():
        return PostgresWorkflowRepository(postgres=postgres)
    return FileWorkflowRepository(root_dir=fallback_dir)


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
    items: list[Any] = []
    with path.open("r", encoding="utf-8") as fr:
        for line in fr:
            if line.strip():
                items.append(loader(json.loads(line)))
    return items


def _latest_by_id(path: Path, loader: Any) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for item in _read_jsonl(path, loader):
        latest[item.id] = item
    return latest


def _citation_payload(citation: Citation) -> dict[str, object]:
    return dict(citation.__dict__)


def _citation_from_payload(payload: dict[str, object]) -> Citation:
    return Citation(
        source_item_id=str(payload.get("source_item_id") or ""),
        chunk_id=str(payload.get("chunk_id") or ""),
        label=str(payload.get("label") or ""),
        url=str(payload.get("url") or ""),
        quote=str(payload.get("quote") or ""),
        score=float(payload["score"]) if payload.get("score") is not None else None,
    )


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _json_payload(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _task_candidate_payload(candidate: TaskCandidate) -> dict[str, object]:
    return {
        "id": candidate.id,
        "title": candidate.title,
        "description": candidate.description,
        "proposed_assignee_user_id": candidate.proposed_assignee_user_id,
        "proposed_due_at": _dt(candidate.proposed_due_at),
        "related_event_id": candidate.related_event_id,
        "evidence": [_citation_payload(citation) for citation in candidate.evidence],
        "confidence": candidate.confidence,
        "status": candidate.status,
        "created_by": candidate.created_by,
        "metadata": dict(candidate.metadata),
        "created_at": _dt(candidate.created_at),
        "updated_at": _dt(candidate.updated_at),
    }


def _task_candidate_from_payload(payload: dict[str, object]) -> TaskCandidate:
    evidence = _json_payload(payload.get("evidence") or [])
    return TaskCandidate(
        id=str(payload["id"]),
        title=str(payload["title"]),
        description=payload.get("description") and str(payload["description"]),
        proposed_assignee_user_id=payload.get("proposed_assignee_user_id")
        and str(payload["proposed_assignee_user_id"]),
        proposed_due_at=_parse_datetime(payload.get("proposed_due_at")),
        related_event_id=payload.get("related_event_id") and str(payload["related_event_id"]),
        evidence=tuple(_citation_from_payload(dict(item)) for item in evidence),
        confidence=str(payload.get("confidence") or "low"),
        status=str(payload.get("status") or "proposed"),
        created_by=str(payload.get("created_by") or "agent"),
        metadata=dict(_json_payload(payload.get("metadata") or {})),
        created_at=_parse_datetime(payload.get("created_at")),
        updated_at=_parse_datetime(payload.get("updated_at")),
    )


def _task_candidate_from_row(row: tuple[object, ...]) -> TaskCandidate:
    return _task_candidate_from_payload(
        {
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "proposed_assignee_user_id": row[3],
            "proposed_due_at": row[4],
            "related_event_id": row[5],
            "evidence": row[6],
            "confidence": row[7],
            "status": row[8],
            "created_by": row[9],
            "metadata": row[10],
            "created_at": row[11],
            "updated_at": row[12],
        }
    )


def _task_payload(task: Task) -> dict[str, object]:
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description,
        "assignee_user_id": task.assignee_user_id,
        "due_at": _dt(task.due_at),
        "related_event_id": task.related_event_id,
        "source_candidate_id": task.source_candidate_id,
        "status": task.status,
        "priority": task.priority,
        "evidence": [_citation_payload(citation) for citation in task.evidence],
        "metadata": dict(task.metadata),
        "created_at": _dt(task.created_at),
        "updated_at": _dt(task.updated_at),
    }


def _task_from_payload(payload: dict[str, object]) -> Task:
    evidence = _json_payload(payload.get("evidence") or [])
    return Task(
        id=str(payload["id"]),
        title=str(payload["title"]),
        description=payload.get("description") and str(payload["description"]),
        assignee_user_id=payload.get("assignee_user_id") and str(payload["assignee_user_id"]),
        due_at=_parse_datetime(payload.get("due_at")),
        related_event_id=payload.get("related_event_id") and str(payload["related_event_id"]),
        source_candidate_id=payload.get("source_candidate_id") and str(payload["source_candidate_id"]),
        status=str(payload.get("status") or "todo"),
        priority=str(payload.get("priority") or "normal"),
        evidence=tuple(_citation_from_payload(dict(item)) for item in evidence),
        metadata=dict(_json_payload(payload.get("metadata") or {})),
        created_at=_parse_datetime(payload.get("created_at")),
        updated_at=_parse_datetime(payload.get("updated_at")),
    )


def _task_from_row(row: tuple[object, ...]) -> Task:
    return _task_from_payload(
        {
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "assignee_user_id": row[3],
            "due_at": row[4],
            "related_event_id": row[5],
            "source_candidate_id": row[6],
            "status": row[7],
            "priority": row[8],
            "evidence": row[9],
            "metadata": row[10],
            "created_at": row[11],
            "updated_at": row[12],
        }
    )


def _event_payload(event: Event) -> dict[str, object]:
    return {
        "id": event.id,
        "title": event.title,
        "summary": event.summary,
        "starts_at": _dt(event.starts_at),
        "ends_at": _dt(event.ends_at),
        "place": event.place,
        "status": event.status,
        "related_source_ids": list(event.related_source_ids),
        "metadata": dict(event.metadata),
        "created_at": _dt(event.created_at),
        "updated_at": _dt(event.updated_at),
    }


def _event_from_payload(payload: dict[str, object]) -> Event:
    return Event(
        id=str(payload["id"]),
        title=str(payload["title"]),
        summary=payload.get("summary") and str(payload["summary"]),
        starts_at=_parse_datetime(payload.get("starts_at")),
        ends_at=_parse_datetime(payload.get("ends_at")),
        place=payload.get("place") and str(payload["place"]),
        status=str(payload.get("status") or "planning"),
        related_source_ids=tuple(str(item) for item in _json_payload(payload.get("related_source_ids") or [])),
        metadata=dict(_json_payload(payload.get("metadata") or {})),
        created_at=_parse_datetime(payload.get("created_at")),
        updated_at=_parse_datetime(payload.get("updated_at")),
    )


def _event_from_row(row: tuple[object, ...]) -> Event:
    return _event_from_payload(
        {
            "id": row[0],
            "title": row[1],
            "summary": row[2],
            "starts_at": row[3],
            "ends_at": row[4],
            "place": row[5],
            "status": row[6],
            "related_source_ids": row[7],
            "metadata": row[8],
            "created_at": row[9],
            "updated_at": row[10],
        }
    )


def _event_candidate_payload(candidate: EventCandidate) -> dict[str, object]:
    return {
        "id": candidate.id,
        "title": candidate.title,
        "summary": candidate.summary,
        "starts_at": _dt(candidate.starts_at),
        "ends_at": _dt(candidate.ends_at),
        "place": candidate.place,
        "related_source_ids": list(candidate.related_source_ids),
        "evidence": [_citation_payload(citation) for citation in candidate.evidence],
        "confidence": candidate.confidence,
        "status": candidate.status,
        "created_by": candidate.created_by,
        "metadata": dict(candidate.metadata),
        "created_at": _dt(candidate.created_at),
        "updated_at": _dt(candidate.updated_at),
    }


def _event_candidate_from_payload(payload: dict[str, object]) -> EventCandidate:
    evidence = _json_payload(payload.get("evidence") or [])
    return EventCandidate(
        id=str(payload["id"]),
        title=str(payload["title"]),
        summary=payload.get("summary") and str(payload["summary"]),
        starts_at=_parse_datetime(payload.get("starts_at")),
        ends_at=_parse_datetime(payload.get("ends_at")),
        place=payload.get("place") and str(payload["place"]),
        related_source_ids=tuple(str(item) for item in _json_payload(payload.get("related_source_ids") or [])),
        evidence=tuple(_citation_from_payload(dict(item)) for item in evidence),
        confidence=str(payload.get("confidence") or "low"),
        status=str(payload.get("status") or "proposed"),
        created_by=str(payload.get("created_by") or "agent"),
        metadata=dict(_json_payload(payload.get("metadata") or {})),
        created_at=_parse_datetime(payload.get("created_at")),
        updated_at=_parse_datetime(payload.get("updated_at")),
    )


def _event_candidate_from_row(row: tuple[object, ...]) -> EventCandidate:
    return _event_candidate_from_payload(
        {
            "id": row[0],
            "title": row[1],
            "summary": row[2],
            "starts_at": row[3],
            "ends_at": row[4],
            "place": row[5],
            "related_source_ids": row[6],
            "evidence": row[7],
            "confidence": row[8],
            "status": row[9],
            "created_by": row[10],
            "metadata": row[11],
            "created_at": row[12],
            "updated_at": row[13],
        }
    )


def _meeting_payload(meeting: Meeting) -> dict[str, object]:
    return {
        "id": meeting.id,
        "title": meeting.title,
        "scheduled_at": _dt(meeting.scheduled_at),
        "related_event_id": meeting.related_event_id,
        "agenda_markdown": meeting.agenda_markdown,
        "minutes_markdown": meeting.minutes_markdown,
        "decisions": list(meeting.decisions),
        "open_questions": list(meeting.open_questions),
        "task_candidate_ids": list(meeting.task_candidate_ids),
        "metadata": dict(meeting.metadata),
        "created_at": _dt(meeting.created_at),
        "updated_at": _dt(meeting.updated_at),
    }


def _meeting_from_payload(payload: dict[str, object]) -> Meeting:
    return Meeting(
        id=str(payload["id"]),
        title=str(payload["title"]),
        scheduled_at=_parse_datetime(payload.get("scheduled_at")),
        related_event_id=payload.get("related_event_id") and str(payload["related_event_id"]),
        agenda_markdown=str(payload.get("agenda_markdown") or ""),
        minutes_markdown=str(payload.get("minutes_markdown") or ""),
        decisions=tuple(str(item) for item in _json_payload(payload.get("decisions") or [])),
        open_questions=tuple(str(item) for item in _json_payload(payload.get("open_questions") or [])),
        task_candidate_ids=tuple(str(item) for item in _json_payload(payload.get("task_candidate_ids") or [])),
        metadata=dict(_json_payload(payload.get("metadata") or {})),
        created_at=_parse_datetime(payload.get("created_at")),
        updated_at=_parse_datetime(payload.get("updated_at")),
    )


def _meeting_sql_values(payload: dict[str, object]) -> tuple[object, ...]:
    return (
        payload["id"],
        payload["title"],
        _parse_datetime(payload["scheduled_at"]),
        payload["related_event_id"],
        payload["agenda_markdown"],
        payload["minutes_markdown"],
        json.dumps(payload["decisions"], ensure_ascii=False),
        json.dumps(payload["open_questions"], ensure_ascii=False),
        json.dumps(payload["task_candidate_ids"], ensure_ascii=False),
        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
        _parse_datetime(payload["created_at"]),
        _parse_datetime(payload["updated_at"]),
    )


def _meeting_from_row(row: tuple[object, ...]) -> Meeting:
    return _meeting_from_payload(
        {
            "id": row[0],
            "title": row[1],
            "scheduled_at": row[2],
            "related_event_id": row[3],
            "agenda_markdown": row[4],
            "minutes_markdown": row[5],
            "decisions": row[6],
            "open_questions": row[7],
            "task_candidate_ids": row[8],
            "metadata": row[9],
            "created_at": row[10],
            "updated_at": row[11],
        }
    )


def _schedule_payload(schedule: ScheduleEvent) -> dict[str, object]:
    return {
        "id": schedule.id,
        "title": schedule.title,
        "starts_at": _dt(schedule.starts_at),
        "ends_at": _dt(schedule.ends_at),
        "place": schedule.place,
        "related_event_id": schedule.related_event_id,
        "status": schedule.status,
        "metadata": dict(schedule.metadata),
        "created_at": _dt(schedule.created_at),
        "updated_at": _dt(schedule.updated_at),
    }


def _schedule_from_payload(payload: dict[str, object]) -> ScheduleEvent:
    return ScheduleEvent(
        id=str(payload["id"]),
        title=str(payload["title"]),
        starts_at=_parse_datetime(payload.get("starts_at")),
        ends_at=_parse_datetime(payload.get("ends_at")),
        place=payload.get("place") and str(payload["place"]),
        related_event_id=payload.get("related_event_id") and str(payload["related_event_id"]),
        status=str(payload.get("status") or "planned"),
        metadata=dict(_json_payload(payload.get("metadata") or {})),
        created_at=_parse_datetime(payload.get("created_at")),
        updated_at=_parse_datetime(payload.get("updated_at")),
    )


def _schedule_from_row(row: tuple[object, ...]) -> ScheduleEvent:
    return _schedule_from_payload(
        {
            "id": row[0],
            "title": row[1],
            "starts_at": row[2],
            "ends_at": row[3],
            "place": row[4],
            "related_event_id": row[5],
            "status": row[6],
            "metadata": row[7],
            "created_at": row[8],
            "updated_at": row[9],
        }
    )


def _schedule_candidate_payload(candidate: ScheduleCandidate) -> dict[str, object]:
    return {
        "id": candidate.id,
        "title": candidate.title,
        "starts_at": _dt(candidate.starts_at),
        "ends_at": _dt(candidate.ends_at),
        "place": candidate.place,
        "related_event_id": candidate.related_event_id,
        "evidence": [_citation_payload(citation) for citation in candidate.evidence],
        "confidence": candidate.confidence,
        "status": candidate.status,
        "created_by": candidate.created_by,
        "metadata": dict(candidate.metadata),
        "created_at": _dt(candidate.created_at),
        "updated_at": _dt(candidate.updated_at),
    }


def _schedule_candidate_from_payload(payload: dict[str, object]) -> ScheduleCandidate:
    evidence = _json_payload(payload.get("evidence") or [])
    return ScheduleCandidate(
        id=str(payload["id"]),
        title=str(payload["title"]),
        starts_at=_parse_datetime(payload.get("starts_at")),
        ends_at=_parse_datetime(payload.get("ends_at")),
        place=payload.get("place") and str(payload["place"]),
        related_event_id=payload.get("related_event_id") and str(payload["related_event_id"]),
        evidence=tuple(_citation_from_payload(dict(item)) for item in evidence),
        confidence=str(payload.get("confidence") or "low"),
        status=str(payload.get("status") or "proposed"),
        created_by=str(payload.get("created_by") or "agent"),
        metadata=dict(_json_payload(payload.get("metadata") or {})),
        created_at=_parse_datetime(payload.get("created_at")),
        updated_at=_parse_datetime(payload.get("updated_at")),
    )


def _schedule_candidate_from_row(row: tuple[object, ...]) -> ScheduleCandidate:
    return _schedule_candidate_from_payload(
        {
            "id": row[0],
            "title": row[1],
            "starts_at": row[2],
            "ends_at": row[3],
            "place": row[4],
            "related_event_id": row[5],
            "evidence": row[6],
            "confidence": row[7],
            "status": row[8],
            "created_by": row[9],
            "metadata": row[10],
            "created_at": row[11],
            "updated_at": row[12],
        }
    )


def _approval_payload(record: ApprovalRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "target_type": record.target_type,
        "target_id": record.target_id,
        "action": record.action,
        "actor_id": record.actor_id,
        "comment": record.comment,
        "before": dict(record.before),
        "after": dict(record.after),
        "evidence": [_citation_payload(citation) for citation in record.evidence],
        "created_at": _dt(record.created_at),
    }


def _approval_from_payload(payload: dict[str, object]) -> ApprovalRecord:
    evidence = _json_payload(payload.get("evidence") or [])
    return ApprovalRecord(
        id=str(payload["id"]),
        target_type=str(payload["target_type"]),
        target_id=str(payload["target_id"]),
        action=str(payload["action"]),
        actor_id=str(payload.get("actor_id") or ""),
        comment=str(payload.get("comment") or ""),
        before=dict(_json_payload(payload.get("before") or {})),
        after=dict(_json_payload(payload.get("after") or {})),
        evidence=tuple(_citation_from_payload(dict(item)) for item in evidence),
        created_at=_parse_datetime(payload.get("created_at")),
    )


def _approval_from_row(row: tuple[object, ...]) -> ApprovalRecord:
    return _approval_from_payload(
        {
            "id": row[0],
            "target_type": row[1],
            "target_id": row[2],
            "action": row[3],
            "actor_id": row[4],
            "comment": row[5],
            "before": row[6],
            "after": row[7],
            "evidence": row[8],
            "created_at": row[9],
        }
    )
