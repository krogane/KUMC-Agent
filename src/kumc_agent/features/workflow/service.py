from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from kumc_agent.domain.models.audit import AuditEvent
from kumc_agent.domain.models.agentic import AgenticSearchRequest
from kumc_agent.domain.models.docgen import DocGenRequest
from kumc_agent.domain.models.operations import (
    WorkflowCandidate,
    WorkflowRun,
)
from kumc_agent.domain.models.retrieval import AccessContext, Citation, RetrievalQuery
from kumc_agent.domain.models.workflow import (
    ApprovalRecord,
    Event,
    EventCandidate,
    Meeting,
    ScheduleCandidate,
    ScheduleEvent,
    Task,
    TaskApprovalBatch,
    TaskChangeCandidate,
    TaskCandidate,
    WorkRequest,
    WorkResponse,
)
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.features.announcement.service import (
    AnnouncementDraftRequest,
    AnnouncementDraftService,
)
from kumc_agent.features.agentic import AgenticSearchService
from kumc_agent.features.docgen.service import DocGenService
from kumc_agent.features.minecraft import MinecraftSupportService
from kumc_agent.features.task_management import (
    DuplicateTaskDetector,
    TaskAccessPolicy,
    TaskExtractionService,
    TaskNotificationPlanner,
)
from kumc_agent.infra.audit.repository import AuditLogRepository
from kumc_agent.infra.operations import OperationsRepository
from kumc_agent.infra.workflow.repository import WorkflowRepository
from kumc_agent.utils.hashing import stable_hash


_TASK_KEYWORDS = (
    "todo",
    "to do",
    "task",
    "タスク",
    "やる",
    "対応",
    "作成",
    "準備",
    "確認",
    "依頼",
    "修正",
    "期限",
    "担当",
)
_DECISION_KEYWORDS = ("決定", "決まり", "決めた", "確定")
_OPEN_QUESTION_KEYWORDS = ("未決", "要確認", "確認事項", "課題", "pending")
_OPEN_TASK_STATUSES = {"todo", "doing", "blocked"}


class WorkflowService:
    def __init__(
        self,
        *,
        repository: WorkflowRepository,
        ask_service: Any | None = None,
        audit_log: AuditLogRepository | None = None,
        agentic_search: AgenticSearchService | None = None,
        docgen: DocGenService | None = None,
        announcement: AnnouncementDraftService | None = None,
        minecraft: MinecraftSupportService | None = None,
        operations: OperationsRepository | None = None,
        member_search_service: Any | None = None,
        image_search_service: Any | None = None,
        task_extractor: TaskExtractionService | None = None,
        task_access_policy: TaskAccessPolicy | None = None,
        task_duplicate_detector: DuplicateTaskDetector | None = None,
        task_notification_planner: TaskNotificationPlanner | None = None,
        llm: LLMPort | None = None,
        prompts_dir: Path | None = None,
        llm_model_name: str = "",
    ) -> None:
        self.repository = repository
        self.ask_service = ask_service
        self.audit_log = audit_log
        self.agentic_search = agentic_search
        self.docgen = docgen
        self.announcement = announcement
        self.minecraft = minecraft
        self.operations = operations
        self.member_search_service = member_search_service
        self.image_search_service = image_search_service
        self.task_extractor = task_extractor or TaskExtractionService(
            llm=llm,
            prompts_dir=prompts_dir,
            model_name=llm_model_name,
        )
        self.task_access_policy = task_access_policy or TaskAccessPolicy()
        self.task_duplicate_detector = task_duplicate_detector or DuplicateTaskDetector()
        self.task_notification_planner = task_notification_planner or TaskNotificationPlanner()

    def run(self, request: WorkRequest) -> WorkResponse:
        run_record = self._start_workflow_run(request)
        try:
            response = self._dispatch(request)
        except Exception as exc:
            self._finish_workflow_run(run_record, status="failed", error=str(exc))
            raise
        self._finish_workflow_run(
            run_record,
            status=_workflow_response_status(response),
            metadata={
                "task_candidates": len(response.task_candidates),
                "task_change_candidates": len(response.task_change_candidates),
                "task_approval_batches": len(response.task_approval_batches),
                "event_candidates": len(response.event_candidates),
                "schedule_candidates": len(response.schedule_candidates),
                "workflow_candidates": len(response.workflow_candidates),
                "assets": len(response.assets),
                "member_profiles": len(response.member_profiles),
                "events": len(response.events),
                "schedules": len(response.schedules),
                "server_operations": len(response.server_operations),
            },
        )
        return response

    def _dispatch(self, request: WorkRequest) -> WorkResponse:
        work_type = request.work_type.strip().lower()
        if work_type == "meeting_prepare":
            return self.meeting_prepare(request)
        if work_type == "meeting_minutes_draft":
            return self.meeting_minutes_draft(request)
        if work_type == "task_extract":
            return self.task_extract(request)
        if work_type == "task_add":
            return self.task_add(request)
        if work_type == "task_list":
            return self.task_list(request)
        if work_type == "task_done":
            return self.task_done(request)
        if work_type == "task_update":
            return self.task_update(request)
        if work_type == "task_delete":
            return self.task_delete(request)
        if work_type == "task_notify_due":
            return self.task_notify_due(request)
        if work_type == "task_batch_approval":
            return self.task_batch_approval(request)
        if work_type == "event_add":
            return self.event_add(request)
        if work_type == "event_list":
            return self.event_list(request)
        if work_type == "event_brief":
            return self.event_brief(request)
        if work_type == "schedule_add":
            return self.schedule_add(request)
        if work_type == "schedule_list":
            return self.schedule_list(request)
        if work_type == "doc_draft":
            return self.doc_draft(request)
        if work_type == "x_draft":
            return self.x_draft(request)
        if work_type == "announcement_draft":
            return self.announcement_draft(request)
        if work_type == "mc_status":
            return self.mc_status(request)
        if work_type == "mc_request":
            return self.mc_request(request)
        if work_type == "image_search":
            return self.image_search(request)
        if work_type == "member_search":
            return self.member_search(request)
        raise ValueError(f"Unsupported work type: {request.work_type}")

    def meeting_prepare(self, request: WorkRequest) -> WorkResponse:
        retrieved = self._retrieve(
            request,
            fallback_query=request.target or request.instruction or "例会 議題 未完了 タスク イベント",
        )
        open_tasks = [
            task
            for task in self.repository.list_tasks()
            if task.status in _OPEN_TASK_STATUSES
        ]
        events = self.repository.list_events()
        schedules = self.repository.list_schedules()
        agenda = self._build_meeting_agenda(
            instruction=request.instruction,
            retrieved_text=retrieved["text"],
            open_tasks=open_tasks,
            events=events,
            schedules=schedules,
        )
        meeting = self.repository.save_meeting(
            Meeting(
                id=stable_hash(f"meeting-prepare:{agenda}")[:32],
                title=_first_nonempty_line(request.instruction) or "例会準備",
                agenda_markdown=agenda,
                task_candidate_ids=tuple(),
                metadata={"source": "meeting_prepare"},
            )
        )
        self._audit("workflow.meeting_prepare", request.access, "succeeded", meeting.id)
        text = _compact_lines(
            [
                "例会準備案を作成しました。",
                f"未完了タスク: {len(open_tasks)}件",
                f"関連イベント: {len(events)}件",
                "詳細に議題案と確認事項をまとめています。",
            ]
        )
        return WorkResponse(
            text=text,
            detail_markdown=agenda,
            tasks=tuple(open_tasks),
            events=tuple(events),
            schedules=tuple(schedules),
            meetings=(meeting,),
            metadata={"retrieved_citations": len(retrieved["citations"])},
        )

    def meeting_minutes_draft(self, request: WorkRequest) -> WorkResponse:
        source_text = self._source_text(request)
        retrieved = self._retrieve(
            request,
            fallback_query=request.target or request.instruction or "議事録 ToDo 決定事項",
        )
        combined = _join_text(source_text, retrieved["text"])
        candidates, extraction_metadata = self._extract_and_store_candidates(
            combined,
            request.access,
            evidence=tuple(retrieved["citations"]),
            metadata={"source": "meeting_minutes_draft"},
        )
        decisions = _matching_lines(combined, _DECISION_KEYWORDS)
        open_questions = _matching_lines(combined, _OPEN_QUESTION_KEYWORDS)
        minutes = self._build_minutes_markdown(
            source_text=combined,
            decisions=decisions,
            open_questions=open_questions,
            candidates=candidates,
        )
        meeting = self.repository.save_meeting(
            Meeting(
                id=stable_hash(f"meeting-minutes:{minutes}")[:32],
                title=_first_nonempty_line(request.instruction) or "議事録下書き",
                minutes_markdown=minutes,
                decisions=tuple(decisions),
                open_questions=tuple(open_questions),
                task_candidate_ids=tuple(candidate.id for candidate in candidates),
                metadata={"source": "meeting_minutes_draft"},
            )
        )
        self._audit("workflow.meeting_minutes_draft", request.access, "succeeded", meeting.id)
        return WorkResponse(
            text=f"議事録下書きを作成し、TaskCandidate を {len(candidates)} 件登録しました。",
            detail_markdown=minutes,
            task_candidates=tuple(candidates),
            meetings=(meeting,),
            metadata={
                "decisions": len(decisions),
                "open_questions": len(open_questions),
                "extraction": extraction_metadata,
            },
        )

    def task_extract(self, request: WorkRequest) -> WorkResponse:
        if not self.task_access_policy.can_create_candidate(request.access):
            return self._task_forbidden_response("workflow.task_extract", request.access)
        source_text = self._source_text(request)
        retrieved = self._retrieve(
            request,
            fallback_query=request.target or request.instruction or "タスク 担当 期限 ToDo",
        )
        combined = _join_text(source_text, retrieved["text"])
        candidates, extraction_metadata = self._extract_and_store_candidates(
            combined,
            request.access,
            evidence=tuple(retrieved["citations"]),
            metadata={"source": "task_extract"},
        )
        detail = self._format_task_candidates(candidates)
        self._audit("workflow.task_extract", request.access, "succeeded", "task_candidates")
        return WorkResponse(
            text=f"TaskCandidate を {len(candidates)} 件登録しました。承認されるまで Task 正本には入りません。",
            detail_markdown=detail,
            task_candidates=tuple(candidates),
            metadata={
                "candidate_count": len(candidates),
                "extraction": extraction_metadata,
            },
        )

    def task_add(self, request: WorkRequest) -> WorkResponse:
        if not self.task_access_policy.can_create_candidate(request.access):
            return self._task_forbidden_response("workflow.task_add", request.access)
        text = self._source_text(request)
        title = _extract_labeled_value(text, ("task", "タスク", "title", "件名"))
        title = title or _task_title(_first_nonempty_line(text) or text)
        if not title:
            return WorkResponse(
                text="タスク名を確認してください。例: タスク: 会場予約",
                detail_markdown="TaskCandidate は作成していません。",
                metadata={
                    "missing_fields": ["title"],
                    "candidate_created": False,
                },
            )
        due_at = _extract_datetime(text)
        due_mentioned = bool(re.search(r"(期限|due)[:：]?\s*\S+", text, flags=re.IGNORECASE))
        if due_mentioned and due_at is None:
            return WorkResponse(
                text="期限の解釈を確認してください。YYYY-MM-DD 形式で指定してください。",
                detail_markdown="TaskCandidate は作成していません。",
                metadata={
                    "ambiguous_fields": ["due_at"],
                    "candidate_created": False,
                },
            )
        assignee = _extract_assignee(text)
        priority = _extract_priority(text)
        related_event_id = _extract_labeled_value(text, ("event", "event_id", "イベントID", "関連イベント"))
        candidate = self.repository.save_task_candidate(
            self._annotate_task_duplicate(
                TaskCandidate(
                    id=stable_hash(
                        f"task-candidate:manual:{title}:{assignee or ''}:{due_at.isoformat() if due_at else ''}:{related_event_id or ''}"
                    )[:32],
                    title=_clean_title(title),
                    description=text.strip() or None,
                    proposed_assignee_user_id=assignee,
                    proposed_due_at=due_at,
                    related_event_id=related_event_id,
                    confidence="high",
                    status="proposed",
                    created_by="user",
                    metadata={
                        "source": "task_add",
                        "created_by_user_id": request.access.user_id,
                        "priority": priority,
                    },
                )
            )
        )
        self._audit("workflow.task_add", request.access, "proposed", candidate.id)
        return WorkResponse(
            text=f"TaskCandidate を作成しました。承認されるまで Task 正本には入りません: {candidate.title}",
            detail_markdown=self._format_task_candidates([candidate]),
            task_candidates=(candidate,),
        )

    def task_list(self, request: WorkRequest) -> WorkResponse:
        if not self.task_access_policy.can_list(request.access):
            return self._task_forbidden_response("workflow.task_list", request.access)
        conditions = _extract_task_list_conditions(request.instruction)
        tasks = self.repository.list_tasks(**conditions)
        candidates = self.repository.list_task_candidates(status="proposed")
        if conditions.get("related_event_id"):
            candidates = [
                candidate
                for candidate in candidates
                if candidate.related_event_id == conditions["related_event_id"]
            ]
        detail = "\n\n".join(
            [
                self._format_tasks(tasks),
                self._format_task_candidates(candidates),
            ]
        )
        return WorkResponse(
            text=f"Task は {len(tasks)} 件、承認待ち TaskCandidate は {len(candidates)} 件です。",
            detail_markdown=detail,
            tasks=tuple(tasks),
            task_candidates=tuple(candidates),
            metadata={"filters": _serializable_task_filters(conditions)},
        )

    def task_done(self, request: WorkRequest) -> WorkResponse:
        target_id = request.target.strip() or _extract_labeled_value(
            request.instruction,
            ("task_id", "id", "タスクID"),
        )
        if not target_id:
            raise ValueError("task_done requires target task id.")
        task = self.repository.get_task(target_id)
        if task is None:
            raise KeyError(target_id)
        if not self.task_access_policy.can_update_task_status(request.access, task):
            return self._task_forbidden_response("workflow.task_done", request.access)
        stored = self.repository.save_task(
            replace(
                task,
                status="done",
                metadata={
                    **task.metadata,
                    "done_by": request.access.user_id,
                    "done_comment": request.instruction,
                },
            )
        )
        self._audit("workflow.task_done", request.access, "succeeded", stored.id)
        return WorkResponse(
            text=f"Task を done にしました: {stored.title}",
            detail_markdown=self._format_tasks([stored]),
            tasks=(stored,),
        )

    def task_update(self, request: WorkRequest) -> WorkResponse:
        target_id = request.target.strip() or _extract_labeled_value(
            request.instruction,
            ("task_id", "id", "タスクID"),
        )
        if not target_id:
            raise ValueError("task_update requires target task id.")
        task = self.repository.get_task(target_id)
        if task is None:
            raise KeyError(target_id)
        if not self.task_access_policy.can_update_task_status(request.access, task):
            return self._task_forbidden_response("workflow.task_update", request.access)
        after = _task_payload_for_change(task)
        title = _extract_labeled_value(request.instruction, ("title", "タイトル", "件名"))
        assignee = _extract_assignee(request.instruction)
        due_at = _extract_datetime(request.instruction)
        status = _extract_task_status(request.instruction)
        priority = _extract_priority(request.instruction)
        description = _extract_labeled_value(request.instruction, ("description", "説明", "本文"))
        if title:
            after["title"] = _clean_title(title)
        if assignee:
            after["assignee_user_id"] = assignee
        if due_at:
            after["due_at"] = due_at.isoformat()
        if status:
            after["status"] = status
        if priority:
            after["priority"] = priority
        if description:
            after["description"] = description
        if after == _task_payload_for_change(task):
            return WorkResponse(
                text="変更内容を解釈できませんでした。title/status/担当/期限/priority などを指定してください。",
                metadata={"candidate_created": False, "missing_fields": ["change"]},
            )
        candidate = self.repository.save_task_change_candidate(
            TaskChangeCandidate(
                id=stable_hash(f"task-change:update:{target_id}:{json.dumps(after, sort_keys=True, default=str)}")[:32],
                task_id=target_id,
                operation="update",
                before=_task_payload_for_change(task),
                after=after,
                reason=request.instruction,
                confidence="high",
                status="proposed",
                created_by="user",
                metadata={"source": "task_update", "created_by_user_id": request.access.user_id},
            )
        )
        self._audit("workflow.task_update", request.access, "proposed", candidate.id)
        return WorkResponse(
            text=f"Task変更候補を作成しました。承認されるまで正本は変更されません: {target_id}",
            detail_markdown=self._format_task_change_candidates([candidate]),
            task_change_candidates=(candidate,),
        )

    def task_delete(self, request: WorkRequest) -> WorkResponse:
        target_id = request.target.strip() or _extract_labeled_value(
            request.instruction,
            ("task_id", "id", "タスクID"),
        )
        if not target_id:
            raise ValueError("task_delete requires target task id.")
        task = self.repository.get_task(target_id)
        if task is None:
            raise KeyError(target_id)
        if not self.task_access_policy.can_update_task_status(request.access, task):
            return self._task_forbidden_response("workflow.task_delete", request.access)
        candidate = self.repository.save_task_change_candidate(
            TaskChangeCandidate(
                id=stable_hash(f"task-change:delete:{target_id}:{request.instruction}")[:32],
                task_id=target_id,
                operation="delete",
                before=_task_payload_for_change(task),
                after={"status": "deleted"},
                reason=request.instruction,
                confidence="high",
                status="proposed",
                created_by="user",
                metadata={"source": "task_delete", "created_by_user_id": request.access.user_id},
            )
        )
        self._audit("workflow.task_delete", request.access, "proposed", candidate.id)
        return WorkResponse(
            text=f"Task削除候補を作成しました。承認されるまで正本は削除されません: {target_id}",
            detail_markdown=self._format_task_change_candidates([candidate]),
            task_change_candidates=(candidate,),
        )

    def task_notify_due(self, request: WorkRequest) -> WorkResponse:
        if not request.access.is_admin:
            return self._task_forbidden_response("workflow.task_notify_due", request.access)
        before_days = _extract_int_labeled_value(request.instruction, ("days", "日数")) or 1
        tasks = self.repository.list_tasks()
        selected = self.task_notification_planner.due_notifications(
            tasks=tasks,
            before_days=before_days,
        )
        stored: list[Task] = []
        now_text = datetime.now(UTC).isoformat()
        for task in selected:
            key = "overdue" if task.due_at and task.due_at < datetime.now(UTC) else "due_soon"
            notifications = dict(task.metadata.get("notifications") or {})
            notifications[key] = now_text
            stored.append(
                self.repository.save_task(
                    replace(
                        task,
                        metadata={
                            **task.metadata,
                            "notifications": notifications,
                        },
                    )
                )
            )
        self._audit("workflow.task_notify_due", request.access, "succeeded", "tasks")
        return WorkResponse(
            text=f"通知対象 Task は {len(stored)} 件です。",
            detail_markdown=self._format_tasks(stored),
            tasks=tuple(stored),
            metadata={"notification_count": len(stored), "before_days": before_days},
        )

    def task_batch_approval(self, request: WorkRequest) -> WorkResponse:
        if not request.access.is_admin:
            return self._task_forbidden_response("workflow.task_batch_approval", request.access)
        candidates = [
            candidate
            for candidate in self.repository.list_task_candidates(status="proposed")
            if candidate.created_by == "agent"
        ]
        change_candidates = self.repository.list_task_change_candidates(status="proposed")
        idempotency_key = stable_hash(
            "task-batch:"
            + ":".join(candidate.id for candidate in candidates)
            + ":"
            + ":".join(candidate.id for candidate in change_candidates)
        )[:32]
        existing = self.repository.get_task_approval_batch(idempotency_key)
        if existing is not None:
            batch = existing
        else:
            batch = self.repository.save_task_approval_batch(
                TaskApprovalBatch(
                    id=idempotency_key,
                    candidate_ids=tuple(candidate.id for candidate in candidates),
                    change_candidate_ids=tuple(candidate.id for candidate in change_candidates),
                    period_start=None,
                    period_end=datetime.now(UTC),
                    notification_channel_id=_extract_labeled_value(
                        request.instruction,
                        ("channel", "channel_id", "通知先"),
                    ),
                    status="pending",
                    metadata={
                        "source": "task_batch_approval",
                        "idempotency_key": idempotency_key,
                    },
                )
            )
        detail = "\n\n".join(
            [
                self._format_task_candidates(candidates),
                self._format_task_change_candidates(change_candidates),
            ]
        )
        self._audit("workflow.task_batch_approval", request.access, "succeeded", batch.id)
        return WorkResponse(
            text=(
                f"Task承認batchを作成しました: {batch.id} "
                f"候補 {len(candidates)} 件 / 変更候補 {len(change_candidates)} 件"
            ),
            detail_markdown=detail,
            task_candidates=tuple(candidates),
            task_change_candidates=tuple(change_candidates),
            task_approval_batches=(batch,),
            metadata={"batch_id": batch.id},
        )

    def event_add(self, request: WorkRequest) -> WorkResponse:
        event = self._event_from_text(self._source_text(request))
        candidate = self.repository.save_event_candidate(
            EventCandidate(
                id=stable_hash(f"event-candidate:{event.title}:{event.starts_at}:{event.place}")[:32],
                title=event.title,
                summary=event.summary,
                starts_at=event.starts_at,
                ends_at=event.ends_at,
                place=event.place,
                related_source_ids=event.related_source_ids,
                confidence="high" if event.starts_at and event.place else "medium",
                status="proposed",
                created_by="user",
                metadata={
                    **event.metadata,
                    "created_by_user_id": request.access.user_id,
                },
            )
        )
        self._audit("workflow.event_add", request.access, "proposed", candidate.id)
        return WorkResponse(
            text=f"EventCandidate を作成しました。承認されるまで Event 正本には入りません: {candidate.title}",
            detail_markdown=self._format_event_candidates([candidate]),
            event_candidates=(candidate,),
        )

    def event_list(self, request: WorkRequest) -> WorkResponse:
        events = self.repository.list_events()
        return WorkResponse(
            text=f"Event は {len(events)} 件あります。",
            detail_markdown=self._format_events(events),
            events=tuple(events),
        )

    def event_brief(self, request: WorkRequest) -> WorkResponse:
        event = self._resolve_event(request)
        related_tasks = (
            self.repository.list_tasks(related_event_id=event.id) if event else []
        )
        if event:
            related_tasks = [
                task for task in related_tasks if task.status in _OPEN_TASK_STATUSES
            ]
        else:
            related_tasks = [
                task
                for task in self.repository.list_tasks()
                if task.status in _OPEN_TASK_STATUSES
            ]
        retrieved = self._retrieve(
            request,
            fallback_query=request.target or (event.title if event else request.instruction) or "イベント 関連資料",
        )
        detail = self._build_event_brief(
            event=event,
            tasks=related_tasks,
            retrieved_text=retrieved["text"],
            citations=tuple(retrieved["citations"]),
        )
        title = event.title if event else "指定イベントなし"
        return WorkResponse(
            text=f"Event brief を作成しました: {title} / 未完了タスク {len(related_tasks)} 件",
            detail_markdown=detail,
            tasks=tuple(related_tasks),
            events=(event,) if event else tuple(),
            metadata={"retrieved_citations": len(retrieved["citations"])},
        )

    def schedule_add(self, request: WorkRequest) -> WorkResponse:
        schedule = self._schedule_from_text(self._source_text(request))
        candidate = self.repository.save_schedule_candidate(
            ScheduleCandidate(
                id=stable_hash(
                    f"schedule-candidate:{schedule.title}:{schedule.starts_at}:{schedule.place}:{schedule.related_event_id}"
                )[:32],
                title=schedule.title,
                starts_at=schedule.starts_at,
                ends_at=schedule.ends_at,
                place=schedule.place,
                related_event_id=schedule.related_event_id,
                confidence="high" if schedule.starts_at and schedule.place else "medium",
                status="proposed",
                created_by="user",
                metadata={
                    **schedule.metadata,
                    "created_by_user_id": request.access.user_id,
                },
            )
        )
        self._audit("workflow.schedule_add", request.access, "proposed", candidate.id)
        return WorkResponse(
            text=f"ScheduleCandidate を作成しました。承認されるまで Schedule 正本には入りません: {candidate.title}",
            detail_markdown=self._format_schedule_candidates([candidate]),
            schedule_candidates=(candidate,),
        )

    def schedule_list(self, request: WorkRequest) -> WorkResponse:
        schedules = self.repository.list_schedules()
        return WorkResponse(
            text=f"Schedule は {len(schedules)} 件あります。",
            detail_markdown=self._format_schedules(schedules),
            schedules=tuple(schedules),
        )

    def doc_draft(self, request: WorkRequest) -> WorkResponse:
        docgen = self.docgen or DocGenService()
        retrieved = self._agentic_or_retrieve(
            request,
            fallback_query=request.target or request.instruction or "週報 意思決定 メモ 根拠",
        )
        title = _extract_labeled_value(request.instruction, ("title", "タイトル", "件名"))
        title = title or _first_nonempty_line(request.instruction) or "資料下書き"
        doc_type = _doc_type(request.instruction)
        document = docgen.run(
            DocGenRequest(
                title=title,
                instruction=request.instruction,
                source_text=_join_text(request.target, retrieved["text"]),
                doc_type=doc_type,
                audience=_extract_labeled_value(request.instruction, ("対象", "audience")) or "",
                purpose=_extract_labeled_value(request.instruction, ("目的", "purpose")) or "",
                citations=tuple(retrieved["citations"]),
                public=False,
            )
        )
        self._audit("workflow.doc_draft", request.access, "succeeded", document.plan.id)
        return WorkResponse(
            text=f"{document.plan.doc_type} の Markdown 下書きを作成しました。",
            detail_markdown=document.markdown,
            warnings=document.warnings,
            metadata={"document_plan_id": document.plan.id},
        )

    def x_draft(self, request: WorkRequest) -> WorkResponse:
        retrieved = self._agentic_or_retrieve(
            request,
            fallback_query=request.target or request.instruction or "告知 SNS 投稿 過去文面",
        )
        source = _join_text(request.instruction, request.target, retrieved["text"])
        candidates = _x_draft_candidates(source)
        best = _pick_x_draft(candidates)
        detail = "\n".join(
            [
                "# X draft tournament",
                "",
                "## Selected",
                best,
                "",
                "## Candidates",
                *[f"{index + 1}. {candidate}" for index, candidate in enumerate(candidates)],
                "",
                "## Fact check",
                "- 外部投稿前に日時・場所・参加条件を確認してください。",
                "- 投稿は実行していません。",
            ]
        )
        self._audit("workflow.x_draft", request.access, "succeeded", "x_draft")
        return WorkResponse(
            text="X 投稿案を作成しました。投稿は実行していません。",
            detail_markdown=detail,
            metadata={"candidate_count": len(candidates), "selected_length": len(best)},
        )

    def announcement_draft(self, request: WorkRequest) -> WorkResponse:
        if self.announcement is None:
            from kumc_agent.infra.announcement.repository import FileAnnouncementRepository

            self.announcement = AnnouncementDraftService(
                repository=FileAnnouncementRepository(root_dir=Path("data/announcement")),
                docgen=self.docgen or DocGenService(),
            )
        retrieved = self._agentic_or_retrieve(
            request,
            fallback_query=request.target or request.instruction or "告知 関連資料 日時 場所",
        )
        title = _extract_labeled_value(request.instruction, ("title", "タイトル", "件名"))
        title = title or _first_nonempty_line(request.instruction) or "告知下書き"
        draft = self.announcement.draft(
            AnnouncementDraftRequest(
                title=title,
                instruction=request.instruction,
                source_text=_join_text(request.target, retrieved["text"]),
                medium=_announcement_medium(request.instruction),
                audience=_extract_labeled_value(request.instruction, ("対象", "audience")) or "",
                created_by=request.access.user_id or "agent",
            )
        )
        self._audit("workflow.announcement_draft", request.access, "succeeded", draft.id)
        return WorkResponse(
            text=f"Announcement draft を保存しました: {draft.title} / status={draft.status}",
            detail_markdown=draft.body_markdown,
            warnings=tuple(finding.message for finding in draft.fact_checks),
            metadata={"announcement_id": draft.id, "status": draft.status},
        )

    def mc_status(self, request: WorkRequest) -> WorkResponse:
        if self.minecraft is None:
            raise RuntimeError("Minecraft support service is not configured.")
        result = self.minecraft.status(access=request.access)
        self._audit("workflow.mc_status", request.access, "succeeded", "minecraft")
        return WorkResponse(
            text=result.text,
            detail_markdown=result.detail_markdown,
            warnings=result.warnings,
        )

    def mc_request(self, request: WorkRequest) -> WorkResponse:
        if self.minecraft is None:
            raise RuntimeError("Minecraft support service is not configured.")
        result = self.minecraft.request(
            instruction=request.instruction,
            target=request.target,
            access=request.access,
        )
        target = result.operation.id if result.operation else "minecraft"
        self._audit("workflow.mc_request", request.access, "dry_run", target)
        return WorkResponse(
            text=result.text,
            detail_markdown=result.detail_markdown,
            server_operations=(result.operation,) if result.operation else tuple(),
            warnings=result.warnings,
            metadata={
                "server_operation_id": result.operation.id if result.operation else "",
                "status": result.operation.status if result.operation else "",
                "risk_level": result.operation.risk_level if result.operation else "",
                "execution_allowed": False,
            },
        )

    def image_search(self, request: WorkRequest) -> WorkResponse:
        query = (request.target or request.instruction).strip()
        if self.operations is None:
            return WorkResponse(
                text="画像検索 repository は未設定です。",
                detail_markdown="Asset repository is not configured.",
                metadata={"route": "image_search", "configured": False},
            )
        if self.image_search_service is not None:
            from kumc_agent.features.image_search import ImageSearchRequest

            result = self.image_search_service.search(
                ImageSearchRequest(
                    query=query,
                    access_context=request.access,
                    metadata={"workflow": "image_search"},
                )
            )
            return WorkResponse(
                text=result.text,
                detail_markdown=result.detail_markdown,
                assets=result.assets,
                metadata=result.metadata,
            )
        assets = tuple(self.operations.list_assets(query=query))
        detail = self._format_assets(list(assets))
        if not assets:
            detail = "\n".join(
                [
                    "# Image Search",
                    "",
                    "該当する Asset は登録されていません。",
                ]
            )
        return WorkResponse(
            text=f"画像候補は {len(assets)} 件です。再利用可否はこの結果では判断しません。",
            detail_markdown=detail,
            assets=assets,
            metadata={
                "route": "image_search",
                "query": query,
                "degraded": True,
                "degraded_reason": "image_search_service_unavailable",
            },
        )

    def member_search(self, request: WorkRequest) -> WorkResponse:
        if self.operations is None:
            return WorkResponse(
                text="メンバー検索 repository は未設定です。",
                detail_markdown="Member profile repository is not configured.",
                metadata={"route": "member_search", "configured": False},
            )
        query = (request.target or request.instruction).strip()
        if self.member_search_service is not None:
            result = self.member_search_service.search(query=query, access=request.access)
            return WorkResponse(
                text=result.text,
                detail_markdown=result.detail_markdown,
                member_profiles=result.profiles,
                metadata=result.metadata,
            )
        if not _can_search_members(request.access):
            return WorkResponse(
                text="権限がありません。",
                detail_markdown="member_search requires configured member search policy. 対象情報の有無は表示しません。",
                metadata={"route": "member_search", "authorized": False},
            )
        profiles = tuple(self.operations.search_member_profiles(query=query))
        return WorkResponse(
            text=f"条件に合うメンバー候補は {len(profiles)} 件です。担当決定には本人または運営確認が必要です。",
            detail_markdown=self._format_member_profiles(list(profiles)),
            member_profiles=profiles,
            metadata={"route": "member_search", "search_conditions": {"fallback_query": query}},
        )

    def approval(
        self,
        *,
        action: str,
        target_type: str,
        target_id: str = "",
        comment: str = "",
        access: AccessContext,
    ) -> WorkResponse:
        normalized_action = action.strip().lower()
        normalized_type = target_type.strip().lower() or "task"
        if normalized_type == "event":
            return self._event_approval(
                action=normalized_action,
                target_id=target_id,
                comment=comment,
                access=access,
            )
        if normalized_type == "schedule":
            return self._schedule_approval(
                action=normalized_action,
                target_id=target_id,
                comment=comment,
                access=access,
            )
        if normalized_type in {
            "announcement",
            "automation_rule",
            "server_operation",
            "finance_record",
            "member_assignment",
            "other",
        }:
            return self._generic_approval(
                action=normalized_action,
                target_type=normalized_type,
                target_id=target_id,
                comment=comment,
                access=access,
            )
        if normalized_type != "task":
            raise ValueError("Unsupported approval target type.")
        if normalized_action == "list":
            if not self.task_access_policy.can_list(access):
                return self._task_forbidden_response("workflow.approval.list", access)
            candidates = self.repository.list_task_candidates(status="proposed")
            change_candidates = self.repository.list_task_change_candidates(status="proposed")
            return WorkResponse(
                text=(
                    f"承認待ち TaskCandidate は {len(candidates)} 件、"
                    f"Task変更候補は {len(change_candidates)} 件です。"
                ),
                detail_markdown="\n\n".join(
                    [
                        self._format_task_candidates(candidates),
                        self._format_task_change_candidates(change_candidates),
                    ]
                ),
                task_candidates=tuple(candidates),
                task_change_candidates=tuple(change_candidates),
            )
        if normalized_action == "show":
            candidate = self.repository.get_task_candidate(target_id) if target_id else None
            change_candidate = (
                self.repository.get_task_change_candidate(target_id) if target_id else None
            )
            task = self.repository.get_task(target_id) if target_id else None
            approvals = self.repository.list_approvals(target_type="task", target_id=target_id)
            if candidate is None and task is None and change_candidate is None:
                raise KeyError(target_id)
            if candidate and not self.task_access_policy.can_show_candidate(access, candidate):
                return self._task_forbidden_response("workflow.approval.show", access)
            if task and not self.task_access_policy.can_show_task(access, task):
                return self._task_forbidden_response("workflow.approval.show", access)
            if change_candidate:
                target_task = self.repository.get_task(change_candidate.task_id)
                if target_task and not self.task_access_policy.can_show_task(access, target_task):
                    return self._task_forbidden_response("workflow.approval.show", access)
            details = []
            if candidate is not None:
                details.append(self._format_task_candidates([candidate]))
            if change_candidate is not None:
                details.append(self._format_task_change_candidates([change_candidate]))
            if task is not None:
                details.append(self._format_tasks([task]))
            if approvals:
                details.append(self._format_approvals(approvals))
            return WorkResponse(
                text=f"task approval target を表示します: {target_id}",
                detail_markdown="\n\n".join(details),
                task_candidates=(candidate,) if candidate else tuple(),
                task_change_candidates=(change_candidate,) if change_candidate else tuple(),
                tasks=(task,) if task else tuple(),
                approvals=tuple(approvals),
            )
        if normalized_action == "edit":
            if self.repository.get_task_change_candidate(target_id) is not None:
                try:
                    change_candidate, record = self._edit_task_change_candidate(
                        target_id=target_id,
                        comment=comment,
                        access=access,
                    )
                except PermissionError:
                    return self._task_forbidden_response("workflow.approval.edit", access)
                self._audit("workflow.approval.edit", access, "succeeded", target_id)
                return WorkResponse(
                    text=f"Task変更候補を編集しました: {change_candidate.task_id}",
                    detail_markdown=self._format_task_change_candidates([change_candidate]),
                    task_change_candidates=(change_candidate,),
                    approvals=(record,),
                )
            try:
                candidate, record = self._edit_task_candidate(
                    target_id=target_id,
                    comment=comment,
                    access=access,
                )
            except PermissionError:
                return self._task_forbidden_response("workflow.approval.edit", access)
            self._audit("workflow.approval.edit", access, "succeeded", target_id)
            return WorkResponse(
                text=f"TaskCandidate を編集しました: {candidate.title}",
                detail_markdown=self._format_task_candidates([candidate]),
                task_candidates=(candidate,),
                approvals=(record,),
            )
        if normalized_action == "approve":
            change_candidate = self.repository.get_task_change_candidate(target_id)
            try:
                if change_candidate is not None:
                    task, record = self._approve_task_change_candidate(
                        target_id=target_id,
                        comment=comment,
                        access=access,
                    )
                else:
                    task, record = self._approve_task_candidate(
                        target_id=target_id,
                        comment=comment,
                        access=access,
                    )
            except PermissionError:
                return self._task_forbidden_response("workflow.approval.approve", access)
            self._audit("workflow.approval.approve", access, "succeeded", target_id)
            return WorkResponse(
                text=f"Task承認を反映しました: {task.title}",
                detail_markdown=self._format_tasks([task]),
                tasks=(task,),
                approvals=(record,),
            )
        if normalized_action == "reject":
            change_candidate = self.repository.get_task_change_candidate(target_id)
            if change_candidate is not None:
                if not self.task_access_policy.can_approve(access):
                    return self._task_forbidden_response("workflow.approval.reject", access)
                rejected_change = self.repository.update_task_change_candidate_status(
                    candidate_id=target_id,
                    status="rejected",
                    metadata={"rejected_by": access.user_id, "rejection_comment": comment},
                )
                record = self.repository.save_approval(
                    ApprovalRecord(
                        id=str(uuid4()),
                        target_type="task",
                        target_id=target_id,
                        action="reject",
                        actor_id=access.user_id,
                        comment=comment,
                        before={},
                        after=asdict(rejected_change),
                        evidence=rejected_change.evidence,
                    )
                )
                self._audit("workflow.approval.reject", access, "succeeded", target_id)
                return WorkResponse(
                    text=f"Task変更候補を却下しました: {rejected_change.task_id}",
                    detail_markdown=self._format_task_change_candidates([rejected_change]),
                    task_change_candidates=(rejected_change,),
                    approvals=(record,),
                )
            candidate_for_policy = self.repository.get_task_candidate(target_id)
            if candidate_for_policy is None:
                raise KeyError(target_id)
            if not self.task_access_policy.can_reject_candidate(access, candidate_for_policy):
                return self._task_forbidden_response("workflow.approval.reject", access)
            candidate = self.repository.update_task_candidate_status(
                candidate_id=target_id,
                status="rejected",
                metadata={"rejected_by": access.user_id, "rejection_comment": comment},
            )
            record = self.repository.save_approval(
                ApprovalRecord(
                    id=str(uuid4()),
                    target_type="task",
                    target_id=target_id,
                    action="reject",
                    actor_id=access.user_id,
                    comment=comment,
                    before={},
                    after=asdict(candidate),
                    evidence=candidate.evidence,
                )
            )
            self._audit("workflow.approval.reject", access, "succeeded", target_id)
            return WorkResponse(
                text=f"TaskCandidate を却下しました: {candidate.title}",
                detail_markdown=self._format_task_candidates([candidate]),
                task_candidates=(candidate,),
                approvals=(record,),
            )
        raise ValueError(f"Unsupported approval action: {action}")

    def _event_approval(
        self,
        *,
        action: str,
        target_id: str,
        comment: str,
        access: AccessContext,
    ) -> WorkResponse:
        if action == "list":
            candidates = self.repository.list_event_candidates(status="proposed")
            return WorkResponse(
                text=f"承認待ち EventCandidate は {len(candidates)} 件です。",
                detail_markdown=self._format_event_candidates(candidates),
                event_candidates=tuple(candidates),
            )
        if action == "show":
            candidate = self.repository.get_event_candidate(target_id) if target_id else None
            event = self.repository.get_event(target_id) if target_id else None
            approvals = self.repository.list_approvals(target_type="event", target_id=target_id)
            if candidate is None and event is None:
                raise KeyError(target_id)
            details = []
            if candidate is not None:
                details.append(self._format_event_candidates([candidate]))
            if event is not None:
                details.append(self._format_events([event]))
            if approvals:
                details.append(self._format_approvals(approvals))
            return WorkResponse(
                text=f"event approval target を表示します: {target_id}",
                detail_markdown="\n\n".join(details),
                event_candidates=(candidate,) if candidate else tuple(),
                events=(event,) if event else tuple(),
                approvals=tuple(approvals),
            )
        if action == "edit":
            candidate, record = self._edit_event_candidate(
                target_id=target_id,
                comment=comment,
                access=access,
            )
            self._audit("workflow.approval.event.edit", access, "succeeded", target_id)
            return WorkResponse(
                text=f"EventCandidate を編集しました: {candidate.title}",
                detail_markdown=self._format_event_candidates([candidate]),
                event_candidates=(candidate,),
                approvals=(record,),
            )
        if action == "approve":
            event, record = self._approve_event_candidate(
                target_id=target_id,
                comment=comment,
                access=access,
            )
            self._audit("workflow.approval.event.approve", access, "succeeded", target_id)
            return WorkResponse(
                text=f"EventCandidate を承認し Event 正本に登録しました: {event.title}",
                detail_markdown=self._format_events([event]),
                events=(event,),
                approvals=(record,),
            )
        if action == "reject":
            candidate = self.repository.update_event_candidate_status(
                candidate_id=target_id,
                status="rejected",
                metadata={"rejected_by": access.user_id, "rejection_comment": comment},
            )
            record = self.repository.save_approval(
                ApprovalRecord(
                    id=str(uuid4()),
                    target_type="event",
                    target_id=target_id,
                    action="reject",
                    actor_id=access.user_id,
                    comment=comment,
                    before={},
                    after=asdict(candidate),
                    evidence=candidate.evidence,
                )
            )
            self._audit("workflow.approval.event.reject", access, "succeeded", target_id)
            return WorkResponse(
                text=f"EventCandidate を却下しました: {candidate.title}",
                detail_markdown=self._format_event_candidates([candidate]),
                event_candidates=(candidate,),
                approvals=(record,),
            )
        raise ValueError(f"Unsupported event approval action: {action}")

    def _schedule_approval(
        self,
        *,
        action: str,
        target_id: str,
        comment: str,
        access: AccessContext,
    ) -> WorkResponse:
        if action == "list":
            candidates = self.repository.list_schedule_candidates(status="proposed")
            return WorkResponse(
                text=f"承認待ち ScheduleCandidate は {len(candidates)} 件です。",
                detail_markdown=self._format_schedule_candidates(candidates),
                schedule_candidates=tuple(candidates),
            )
        if action == "show":
            candidate = self.repository.get_schedule_candidate(target_id) if target_id else None
            schedule = self.repository.get_schedule(target_id) if target_id else None
            approvals = self.repository.list_approvals(target_type="schedule", target_id=target_id)
            if candidate is None and schedule is None:
                raise KeyError(target_id)
            details = []
            if candidate is not None:
                details.append(self._format_schedule_candidates([candidate]))
            if schedule is not None:
                details.append(self._format_schedules([schedule]))
            if approvals:
                details.append(self._format_approvals(approvals))
            return WorkResponse(
                text=f"schedule approval target を表示します: {target_id}",
                detail_markdown="\n\n".join(details),
                schedule_candidates=(candidate,) if candidate else tuple(),
                schedules=(schedule,) if schedule else tuple(),
                approvals=tuple(approvals),
            )
        if action == "edit":
            candidate, record = self._edit_schedule_candidate(
                target_id=target_id,
                comment=comment,
                access=access,
            )
            self._audit("workflow.approval.schedule.edit", access, "succeeded", target_id)
            return WorkResponse(
                text=f"ScheduleCandidate を編集しました: {candidate.title}",
                detail_markdown=self._format_schedule_candidates([candidate]),
                schedule_candidates=(candidate,),
                approvals=(record,),
            )
        if action == "approve":
            schedule, record = self._approve_schedule_candidate(
                target_id=target_id,
                comment=comment,
                access=access,
            )
            self._audit("workflow.approval.schedule.approve", access, "succeeded", target_id)
            return WorkResponse(
                text=f"ScheduleCandidate を承認し Schedule 正本に登録しました: {schedule.title}",
                detail_markdown=self._format_schedules([schedule]),
                schedules=(schedule,),
                approvals=(record,),
            )
        if action == "reject":
            candidate = self.repository.update_schedule_candidate_status(
                candidate_id=target_id,
                status="rejected",
                metadata={"rejected_by": access.user_id, "rejection_comment": comment},
            )
            record = self.repository.save_approval(
                ApprovalRecord(
                    id=str(uuid4()),
                    target_type="schedule",
                    target_id=target_id,
                    action="reject",
                    actor_id=access.user_id,
                    comment=comment,
                    before={},
                    after=asdict(candidate),
                    evidence=candidate.evidence,
                )
            )
            self._audit("workflow.approval.schedule.reject", access, "succeeded", target_id)
            return WorkResponse(
                text=f"ScheduleCandidate を却下しました: {candidate.title}",
                detail_markdown=self._format_schedule_candidates([candidate]),
                schedule_candidates=(candidate,),
                approvals=(record,),
            )
        raise ValueError(f"Unsupported schedule approval action: {action}")

    def _generic_approval(
        self,
        *,
        action: str,
        target_type: str,
        target_id: str,
        comment: str,
        access: AccessContext,
    ) -> WorkResponse:
        approvals = self.repository.list_approvals(
            target_type=target_type,
            target_id=target_id or None,
        )
        candidates = (
            self.operations.list_workflow_candidates(
                candidate_type=target_type,
                status="proposed",
            )
            if self.operations and action == "list"
            else []
        )
        if action == "list":
            return WorkResponse(
                text=f"承認待ち {target_type} candidate は {len(candidates)} 件です。",
                detail_markdown="\n\n".join(
                    [
                        self._format_workflow_candidates(candidates),
                        self._format_approvals(approvals),
                    ]
                ),
                workflow_candidates=tuple(candidates),
                approvals=tuple(approvals),
                metadata={"target_type": target_type, "generic_approval": True},
            )
        if action == "show":
            return WorkResponse(
                text=f"{target_type} approval target を表示します: {target_id}",
                detail_markdown=self._format_approvals(approvals),
                approvals=tuple(approvals),
                metadata={"target_type": target_type, "target_id": target_id},
            )
        if action in {"approve", "reject", "edit"}:
            status = "approved" if action == "approve" else "rejected" if action == "reject" else "edited"
            record = self.repository.save_approval(
                ApprovalRecord(
                    id=str(uuid4()),
                    target_type=target_type,
                    target_id=target_id,
                    action=action,
                    actor_id=access.user_id,
                    comment=comment,
                    before={},
                    after={
                        "status": status,
                        "side_effects": "none",
                        "note": "generic approval record only; no external action executed",
                    },
                )
            )
            self._audit(f"workflow.approval.{target_type}.{action}", access, "recorded", target_id)
            return WorkResponse(
                text=f"{target_type} approval record を保存しました。外部副作用は実行していません: {action}",
                detail_markdown=self._format_approvals([record]),
                approvals=(record,),
                metadata={"target_type": target_type, "side_effects": "none"},
            )
        raise ValueError(f"Unsupported generic approval action: {action}")

    def _edit_event_candidate(
        self,
        *,
        target_id: str,
        comment: str,
        access: AccessContext,
    ) -> tuple[EventCandidate, ApprovalRecord]:
        candidate = self.repository.get_event_candidate(target_id)
        if candidate is None:
            raise KeyError(target_id)
        if candidate.status not in {"proposed", "approved"}:
            raise ValueError(f"EventCandidate is not editable: {candidate.status}")
        title = _extract_labeled_value(comment, ("イベント", "event", "title", "件名", "名前"))
        summary = _extract_labeled_value(comment, ("summary", "概要", "説明", "本文"))
        place = _extract_labeled_value(comment, ("場所", "会場", "place"))
        starts_at = _extract_datetime(comment) or candidate.starts_at
        edited = self.repository.save_event_candidate(
            replace(
                candidate,
                title=_clean_title(title) if title else candidate.title,
                summary=summary or candidate.summary,
                starts_at=starts_at,
                place=place or candidate.place,
                metadata={
                    **candidate.metadata,
                    "edited_by": access.user_id,
                    "edit_comment": comment,
                },
            )
        )
        record = self.repository.save_approval(
            ApprovalRecord(
                id=str(uuid4()),
                target_type="event",
                target_id=target_id,
                action="edit",
                actor_id=access.user_id,
                comment=comment,
                before=asdict(candidate),
                after=asdict(edited),
                evidence=edited.evidence,
            )
        )
        return edited, record

    def _approve_event_candidate(
        self,
        *,
        target_id: str,
        comment: str,
        access: AccessContext,
    ) -> tuple[Event, ApprovalRecord]:
        candidate = self.repository.get_event_candidate(target_id)
        if candidate is None:
            raise KeyError(target_id)
        if candidate.status not in {"proposed", "approved"}:
            raise ValueError(f"EventCandidate is not approvable: {candidate.status}")
        event = self.repository.save_event(
            Event(
                id=stable_hash(f"event:{candidate.id}")[:32],
                title=candidate.title,
                summary=candidate.summary,
                starts_at=candidate.starts_at,
                ends_at=candidate.ends_at,
                place=candidate.place,
                status="planning",
                related_source_ids=candidate.related_source_ids,
                metadata={
                    **candidate.metadata,
                    "approved_by": access.user_id,
                    "source_candidate_id": candidate.id,
                },
            )
        )
        merged = self.repository.update_event_candidate_status(
            candidate_id=candidate.id,
            status="merged",
            metadata={"merged_event_id": event.id, "approved_by": access.user_id},
        )
        record = self.repository.save_approval(
            ApprovalRecord(
                id=str(uuid4()),
                target_type="event",
                target_id=candidate.id,
                action="approve",
                actor_id=access.user_id,
                comment=comment,
                before=asdict(candidate),
                after={"candidate": asdict(merged), "event": asdict(event)},
                evidence=candidate.evidence,
            )
        )
        return event, record

    def _edit_schedule_candidate(
        self,
        *,
        target_id: str,
        comment: str,
        access: AccessContext,
    ) -> tuple[ScheduleCandidate, ApprovalRecord]:
        candidate = self.repository.get_schedule_candidate(target_id)
        if candidate is None:
            raise KeyError(target_id)
        if candidate.status not in {"proposed", "approved"}:
            raise ValueError(f"ScheduleCandidate is not editable: {candidate.status}")
        title = _extract_labeled_value(comment, ("予定", "schedule", "title", "件名", "名前"))
        place = _extract_labeled_value(comment, ("場所", "会場", "place"))
        related_event_id = _extract_labeled_value(comment, ("event_id", "イベントID"))
        starts_at = _extract_datetime(comment) or candidate.starts_at
        edited = self.repository.save_schedule_candidate(
            replace(
                candidate,
                title=_clean_title(title) if title else candidate.title,
                starts_at=starts_at,
                place=place or candidate.place,
                related_event_id=related_event_id or candidate.related_event_id,
                metadata={
                    **candidate.metadata,
                    "edited_by": access.user_id,
                    "edit_comment": comment,
                },
            )
        )
        record = self.repository.save_approval(
            ApprovalRecord(
                id=str(uuid4()),
                target_type="schedule",
                target_id=target_id,
                action="edit",
                actor_id=access.user_id,
                comment=comment,
                before=asdict(candidate),
                after=asdict(edited),
                evidence=edited.evidence,
            )
        )
        return edited, record

    def _approve_schedule_candidate(
        self,
        *,
        target_id: str,
        comment: str,
        access: AccessContext,
    ) -> tuple[ScheduleEvent, ApprovalRecord]:
        candidate = self.repository.get_schedule_candidate(target_id)
        if candidate is None:
            raise KeyError(target_id)
        if candidate.status not in {"proposed", "approved"}:
            raise ValueError(f"ScheduleCandidate is not approvable: {candidate.status}")
        schedule = self.repository.save_schedule(
            ScheduleEvent(
                id=stable_hash(f"schedule:{candidate.id}")[:32],
                title=candidate.title,
                starts_at=candidate.starts_at,
                ends_at=candidate.ends_at,
                place=candidate.place,
                related_event_id=candidate.related_event_id,
                status="planned",
                metadata={
                    **candidate.metadata,
                    "approved_by": access.user_id,
                    "source_candidate_id": candidate.id,
                },
            )
        )
        merged = self.repository.update_schedule_candidate_status(
            candidate_id=candidate.id,
            status="merged",
            metadata={"merged_schedule_id": schedule.id, "approved_by": access.user_id},
        )
        record = self.repository.save_approval(
            ApprovalRecord(
                id=str(uuid4()),
                target_type="schedule",
                target_id=candidate.id,
                action="approve",
                actor_id=access.user_id,
                comment=comment,
                before=asdict(candidate),
                after={"candidate": asdict(merged), "schedule": asdict(schedule)},
                evidence=candidate.evidence,
            )
        )
        return schedule, record

    def _edit_task_candidate(
        self,
        *,
        target_id: str,
        comment: str,
        access: AccessContext,
    ) -> tuple[TaskCandidate, ApprovalRecord]:
        candidate = self.repository.get_task_candidate(target_id)
        if candidate is None:
            raise KeyError(target_id)
        if candidate.status not in {"proposed", "approved"}:
            raise ValueError(f"TaskCandidate is not editable: {candidate.status}")
        if not self.task_access_policy.can_edit_candidate(access, candidate):
            raise PermissionError("TaskCandidate edit is not authorized.")
        title = _extract_labeled_value(comment, ("title", "タイトル", "件名"))
        assignee = _extract_assignee(comment)
        due_at = _extract_datetime(comment) or candidate.proposed_due_at
        description = _extract_labeled_value(comment, ("description", "説明", "本文"))
        edited = self.repository.save_task_candidate(
            replace(
                candidate,
                title=_clean_title(title) if title else candidate.title,
                description=description or candidate.description,
                proposed_assignee_user_id=assignee or candidate.proposed_assignee_user_id,
                proposed_due_at=due_at,
                metadata={
                    **candidate.metadata,
                    "edited_by": access.user_id,
                    "edit_comment": comment,
                },
            )
        )
        record = self.repository.save_approval(
            ApprovalRecord(
                id=str(uuid4()),
                target_type="task",
                target_id=target_id,
                action="edit",
                actor_id=access.user_id,
                comment=comment,
                before=asdict(candidate),
                after=asdict(edited),
                evidence=edited.evidence,
            )
        )
        return edited, record

    def _edit_task_change_candidate(
        self,
        *,
        target_id: str,
        comment: str,
        access: AccessContext,
    ) -> tuple[TaskChangeCandidate, ApprovalRecord]:
        candidate = self.repository.get_task_change_candidate(target_id)
        if candidate is None:
            raise KeyError(target_id)
        if candidate.status not in {"proposed", "approved"}:
            raise ValueError(f"TaskChangeCandidate is not editable: {candidate.status}")
        task = self.repository.get_task(candidate.task_id)
        if task is None:
            raise KeyError(candidate.task_id)
        if not self.task_access_policy.can_update_task_status(access, task):
            raise PermissionError("TaskChangeCandidate edit is not authorized.")
        after = dict(candidate.after)
        title = _extract_labeled_value(comment, ("title", "タイトル", "件名"))
        assignee = _extract_assignee(comment)
        due_at = _extract_datetime(comment)
        status = _extract_task_status(comment)
        priority = _extract_priority(comment)
        if title:
            after["title"] = _clean_title(title)
        if assignee:
            after["assignee_user_id"] = assignee
        if due_at:
            after["due_at"] = due_at.isoformat()
        if status:
            after["status"] = status
        if priority:
            after["priority"] = priority
        edited = self.repository.save_task_change_candidate(
            replace(
                candidate,
                after=after,
                reason=comment or candidate.reason,
                metadata={
                    **candidate.metadata,
                    "edited_by": access.user_id,
                    "edit_comment": comment,
                },
            )
        )
        record = self.repository.save_approval(
            ApprovalRecord(
                id=str(uuid4()),
                target_type="task",
                target_id=target_id,
                action="edit",
                actor_id=access.user_id,
                comment=comment,
                before=asdict(candidate),
                after=asdict(edited),
                evidence=edited.evidence,
            )
        )
        return edited, record

    def _approve_task_candidate(
        self,
        *,
        target_id: str,
        comment: str,
        access: AccessContext,
    ) -> tuple[Task, ApprovalRecord]:
        candidate = self.repository.get_task_candidate(target_id)
        if candidate is None:
            raise KeyError(target_id)
        if candidate.status not in {"proposed", "approved"}:
            raise ValueError(f"TaskCandidate is not approvable: {candidate.status}")
        if not self.task_access_policy.can_approve(access):
            raise PermissionError("TaskCandidate approve is not authorized.")
        task = self.repository.save_task(
            Task(
                id=stable_hash(f"task:{candidate.id}")[:32],
                title=candidate.title,
                description=candidate.description,
                assignee_user_id=candidate.proposed_assignee_user_id,
                due_at=candidate.proposed_due_at,
                related_event_id=candidate.related_event_id,
                source_candidate_id=candidate.id,
                status="todo",
                evidence=candidate.evidence,
                metadata={"approved_by": access.user_id},
            )
        )
        merged = self.repository.update_task_candidate_status(
            candidate_id=candidate.id,
            status="merged",
            metadata={"merged_task_id": task.id, "approved_by": access.user_id},
        )
        record = self.repository.save_approval(
            ApprovalRecord(
                id=str(uuid4()),
                target_type="task",
                target_id=candidate.id,
                action="approve",
                actor_id=access.user_id,
                comment=comment,
                before=asdict(candidate),
                after={"candidate": asdict(merged), "task": asdict(task)},
                evidence=candidate.evidence,
            )
        )
        return task, record

    def _approve_task_change_candidate(
        self,
        *,
        target_id: str,
        comment: str,
        access: AccessContext,
    ) -> tuple[Task, ApprovalRecord]:
        candidate = self.repository.get_task_change_candidate(target_id)
        if candidate is None:
            raise KeyError(target_id)
        if candidate.status not in {"proposed", "approved"}:
            raise ValueError(f"TaskChangeCandidate is not approvable: {candidate.status}")
        if not self.task_access_policy.can_approve(access):
            raise PermissionError("TaskChangeCandidate approve is not authorized.")
        task = self.repository.get_task(candidate.task_id)
        if task is None:
            raise KeyError(candidate.task_id)
        if candidate.operation == "delete":
            updated = replace(
                task,
                status="deleted",
                metadata={
                    **task.metadata,
                    "deleted_by": access.user_id,
                    "delete_candidate_id": candidate.id,
                    "delete_reason": candidate.reason,
                },
            )
        else:
            updated = _replace_task_from_payload(
                task,
                candidate.after,
                metadata={
                    **task.metadata,
                    "updated_by": access.user_id,
                    "change_candidate_id": candidate.id,
                    "change_reason": candidate.reason,
                },
            )
        stored = self.repository.save_task(updated)
        merged = self.repository.update_task_change_candidate_status(
            candidate_id=candidate.id,
            status="merged",
            metadata={"merged_task_id": stored.id, "approved_by": access.user_id},
        )
        record = self.repository.save_approval(
            ApprovalRecord(
                id=str(uuid4()),
                target_type="task",
                target_id=candidate.id,
                action="approve",
                actor_id=access.user_id,
                comment=comment,
                before=asdict(candidate),
                after={"candidate": asdict(merged), "task": asdict(stored)},
                evidence=candidate.evidence,
            )
        )
        return stored, record

    def _extract_and_store_candidates(
        self,
        text: str,
        access: AccessContext,
        *,
        evidence: tuple[Citation, ...],
        metadata: dict[str, Any],
    ) -> tuple[list[TaskCandidate], dict[str, Any]]:
        result = self.task_extractor.extract(
            text=text,
            evidence=evidence,
            access=access,
            metadata=metadata,
        )
        stored: list[TaskCandidate] = []
        for candidate in result.candidates:
            stored.append(self.repository.save_task_candidate(self._annotate_task_duplicate(candidate)))
        return stored, result.metadata

    def _extract_task_candidates(
        self,
        text: str,
        *,
        evidence: tuple[Citation, ...],
        metadata: dict[str, Any],
    ) -> list[TaskCandidate]:
        candidates: list[TaskCandidate] = []
        seen_titles: set[str] = set()
        for line in _candidate_lines(text):
            if re.match(r"^(未決|要確認|確認事項)[:：]", line):
                continue
            if not any(keyword.lower() in line.lower() for keyword in _TASK_KEYWORDS):
                continue
            title = _task_title(line)
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            assignee = _extract_assignee(line)
            due_at = _extract_datetime(line)
            confidence = "high" if assignee and due_at else "medium" if assignee or due_at else "low"
            candidate_id = stable_hash(
                f"task-candidate:{title}:{assignee or ''}:{due_at.isoformat() if due_at else ''}"
            )[:32]
            candidates.append(
                TaskCandidate(
                    id=candidate_id,
                    title=title,
                    description=line,
                    proposed_assignee_user_id=assignee,
                    proposed_due_at=due_at,
                    evidence=evidence[:5],
                    confidence=confidence,
                    status="proposed",
                    created_by="agent",
                    metadata=metadata | {"extractor": "heuristic_v1"},
                )
            )
        return candidates

    def _retrieve(self, request: WorkRequest, *, fallback_query: str) -> dict[str, object]:
        if self.ask_service is None:
            return {"text": "", "citations": tuple()}
        query_text = (fallback_query or "").strip()
        if not query_text:
            return {"text": "", "citations": tuple()}
        response = self.ask_service.ask(
            RetrievalQuery(
                text=query_text,
                source_filter="all",
                mode="search_only",
                depth="normal",
                access=request.access,
            )
        )
        return {
            "text": response.detail_markdown or response.text,
            "citations": tuple(response.citations),
        }

    def _agentic_or_retrieve(self, request: WorkRequest, *, fallback_query: str) -> dict[str, object]:
        if self.agentic_search is None:
            return self._retrieve(request, fallback_query=fallback_query)
        response = self.agentic_search.search(
            AgenticSearchRequest(
                query=fallback_query,
                source_filter="all",
                access=request.access,
            )
        )
        return {
            "text": response.detail_markdown or response.text,
            "citations": tuple(response.citations),
        }

    def _source_text(self, request: WorkRequest) -> str:
        return _join_text(request.instruction, request.target)

    def _event_from_text(self, text: str) -> Event:
        title = _extract_labeled_value(text, ("イベント", "件名", "title", "名前"))
        title = title or _first_nonempty_line(text) or "Untitled event"
        starts_at = _extract_datetime(text)
        place = _extract_labeled_value(text, ("場所", "会場", "place"))
        return Event(
            id=stable_hash(f"event:{title}:{starts_at.isoformat() if starts_at else ''}")[:32],
            title=_clean_title(title),
            summary=text.strip() or None,
            starts_at=starts_at,
            place=place,
            status="planning",
            metadata={"source": "event_add"},
        )

    def _schedule_from_text(self, text: str) -> ScheduleEvent:
        title = _extract_labeled_value(text, ("予定", "件名", "title", "名前"))
        title = title or _first_nonempty_line(text) or "Untitled schedule"
        starts_at = _extract_datetime(text)
        place = _extract_labeled_value(text, ("場所", "会場", "place"))
        related_event_id = _extract_labeled_value(text, ("event_id", "イベントID"))
        return ScheduleEvent(
            id=stable_hash(f"schedule:{title}:{starts_at.isoformat() if starts_at else ''}")[:32],
            title=_clean_title(title),
            starts_at=starts_at,
            place=place,
            related_event_id=related_event_id,
            status="planned",
            metadata={"source": "schedule_add"},
        )

    def _resolve_event(self, request: WorkRequest) -> Event | None:
        target = (request.target or request.instruction).strip()
        if target:
            direct = self.repository.get_event(target)
            if direct:
                return direct
        events = self.repository.list_events()
        if not events:
            return None
        lowered = target.lower()
        for event in events:
            if lowered and (lowered in event.title.lower() or event.id.startswith(lowered)):
                return event
        return events[0]

    def _build_meeting_agenda(
        self,
        *,
        instruction: str,
        retrieved_text: str,
        open_tasks: list[Task],
        events: list[Event],
        schedules: list[ScheduleEvent],
    ) -> str:
        task_lines = [_task_line(task) for task in open_tasks[:10]] or ["- 未完了タスクはありません。"]
        event_lines = [_event_line(event) for event in events[:10]] or ["- 関連イベントは未登録です。"]
        schedule_lines = [_schedule_line(schedule) for schedule in schedules[:10]] or ["- 予定は未登録です。"]
        retrieved_lines = _candidate_lines(retrieved_text)[:8]
        related = [f"- {line}" for line in retrieved_lines] or ["- 関連資料からの追加論点は見つかりませんでした。"]
        return "\n".join(
            [
                "# 例会準備",
                "",
                "## 議題案",
                "- 未完了タスクの進捗確認",
                "- 直近イベントと予定の確認",
                "- 告知・資料・担当の抜け漏れ確認",
                "",
                "## 未完了タスク",
                *task_lines,
                "",
                "## イベント",
                *event_lines,
                "",
                "## 予定",
                *schedule_lines,
                "",
                "## 関連資料からの確認事項",
                *related,
                "",
                "## 告知文案",
                instruction.strip() or "次回例会では、未完了タスクと直近イベントの確認を行います。",
            ]
        )

    def _build_minutes_markdown(
        self,
        *,
        source_text: str,
        decisions: list[str],
        open_questions: list[str],
        candidates: list[TaskCandidate],
    ) -> str:
        return "\n".join(
            [
                "# 議事録下書き",
                "",
                "## 決定事項",
                *([f"- {line}" for line in decisions] or ["- 決定事項は抽出されませんでした。"]),
                "",
                "## 未決事項・確認事項",
                *([f"- {line}" for line in open_questions] or ["- 未決事項は抽出されませんでした。"]),
                "",
                "## ToDo 候補",
                *([f"- {candidate.title} ({candidate.confidence})" for candidate in candidates] or ["- ToDo 候補は抽出されませんでした。"]),
                "",
                "## 元テキスト抜粋",
                *_quote_lines(source_text, limit=8),
            ]
        )

    def _build_event_brief(
        self,
        *,
        event: Event | None,
        tasks: list[Task],
        retrieved_text: str,
        citations: tuple[Citation, ...],
    ) -> str:
        header = _event_line(event) if event else "- Event は未指定です。"
        return "\n".join(
            [
                "# Event brief",
                "",
                "## Event",
                header,
                "",
                "## 未完了タスク",
                *([_task_line(task) for task in tasks] or ["- 未完了タスクはありません。"]),
                "",
                "## 関連資料",
                *([f"- {line}" for line in _candidate_lines(retrieved_text)[:8]] or ["- 関連資料は見つかりませんでした。"]),
                "",
                "## 根拠",
                *([f"- {citation.label or citation.chunk_id} {citation.url}".strip() for citation in citations[:8]] or ["- citation はありません。"]),
            ]
        )

    def _format_task_candidates(self, candidates: list[TaskCandidate]) -> str:
        if not candidates:
            return "TaskCandidate はありません。"
        lines = ["# TaskCandidate"]
        for candidate in candidates:
            due = candidate.proposed_due_at.isoformat() if candidate.proposed_due_at else "未定"
            assignee = candidate.proposed_assignee_user_id or "未定"
            priority = candidate.metadata.get("priority") or "normal"
            duplicate_count = len(candidate.metadata.get("duplicate_candidates") or [])
            duplicate_note = f" / duplicates: {duplicate_count}" if duplicate_count else ""
            lines.append(
                f"- `{candidate.id}` {candidate.title} / 担当: {assignee} / 期限: {due} / priority: {priority} / status: {candidate.status} / confidence: {candidate.confidence}{duplicate_note}"
            )
        return "\n".join(lines)

    def _format_task_change_candidates(self, candidates: list[TaskChangeCandidate]) -> str:
        if not candidates:
            return "TaskChangeCandidate はありません。"
        lines = ["# TaskChangeCandidate"]
        for candidate in candidates:
            lines.append(
                f"- `{candidate.id}` task={candidate.task_id} / operation: {candidate.operation} / status: {candidate.status} / confidence: {candidate.confidence}"
            )
            if candidate.reason:
                lines.append(f"  - reason: {_truncate(candidate.reason, 160)}")
        return "\n".join(lines)

    def _format_tasks(self, tasks: list[Task]) -> str:
        if not tasks:
            return "Task はありません。"
        return "\n".join(["# Task", *[_task_line(task) for task in tasks]])

    def _format_approvals(self, approvals: list[ApprovalRecord]) -> str:
        if not approvals:
            return "ApprovalRecord はありません。"
        lines = ["# ApprovalRecord"]
        for record in approvals:
            created = record.created_at.isoformat() if record.created_at else "未記録"
            lines.append(
                f"- `{record.id}` action={record.action} actor={record.actor_id} created={created} comment={record.comment}"
            )
        return "\n".join(lines)

    def _format_events(self, events: list[Event]) -> str:
        if not events:
            return "Event はありません。"
        return "\n".join(["# Event", *[_event_line(event) for event in events]])

    def _format_event_candidates(self, candidates: list[EventCandidate]) -> str:
        if not candidates:
            return "EventCandidate はありません。"
        lines = ["# EventCandidate"]
        for candidate in candidates:
            starts = candidate.starts_at.isoformat() if candidate.starts_at else "未定"
            place = candidate.place or "未定"
            lines.append(
                f"- `{candidate.id}` {candidate.title} / 日時: {starts} / 場所: {place} / status: {candidate.status} / confidence: {candidate.confidence}"
            )
        return "\n".join(lines)

    def _format_schedules(self, schedules: list[ScheduleEvent]) -> str:
        if not schedules:
            return "Schedule はありません。"
        return "\n".join(["# Schedule", *[_schedule_line(schedule) for schedule in schedules]])

    def _format_schedule_candidates(self, candidates: list[ScheduleCandidate]) -> str:
        if not candidates:
            return "ScheduleCandidate はありません。"
        lines = ["# ScheduleCandidate"]
        for candidate in candidates:
            starts = candidate.starts_at.isoformat() if candidate.starts_at else "未定"
            place = candidate.place or "未定"
            related = candidate.related_event_id or "未定"
            lines.append(
                f"- `{candidate.id}` {candidate.title} / 日時: {starts} / 場所: {place} / event: {related} / status: {candidate.status} / confidence: {candidate.confidence}"
            )
        return "\n".join(lines)

    def _format_workflow_candidates(self, candidates: list[WorkflowCandidate]) -> str:
        if not candidates:
            return "WorkflowCandidate はありません。"
        lines = ["# WorkflowCandidate"]
        for candidate in candidates:
            lines.append(
                f"- `{candidate.id}` {candidate.title} / type: {candidate.candidate_type} / status: {candidate.status} / confidence: {candidate.confidence}"
            )
        return "\n".join(lines)

    def _format_assets(self, assets: list[object]) -> str:
        if not assets:
            return "Asset はありません。"
        lines = ["# Image Search"]
        for asset in assets:
            metadata = dict(getattr(asset, "metadata", {}) or {})
            source = metadata.get("source_label") or getattr(asset, "source_kind", "")
            source_url = metadata.get("source_url") or getattr(asset, "uri", "")
            description = getattr(asset, "description", "") or metadata.get("caption") or ""
            lines.append(
                f"- `{asset.id}` {asset.title or asset.uri or 'untitled'} / source: {source}"
            )
            if description:
                lines.append(f"  - 説明: {_truncate(str(description), 160)}")
            if source_url:
                lines.append(f"  - 出典: {source_url}")
        lines.append("")
        lines.append("この結果は画像候補の提示のみで、外部公開・転載・再利用の可否は判断しません。")
        return "\n".join(lines)

    def _format_member_profiles(self, profiles: list[object]) -> str:
        if not profiles:
            return "MemberProfile はありません。"
        lines = ["# MemberProfile"]
        for profile in profiles:
            skills = ", ".join(profile.skills) if profile.skills else "未登録"
            roles = ", ".join(profile.roles) if profile.roles else "未登録"
            lines.append(
                f"- `{profile.id}` {profile.display_name or profile.discord_user_id or 'unnamed'} / roles: {roles} / skills: {skills}"
            )
        lines.append("")
        lines.append("担当決定には本人または運営確認が必要です。")
        return "\n".join(lines)

    def _start_workflow_run(self, request: WorkRequest) -> WorkflowRun | None:
        if self.operations is None:
            return None
        return self.operations.save_workflow_run(
            WorkflowRun(
                workflow_id=request.work_type.strip().lower() or "unknown",
                trigger="manual",
                actor_user_id=request.access.user_id,
                guild_id=request.access.guild_id,
                input={
                    "work_type": request.work_type,
                    "instruction": request.instruction,
                    "target": request.target,
                    "output_format": request.output_format,
                },
                status="running",
            )
        )

    def _annotate_task_duplicate(self, candidate: TaskCandidate) -> TaskCandidate:
        return self.task_duplicate_detector.annotate(
            candidate,
            existing_candidates=self.repository.list_task_candidates(),
            existing_tasks=self.repository.list_tasks(include_deleted=False),
        )

    def _task_forbidden_response(self, action: str, access: AccessContext) -> WorkResponse:
        self._audit(action, access, "denied", "task")
        return WorkResponse(
            text="権限がありません。対象情報の有無は表示しません。",
            detail_markdown="",
            metadata=self.task_access_policy.forbidden_response_metadata(),
        )

    def _finish_workflow_run(
        self,
        run: WorkflowRun | None,
        *,
        status: str,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.operations is None or run is None:
            return
        self.operations.save_workflow_run(
            replace(
                run,
                status=status,
                error=error,
                metadata={**run.metadata, **(metadata or {})},
            )
        )

    def _audit(
        self,
        action: str,
        access: AccessContext,
        outcome: str,
        target: str,
    ) -> None:
        if self.audit_log is None:
            return
        self.audit_log.append(
            AuditEvent(
                action=action,
                actor_id=access.user_id,
                actor_type="discord_user" if access.user_id else "service",
                outcome=outcome,
                target=target,
                risk_level="medium",
            )
        )


def _candidate_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip("-*・ ")
        if len(line) < 3 or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _matching_lines(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [
        line
        for line in _candidate_lines(text)
        if any(keyword.lower() in line.lower() for keyword in keywords)
    ]


def _task_title(line: str) -> str:
    cleaned = re.sub(r"^\s*(TODO|ToDo|todo|タスク)[:：]?\s*", "", line).strip()
    cleaned = re.sub(r"担当[:：]\s*[^\s、。]+", "", cleaned)
    cleaned = re.sub(r"期限[:：]\s*[^\s、。]+", "", cleaned)
    return _clean_title(cleaned[:120])


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" -:：、。"))


def _extract_assignee(line: str) -> str | None:
    match = re.search(r"(?:担当|assignee)[:：]\s*(@?[A-Za-z0-9_.\-\u3040-\u30ff\u3400-\u9fff]+)", line)
    if match:
        return match.group(1).lstrip("@")
    match = re.search(r"@([A-Za-z0-9_.\-]+)", line)
    return match.group(1) if match else None


def _extract_priority(text: str) -> str:
    match = re.search(r"(?:priority|優先度)[:：]\s*([A-Za-z\u3040-\u30ff\u3400-\u9fff]+)", text, flags=re.IGNORECASE)
    value = match.group(1) if match else _extract_labeled_value(text, ("priority", "優先度"))
    lowered = (value or text or "").lower()
    if "urgent" in lowered or "至急" in lowered:
        return "urgent"
    if "high" in lowered or "高" in lowered:
        return "high"
    if "low" in lowered or "低" in lowered:
        return "low"
    if "normal" in lowered or "通常" in lowered:
        return "normal"
    return "normal"


def _extract_task_status(text: str) -> str | None:
    match = re.search(r"(?:status|状態)[:：]\s*([A-Za-z_]+|未着手|対応中|保留|完了)", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).lower()
    mapping = {
        "未着手": "todo",
        "対応中": "doing",
        "保留": "blocked",
        "完了": "done",
    }
    return mapping.get(value, value)


def _extract_task_list_conditions(text: str) -> dict[str, object]:
    status = _extract_task_status(text)
    assignee = _extract_assignee(text) or _extract_labeled_value(text, ("assignee", "担当"))
    related_event_id = _extract_labeled_value(text, ("event", "event_id", "イベントID", "関連イベント"))
    priority = _extract_labeled_value(text, ("priority", "優先度"))
    due_to = _extract_datetime(text) if re.search(r"(期限|due|まで)", text, flags=re.IGNORECASE) else None
    conditions: dict[str, object] = {}
    if status:
        conditions["status"] = status
    if assignee:
        conditions["assignee_user_id"] = assignee.lstrip("@")
    if related_event_id:
        conditions["related_event_id"] = related_event_id
    if priority:
        conditions["priority"] = priority.lower()
    if due_to:
        conditions["due_to"] = due_to
    return conditions


def _serializable_task_filters(filters: dict[str, object]) -> dict[str, object]:
    return {
        key: value.isoformat() if hasattr(value, "isoformat") else value
        for key, value in filters.items()
    }


def _extract_int_labeled_value(text: str, labels: tuple[str, ...]) -> int | None:
    value = _extract_labeled_value(text, labels)
    if not value:
        return None
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else None


def _task_payload_for_change(task: Task) -> dict[str, object]:
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description,
        "assignee_user_id": task.assignee_user_id,
        "due_at": task.due_at.isoformat() if task.due_at else None,
        "related_event_id": task.related_event_id,
        "status": task.status,
        "priority": task.priority,
    }


def _replace_task_from_payload(
    task: Task,
    payload: dict[str, object],
    *,
    metadata: dict[str, Any],
) -> Task:
    due_at = payload.get("due_at")
    parsed_due = task.due_at
    if isinstance(due_at, datetime):
        parsed_due = due_at
    elif isinstance(due_at, str) and due_at:
        parsed_due = datetime.fromisoformat(due_at.replace("Z", "+00:00"))
    return replace(
        task,
        title=str(payload.get("title") or task.title),
        description=payload.get("description") and str(payload["description"]),
        assignee_user_id=payload.get("assignee_user_id") and str(payload["assignee_user_id"]),
        due_at=parsed_due,
        related_event_id=payload.get("related_event_id") and str(payload["related_event_id"]),
        status=str(payload.get("status") or task.status),
        priority=str(payload.get("priority") or task.priority),
        metadata=metadata,
    )


def _can_search_members(access: AccessContext) -> bool:
    roles = {role.lower() for role in access.role_ids}
    return access.is_admin or "admin" in roles or "organizer" in roles


def _workflow_response_status(response: WorkResponse) -> str:
    if (
        response.server_operations
        or response.task_candidates
        or response.task_change_candidates
        or response.task_approval_batches
        or response.event_candidates
        or response.schedule_candidates
    ):
        return "waiting_approval"
    if response.workflow_candidates:
        return "waiting_approval"
    return "succeeded"


def _extract_datetime(text: str) -> datetime | None:
    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?", text)
    if match:
        year, month, day, hour, minute = match.groups()
        return datetime(
            int(year),
            int(month),
            int(day),
            int(hour or 0),
            int(minute or 0),
            tzinfo=UTC,
        )
    match = re.search(r"(\d{1,2})月(\d{1,2})日(?:\s*(\d{1,2})[:時](\d{1,2})?)?", text)
    if match:
        month, day, hour, minute = match.groups()
        now = datetime.now(UTC)
        return datetime(
            now.year,
            int(month),
            int(day),
            int(hour or 0),
            int(minute or 0),
            tzinfo=UTC,
        )
    match = re.search(r"(\d{1,2})/(\d{1,2})(?:\s*(\d{1,2}):(\d{2}))?", text)
    if match:
        month, day, hour, minute = match.groups()
        now = datetime.now(UTC)
        return datetime(
            now.year,
            int(month),
            int(day),
            int(hour or 0),
            int(minute or 0),
            tzinfo=UTC,
        )
    return None


def _extract_labeled_value(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        match = re.search(rf"{re.escape(label)}[:：]\s*([^\n、。]+)", text, flags=re.IGNORECASE)
        if match:
            return _clean_title(match.group(1))
    return None


def _first_nonempty_line(text: str) -> str:
    for line in _candidate_lines(text):
        return _clean_title(line)
    return ""


def _join_text(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _compact_lines(lines: list[str]) -> str:
    return "\n".join(line for line in lines if line)


def _truncate(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _quote_lines(text: str, *, limit: int) -> list[str]:
    lines = _candidate_lines(text)[:limit]
    return [f"> {line}" for line in lines] if lines else ["> 抜粋はありません。"]


def _task_line(task: Task) -> str:
    due = task.due_at.isoformat() if task.due_at else "未定"
    assignee = task.assignee_user_id or "未定"
    return f"- `{task.id}` {task.title} / 担当: {assignee} / 期限: {due} / priority: {task.priority} / status: {task.status}"


def _event_line(event: Event | None) -> str:
    if event is None:
        return "- Event は未指定です。"
    start = event.starts_at.isoformat() if event.starts_at else "日時未定"
    place = event.place or "場所未定"
    return f"- `{event.id}` {event.title} / {start} / {place} / status: {event.status}"


def _schedule_line(schedule: ScheduleEvent) -> str:
    start = schedule.starts_at.isoformat() if schedule.starts_at else "日時未定"
    place = schedule.place or "場所未定"
    return f"- `{schedule.id}` {schedule.title} / {start} / {place} / status: {schedule.status}"


def _doc_type(text: str) -> str:
    lowered = text.lower()
    if "週報" in text or "weekly" in lowered:
        return "weekly_report"
    if "意思決定" in text or "decision" in lowered or "決定メモ" in text:
        return "decision_memo"
    if "告知" in text or "announcement" in lowered:
        return "announcement"
    return "generic"


def _announcement_medium(text: str) -> str:
    lowered = text.lower()
    if "x" in lowered or "twitter" in lowered:
        return "x"
    if "blog" in lowered or "ブログ" in text:
        return "blog"
    return "discord"


def _x_draft_candidates(text: str) -> list[str]:
    safe = re.sub(r"\s+", " ", text).strip()
    if not safe:
        safe = "KUMC の活動についてのお知らせです。詳細は追って共有します。"
    base = safe[:180]
    candidates = [
        f"{base}\n#KUMC #Minecraft",
        f"【お知らせ】{base}",
        f"{base}\n参加前に日時・場所・条件をご確認ください。",
    ]
    return [_fit_x(candidate) for candidate in candidates]


def _pick_x_draft(candidates: list[str]) -> str:
    def score(candidate: str) -> tuple[int, int]:
        has_check = int("確認" in candidate or "日時" in candidate)
        return (has_check, -abs(180 - len(candidate)))

    return sorted(candidates, key=score, reverse=True)[0]


def _fit_x(text: str) -> str:
    if len(text) <= 280:
        return text
    return text[:277].rstrip() + "..."
