from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from kumc_agent.domain.models.audit import AuditEvent
from kumc_agent.domain.models.agentic import AgenticSearchRequest
from kumc_agent.domain.models.docgen import DocGenRequest
from kumc_agent.domain.models.operations import (
    AssetUsageRequest,
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
    TaskCandidate,
    WorkRequest,
    WorkResponse,
)
from kumc_agent.features.announcement.service import (
    AnnouncementDraftRequest,
    AnnouncementDraftService,
)
from kumc_agent.features.agentic import AgenticSearchService
from kumc_agent.features.docgen.service import DocGenService
from kumc_agent.features.minecraft import MinecraftSupportService
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
                "event_candidates": len(response.event_candidates),
                "schedule_candidates": len(response.schedule_candidates),
                "workflow_candidates": len(response.workflow_candidates),
                "assets": len(response.assets),
                "asset_usage_requests": len(response.asset_usage_requests),
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
        if work_type == "image_usage_request":
            return self.image_usage_request(request)
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
        candidates = self._extract_and_store_candidates(
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
            metadata={"decisions": len(decisions), "open_questions": len(open_questions)},
        )

    def task_extract(self, request: WorkRequest) -> WorkResponse:
        source_text = self._source_text(request)
        retrieved = self._retrieve(
            request,
            fallback_query=request.target or request.instruction or "タスク 担当 期限 ToDo",
        )
        combined = _join_text(source_text, retrieved["text"])
        candidates = self._extract_and_store_candidates(
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
            metadata={"candidate_count": len(candidates)},
        )

    def task_add(self, request: WorkRequest) -> WorkResponse:
        text = self._source_text(request)
        title = _extract_labeled_value(text, ("task", "タスク", "title", "件名"))
        title = title or _task_title(_first_nonempty_line(text) or text)
        if not title:
            raise ValueError("task_add requires a task title.")
        due_at = _extract_datetime(text)
        assignee = _extract_assignee(text)
        candidate = self.repository.save_task_candidate(
            TaskCandidate(
                id=stable_hash(
                    f"task-candidate:manual:{title}:{assignee or ''}:{due_at.isoformat() if due_at else ''}"
                )[:32],
                title=_clean_title(title),
                description=text.strip() or None,
                proposed_assignee_user_id=assignee,
                proposed_due_at=due_at,
                confidence="high",
                status="proposed",
                created_by="user",
                metadata={"source": "task_add", "created_by_user_id": request.access.user_id},
            )
        )
        self._audit("workflow.task_add", request.access, "proposed", candidate.id)
        return WorkResponse(
            text=f"TaskCandidate を作成しました。承認されるまで Task 正本には入りません: {candidate.title}",
            detail_markdown=self._format_task_candidates([candidate]),
            task_candidates=(candidate,),
        )

    def task_list(self, request: WorkRequest) -> WorkResponse:
        status = _extract_labeled_value(request.instruction, ("status", "状態"))
        tasks = self.repository.list_tasks(status=status)
        candidates = self.repository.list_task_candidates(status="proposed")
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
        assets = tuple(self.operations.list_assets(query=query))
        detail = self._format_assets(list(assets))
        if not assets:
            detail = "\n".join(
                [
                    "# Image Search",
                    "",
                    "該当する Asset は登録されていません。",
                    "外部画像を再利用可能とは判断していません。",
                ]
            )
        return WorkResponse(
            text=f"画像候補は {len(assets)} 件です。利用する場合は image_usage_request で承認依頼を作成してください。",
            detail_markdown=detail,
            assets=assets,
            metadata={"route": "image_search", "query": query},
        )

    def image_usage_request(self, request: WorkRequest) -> WorkResponse:
        if self.operations is None:
            return WorkResponse(
                text="画像利用申請 repository は未設定です。",
                detail_markdown="Asset usage repository is not configured.",
                metadata={"route": "image_usage_request", "configured": False},
            )
        asset_id = request.target.strip() or _extract_labeled_value(
            request.instruction,
            ("asset_id", "asset", "画像ID"),
        ) or ""
        asset = self.operations.get_asset(asset_id) if asset_id else None
        purpose = _extract_labeled_value(request.instruction, ("purpose", "目的", "用途")) or request.instruction
        medium = _extract_labeled_value(request.instruction, ("medium", "媒体", "掲載先")) or ""
        usage = self.operations.save_asset_usage_request(
            AssetUsageRequest(
                id=stable_hash(f"asset-usage:{asset_id}:{purpose}:{medium}:{request.access.user_id}")[:32],
                asset_id=asset_id,
                purpose=purpose.strip(),
                medium=medium.strip(),
                requested_by=request.access.user_id,
                status="proposed",
                needs_owner_check=True,
                needs_people_check=True if asset is None else bool(asset.contains_people),
                payload={"instruction": request.instruction, "target": request.target},
                metadata={
                    "asset_found": asset is not None,
                    "rights_status": asset.rights_status if asset else "unknown",
                },
            )
        )
        candidate = self.operations.save_workflow_candidate(
            WorkflowCandidate(
                id=stable_hash(f"workflow-candidate:asset_usage:{usage.id}")[:32],
                candidate_type="asset_usage",
                title=f"Asset usage request: {usage.asset_id or 'unresolved asset'}",
                payload=asdict(usage),
                confidence="medium" if asset else "low",
                status="proposed",
                created_by=request.access.user_id or "agent",
                metadata={"target_type": "asset_usage", "target_id": usage.id},
            )
        )
        return WorkResponse(
            text=f"AssetUsageRequest を作成しました。承認前に外部公開可能とは判断しません: {usage.id}",
            detail_markdown=self._format_asset_usage_requests([usage]),
            workflow_candidates=(candidate,),
            asset_usage_requests=(usage,),
            assets=(asset,) if asset else tuple(),
            metadata={"route": "image_usage_request", "approval_required": True},
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
            "asset_usage",
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
            candidates = self.repository.list_task_candidates(status="proposed")
            return WorkResponse(
                text=f"承認待ち TaskCandidate は {len(candidates)} 件です。",
                detail_markdown=self._format_task_candidates(candidates),
                task_candidates=tuple(candidates),
            )
        if normalized_action == "show":
            candidate = self.repository.get_task_candidate(target_id) if target_id else None
            task = self.repository.get_task(target_id) if target_id else None
            approvals = self.repository.list_approvals(target_type="task", target_id=target_id)
            if candidate is None and task is None:
                raise KeyError(target_id)
            details = []
            if candidate is not None:
                details.append(self._format_task_candidates([candidate]))
            if task is not None:
                details.append(self._format_tasks([task]))
            if approvals:
                details.append(self._format_approvals(approvals))
            return WorkResponse(
                text=f"task approval target を表示します: {target_id}",
                detail_markdown="\n\n".join(details),
                task_candidates=(candidate,) if candidate else tuple(),
                tasks=(task,) if task else tuple(),
                approvals=tuple(approvals),
            )
        if normalized_action == "edit":
            candidate, record = self._edit_task_candidate(
                target_id=target_id,
                comment=comment,
                access=access,
            )
            self._audit("workflow.approval.edit", access, "succeeded", target_id)
            return WorkResponse(
                text=f"TaskCandidate を編集しました: {candidate.title}",
                detail_markdown=self._format_task_candidates([candidate]),
                task_candidates=(candidate,),
                approvals=(record,),
            )
        if normalized_action == "approve":
            task, record = self._approve_task_candidate(
                target_id=target_id,
                comment=comment,
                access=access,
            )
            self._audit("workflow.approval.approve", access, "succeeded", target_id)
            return WorkResponse(
                text=f"TaskCandidate を承認し Task 正本に登録しました: {task.title}",
                detail_markdown=self._format_tasks([task]),
                tasks=(task,),
                approvals=(record,),
            )
        if normalized_action == "reject":
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

    def _extract_and_store_candidates(
        self,
        text: str,
        access: AccessContext,
        *,
        evidence: tuple[Citation, ...],
        metadata: dict[str, Any],
    ) -> list[TaskCandidate]:
        candidates = self._extract_task_candidates(text, evidence=evidence, metadata=metadata)
        stored: list[TaskCandidate] = []
        for candidate in candidates:
            stored.append(self.repository.save_task_candidate(candidate))
        return stored

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
            lines.append(
                f"- `{candidate.id}` {candidate.title} / 担当: {assignee} / 期限: {due} / status: {candidate.status} / confidence: {candidate.confidence}"
            )
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
        lines = ["# Asset"]
        for asset in assets:
            lines.append(
                f"- `{asset.id}` {asset.title or asset.uri or 'untitled'} / source: {asset.source_kind} / rights: {asset.rights_status} / people: {asset.contains_people}"
            )
        return "\n".join(lines)

    def _format_asset_usage_requests(self, requests: list[AssetUsageRequest]) -> str:
        if not requests:
            return "AssetUsageRequest はありません。"
        lines = ["# AssetUsageRequest"]
        for request in requests:
            lines.append(
                f"- `{request.id}` asset={request.asset_id or '未指定'} / medium={request.medium or '未定'} / status={request.status} / owner_check={request.needs_owner_check} / people_check={request.needs_people_check}"
            )
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


def _can_search_members(access: AccessContext) -> bool:
    roles = {role.lower() for role in access.role_ids}
    return access.is_admin or "admin" in roles or "organizer" in roles


def _workflow_response_status(response: WorkResponse) -> str:
    if response.server_operations or response.task_candidates or response.event_candidates or response.schedule_candidates:
        return "waiting_approval"
    if response.asset_usage_requests or response.workflow_candidates:
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


def _quote_lines(text: str, *, limit: int) -> list[str]:
    lines = _candidate_lines(text)[:limit]
    return [f"> {line}" for line in lines] if lines else ["> 抜粋はありません。"]


def _task_line(task: Task) -> str:
    due = task.due_at.isoformat() if task.due_at else "未定"
    assignee = task.assignee_user_id or "未定"
    return f"- `{task.id}` {task.title} / 担当: {assignee} / 期限: {due} / status: {task.status}"


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
