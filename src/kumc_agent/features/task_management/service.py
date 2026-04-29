from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import re
from typing import Any

from kumc_agent.domain.models.retrieval import AccessContext, Citation
from kumc_agent.domain.models.workflow import Task, TaskCandidate, TaskChangeCandidate
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.utils.hashing import stable_hash


SCHEMA_VERSION = "workflow_extraction.v1"


@dataclass(frozen=True)
class TaskExtractionResult:
    candidates: tuple[TaskCandidate, ...]
    metadata: dict[str, Any]
    change_candidates: tuple[TaskChangeCandidate, ...] = tuple()
    ignored_items: tuple[dict[str, Any], ...] = tuple()


class TaskExtractionService:
    def __init__(
        self,
        *,
        llm: LLMPort | None = None,
        prompts_dir: Path | None = None,
        prompt_name: str = "task_extraction.md",
        model_name: str = "",
    ) -> None:
        self._llm = llm
        self._prompts_dir = prompts_dir
        self._prompt_name = prompt_name
        self._model_name = model_name

    def extract(
        self,
        *,
        text: str,
        evidence: tuple[Citation, ...],
        access: AccessContext,
        metadata: dict[str, Any],
        existing_tasks: tuple[Task, ...] = tuple(),
    ) -> TaskExtractionResult:
        source_text = _safe_context(text)
        base_metadata = {
            **metadata,
            "schema_version": SCHEMA_VERSION,
            "extractor": "task_llm",
            "extractor_model": self._model_name,
            "prompt_version": self._prompt_name,
        }
        if self._llm is None:
            return TaskExtractionResult(
                candidates=tuple(),
                metadata={
                    **base_metadata,
                    "degraded": True,
                    "degraded_reason": "llm_unavailable",
                },
            )
        if not source_text.strip():
            return TaskExtractionResult(
                candidates=tuple(),
                metadata={
                    **base_metadata,
                    "degraded": True,
                    "degraded_reason": "empty_input",
                },
            )
        try:
            raw = self._llm.generate(
                system_prompt=self._prompt(),
                user_prompt=json.dumps(
                    {
                        "text": source_text,
                        "evidence": [_citation_payload(item) for item in evidence[:8]],
                        "existing_tasks": [
                            _task_payload_for_extraction(task) for task in existing_tasks[:50]
                        ],
                        "expected_operation": metadata.get("expected_operation") or "",
                        "actor_user_id": access.user_id,
                    },
                    ensure_ascii=False,
                ),
                temperature=0.0,
                max_output_tokens=2048,
            )
            payload = _extract_json_object(raw)
            ignored_items = _ignored_items(payload)
            if bool(payload.get("degraded")):
                return TaskExtractionResult(
                    candidates=tuple(),
                    change_candidates=tuple(),
                    ignored_items=ignored_items,
                    metadata={
                        **base_metadata,
                        "degraded": True,
                        "degraded_reason": str(payload.get("degraded_reason") or "llm_degraded"),
                        "candidate_count": 0,
                        "change_candidate_count": 0,
                        "ignored_items": list(ignored_items),
                    },
                )
            items = _new_items(payload, legacy_key="tasks", item_type="task")
            candidates = tuple(
                candidate
                for candidate in (
                    self._candidate_from_payload(
                        item,
                        evidence=evidence,
                        source_text=source_text,
                        base_metadata=base_metadata,
                    )
                    for item in items
                    if isinstance(item, dict)
                )
                if candidate is not None
            )
            expected_operation = str(metadata.get("expected_operation") or "").strip()
            raw_changes = _change_items(payload, legacy_key="task_changes", item_type="task")
            change_candidates = tuple(
                candidate
                for candidate in (
                    self._change_candidate_from_payload(
                        item,
                        evidence=evidence,
                        source_text=source_text,
                        existing_tasks=existing_tasks,
                        expected_operation=expected_operation,
                        created_by=str(metadata.get("created_by") or "agent"),
                        base_metadata=base_metadata,
                    )
                    for item in raw_changes
                    if isinstance(item, dict)
                )
                if candidate is not None
            )
            return TaskExtractionResult(
                candidates=candidates,
                change_candidates=change_candidates,
                ignored_items=ignored_items,
                metadata={
                    **base_metadata,
                    "candidate_count": len(candidates),
                    "change_candidate_count": len(change_candidates),
                    "ignored_items": list(ignored_items),
                },
            )
        except Exception as exc:
            return TaskExtractionResult(
                candidates=tuple(),
                metadata={
                    **base_metadata,
                    "degraded": True,
                    "degraded_reason": type(exc).__name__,
                },
            )

    def _candidate_from_payload(
        self,
        payload: dict[str, Any],
        *,
        evidence: tuple[Citation, ...],
        source_text: str,
        base_metadata: dict[str, Any],
    ) -> TaskCandidate | None:
        title = _clean_title(str(payload.get("title") or ""))
        if not title:
            return None
        confidence = str(payload.get("confidence") or "medium").lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        evidence_refs = payload.get("evidence")
        task_evidence = evidence[:5] or _synthetic_evidence(
            title=title,
            source_text=source_text,
            evidence_refs=evidence_refs,
        )
        if not task_evidence:
            return None
        assignee = payload.get("assignee_user_id") or payload.get("proposed_assignee_user_id")
        due_at = _parse_datetime(payload.get("due_at") or payload.get("proposed_due_at"))
        priority = str(payload.get("priority") or "normal").lower()
        if priority not in {"low", "normal", "high", "urgent"}:
            priority = "normal"
        candidate_id = stable_hash(
            "task-candidate:llm:"
            f"{title}:{assignee or ''}:{due_at.isoformat() if due_at else ''}:"
            f"{payload.get('related_event_id') or ''}"
        )[:32]
        return TaskCandidate(
            id=candidate_id,
            title=title,
            description=str(payload.get("description") or "").strip() or None,
            proposed_assignee_user_id=str(assignee).lstrip("@") if assignee else None,
            proposed_due_at=due_at,
            related_event_id=(
                str(payload.get("related_event_id")) if payload.get("related_event_id") else None
            ),
            evidence=task_evidence,
            confidence=confidence,
            status="proposed",
            created_by="agent",
            metadata={
                **base_metadata,
                "priority": priority,
                "evidence_refs": evidence_refs if isinstance(evidence_refs, list) else [],
            },
        )

    def _change_candidate_from_payload(
        self,
        payload: dict[str, Any],
        *,
        evidence: tuple[Citation, ...],
        source_text: str,
        existing_tasks: tuple[Task, ...],
        expected_operation: str,
        created_by: str,
        base_metadata: dict[str, Any],
    ) -> TaskChangeCandidate | None:
        operation = str(payload.get("operation") or "").strip().lower()
        if operation in {"complete", "done"}:
            operation = "update"
        if operation not in {"update", "delete"}:
            return None
        if expected_operation and operation != expected_operation:
            return None
        task = _resolve_existing_task(payload, existing_tasks)
        if task is None:
            return None
        before = _task_payload_for_extraction(task)
        after = dict(before)
        raw_after = payload.get("after")
        if isinstance(raw_after, dict):
            after.update(_clean_task_change_payload(raw_after))
        after.update(_clean_task_change_payload(payload))
        if operation == "delete":
            after["status"] = "deleted"
        if operation == "update" and after == before:
            return None
        evidence_refs = payload.get("evidence")
        change_evidence = evidence[:5] or _synthetic_evidence(
            title=task.title,
            source_text=source_text,
            evidence_refs=evidence_refs,
        )
        if not change_evidence:
            return None
        confidence = str(payload.get("confidence") or "medium").lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        reason = str(payload.get("reason") or payload.get("description") or "").strip()
        candidate_id = stable_hash(
            "task-change:llm:"
            f"{task.id}:{operation}:{json.dumps(after, sort_keys=True, default=str)}"
        )[:32]
        return TaskChangeCandidate(
            id=candidate_id,
            task_id=task.id,
            operation=operation,
            before=before,
            after=after,
            reason=reason,
            evidence=change_evidence,
            confidence=confidence,
            status="proposed",
            created_by=created_by if created_by in {"agent", "user"} else "agent",
            metadata={
                **base_metadata,
                "evidence_refs": evidence_refs if isinstance(evidence_refs, list) else [],
            },
        )

    def _prompt(self) -> str:
        if self._prompts_dir is None:
            return _DEFAULT_PROMPT
        path = self._prompts_dir / self._prompt_name
        if not path.exists():
            return _DEFAULT_PROMPT
        return path.read_text(encoding="utf-8")


class DuplicateTaskDetector:
    def annotate(
        self,
        candidate: TaskCandidate,
        *,
        existing_candidates: list[TaskCandidate],
        existing_tasks: list[Task],
    ) -> TaskCandidate:
        duplicates: list[dict[str, Any]] = []
        normalized_title = _normalize_title(candidate.title)
        for other in existing_candidates:
            if other.id == candidate.id or other.status not in {"proposed", "approved"}:
                continue
            score = _duplicate_score(
                normalized_title,
                _normalize_title(other.title),
                candidate.proposed_assignee_user_id,
                other.proposed_assignee_user_id,
                candidate.proposed_due_at,
                other.proposed_due_at,
                candidate.related_event_id,
                other.related_event_id,
            )
            if score >= 0.72:
                duplicates.append(
                    {
                        "target_type": "task_candidate",
                        "target_id": other.id,
                        "score": round(score, 3),
                        "reason": _duplicate_reason(
                            candidate.proposed_assignee_user_id,
                            other.proposed_assignee_user_id,
                            candidate.proposed_due_at,
                            other.proposed_due_at,
                            candidate.related_event_id,
                            other.related_event_id,
                        ),
                    }
                )
        for task in existing_tasks:
            score = _duplicate_score(
                normalized_title,
                _normalize_title(task.title),
                candidate.proposed_assignee_user_id,
                task.assignee_user_id,
                candidate.proposed_due_at,
                task.due_at,
                candidate.related_event_id,
                task.related_event_id,
            )
            if score >= 0.72:
                duplicates.append(
                    {
                        "target_type": "task",
                        "target_id": task.id,
                        "score": round(score, 3),
                        "reason": _duplicate_reason(
                            candidate.proposed_assignee_user_id,
                            task.assignee_user_id,
                            candidate.proposed_due_at,
                            task.due_at,
                            candidate.related_event_id,
                            task.related_event_id,
                        ),
                    }
                )
        if not duplicates:
            return candidate
        return TaskCandidate(
            **{
                **candidate.__dict__,
                "metadata": {
                    **candidate.metadata,
                    "duplicate_candidates": duplicates[:10],
                },
            }
        )


class TaskAccessPolicy:
    def __init__(
        self,
        *,
        admin_user_ids: tuple[str, ...] = tuple(),
        admin_role_ids: tuple[str, ...] = tuple(),
    ) -> None:
        self._admin_user_ids = {str(value) for value in admin_user_ids if str(value)}
        self._admin_role_ids = {str(value).lower() for value in admin_role_ids if str(value)}

    def is_admin(self, access: AccessContext) -> bool:
        roles = {role.lower() for role in access.role_ids}
        return (
            access.is_admin
            or bool(access.user_id and access.user_id in self._admin_user_ids)
            or bool(roles & self._admin_role_ids)
            or "admin" in roles
        )

    def can_create_candidate(self, access: AccessContext) -> bool:
        return bool(access.user_id) or self.is_admin(access)

    def can_list(self, access: AccessContext) -> bool:
        return self.is_admin(access) or _has_task_role(access)

    def can_show_candidate(self, access: AccessContext, candidate: TaskCandidate) -> bool:
        return (
            self.is_admin(access)
            or candidate.metadata.get("created_by_user_id") == access.user_id
            or candidate.proposed_assignee_user_id == access.user_id
        )

    def can_edit_candidate(self, access: AccessContext, candidate: TaskCandidate) -> bool:
        return self.is_admin(access) or candidate.metadata.get("created_by_user_id") == access.user_id

    def can_approve(self, access: AccessContext) -> bool:
        return self.is_admin(access)

    def can_reject_candidate(self, access: AccessContext, candidate: TaskCandidate) -> bool:
        return self.is_admin(access) or candidate.metadata.get("created_by_user_id") == access.user_id

    def can_show_task(self, access: AccessContext, task: Task) -> bool:
        return self.is_admin(access) or task.assignee_user_id == access.user_id or _has_task_role(access)

    def can_update_task_status(self, access: AccessContext, task: Task) -> bool:
        return self.is_admin(access) or task.assignee_user_id == access.user_id

    def forbidden_response_metadata(self) -> dict[str, Any]:
        return {"authorized": False}


class TaskNotificationPlanner:
    def planned_notifications(
        self,
        *,
        tasks: list[Task],
        now: datetime | None = None,
        before_days: int = 1,
    ) -> list[tuple[Task, str]]:
        current = now or datetime.now(UTC)
        horizon = current + timedelta(days=max(0, before_days))
        selected: list[tuple[Task, str]] = []
        for task in tasks:
            if task.status not in {"todo", "doing", "blocked"}:
                continue
            notifications = task.metadata.get("notifications")
            sent = notifications if isinstance(notifications, dict) else {}
            if not task.assignee_user_id and not sent.get("unassigned"):
                selected.append((task, "unassigned"))
            if task.status == "blocked" and not sent.get("blocked_check"):
                selected.append((task, "blocked_check"))
            if task.due_at is None:
                continue
            if task.due_at < current:
                key = "overdue"
            elif task.due_at <= horizon:
                key = "due_soon"
            else:
                continue
            if not sent.get(key):
                selected.append((task, key))
        return selected

    def due_notifications(
        self,
        *,
        tasks: list[Task],
        now: datetime | None = None,
        before_days: int = 1,
    ) -> list[Task]:
        return [
            task
            for task, kind in self.planned_notifications(
                tasks=tasks,
                now=now,
                before_days=before_days,
            )
            if kind in {"due_soon", "overdue"}
        ]


def _has_task_role(access: AccessContext) -> bool:
    roles = {role.lower() for role in access.role_ids}
    return "admin" in roles or "organizer" in roles or "task_manager" in roles


def _synthetic_evidence(
    *,
    title: str,
    source_text: str,
    evidence_refs: object,
) -> tuple[Citation, ...]:
    refs = evidence_refs if isinstance(evidence_refs, list) else []
    if not refs:
        return tuple()
    label = str(refs[0] or "input evidence")
    safe_quote = _safe_context(source_text, limit=360)
    return (
        Citation(
            source_item_id=stable_hash(f"task-evidence:{title}:{safe_quote}")[:32],
            chunk_id="llm-evidence",
            label=label[:120],
            quote=safe_quote,
            metadata={"synthetic": True},
        ),
    )


def _new_items(
    payload: dict[str, Any],
    *,
    legacy_key: str,
    item_type: str,
) -> list[dict[str, Any]]:
    raw_items = payload.get("new_items")
    legacy = False
    if raw_items is None:
        raw_items = payload.get(legacy_key)
        legacy = True
    if raw_items is None and ("change_items" in payload or "task_changes" in payload):
        raw_items = []
        legacy = False
    if not isinstance(raw_items, list):
        raise ValueError(f"{legacy_key} or new_items must be a list")
    return [
        item
        for item in raw_items
        if isinstance(item, dict)
        and (legacy or str(item.get("item_type") or "").strip().lower() == item_type)
    ]


def _change_items(
    payload: dict[str, Any],
    *,
    legacy_key: str,
    item_type: str,
) -> list[dict[str, Any]]:
    raw_items = payload.get("change_items")
    legacy = False
    if raw_items is None:
        raw_items = payload.get(legacy_key, [])
        legacy = True
    if not isinstance(raw_items, list):
        raise ValueError(f"{legacy_key} or change_items must be a list")
    return [
        item
        for item in raw_items
        if isinstance(item, dict)
        and (legacy or str(item.get("item_type") or "").strip().lower() == item_type)
    ]


def _ignored_items(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_items = payload.get("ignored_items")
    if not isinstance(raw_items, list):
        return tuple()
    ignored: list[dict[str, Any]] = []
    for item in raw_items[:20]:
        if not isinstance(item, dict):
            continue
        ignored.append(
            {
                "reason": _safe_context(str(item.get("reason") or ""), limit=160),
                "text_excerpt": _safe_context(str(item.get("text_excerpt") or ""), limit=240),
            }
        )
    return tuple(ignored)


def _task_payload_for_extraction(task: Task) -> dict[str, object]:
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


def _resolve_existing_task(payload: dict[str, Any], tasks: tuple[Task, ...]) -> Task | None:
    task_id = str(
        payload.get("target_id") or payload.get("task_id") or payload.get("id") or ""
    ).strip()
    if task_id:
        for task in tasks:
            if task.id == task_id:
                return task
    title = _normalize_title(str(payload.get("title") or payload.get("task_title") or ""))
    matches = [task for task in tasks if title and _normalize_title(task.title) == title]
    assignee = payload.get("assignee_user_id") or payload.get("proposed_assignee_user_id")
    if assignee:
        assignee_text = str(assignee).lstrip("@")
        assigned = [task for task in matches if task.assignee_user_id == assignee_text]
        if len(assigned) == 1:
            return assigned[0]
    due_at = _parse_datetime(payload.get("due_at") or payload.get("proposed_due_at"))
    if due_at:
        dated = [
            task
            for task in matches
            if task.due_at is not None and task.due_at.date() == due_at.date()
        ]
        if len(dated) == 1:
            return dated[0]
    if len(matches) == 1:
        return matches[0]
    return None


def _clean_task_change_payload(payload: dict[str, Any]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for key in ("title", "description", "related_event_id"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            cleaned[key] = _clean_title(str(value)) if key == "title" else str(value).strip()
    assignee = payload.get("assignee_user_id") or payload.get("proposed_assignee_user_id")
    if assignee is not None and str(assignee).strip():
        cleaned["assignee_user_id"] = str(assignee).lstrip("@")
    due_at = _parse_datetime(payload.get("due_at") or payload.get("proposed_due_at"))
    if due_at is not None:
        cleaned["due_at"] = due_at.isoformat()
    status = str(payload.get("status") or "").strip().lower()
    if status in {"todo", "doing", "blocked", "done", "deleted"}:
        cleaned["status"] = status
    priority = str(payload.get("priority") or "").strip().lower()
    if priority in {"low", "normal", "high", "urgent"}:
        cleaned["priority"] = priority
    return cleaned


def _citation_payload(citation: Citation) -> dict[str, object]:
    return {
        "source_item_id": citation.source_item_id,
        "chunk_id": citation.chunk_id,
        "label": citation.label,
        "url": citation.url,
        "quote": _safe_context(citation.quote, limit=240),
        "score": citation.score,
    }


def _safe_context(text: str, *, limit: int = 6000) -> str:
    masked = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text or "",
    )
    normalized = re.sub(r"\s+", " ", masked).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?", text)
        if not match:
            return None
        year, month, day, hour, minute = match.groups()
        return datetime(
            int(year),
            int(month),
            int(day),
            int(hour or 0),
            int(minute or 0),
            tzinfo=UTC,
        )


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" -:：、。"))


def _normalize_title(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.lower())


def _duplicate_score(
    left_title: str,
    right_title: str,
    left_assignee: str | None,
    right_assignee: str | None,
    left_due: datetime | None,
    right_due: datetime | None,
    left_event: str | None,
    right_event: str | None,
) -> float:
    if not left_title or not right_title:
        return 0.0
    title_score = _jaccard(left_title, right_title)
    score = title_score * 0.62
    if left_assignee and right_assignee and left_assignee == right_assignee:
        score += 0.14
    if left_due and right_due and left_due.date() == right_due.date():
        score += 0.14
    if left_event and right_event and left_event == right_event:
        score += 0.10
    return min(1.0, score)


def _duplicate_reason(
    left_assignee: str | None,
    right_assignee: str | None,
    left_due: datetime | None,
    right_due: datetime | None,
    left_event: str | None,
    right_event: str | None,
) -> str:
    reasons = ["title_similarity"]
    if left_assignee and right_assignee and left_assignee == right_assignee:
        reasons.append("same_assignee")
    if left_due and right_due and left_due.date() == right_due.date():
        reasons.append("same_due_date")
    if left_event and right_event and left_event == right_event:
        reasons.append("same_related_event")
    return ",".join(reasons)


def _jaccard(left: str, right: str) -> float:
    if left == right:
        return 1.0
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


_DEFAULT_PROMPT = """\
あなたはKUMC-Agentのタスク抽出専用コンポーネントです。
入力文から、承認待ち候補にする具体的なタスクだけを抽出してください。
既存Taskの変更・完了・削除は新規タスクではなくchange_itemsに出してください。
必ずJSONオブジェクトだけを返してください。schema_versionは "workflow_extraction.v1" です。
schema:
{
  "schema_version": "workflow_extraction.v1",
  "new_items": [
    {
      "item_type": "task",
      "title": "string",
      "description": "string",
      "assignee_user_id": "string|null",
      "due_at": "YYYY-MM-DDTHH:MM:SS+00:00|null",
      "related_event_id": "string|null",
      "priority": "low|normal|high|urgent",
      "confidence": "low|medium|high",
      "evidence": ["short evidence labels"]
    }
  ],
  "change_items": [
    {
      "item_type": "task",
      "target_id": "既存Task id",
      "operation": "update|delete",
      "after": {
        "status": "todo|doing|blocked|done|deleted",
        "assignee_user_id": "string|null",
        "due_at": "YYYY-MM-DDTHH:MM:SS+00:00|null",
        "priority": "low|normal|high|urgent"
      },
      "reason": "変更理由",
      "confidence": "low|medium|high",
      "evidence": ["short evidence labels"]
    }
  ],
  "ignored_items": [],
  "degraded": false
}
"""
