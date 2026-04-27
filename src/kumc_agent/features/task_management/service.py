from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import re
from typing import Any

from kumc_agent.domain.models.retrieval import AccessContext, Citation
from kumc_agent.domain.models.workflow import Task, TaskCandidate
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class TaskExtractionResult:
    candidates: tuple[TaskCandidate, ...]
    metadata: dict[str, Any]


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
    ) -> TaskExtractionResult:
        source_text = _safe_context(text)
        if self._llm is None:
            return TaskExtractionResult(
                candidates=tuple(),
                metadata={
                    **metadata,
                    "extractor": "task_llm",
                    "degraded": True,
                    "degraded_reason": "llm_unavailable",
                },
            )
        if not source_text.strip():
            return TaskExtractionResult(
                candidates=tuple(),
                metadata={
                    **metadata,
                    "extractor": "task_llm",
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
                        "actor_user_id": access.user_id,
                    },
                    ensure_ascii=False,
                ),
                temperature=0.0,
                max_output_tokens=2048,
            )
            payload = _extract_json_object(raw)
            items = payload.get("tasks")
            if not isinstance(items, list):
                raise ValueError("tasks must be a list")
            candidates = tuple(
                candidate
                for candidate in (
                    self._candidate_from_payload(
                        item,
                        evidence=evidence,
                        base_metadata={
                            **metadata,
                            "extractor": "task_llm",
                            "extractor_model": self._model_name,
                            "prompt_version": self._prompt_name,
                        },
                    )
                    for item in items
                    if isinstance(item, dict)
                )
                if candidate is not None
            )
            return TaskExtractionResult(
                candidates=candidates,
                metadata={
                    **metadata,
                    "extractor": "task_llm",
                    "extractor_model": self._model_name,
                    "prompt_version": self._prompt_name,
                    "candidate_count": len(candidates),
                },
            )
        except Exception as exc:
            return TaskExtractionResult(
                candidates=tuple(),
                metadata={
                    **metadata,
                    "extractor": "task_llm",
                    "extractor_model": self._model_name,
                    "prompt_version": self._prompt_name,
                    "degraded": True,
                    "degraded_reason": type(exc).__name__,
                },
            )

    def _candidate_from_payload(
        self,
        payload: dict[str, Any],
        *,
        evidence: tuple[Citation, ...],
        base_metadata: dict[str, Any],
    ) -> TaskCandidate | None:
        title = _clean_title(str(payload.get("title") or ""))
        if not title:
            return None
        confidence = str(payload.get("confidence") or "medium").lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        task_evidence = evidence[:5]
        evidence_refs = payload.get("evidence")
        if not task_evidence and not evidence_refs:
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
                    {"target_type": "task_candidate", "target_id": other.id, "score": round(score, 3)}
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
                duplicates.append({"target_type": "task", "target_id": task.id, "score": round(score, 3)})
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
    def can_create_candidate(self, access: AccessContext) -> bool:
        return bool(access.user_id) or access.is_admin

    def can_list(self, access: AccessContext) -> bool:
        return access.is_admin or _has_task_role(access)

    def can_show_candidate(self, access: AccessContext, candidate: TaskCandidate) -> bool:
        return (
            access.is_admin
            or candidate.metadata.get("created_by_user_id") == access.user_id
            or candidate.proposed_assignee_user_id == access.user_id
        )

    def can_edit_candidate(self, access: AccessContext, candidate: TaskCandidate) -> bool:
        return access.is_admin or candidate.metadata.get("created_by_user_id") == access.user_id

    def can_approve(self, access: AccessContext) -> bool:
        return access.is_admin

    def can_reject_candidate(self, access: AccessContext, candidate: TaskCandidate) -> bool:
        return access.is_admin or candidate.metadata.get("created_by_user_id") == access.user_id

    def can_show_task(self, access: AccessContext, task: Task) -> bool:
        return access.is_admin or task.assignee_user_id == access.user_id or _has_task_role(access)

    def can_update_task_status(self, access: AccessContext, task: Task) -> bool:
        return access.is_admin or task.assignee_user_id == access.user_id

    def forbidden_response_metadata(self) -> dict[str, Any]:
        return {"authorized": False}


class TaskNotificationPlanner:
    def due_notifications(
        self,
        *,
        tasks: list[Task],
        now: datetime | None = None,
        before_days: int = 1,
    ) -> list[Task]:
        current = now or datetime.now(UTC)
        horizon = current + timedelta(days=max(0, before_days))
        selected: list[Task] = []
        for task in tasks:
            if task.status not in {"todo", "doing", "blocked"} or task.due_at is None:
                continue
            notifications = task.metadata.get("notifications")
            sent = notifications if isinstance(notifications, dict) else {}
            if task.due_at < current:
                key = "overdue"
            elif task.due_at <= horizon:
                key = "due_soon"
            else:
                continue
            if sent.get(key):
                continue
            selected.append(task)
        return selected


def _has_task_role(access: AccessContext) -> bool:
    roles = {role.lower() for role in access.role_ids}
    return "admin" in roles or "organizer" in roles or "task_manager" in roles


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
入力文から、実行すべき具体的なタスクだけを抽出してください。
未決事項、質問、単なる告知、予定そのものはタスクにしないでください。
必ずJSONオブジェクトだけを返してください。
schema:
{
  "tasks": [
    {
      "title": "string",
      "description": "string",
      "assignee_user_id": "string|null",
      "due_at": "YYYY-MM-DDTHH:MM:SS+00:00|null",
      "related_event_id": "string|null",
      "priority": "low|normal|high|urgent",
      "confidence": "low|medium|high",
      "evidence": ["short evidence labels"]
    }
  ]
}
"""
