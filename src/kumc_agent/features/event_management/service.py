from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
import re
from typing import Any

from kumc_agent.domain.models.retrieval import AccessContext, Citation
from kumc_agent.domain.models.workflow import Event, EventCandidate, EventChangeCandidate
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class EventExtractionResult:
    candidates: tuple[EventCandidate, ...]
    metadata: dict[str, Any]
    change_candidates: tuple[EventChangeCandidate, ...] = tuple()


class EventExtractionService:
    def __init__(
        self,
        *,
        llm: LLMPort | None = None,
        prompts_dir: Path | None = None,
        prompt_name: str = "event_extraction.md",
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
        existing_events: tuple[Event, ...] = tuple(),
    ) -> EventExtractionResult:
        source_text = _safe_context(text)
        base_metadata = {
            **metadata,
            "extractor": "event_llm",
            "extractor_model": self._model_name,
            "prompt_version": self._prompt_name,
        }
        if self._llm is None:
            return EventExtractionResult(
                candidates=tuple(),
                metadata={
                    **base_metadata,
                    "degraded": True,
                    "degraded_reason": "llm_unavailable",
                },
            )
        if not source_text.strip():
            return EventExtractionResult(
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
                        "existing_events": [
                            _event_payload_for_extraction(event) for event in existing_events[:50]
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
            items = payload.get("new_events")
            if items is None:
                items = payload.get("events")
            if not isinstance(items, list):
                raise ValueError("new_events must be a list")
            candidates = tuple(
                candidate
                for candidate in (
                    self._candidate_from_payload(
                        item,
                        evidence=evidence,
                        source_text=source_text,
                        created_by=str(metadata.get("created_by") or "agent"),
                        base_metadata=base_metadata,
                    )
                    for item in items
                    if isinstance(item, dict)
                )
                if candidate is not None
            )
            raw_changes = payload.get("event_changes")
            if raw_changes is None:
                raw_changes = []
            if not isinstance(raw_changes, list):
                raise ValueError("event_changes must be a list")
            expected_operation = str(metadata.get("expected_operation") or "").strip()
            change_candidates = tuple(
                candidate
                for candidate in (
                    self._change_candidate_from_payload(
                        item,
                        evidence=evidence,
                        source_text=source_text,
                        existing_events=existing_events,
                        expected_operation=expected_operation,
                        created_by=str(metadata.get("created_by") or "agent"),
                        base_metadata=base_metadata,
                    )
                    for item in raw_changes
                    if isinstance(item, dict)
                )
                if candidate is not None
            )
            ignored_items = payload.get("ignored_items")
            return EventExtractionResult(
                candidates=candidates,
                change_candidates=change_candidates,
                metadata={
                    **base_metadata,
                    "candidate_count": len(candidates),
                    "change_candidate_count": len(change_candidates),
                    "ignored_items": ignored_items if isinstance(ignored_items, list) else [],
                },
            )
        except Exception as exc:
            return EventExtractionResult(
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
        created_by: str,
        base_metadata: dict[str, Any],
    ) -> EventCandidate | None:
        title = _clean_title(str(payload.get("title") or ""))
        starts_at = _parse_datetime(payload.get("starts_at"))
        if not title or starts_at is None:
            return None
        confidence = str(payload.get("confidence") or "medium").lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        evidence_refs = payload.get("evidence")
        event_evidence = evidence[:5] or _synthetic_evidence(
            title=title,
            source_text=source_text,
            evidence_refs=evidence_refs,
        )
        if not event_evidence:
            return None
        ends_at = _parse_datetime(payload.get("ends_at"))
        place = str(payload.get("place") or "").strip() or None
        related_sources = payload.get("related_source_ids")
        if not isinstance(related_sources, list):
            related_sources = []
        candidate_id = stable_hash(
            "event-candidate:llm:"
            f"{title}:{starts_at.isoformat()}:{ends_at.isoformat() if ends_at else ''}:{place or ''}"
        )[:32]
        return EventCandidate(
            id=candidate_id,
            title=title,
            summary=str(payload.get("summary") or "").strip() or None,
            starts_at=starts_at,
            ends_at=ends_at,
            place=place,
            related_source_ids=tuple(str(item) for item in related_sources),
            evidence=event_evidence,
            confidence=confidence,
            status="proposed",
            created_by=created_by if created_by in {"agent", "user"} else "agent",
            metadata={
                **base_metadata,
                "evidence_refs": evidence_refs if isinstance(evidence_refs, list) else [],
                "related_task_query": str(payload.get("related_task_query") or "").strip(),
            },
        )

    def _change_candidate_from_payload(
        self,
        payload: dict[str, Any],
        *,
        evidence: tuple[Citation, ...],
        source_text: str,
        existing_events: tuple[Event, ...],
        expected_operation: str,
        created_by: str,
        base_metadata: dict[str, Any],
    ) -> EventChangeCandidate | None:
        operation = str(payload.get("operation") or "").strip().lower()
        if operation == "cancel":
            operation = "delete"
        if operation not in {"update", "delete"}:
            return None
        if expected_operation and operation != expected_operation:
            return None
        event = _resolve_existing_event(payload, existing_events)
        if event is None:
            return None
        before = _event_payload_for_extraction(event)
        after = dict(before)
        raw_after = payload.get("after")
        if isinstance(raw_after, dict):
            after.update(_clean_event_change_payload(raw_after))
        after.update(_clean_event_change_payload(payload))
        if operation == "delete":
            after["status"] = "canceled"
        if operation == "update" and after == before:
            return None
        evidence_refs = payload.get("evidence")
        change_evidence = evidence[:5] or _synthetic_evidence(
            title=event.title,
            source_text=source_text,
            evidence_refs=evidence_refs,
        )
        if not change_evidence:
            return None
        confidence = str(payload.get("confidence") or "medium").lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        reason = str(payload.get("reason") or payload.get("summary") or "").strip()
        candidate_id = stable_hash(
            "event-change:llm:"
            f"{event.id}:{operation}:{json.dumps(after, sort_keys=True, default=str)}"
        )[:32]
        return EventChangeCandidate(
            id=candidate_id,
            event_id=event.id,
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


class DuplicateEventDetector:
    def annotate(
        self,
        candidate: EventCandidate,
        *,
        existing_candidates: list[EventCandidate],
        existing_events: list[Event],
    ) -> EventCandidate:
        duplicates: list[dict[str, Any]] = []
        normalized_title = _normalize_title(candidate.title)
        for other in existing_candidates:
            if other.status not in {"proposed", "approved"}:
                continue
            score = (
                1.0
                if other.id == candidate.id
                else _duplicate_score(
                    normalized_title,
                    _normalize_title(other.title),
                    candidate.starts_at,
                    other.starts_at,
                    candidate.place,
                    other.place,
                )
            )
            if score >= 0.72:
                duplicates.append(
                    {
                        "target_type": "event_candidate",
                        "target_id": other.id,
                        "score": round(score, 3),
                    }
                )
        for event in existing_events:
            if event.status == "canceled":
                continue
            score = _duplicate_score(
                normalized_title,
                _normalize_title(event.title),
                candidate.starts_at,
                event.starts_at,
                candidate.place,
                event.place,
            )
            if score >= 0.72:
                duplicates.append(
                    {"target_type": "event", "target_id": event.id, "score": round(score, 3)}
                )
        if not duplicates:
            return candidate
        return EventCandidate(
            **{
                **candidate.__dict__,
                "metadata": {
                    **candidate.metadata,
                    "duplicate_candidates": duplicates[:10],
                },
            }
        )


class EventAccessPolicy:
    def __init__(
        self,
        *,
        admin_user_ids: tuple[str, ...] = tuple(),
        admin_role_ids: tuple[str, ...] = tuple(),
    ) -> None:
        self._admin_user_ids = {str(value) for value in admin_user_ids if str(value)}
        self._admin_role_ids = {str(value).lower() for value in admin_role_ids if str(value)}

    def can_manage(self, access: AccessContext) -> bool:
        roles = {role.lower() for role in access.role_ids}
        return (
            access.is_admin
            or bool(access.user_id and access.user_id in self._admin_user_ids)
            or bool(roles & self._admin_role_ids)
            or _has_event_admin_role(access)
        )

    def forbidden_response_metadata(self) -> dict[str, Any]:
        return {"authorized": False}


class EventNotificationPlanner:
    def notifications(
        self,
        *,
        events: list[Event],
        now: datetime | None = None,
        before_days: int = 1,
        kind: str = "before",
    ) -> list[Event]:
        current = now or datetime.now(UTC)
        selected: list[Event] = []
        for event in events:
            if event.status not in {"planning", "announced"} or event.starts_at is None:
                continue
            key = self.notification_key(event=event, now=current, before_days=before_days, kind=kind)
            if key is None:
                continue
            notifications = event.metadata.get("notifications")
            sent = notifications if isinstance(notifications, dict) else {}
            if sent.get(key):
                continue
            selected.append(event)
        return selected

    def notification_key(
        self,
        *,
        event: Event,
        now: datetime,
        before_days: int,
        kind: str,
    ) -> str | None:
        if event.starts_at is None:
            return None
        now_tz = now.tzinfo or UTC
        starts_at = event.starts_at
        if starts_at.tzinfo is None:
            starts_at = starts_at.replace(tzinfo=now_tz)
        else:
            starts_at = starts_at.astimezone(now_tz)
        event_day = starts_at.date()
        current_day = now.date()
        if kind == "day_of":
            if event_day != current_day:
                return None
            return f"day_of:{event_day.isoformat()}"
        if kind == "completion":
            if event_day != current_day:
                return None
            return f"completion:{event_day.isoformat()}"
        target_day = current_day + timedelta(days=max(0, before_days))
        if event_day != target_day:
            return None
        return f"before:{max(0, before_days)}:{event_day.isoformat()}"


def _has_event_admin_role(access: AccessContext) -> bool:
    roles = {role.lower() for role in access.role_ids}
    return "admin" in roles or "organizer" in roles or "event_manager" in roles


def _synthetic_evidence(
    *,
    title: str,
    source_text: str,
    evidence_refs: object,
) -> tuple[Citation, ...]:
    refs = evidence_refs if isinstance(evidence_refs, list) else []
    if not refs:
        return tuple()
    safe_quote = _safe_context(source_text, limit=360)
    return (
        Citation(
            source_item_id=stable_hash(f"event-evidence:{title}:{safe_quote}")[:32],
            chunk_id="llm-evidence",
            label=str(refs[0] or "input evidence")[:120],
            quote=safe_quote,
            metadata={"synthetic": True},
        ),
    )


def _event_payload_for_extraction(event: Event) -> dict[str, object]:
    return {
        "id": event.id,
        "title": event.title,
        "summary": event.summary,
        "starts_at": event.starts_at.isoformat() if event.starts_at else None,
        "ends_at": event.ends_at.isoformat() if event.ends_at else None,
        "place": event.place,
        "status": event.status,
        "related_source_ids": list(event.related_source_ids),
    }


def _resolve_existing_event(payload: dict[str, Any], events: tuple[Event, ...]) -> Event | None:
    event_id = str(payload.get("event_id") or payload.get("id") or "").strip()
    if event_id:
        for event in events:
            if event.id == event_id:
                return event
    title = _normalize_title(str(payload.get("title") or payload.get("event_title") or ""))
    starts_at = _parse_datetime(payload.get("starts_at"))
    matches = [
        event
        for event in events
        if title and _normalize_title(event.title) == title
    ]
    if starts_at:
        dated = [event for event in matches if event.starts_at and event.starts_at.date() == starts_at.date()]
        if len(dated) == 1:
            return dated[0]
    if len(matches) == 1:
        return matches[0]
    return None


def _clean_event_change_payload(payload: dict[str, Any]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for key in ("title", "summary", "place", "status"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            cleaned[key] = _clean_title(str(value)) if key == "title" else str(value).strip()
    for key in ("starts_at", "ends_at"):
        parsed = _parse_datetime(payload.get(key))
        if parsed is not None:
            cleaned[key] = parsed.isoformat()
    related_sources = payload.get("related_source_ids")
    if isinstance(related_sources, list):
        cleaned["related_source_ids"] = [str(item) for item in related_sources]
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


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("JSON object not found")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?", text)
        if not match:
            return None
        year, month, day, hour, minute = match.groups()
        parsed = datetime(
            int(year),
            int(month),
            int(day),
            int(hour or 0),
            int(minute or 0),
        )
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" -:：、。"))


def _normalize_title(value: str) -> str:
    return re.sub(r"[\s\-_:：、。・]+", "", value.lower())


def _duplicate_score(
    left_title: str,
    right_title: str,
    left_start: datetime | None,
    right_start: datetime | None,
    left_place: str | None,
    right_place: str | None,
) -> float:
    if not left_title or not right_title:
        title_score = 0.0
    elif left_title == right_title:
        title_score = 0.6
    elif left_title in right_title or right_title in left_title:
        title_score = 0.45
    else:
        common = len(set(left_title) & set(right_title))
        total = max(1, len(set(left_title) | set(right_title)))
        title_score = 0.35 * (common / total)
    date_score = 0.0
    if left_start and right_start:
        delta = abs((left_start - right_start).total_seconds())
        if delta <= 3600:
            date_score = 0.3
        elif delta <= 86400:
            date_score = 0.18
    place_score = 0.0
    left_place_norm = _normalize_title(left_place or "")
    right_place_norm = _normalize_title(right_place or "")
    if left_place_norm and right_place_norm:
        if left_place_norm == right_place_norm:
            place_score = 0.1
        elif left_place_norm in right_place_norm or right_place_norm in left_place_norm:
            place_score = 0.06
    return min(1.0, title_score + date_score + place_score)


_DEFAULT_PROMPT = """\
あなたはKUMCのイベント管理用抽出器です。
入力本文から、正本登録前にadmin承認へ回すべきイベント候補だけを抽出してください。
タスク単体、雑談、未決事項、日時のない一般告知はイベントにしないでください。

JSONのみを返してください:
{
  "new_events": [
    {
      "title": "イベント名",
      "summary": "短い概要",
      "starts_at": "2026-05-05T14:00:00+09:00",
      "ends_at": null,
      "place": "場所",
      "related_source_ids": [],
      "related_task_query": "関連タスク検索条件",
      "confidence": "low|medium|high",
      "evidence": ["根拠ラベル"]
    }
  ],
  "event_changes": [
    {
      "event_id": "既存Event id",
      "operation": "update|delete",
      "after": {
        "starts_at": "2026-05-05T15:00:00+09:00",
        "place": "変更後場所",
        "status": "planning|announced|done|canceled"
      },
      "reason": "変更理由",
      "confidence": "low|medium|high",
      "evidence": ["根拠ラベル"]
    }
  ],
  "ignored_items": [],
  "degraded": false
}
"""
