from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
import re
from typing import Any

from kumc_agent.domain.models.retrieval import AccessContext, Citation
from kumc_agent.domain.models.workflow import Event, EventCandidate
from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class EventExtractionResult:
    candidates: tuple[EventCandidate, ...]
    metadata: dict[str, Any]


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
                        "actor_user_id": access.user_id,
                    },
                    ensure_ascii=False,
                ),
                temperature=0.0,
                max_output_tokens=2048,
            )
            payload = _extract_json_object(raw)
            items = payload.get("events")
            if not isinstance(items, list):
                raise ValueError("events must be a list")
            candidates = tuple(
                candidate
                for candidate in (
                    self._candidate_from_payload(
                        item,
                        evidence=evidence,
                        base_metadata=base_metadata,
                    )
                    for item in items
                    if isinstance(item, dict)
                )
                if candidate is not None
            )
            return EventExtractionResult(
                candidates=candidates,
                metadata={**base_metadata, "candidate_count": len(candidates)},
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
        base_metadata: dict[str, Any],
    ) -> EventCandidate | None:
        title = _clean_title(str(payload.get("title") or ""))
        starts_at = _parse_datetime(payload.get("starts_at"))
        if not title or starts_at is None:
            return None
        confidence = str(payload.get("confidence") or "medium").lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        event_evidence = evidence[:5]
        evidence_refs = payload.get("evidence")
        if not event_evidence and not evidence_refs:
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
            created_by="agent",
            metadata={
                **base_metadata,
                "evidence_refs": evidence_refs if isinstance(evidence_refs, list) else [],
                "related_task_query": str(payload.get("related_task_query") or "").strip(),
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
    def can_manage(self, access: AccessContext) -> bool:
        return access.is_admin or _has_event_admin_role(access)

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
        event_day = event.starts_at.date()
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
        if event_day < current_day or event_day > target_day:
            return None
        return f"before:{max(0, before_days)}:{event_day.isoformat()}"


def _has_event_admin_role(access: AccessContext) -> bool:
    roles = {role.lower() for role in access.role_ids}
    return "admin" in roles or "organizer" in roles or "event_manager" in roles


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
  "events": [
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
  ]
}
"""
