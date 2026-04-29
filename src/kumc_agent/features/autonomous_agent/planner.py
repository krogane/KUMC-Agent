from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from kumc_agent.domain.models.autonomous_agent import (
    AutonomousCheck,
    AutonomousPlan,
    AutonomousQuery,
    SnapshotItem,
)
from kumc_agent.features.autonomous_agent.llm import (
    AutonomousLLMConfig,
    dump_value,
    llm_generate,
    load_json_object,
    read_prompt,
    string_tuple,
)
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class PlannerConfig:
    notification_channel_id: str = ""
    max_replans: int = 1
    duplicate_suppression_hours: int = 24


class AutonomousPlanner:
    def __init__(
        self,
        *,
        config: PlannerConfig | None = None,
        llm: object | None = None,
        llm_config: AutonomousLLMConfig | None = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self.llm = llm
        self.llm_config = llm_config or AutonomousLLMConfig()

    def plan(self, snapshot) -> AutonomousPlan:
        deterministic = self._deterministic_plan(snapshot)
        llm_plan = self._plan_with_llm(snapshot=snapshot, deterministic=deterministic)
        if llm_plan is None:
            return deterministic
        return _merge_plans(deterministic, llm_plan)

    def _deterministic_plan(self, snapshot) -> AutonomousPlan:
        checks: list[AutonomousCheck] = []
        queries: list[AutonomousQuery] = []
        target_refs: list[str] = []
        warnings: list[str] = list(snapshot.warnings)
        suppressed = _suppressed_refs(
            snapshot.recent_runs,
            duplicate_suppression_hours=self.config.duplicate_suppression_hours,
        )

        for item in snapshot.tasks_due_soon:
            ref = f"task:{item.id}"
            target_refs.append(ref)
            if ref in suppressed:
                warnings.append(f"duplicate_notification_suppressed:{ref}")
                continue
            checks.append(_check("task_due_soon", ref, f"期限が近いタスクを確認: {item.title}"))

        for item in snapshot.tasks_overdue:
            ref = f"task:{item.id}"
            target_refs.append(ref)
            if ref in suppressed:
                warnings.append(f"duplicate_notification_suppressed:{ref}")
                continue
            checks.append(
                _check(
                    "task_overdue",
                    ref,
                    f"期限超過タスクの完了状況を確認: {item.title}",
                    risk="medium",
                )
            )

        for item in snapshot.tasks_stale:
            ref = f"task:{item.id}"
            target_refs.append(ref)
            checks.append(_check("task_stale", ref, f"停滞中タスクを確認: {item.title}", risk="medium"))

        for item in snapshot.events_without_tasks:
            ref = f"event:{item.id}"
            target_refs.append(ref)
            queries.append(
                _query(
                    "event_prepare_tasks",
                    f"イベント {item.id} ({item.title}) の準備タスク候補を作成して",
                    source="event",
                    work_type="task_add",
                    target_refs=(ref,),
                )
            )

        for item in snapshot.events_missing_details:
            ref = f"event:{item.id}"
            target_refs.append(ref)
            checks.append(
                _check(
                    "event_missing_details",
                    ref,
                    f"日時または場所が未定のイベントを確認: {item.title}",
                    risk="medium",
                )
            )
            queries.append(
                _query(
                    "event_update_candidate",
                    f"イベント {item.id} ({item.title}) の不足情報を確認する通知案または変更候補を作成して",
                    source="event",
                    work_type="event_update",
                    target_refs=(ref,),
                )
            )

        for item in snapshot.rag_delta:
            ref = f"source:{item.id}"
            target_refs.append(ref)
            queries.append(
                _query(
                    "rag_delta_extract",
                    f"本日の資料差分 {item.id} からタスク・イベント候補を抽出して",
                    source="all",
                    work_type="task_extract",
                    target_refs=(ref,),
                )
            )

        for item in snapshot.server_ops:
            ref = f"server_operation:{item.id}"
            target_refs.append(ref)
            checks.append(
                _check(
                    "server_operation_approval",
                    ref,
                    f"未承認サーバー操作の承認依頼を確認: {item.title}",
                    risk=item.risk or "medium",
                    side_effect_boundary="approval_required",
                )
            )

        for item in snapshot.automation_runs:
            ref = f"automation_run:{item.id}"
            target_refs.append(ref)
            checks.append(
                _check(
                    "automation_run_attention",
                    ref,
                    f"確認が必要な automation run: {item.title}",
                    risk="medium",
                    side_effect_boundary="approval_required",
                )
            )

        max_risk = _max_risk([check.risk for check in checks] + [query.risk for query in queries])
        boundary = "approval_required" if max_risk in {"high", "critical", "medium"} else "candidate_only"
        return AutonomousPlan(
            checks=tuple(checks),
            required_queries=tuple(queries),
            target_refs=tuple(dict.fromkeys(target_refs)),
            success_criteria=_success_criteria(checks, queries),
            risk=max_risk,
            side_effect_boundary=boundary,
            notification_policy={
                "channel_id": self.config.notification_channel_id,
                "duplicate_suppression": True,
            },
            retry_policy={"max_replans": self.config.max_replans},
            warnings=tuple(warnings),
            metadata={
                "planner": "deterministic_v1",
                "check_count": len(checks),
                "query_count": len(queries),
            },
        )

    def _plan_with_llm(self, *, snapshot, deterministic: AutonomousPlan) -> AutonomousPlan | None:
        if not self.llm_config.enabled or self.llm is None:
            return None
        system_prompt = read_prompt(
            self.llm_config.prompts_dir,
            self.llm_config.prompt_name,
            fallback="Return autonomous agent planning JSON only.",
        )
        payload = {
            "snapshot": dump_value(snapshot),
            "deterministic_plan": dump_value(deterministic),
            "allowed_decisions": ["candidate_only", "approval_required"],
            "required_contract": {
                "no_direct_side_effects": True,
                "external_posting": "proposal_only",
                "server_operations": "approval_required",
                "master_updates": "candidate_only",
            },
        }
        for _ in range(max(1, self.llm_config.max_retries)):
            raw = llm_generate(
                self.llm,
                system_prompt=system_prompt,
                user_payload=payload,
                temperature=self.llm_config.temperature,
                max_output_tokens=self.llm_config.max_output_tokens,
            )
            parsed = load_json_object(raw)
            if parsed is None:
                continue
            plan = self._plan_from_payload(parsed, deterministic=deterministic)
            if plan is not None:
                return plan
        return None

    def _plan_from_payload(
        self,
        payload: dict[str, Any],
        *,
        deterministic: AutonomousPlan,
    ) -> AutonomousPlan | None:
        checks: list[AutonomousCheck] = []
        for raw in payload.get("checks") or []:
            if not isinstance(raw, dict):
                continue
            target_ref = str(raw.get("target_ref") or "").strip()
            kind = str(raw.get("kind") or "").strip()
            reason = str(raw.get("reason") or "").strip()
            if not target_ref or not kind or not reason:
                continue
            risk = _risk(str(raw.get("risk") or "low"))
            boundary = _boundary(str(raw.get("side_effect_boundary") or "candidate_only"), risk=risk)
            checks.append(
                AutonomousCheck(
                    id=str(raw.get("id") or stable_hash(f"llm-check:{kind}:{target_ref}:{reason}")[:24]),
                    kind=kind,
                    target_ref=target_ref,
                    reason=reason,
                    risk=risk,
                    side_effect_boundary=boundary,
                    metadata=_dict(raw.get("metadata")),
                )
            )
        queries: list[AutonomousQuery] = []
        for raw in payload.get("required_queries") or []:
            if not isinstance(raw, dict):
                continue
            query = str(raw.get("query") or "").strip()
            if not query:
                continue
            target_refs = string_tuple(raw.get("target_refs"))
            work_type = str(raw.get("work_type") or "").strip()
            raw_risk = _risk(str(raw.get("risk") or "low"))
            risk = _boundary(
                str(raw.get("side_effect_boundary") or raw.get("risk") or "candidate_only"),
                risk=raw_risk,
            )
            queries.append(
                AutonomousQuery(
                    id=str(raw.get("id") or stable_hash(f"llm-query:{query}:{':'.join(target_refs)}")[:24]),
                    query=query,
                    source=str(raw.get("source") or "all"),
                    mode=str(raw.get("mode") or "careful"),
                    depth=str(raw.get("depth") or "normal"),
                    target_refs=target_refs,
                    work_type=work_type,
                    risk=risk,
                    metadata=_dict(raw.get("metadata")),
                )
            )
        if not checks and not queries:
            return None
        risks = [check.risk for check in checks] + [
            "medium" if query.risk == "approval_required" else "low"
            for query in queries
        ]
        max_risk = _risk(str(payload.get("risk") or _max_risk(risks)))
        boundary = _boundary(
            str(payload.get("side_effect_boundary") or deterministic.side_effect_boundary),
            risk=max_risk,
        )
        target_refs = tuple(
            dict.fromkeys(
                [
                    *string_tuple(payload.get("target_refs")),
                    *(check.target_ref for check in checks),
                    *(ref for query in queries for ref in query.target_refs),
                ]
            )
        )
        return AutonomousPlan(
            checks=tuple(checks),
            required_queries=tuple(queries),
            target_refs=target_refs,
            success_criteria=(
                string_tuple(payload.get("success_criteria"))
                or _success_criteria(checks, queries)
            ),
            risk=max_risk,
            side_effect_boundary=boundary,
            notification_policy=(
                _dict(payload.get("notification_policy"))
                or dict(deterministic.notification_policy)
            ),
            retry_policy=_dict(payload.get("retry_policy")) or dict(deterministic.retry_policy),
            warnings=string_tuple(payload.get("warnings")),
            metadata={
                **_dict(payload.get("metadata")),
                "planner": "llm",
                "fallback_guard": "deterministic_merge",
            },
        )


def _check(
    kind: str,
    target_ref: str,
    reason: str,
    *,
    risk: str = "low",
    side_effect_boundary: str = "candidate_only",
) -> AutonomousCheck:
    return AutonomousCheck(
        id=stable_hash(f"autonomous-check:{kind}:{target_ref}:{reason}")[:24],
        kind=kind,
        target_ref=target_ref,
        reason=reason,
        risk=risk,
        side_effect_boundary=side_effect_boundary,
    )


def _query(
    kind: str,
    query: str,
    *,
    source: str,
    work_type: str,
    target_refs: tuple[str, ...],
) -> AutonomousQuery:
    return AutonomousQuery(
        id=stable_hash(f"autonomous-query:{kind}:{query}:{':'.join(target_refs)}")[:24],
        query=query,
        source=source,
        work_type=work_type,
        target_refs=target_refs,
        risk="candidate_only",
        metadata={"kind": kind},
    )


def _suppressed_refs(
    items: tuple[SnapshotItem, ...],
    *,
    duplicate_suppression_hours: int,
) -> set[str]:
    refs: set[str] = set()
    since = datetime.now(UTC) - timedelta(hours=max(1, int(duplicate_suppression_hours)))
    for item in items:
        updated_at = _parse_datetime(item.metadata.get("updated_at") or item.metadata.get("created_at"))
        if updated_at is not None and updated_at < since:
            continue
        raw = item.metadata.get("notification_target_refs")
        if isinstance(raw, list):
            refs.update(str(value) for value in raw)
    return refs


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _success_criteria(checks: list[AutonomousCheck], queries: list[AutonomousQuery]) -> tuple[str, ...]:
    criteria = ["no_direct_side_effects"]
    if checks:
        criteria.append("notification_or_approval_proposals_created")
    if queries:
        criteria.append("tool_results_normalized")
    return tuple(criteria)


def _max_risk(values: list[str]) -> str:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    if not values:
        return "low"
    return max((value if value in order else "low" for value in values), key=lambda item: order[item])


def _merge_plans(deterministic: AutonomousPlan, llm_plan: AutonomousPlan) -> AutonomousPlan:
    checks = tuple(_unique_by_id([*deterministic.checks, *llm_plan.checks]))
    queries = tuple(_unique_by_id([*deterministic.required_queries, *llm_plan.required_queries]))
    target_refs = tuple(dict.fromkeys([*deterministic.target_refs, *llm_plan.target_refs]))
    risk = _max_risk([deterministic.risk, llm_plan.risk])
    boundary = "approval_required" if (
        deterministic.side_effect_boundary == "approval_required"
        or llm_plan.side_effect_boundary == "approval_required"
        or risk in {"medium", "high", "critical"}
    ) else "candidate_only"
    return AutonomousPlan(
        checks=checks,
        required_queries=queries,
        target_refs=target_refs,
        success_criteria=tuple(dict.fromkeys([*deterministic.success_criteria, *llm_plan.success_criteria])),
        risk=risk,
        side_effect_boundary=boundary,
        notification_policy={**deterministic.notification_policy, **llm_plan.notification_policy},
        retry_policy={**deterministic.retry_policy, **llm_plan.retry_policy},
        warnings=tuple(dict.fromkeys([*deterministic.warnings, *llm_plan.warnings])),
        metadata={**deterministic.metadata, **llm_plan.metadata, "planner": "llm_with_deterministic_guard"},
    )


def _unique_by_id(items: list[object]) -> list[object]:
    seen: set[str] = set()
    unique: list[object] = []
    for item in items:
        item_id = str(getattr(item, "id", "") or "")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        unique.append(item)
    return unique


def _risk(value: str) -> str:
    normalized = str(value or "low").lower()
    return normalized if normalized in {"low", "medium", "high", "critical"} else "low"


def _boundary(value: str, *, risk: str) -> str:
    normalized = str(value or "").lower()
    if normalized == "approval_required" or risk in {"medium", "high", "critical"}:
        return "approval_required"
    return "candidate_only"


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
