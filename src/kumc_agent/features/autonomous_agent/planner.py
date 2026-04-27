from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.domain.models.autonomous_agent import (
    AutonomousCheck,
    AutonomousPlan,
    AutonomousQuery,
    SnapshotItem,
)
from kumc_agent.utils.hashing import stable_hash


@dataclass(frozen=True)
class PlannerConfig:
    notification_channel_id: str = ""
    max_replans: int = 1


class AutonomousPlanner:
    def __init__(self, *, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def plan(self, snapshot) -> AutonomousPlan:
        checks: list[AutonomousCheck] = []
        queries: list[AutonomousQuery] = []
        target_refs: list[str] = []
        warnings: list[str] = list(snapshot.warnings)
        suppressed = _suppressed_refs(snapshot.recent_runs)

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


def _suppressed_refs(items: tuple[SnapshotItem, ...]) -> set[str]:
    refs: set[str] = set()
    for item in items:
        raw = item.metadata.get("notification_target_refs")
        if isinstance(raw, list):
            refs.update(str(value) for value in raw)
    return refs


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
