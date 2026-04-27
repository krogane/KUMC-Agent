from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.domain.models.autonomous_agent import (
    ApprovalRequestProposal,
    AutonomousDecision,
    AutonomousPlan,
    AutonomousToolResult,
    NotificationProposal,
)
from kumc_agent.features.autonomous_agent.sanitizer import safe_text
from kumc_agent.utils.hashing import stable_hash

_SIDE_EFFECT_MARKERS = {
    "external_posted",
    "server_operation_executed",
    "master_task_updated",
    "master_event_updated",
    "executed",
    "sent",
}


@dataclass(frozen=True)
class VerifierConfig:
    notification_channel_id: str = ""
    require_citations_for_candidates: bool = True


class AutonomousVerifier:
    def __init__(self, *, config: VerifierConfig | None = None) -> None:
        self.config = config or VerifierConfig()

    def verify(
        self,
        *,
        plan: AutonomousPlan,
        tool_results: tuple[AutonomousToolResult, ...],
    ) -> AutonomousDecision:
        warnings: list[str] = list(plan.warnings)
        conflicts: list[str] = []
        missing: list[str] = []
        satisfied: list[str] = []

        if _has_forbidden_side_effect(tool_results):
            conflicts.append("forbidden_side_effect_detected")
        else:
            satisfied.append("no_direct_side_effects")

        if plan.required_queries and not tool_results:
            missing.append("tool_results")
        elif plan.required_queries:
            satisfied.append("tool_results_normalized")

        warnings.extend(warning for result in tool_results for warning in result.warnings)

        candidate_refs = tuple(
            dict.fromkeys(
                candidate_id
                for result in tool_results
                for candidate_id in result.candidate_ids
            )
        )
        approval_ids = tuple(
            dict.fromkeys(
                approval_id
                for result in tool_results
                for approval_id in result.approval_ids
            )
        )
        if (
            candidate_refs
            and self.config.require_citations_for_candidates
            and not any(result.citations for result in tool_results)
        ):
            warnings.append("candidate_citations_missing")

        notifications = self._notification_proposals(plan)
        approvals = self._approval_requests(plan, approval_ids=approval_ids)
        if notifications or approvals or candidate_refs:
            satisfied.append("notification_or_approval_proposals_created")

        if notifications and not self.config.notification_channel_id:
            warnings.append("notification_channel_unconfigured")

        decision = "noop"
        if conflicts:
            decision = "noop"
        elif missing and _can_retry(plan):
            decision = "retry_search"
        elif approvals:
            decision = "request_approval"
        elif candidate_refs:
            decision = "create_candidates"
        elif notifications:
            decision = "notify"

        return AutonomousDecision(
            decision=decision,
            satisfied=tuple(dict.fromkeys(satisfied)),
            missing=tuple(dict.fromkeys(missing)),
            conflicts=tuple(dict.fromkeys(conflicts)),
            notification_proposals=tuple(notifications),
            approval_requests=tuple(approvals),
            candidate_refs=candidate_refs,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                "verifier": "deterministic_v1",
                "tool_result_count": len(tool_results),
                "approval_ids": list(approval_ids),
            },
        )

    def _notification_proposals(self, plan: AutonomousPlan) -> list[NotificationProposal]:
        proposals: list[NotificationProposal] = []
        for check in plan.checks:
            if check.kind in {
                "task_due_soon",
                "task_overdue",
                "task_stale",
                "event_missing_details",
                "automation_run_attention",
            }:
                body = safe_text(f"{check.reason}\n対象: {check.target_ref}", limit=800)
                proposals.append(
                    NotificationProposal(
                        id=stable_hash(f"notification:{check.id}:{body}")[:24],
                        target_channel_id=self.config.notification_channel_id,
                        body=body,
                        target_refs=(check.target_ref,),
                        risk=check.risk,
                        metadata={"check_id": check.id},
                    )
                )
        return proposals

    def _approval_requests(
        self,
        plan: AutonomousPlan,
        *,
        approval_ids: tuple[str, ...],
    ) -> list[ApprovalRequestProposal]:
        proposals: list[ApprovalRequestProposal] = []
        for check in plan.checks:
            if check.side_effect_boundary == "approval_required" or check.kind == "server_operation_approval":
                target_type, _, target_id = check.target_ref.partition(":")
                proposals.append(
                    ApprovalRequestProposal(
                        id=stable_hash(f"approval-request:{check.id}:{check.target_ref}")[:24],
                        target_type=target_type or "unknown",
                        target_id=target_id or check.target_ref,
                        reason=safe_text(check.reason, limit=600),
                        risk=check.risk,
                        metadata={
                            "check_id": check.id,
                            "existing_approval_ids": list(approval_ids),
                        },
                    )
                )
        return proposals


def _has_forbidden_side_effect(results: tuple[AutonomousToolResult, ...]) -> bool:
    for result in results:
        metadata_text = " ".join(str(value).lower() for value in result.metadata.values())
        if any(marker in metadata_text for marker in _SIDE_EFFECT_MARKERS):
            side_effects = str(result.metadata.get("side_effects") or "").lower()
            if side_effects not in {"none", "candidate_or_approval_only"}:
                return True
    return False


def _can_retry(plan: AutonomousPlan) -> bool:
    try:
        return int(plan.retry_policy.get("max_replans", 0)) > 0
    except Exception:
        return False
