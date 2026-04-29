from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any

from kumc_agent.domain.models.autonomous_agent import (
    ApprovalRequestProposal,
    AutonomousDecision,
    AutonomousPlan,
    AutonomousToolResult,
    NotificationProposal,
)
from kumc_agent.features.autonomous_agent.llm import (
    AutonomousLLMConfig,
    dump_value,
    llm_generate,
    load_json_object,
    read_prompt,
    string_tuple,
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
_FORBIDDEN_SIDE_EFFECTS = {"master_write", "external_post", "server_execute"}
_SENSITIVE_PATTERNS = (
    ("secret_like_payload", re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\s*[:=]")),
    ("internal_ip_payload", re.compile(r"(\[internal-ip\]|\b(?:10|172\.(?:1[6-9]|2\d|3[0-1])|192\.168)\.\d{1,3}\.\d{1,3}\b)")),
    ("invite_url_payload", re.compile(r"(?i)\b(?:https?://)?(?:discord\.gg|discord(?:app)?\.com/invite)/[A-Za-z0-9-]+")),
    ("personal_contact_payload", re.compile(r"(?i)([\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b0\d{1,4}[- ]?\d{1,4}[- ]?\d{3,4}\b)")),
)


@dataclass(frozen=True)
class VerifierConfig:
    notification_channel_id: str = ""
    require_citations_for_candidates: bool = True


class AutonomousVerifier:
    def __init__(
        self,
        *,
        config: VerifierConfig | None = None,
        llm: object | None = None,
        llm_config: AutonomousLLMConfig | None = None,
    ) -> None:
        self.config = config or VerifierConfig()
        self.llm = llm
        self.llm_config = llm_config or AutonomousLLMConfig()

    def verify(
        self,
        *,
        plan: AutonomousPlan,
        tool_results: tuple[AutonomousToolResult, ...],
    ) -> AutonomousDecision:
        deterministic = self._deterministic_verify(plan=plan, tool_results=tool_results)
        llm_decision = self._verify_with_llm(
            plan=plan,
            tool_results=tool_results,
            deterministic=deterministic,
        )
        if llm_decision is None:
            return deterministic
        return _merge_decisions(deterministic, llm_decision)

    def _deterministic_verify(
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
        conflicts.extend(_safety_conflicts(plan=plan, tool_results=tool_results))

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

    def _verify_with_llm(
        self,
        *,
        plan: AutonomousPlan,
        tool_results: tuple[AutonomousToolResult, ...],
        deterministic: AutonomousDecision,
    ) -> AutonomousDecision | None:
        if not self.llm_config.enabled or self.llm is None:
            return None
        system_prompt = read_prompt(
            self.llm_config.prompts_dir,
            self.llm_config.prompt_name,
            fallback="Return autonomous agent verification JSON only.",
        )
        payload = {
            "plan": dump_value(plan),
            "tool_results": [dump_value(result) for result in tool_results],
            "deterministic_decision": dump_value(deterministic),
            "allowed_decisions": ["retry_search", "noop", "notify", "request_approval", "create_candidates"],
            "required_contract": {
                "no_external_posting": True,
                "no_server_execution": True,
                "no_master_write": True,
                "candidate_or_approval_only": True,
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
            decision = str(parsed.get("decision") or "")
            if decision not in {"retry_search", "noop", "notify", "request_approval", "create_candidates"}:
                continue
            return AutonomousDecision(
                decision=decision,
                satisfied=string_tuple(parsed.get("satisfied")),
                missing=string_tuple(parsed.get("missing")),
                conflicts=string_tuple(parsed.get("conflicts")),
                warnings=string_tuple(parsed.get("warnings")),
                notification_proposals=tuple(),
                approval_requests=tuple(),
                candidate_refs=tuple(),
                metadata={
                    **_dict(parsed.get("metadata")),
                    "verifier": "llm",
                    "fallback_guard": "deterministic_merge",
                },
            )
        return None

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
        side_effects = str(result.metadata.get("side_effects") or "").lower()
        if side_effects in _FORBIDDEN_SIDE_EFFECTS:
            return True
        if _int(result.metadata.get("master_write_count")) > 0:
            return True
        if _int(result.metadata.get("external_delivery_count")) > 0:
            return True
        if _int(result.metadata.get("server_execute_count")) > 0:
            return True
        contract = result.metadata.get("side_effect_contract")
        if isinstance(contract, dict) and contract.get("allowed") is False:
            return True
        metadata_text = " ".join(str(value).lower() for value in result.metadata.values())
        if any(marker in metadata_text for marker in _SIDE_EFFECT_MARKERS):
            if side_effects not in {"none", "candidate_or_approval_only"}:
                return True
    return False


def _safety_conflicts(
    *,
    plan: AutonomousPlan,
    tool_results: tuple[AutonomousToolResult, ...],
) -> list[str]:
    text = json.dumps(
        {
            "plan": asdict(plan),
            "tool_results": [asdict(result) for result in tool_results],
        },
        ensure_ascii=False,
        default=str,
    )
    conflicts: list[str] = []
    for label, pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            conflicts.append(label)
    return conflicts


def _can_retry(plan: AutonomousPlan) -> bool:
    try:
        return int(plan.retry_policy.get("max_replans", 0)) > 0
    except Exception:
        return False


def _merge_decisions(deterministic: AutonomousDecision, llm_decision: AutonomousDecision) -> AutonomousDecision:
    conflicts = tuple(dict.fromkeys([*deterministic.conflicts, *llm_decision.conflicts]))
    missing = tuple(dict.fromkeys([*deterministic.missing, *llm_decision.missing]))
    warnings = tuple(dict.fromkeys([*deterministic.warnings, *llm_decision.warnings]))
    decision = llm_decision.decision
    if conflicts:
        decision = "noop"
    elif deterministic.approval_requests and decision not in {"retry_search", "noop"}:
        decision = "request_approval"
    elif deterministic.candidate_refs and decision not in {"retry_search", "noop", "request_approval"}:
        decision = "create_candidates"
    elif deterministic.notification_proposals and decision == "noop":
        decision = deterministic.decision
    return AutonomousDecision(
        decision=decision,
        satisfied=tuple(dict.fromkeys([*deterministic.satisfied, *llm_decision.satisfied])),
        missing=missing,
        conflicts=conflicts,
        notification_proposals=deterministic.notification_proposals,
        approval_requests=deterministic.approval_requests,
        candidate_refs=deterministic.candidate_refs,
        warnings=warnings,
        metadata={**deterministic.metadata, **llm_decision.metadata, "verifier": "llm_with_deterministic_guard"},
    )


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
