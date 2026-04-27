from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any
from uuid import uuid4

from kumc_agent.domain.models.autonomous_agent import AutonomousQuery, AutonomousToolResult
from kumc_agent.domain.models.integrated_input import IntegratedInputRequest
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.workflow import WorkRequest
from kumc_agent.features.autonomous_agent.sanitizer import sanitize_autonomous_metadata

_SAFE_WORK_TYPES = {
    "task_list",
    "event_list",
    "event_brief",
    "schedule_list",
    "mc_status",
    "task_add",
    "task_extract",
    "task_update",
    "task_delete",
    "task_batch_approval",
    "event_add",
    "event_extract",
    "event_update",
    "event_delete",
    "event_batch_approval",
    "schedule_add",
    "mc_request",
}


class AutonomousIntegratedInputAdapter:
    def __init__(
        self,
        *,
        integrated_input: object | None = None,
        workflow_service: object | None = None,
    ) -> None:
        self.integrated_input = integrated_input
        self.workflow_service = workflow_service

    def run_query(
        self,
        query: AutonomousQuery,
        *,
        access: AccessContext,
        dry_run: bool,
    ) -> AutonomousToolResult:
        if dry_run and query.work_type not in {"task_list", "event_list", "event_brief", "schedule_list", "mc_status"}:
            return AutonomousToolResult(
                id=str(uuid4()),
                tool_name="dry_run_adapter",
                status="skipped",
                query_id=query.id,
                target_refs=query.target_refs,
                warnings=("dry_run_skipped_candidate_creation",),
                metadata={
                    "planned_work_type": query.work_type,
                    "query": query.query,
                    "side_effects": "none",
                },
            )
        if self.integrated_input is not None:
            return self._run_integrated(query, access=access)
        if self.workflow_service is not None:
            return self._run_workflow(query, access=access)
        return AutonomousToolResult(
            id=str(uuid4()),
            tool_name="unconfigured_adapter",
            status="failed",
            query_id=query.id,
            target_refs=query.target_refs,
            warnings=("integrated_input_unconfigured",),
            metadata={"side_effects": "none"},
        )

    def _run_integrated(self, query: AutonomousQuery, *, access: AccessContext) -> AutonomousToolResult:
        try:
            response = self.integrated_input.execute(  # type: ignore[attr-defined]
                IntegratedInputRequest(
                    text=query.query,
                    source=query.source,
                    mode=query.mode,
                    depth=query.depth,
                    access=access,
                    frontend="autonomous_agent",
                    metadata={
                        "autonomous_query_id": query.id,
                        "side_effect_boundary": "candidate_only",
                    },
                )
            )
        except Exception as exc:
            return _failed_result("integrated_input", query, exc)
        return _result_from_response("integrated_input", query, response)

    def _run_workflow(self, query: AutonomousQuery, *, access: AccessContext) -> AutonomousToolResult:
        work_type = (query.work_type or "").strip()
        if work_type not in _SAFE_WORK_TYPES:
            return AutonomousToolResult(
                id=str(uuid4()),
                tool_name="workflow_query",
                status="blocked",
                query_id=query.id,
                target_refs=query.target_refs,
                warnings=(f"unsafe_work_type_blocked:{work_type}",),
                metadata={"side_effects": "none"},
            )
        try:
            response = self.workflow_service.run(  # type: ignore[attr-defined]
                WorkRequest(
                    work_type=work_type,
                    instruction=query.query,
                    access=access,
                )
            )
        except Exception as exc:
            return _failed_result("workflow_query", query, exc, work_type=work_type)
        return _result_from_response("workflow_query", query, response, work_type=work_type)


def _result_from_response(
    tool_name: str,
    query: AutonomousQuery,
    response: object,
    *,
    work_type: str = "",
) -> AutonomousToolResult:
    payload = _payload(response)
    candidate_ids = _ids(
        payload,
        "task_candidates",
        "task_change_candidates",
        "event_candidates",
        "event_change_candidates",
        "schedule_candidates",
    )
    approval_ids = _ids(payload, "task_approval_batches", "event_approval_batches", "approvals")
    citations = tuple(getattr(response, "citations", tuple()) or tuple())
    warnings = tuple(str(value) for value in getattr(response, "warnings", tuple()) or tuple())
    return AutonomousToolResult(
        id=str(uuid4()),
        tool_name=tool_name,
        status="succeeded",
        query_id=query.id,
        target_refs=query.target_refs,
        candidate_ids=candidate_ids,
        approval_ids=approval_ids,
        citations=citations,
        warnings=warnings,
        metadata={
            "work_type": work_type,
            "result_counts": {
                "candidate_ids": len(candidate_ids),
                "approval_ids": len(approval_ids),
                "citations": len(citations),
            },
            "response_metadata": sanitize_autonomous_metadata(
                getattr(response, "metadata", {}) or {}
            ),
            "side_effects": "candidate_or_approval_only",
        },
    )


def _failed_result(
    tool_name: str,
    query: AutonomousQuery,
    exc: Exception,
    *,
    work_type: str = "",
) -> AutonomousToolResult:
    return AutonomousToolResult(
        id=str(uuid4()),
        tool_name=tool_name,
        status="failed",
        query_id=query.id,
        target_refs=query.target_refs,
        warnings=(f"{tool_name}_failed:{type(exc).__name__}",),
        metadata={
            "work_type": work_type,
            "error": str(exc),
            "side_effects": "none",
        },
    )


def _payload(response: object) -> dict[str, Any]:
    if hasattr(response, "to_payload"):
        raw = response.to_payload()  # type: ignore[attr-defined]
        return raw if isinstance(raw, dict) else {}
    if is_dataclass(response):
        return asdict(response)
    raw = getattr(response, "__dict__", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _ids(payload: dict[str, Any], *keys: str) -> tuple[str, ...]:
    ids: list[str] = []
    for key in keys:
        value = payload.get(key) or []
        for item in value:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
            elif hasattr(item, "id"):
                ids.append(str(item.id))
    return tuple(dict.fromkeys(ids))
