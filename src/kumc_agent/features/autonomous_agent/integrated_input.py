from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any
from uuid import uuid4

from kumc_agent.domain.models.autonomous_agent import AutonomousQuery, AutonomousToolResult
from kumc_agent.domain.models.integrated_input import IntegratedInputRequest
from kumc_agent.domain.models.retrieval import AccessContext, RetrievalQuery
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
        automation_service: object | None = None,
        retrieval_service: object | None = None,
        server_operation_repository: object | None = None,
    ) -> None:
        self.integrated_input = integrated_input
        self.workflow_service = workflow_service
        self.automation_service = automation_service
        self.retrieval_service = retrieval_service
        self.server_operation_repository = server_operation_repository

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
                    "side_effect_contract": _side_effect_contract("none"),
                    "master_write_count": 0,
                    "external_delivery_count": 0,
                    "server_execute_count": 0,
                },
            )
        if query.work_type == "automation_dry_run" or query.metadata.get("adapter") == "automation":
            return self._run_automation(query, access=access)
        if query.work_type == "retrieval_summary" or query.metadata.get("adapter") == "retrieval":
            return self._run_retrieval(query, access=access)
        if query.work_type == "server_pending" or query.metadata.get("adapter") == "server":
            return self._run_server_pending(query)
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
            metadata={
                "side_effects": "none",
                "side_effect_contract": _side_effect_contract("none"),
                "master_write_count": 0,
                "external_delivery_count": 0,
                "server_execute_count": 0,
            },
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
                metadata={
                    "side_effects": "none",
                    "side_effect_contract": _side_effect_contract("none"),
                    "master_write_count": 0,
                    "external_delivery_count": 0,
                    "server_execute_count": 0,
                },
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

    def _run_automation(self, query: AutonomousQuery, *, access: AccessContext) -> AutonomousToolResult:
        if self.automation_service is None:
            return _blocked_result("automation_query", query, "automation_service_unconfigured")
        rule_id = str(query.metadata.get("rule_id") or "").strip()
        if not rule_id:
            for ref in query.target_refs:
                kind, _, value = ref.partition(":")
                if kind in {"automation_rule", "automation"} and value:
                    rule_id = value
                    break
        if not rule_id:
            return _blocked_result("automation_query", query, "automation_rule_id_missing")
        try:
            response = self.automation_service.dry_run(  # type: ignore[attr-defined]
                rule_id=rule_id,
                trigger_key=str(query.metadata.get("trigger_key") or query.id),
                access=access,
            )
        except Exception as exc:
            return _failed_result("automation_query", query, exc, work_type="automation_dry_run")
        return _result_from_response("automation_query", query, response, work_type="automation_dry_run")

    def _run_retrieval(self, query: AutonomousQuery, *, access: AccessContext) -> AutonomousToolResult:
        if self.retrieval_service is None:
            return _blocked_result("retrieval_query", query, "retrieval_service_unconfigured")
        try:
            response = self.retrieval_service.ask(  # type: ignore[attr-defined]
                RetrievalQuery(
                    text=query.query,
                    source_filter=query.source or "all",
                    mode="answer",
                    depth=query.depth,
                    access=access,
                )
            )
        except Exception as exc:
            return _failed_result("retrieval_query", query, exc, work_type="retrieval_summary")
        return _result_from_response("retrieval_query", query, response, work_type="retrieval_summary")

    def _run_server_pending(self, query: AutonomousQuery) -> AutonomousToolResult:
        if self.server_operation_repository is None:
            return _blocked_result("server_pending_query", query, "server_operation_repository_unconfigured")
        try:
            operations = self.server_operation_repository.list_pending_for_approval()  # type: ignore[attr-defined]
        except Exception as exc:
            return _failed_result("server_pending_query", query, exc, work_type="server_pending")
        server_operations = [
            {
                "id": str(getattr(operation, "id", "")),
                "server_name": str(getattr(operation, "server_name", "")),
                "operation": str(getattr(operation, "operation", "")),
                "status": str(getattr(operation, "status", "")),
                "risk_level": str(getattr(operation, "risk_level", "")),
            }
            for operation in operations
        ]
        return AutonomousToolResult(
            id=str(uuid4()),
            tool_name="server_pending_query",
            status="succeeded",
            query_id=query.id,
            target_refs=query.target_refs,
            metadata={
                "work_type": "server_pending",
                "server_operations": server_operations,
                "server_operation_ids": [item["id"] for item in server_operations if item.get("id")],
                "side_effects": "none",
                "side_effect_contract": _side_effect_contract("none"),
                "master_write_count": 0,
                "external_delivery_count": 0,
                "server_execute_count": 0,
                "result_counts": {"server_operations": len(server_operations)},
            },
        )


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
    task_candidate_ids = _ids(payload, "task_candidates", "task_change_candidates")
    event_candidate_ids = _ids(payload, "event_candidates", "event_change_candidates")
    automation_run_ids = _ids(payload, "runs")
    server_operation_ids = _ids(payload, "server_operations")
    citations = tuple(getattr(response, "citations", tuple()) or tuple())
    warnings = tuple(str(value) for value in getattr(response, "warnings", tuple()) or tuple())
    metadata = sanitize_autonomous_metadata(getattr(response, "metadata", {}) or {})
    read_work_types = {"task_list", "event_list", "event_brief", "schedule_list", "mc_status", "retrieval_summary"}
    master_write_count = 0 if work_type in read_work_types else len(_ids(payload, "tasks", "events", "schedules"))
    external_delivery_count = _external_delivery_count(metadata)
    server_execute_count = _server_execute_count(metadata)
    side_effects = _side_effects_value(
        candidate_ids=candidate_ids,
        approval_ids=approval_ids,
        master_write_count=master_write_count,
        external_delivery_count=external_delivery_count,
        server_execute_count=server_execute_count,
    )
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
                "automation_runs": len(automation_run_ids),
                "server_operations": len(server_operation_ids),
            },
            "task_candidate_ids": list(task_candidate_ids),
            "event_candidate_ids": list(event_candidate_ids),
            "automation_run_ids": list(automation_run_ids),
            "server_operation_ids": list(server_operation_ids),
            "task_candidates": _summaries(payload, "task_candidates", "task_change_candidates"),
            "event_candidates": _summaries(payload, "event_candidates", "event_change_candidates"),
            "automation_runs": _summaries(payload, "runs"),
            "server_operations": _summaries(payload, "server_operations"),
            "response_metadata": metadata,
            "side_effects": side_effects,
            "side_effect_contract": _side_effect_contract(side_effects),
            "master_write_count": master_write_count,
            "external_delivery_count": external_delivery_count,
            "server_execute_count": server_execute_count,
        },
    )


def _blocked_result(tool_name: str, query: AutonomousQuery, reason: str) -> AutonomousToolResult:
    return AutonomousToolResult(
        id=str(uuid4()),
        tool_name=tool_name,
        status="blocked",
        query_id=query.id,
        target_refs=query.target_refs,
        warnings=(reason,),
        metadata={
            "side_effects": "none",
            "side_effect_contract": _side_effect_contract("none"),
            "master_write_count": 0,
            "external_delivery_count": 0,
            "server_execute_count": 0,
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
            "side_effect_contract": _side_effect_contract("none"),
            "master_write_count": 0,
            "external_delivery_count": 0,
            "server_execute_count": 0,
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


def _summaries(payload: dict[str, Any], *keys: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for key in keys:
        value = payload.get(key) or []
        for item in value:
            raw = item if isinstance(item, dict) else getattr(item, "__dict__", {})
            if not isinstance(raw, dict):
                continue
            item_id = str(raw.get("id") or "").strip()
            if not item_id:
                continue
            items.append(
                sanitize_autonomous_metadata(
                    {
                        "id": item_id,
                        "title": raw.get("title") or raw.get("name") or "",
                        "status": raw.get("status") or "",
                        "kind": key,
                    }
                )
            )
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for item in items:
        item_id = str(item.get("id") or "")
        if item_id in seen:
            continue
        seen.add(item_id)
        unique.append(item)
    return unique


def _side_effect_contract(side_effects: str) -> dict[str, object]:
    return {
        "side_effects": side_effects,
        "allowed": side_effects in {"none", "candidate_or_approval_only"},
        "external_post": side_effects == "external_post",
        "server_execute": side_effects == "server_execute",
        "master_write": side_effects == "master_write",
    }


def _side_effects_value(
    *,
    candidate_ids: tuple[str, ...],
    approval_ids: tuple[str, ...],
    master_write_count: int,
    external_delivery_count: int,
    server_execute_count: int,
) -> str:
    if server_execute_count > 0:
        return "server_execute"
    if external_delivery_count > 0:
        return "external_post"
    if master_write_count > 0:
        return "master_write"
    if candidate_ids or approval_ids:
        return "candidate_or_approval_only"
    return "none"


def _external_delivery_count(metadata: dict[str, object]) -> int:
    side_effects = str(metadata.get("side_effects") or metadata.get("external_side_effects") or "").lower()
    if side_effects in {"external_post", "sent", "notification_state_recorded"}:
        return 1
    return _int(metadata.get("external_delivery_count"))


def _server_execute_count(metadata: dict[str, object]) -> int:
    side_effects = str(metadata.get("side_effects") or metadata.get("external_side_effects") or "").lower()
    if side_effects in {"server_execute", "server_operation_executed"}:
        return 1
    return _int(metadata.get("server_execute_count"))


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
