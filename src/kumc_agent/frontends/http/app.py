from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass

from kumc_agent.domain.models.integrated_input import IntegratedInputRequest
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.source import BackfillScope
from kumc_agent.domain.models.workflow import WorkRequest
from kumc_agent.features.foundation.payload_sanitizer import (
    compact_payload_text,
    mask_payload_secret,
    sanitize_payload,
    sanitize_payload_metadata,
)


def _source_filter(payload: dict[str, object]) -> tuple[str, ...]:
    value = payload.get("source_filter") or payload.get("source_filters") or payload.get("source") or ()
    if isinstance(value, str):
        return (value,) if value.strip() else tuple()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item).strip())
    return tuple()


def _access(payload: dict[str, object] | None = None) -> AccessContext:
    payload = payload or {}
    role_ids = payload.get("role_ids") or payload.get("roles") or []
    if isinstance(role_ids, str):
        role_ids = [role_ids]
    return AccessContext(
        user_id=str(payload.get("user_id") or ""),
        guild_id=str(payload.get("guild_id") or ""),
        role_ids=tuple(str(role) for role in role_ids),
        is_admin=bool(payload.get("admin") or payload.get("is_admin")),
    )


def _dump_items(items: object) -> list[dict[str, object]]:
    return [
        _dump_workflow_item(item)
        for item in items or []
    ]


def _dump_workflow_item(item: object) -> dict[str, object]:
    raw = asdict(item) if is_dataclass(item) else getattr(item, "__dict__", {})
    payload = {
        key: value.isoformat() if hasattr(value, "isoformat") else value
        for key, value in raw.items()
    }
    payload = _sanitize_payload_value(payload)
    if "metadata" in payload:
        payload["metadata"] = _sanitize_payload_metadata(payload.get("metadata"))
    if "media_type" not in payload or "metadata" not in payload:
        return payload
    metadata = dict(payload.get("metadata") or {})
    for key in ("downloaded_image_path", "original_image_ref"):
        metadata.pop(key, None)
    for key, limit in (("ocr_text", 800), ("surrounding_text", 1200)):
        value = metadata.get(key)
        if isinstance(value, str):
            metadata[key] = _compact_payload_text(_mask_payload_secret(value), limit)
    payload["metadata"] = metadata
    return payload


def _sanitize_payload_metadata(value: object) -> dict[str, object]:
    return sanitize_payload_metadata(value)


def _sanitize_payload_value(value: object) -> object:
    return sanitize_payload(value)


def _compact_payload_text(text: str, limit: int) -> str:
    return compact_payload_text(text, limit)


def _mask_payload_secret(text: str) -> str:
    return mask_payload_secret(text)


def _workflow_payload(response: object) -> dict[str, object]:
    return {
        "text": getattr(response, "text", ""),
        "detail_markdown": getattr(response, "detail_markdown", ""),
        "task_candidates": _dump_items(getattr(response, "task_candidates", ())),
        "task_change_candidates": _dump_items(getattr(response, "task_change_candidates", ())),
        "task_approval_batches": _dump_items(getattr(response, "task_approval_batches", ())),
        "event_candidates": _dump_items(getattr(response, "event_candidates", ())),
        "event_change_candidates": _dump_items(getattr(response, "event_change_candidates", ())),
        "event_approval_batches": _dump_items(getattr(response, "event_approval_batches", ())),
        "schedule_candidates": _dump_items(getattr(response, "schedule_candidates", ())),
        "workflow_candidates": _dump_items(getattr(response, "workflow_candidates", ())),
        "assets": _dump_items(getattr(response, "assets", ())),
        "member_profiles": _dump_items(getattr(response, "member_profiles", ())),
        "tasks": _dump_items(getattr(response, "tasks", ())),
        "events": _dump_items(getattr(response, "events", ())),
        "schedules": _dump_items(getattr(response, "schedules", ())),
        "meetings": _dump_items(getattr(response, "meetings", ())),
        "approvals": _dump_items(getattr(response, "approvals", ())),
        "server_operations": _dump_items(getattr(response, "server_operations", ())),
        "warnings": list(getattr(response, "warnings", ())),
        "metadata": _sanitize_payload_metadata(getattr(response, "metadata", {}) or {}),
    }


def _comprehensive_payload(response: object) -> dict[str, object]:
    return {
        "text": getattr(response, "text", ""),
        "detail_markdown": getattr(response, "detail_markdown", ""),
        "citations": _sanitize_payload_value(
            [getattr(citation, "__dict__", {}) for citation in getattr(response, "citations", ())]
        ),
        "confidence": getattr(response, "confidence", "low"),
        "task_candidates": _sanitize_payload_value(getattr(response, "task_candidates", ())),
        "event_candidates": _sanitize_payload_value(getattr(response, "event_candidates", ())),
        "server_operations": _sanitize_payload_value(getattr(response, "server_operations", ())),
        "assets": _sanitize_payload_value(getattr(response, "assets", ())),
        "member_profiles": _sanitize_payload_value(getattr(response, "member_profiles", ())),
        "warnings": list(getattr(response, "warnings", ())),
        "metadata": _sanitize_payload_metadata(getattr(response, "metadata", {}) or {}),
    }


def _automation_payload(response: object) -> dict[str, object]:
    return {
        "text": getattr(response, "text", ""),
        "detail_markdown": getattr(response, "detail_markdown", ""),
        "rules": _dump_items(getattr(response, "rules", ())),
        "runs": _dump_items(getattr(response, "runs", ())),
        "warnings": list(getattr(response, "warnings", ())),
        "metadata": dict(getattr(response, "metadata", {}) or {}),
    }


def _foundation_context(context: object) -> object:
    return getattr(context, "foundation", context)


def create_app(context: object):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - depends on deployment env
        raise RuntimeError("fastapi is required to run the API app.") from exc

    app = FastAPI(title="KUMC-Agent API", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        return _foundation_context(context).health.check(actor_id="api", actor_type="service").as_dict()

    @app.post("/admin/action/health")
    def admin_health() -> dict[str, object]:
        report = _foundation_context(context).health.check(actor_id="api-admin", actor_type="service")
        if report.status == "unhealthy":
            raise HTTPException(status_code=503, detail=report.as_dict())
        return report.as_dict()

    @app.post("/ask")
    def ask(payload: dict[str, object]) -> dict[str, object]:
        question = str(payload.get("question") or payload.get("query") or "")
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        access = _access(payload)
        response = context.integrated_input.integrated_input.execute(
            IntegratedInputRequest(
                text=question,
                source=str(payload.get("source") or "all"),
                mode=str(payload.get("mode") or "answer"),
                depth=str(payload.get("depth") or "normal"),
                user_id=access.user_id,
                guild_id=access.guild_id,
                role_ids=access.role_ids,
                is_admin=access.is_admin,
                access=access,
                frontend="http",
                metadata={
                    "route_hint": str(payload.get("route") or ""),
                    "required_features_hint": payload.get("required_features") or (),
                },
            )
        )
        return response.to_payload()

    @app.post("/work")
    def work(payload: dict[str, object]) -> dict[str, object]:
        response = context.workflow.workflow.run(
            WorkRequest(
                work_type=str(payload.get("type") or payload.get("work_type") or ""),
                instruction=str(payload.get("instruction") or ""),
                target=str(payload.get("target") or ""),
                output_format=str(payload.get("format") or payload.get("output_format") or "markdown"),
                source_filter=_source_filter(payload),
                limit=int(payload["limit"]) if payload.get("limit") not in (None, "") else None,
                access=_access(payload),
            )
        )
        return _workflow_payload(response)

    @app.get("/agent/runs")
    def agent_runs(limit: int = 20) -> dict[str, object]:
        runs = context.agentic.trace_repository.latest_runs(limit=limit)
        return {
            "runs": [
                {
                    "id": run.id,
                    "query": run.query,
                    "status": run.status,
                    "confidence": run.confidence,
                    "metadata": _sanitize_payload_metadata(run.metadata),
                    "created_at": run.created_at,
                    "updated_at": run.updated_at,
                }
                for run in runs
            ],
            "metadata": {"limit": limit},
        }

    @app.get("/agent/runs/{run_id}")
    def agent_run(run_id: str) -> dict[str, object]:
        run = context.agentic.trace_repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="agent run not found")
        steps = context.agentic.trace_repository.list_steps(run_id)
        return {
            "run": {
                "id": run.id,
                "query": run.query,
                "status": run.status,
                "answer": run.answer,
                "confidence": run.confidence,
                "citations": _sanitize_payload_value([citation.__dict__ for citation in run.citations]),
                "metadata": _sanitize_payload_metadata(run.metadata),
                "created_at": run.created_at,
                "updated_at": run.updated_at,
            },
            "steps": [
                {
                    "id": step.id,
                    "state": step.state,
                    "status": step.status,
                    "input": _sanitize_payload_value(step.input),
                    "output": _sanitize_payload_value(step.output),
                    "cost_usd": step.cost_usd,
                    "created_at": step.created_at,
                }
                for step in steps
            ],
            "metadata": {},
        }

    @app.post("/approval")
    def approval(payload: dict[str, object]) -> dict[str, object]:
        try:
            response = context.workflow.workflow.approval(
                action=str(payload.get("action") or ""),
                target_type=str(payload.get("type") or payload.get("target_type") or "task"),
                target_id=str(payload.get("target_id") or ""),
                comment=str(payload.get("comment") or ""),
                access=_access(payload),
            )
            return _workflow_payload(response)
        except KeyError:
            return {
                "text": "対象が見つからないか、表示権限がありません。",
                "metadata": {"error": "not_found"},
            }
        except ValueError as exc:
            return {
                "text": str(exc),
                "metadata": {"error": "bad_request"},
            }

    @app.post("/automation")
    def automation(payload: dict[str, object]) -> dict[str, object]:
        access = _access(payload)
        action = str(payload.get("action") or "")
        rule_id = str(payload.get("rule_id") or "")
        if action == "list":
            response = context.automation.automation.list_rules()
        elif action == "show":
            response = context.automation.automation.show(rule_id=rule_id)
        elif action == "enable":
            response = context.automation.automation.enable(rule_id=rule_id, access=access)
        elif action == "disable":
            response = context.automation.automation.disable(rule_id=rule_id, access=access)
        elif action == "set_mode":
            response = context.automation.automation.set_mode(
                rule_id=rule_id,
                mode=str(payload.get("mode") or "dry_run"),
                access=access,
            )
        elif action == "dry_run":
            response = context.automation.automation.dry_run(
                rule_id=rule_id,
                trigger_key=str(payload.get("trigger_key") or "manual"),
                idempotency_key=str(payload.get("idempotency_key") or ""),
                access=access,
            )
        elif action == "run":
            response = context.automation.automation.run(
                rule_id=rule_id,
                trigger_key=str(payload.get("trigger_key") or "manual"),
                idempotency_key=str(payload.get("idempotency_key") or ""),
                access=access,
            )
        else:
            raise HTTPException(status_code=400, detail="unsupported automation action")
        return _automation_payload(response)

    @app.post("/admin/action/{action}")
    def admin_action(action: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        payload = payload or {}
        if action == "health":
            return context.foundation.health.check(actor_id="api-admin", actor_type="service").as_dict()
        if action == "readiness":
            return context.automation.readiness.report().as_dict()
        if action in {"sync", "reindex"}:
            source = str(payload.get("scope") or "").strip()
            if source == "member_profiles":
                guild_ids = [
                    str(value)
                    for value in context.foundation.config.security.effective_member_profile_guild_ids()
                ]
                results = [
                    context.workflow.member_profile_builder.rebuild_guild(guild_id=guild_id).__dict__
                    for guild_id in guild_ids
                    if context.workflow.member_profile_builder is not None
                ]
                return {
                    "action": action,
                    "source_kind": "member_profiles",
                    "results": results,
                    "metadata": {"guild_ids": guild_ids},
                }
            results = asyncio.run(
                context.ingestion.service.backfill_many(
                    source_kinds=(source,) if source else tuple(),
                    scope=BackfillScope(force=action == "reindex"),
                )
            )
            return {"action": action, "results": [result.__dict__ for result in results], "metadata": {}}
        if action == "feature_flags":
            return {"feature_flags": context.foundation.feature_flags.modes(), "metadata": {}}
        if action == "permissions":
            security = context.foundation.config.security
            return {
                "maintenance_command_author_ids": security.maintenance_command_author_ids,
                "discord_guild_allow_list": security.discord_guild_allow_list,
                "discord_member_profile_guild_ids": security.discord_member_profile_guild_ids,
                "effective_member_profile_guild_ids": security.effective_member_profile_guild_ids(),
                "metadata": {},
            }
        if action == "cost_report":
            return context.automation.readiness.cost_report()
        if action == "eval":
            return {"action": "eval", "mode": "local_harness", "metadata": {}}
        raise HTTPException(status_code=400, detail="unsupported admin action")

    return app


__all__ = ["create_app"]
