from __future__ import annotations

import asyncio

from kumc_agent.domain.models.agentic import AgenticSearchRequest
from kumc_agent.domain.models.retrieval import AccessContext, RetrievalQuery
from kumc_agent.domain.models.source import BackfillScope
from kumc_agent.domain.models.workflow import WorkRequest


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
        {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in getattr(item, "__dict__", {}).items()
        }
        for item in items or []
    ]


def _workflow_payload(response: object) -> dict[str, object]:
    return {
        "text": getattr(response, "text", ""),
        "detail_markdown": getattr(response, "detail_markdown", ""),
        "task_candidates": _dump_items(getattr(response, "task_candidates", ())),
        "event_candidates": _dump_items(getattr(response, "event_candidates", ())),
        "schedule_candidates": _dump_items(getattr(response, "schedule_candidates", ())),
        "workflow_candidates": _dump_items(getattr(response, "workflow_candidates", ())),
        "assets": _dump_items(getattr(response, "assets", ())),
        "asset_usage_requests": _dump_items(getattr(response, "asset_usage_requests", ())),
        "member_profiles": _dump_items(getattr(response, "member_profiles", ())),
        "tasks": _dump_items(getattr(response, "tasks", ())),
        "events": _dump_items(getattr(response, "events", ())),
        "schedules": _dump_items(getattr(response, "schedules", ())),
        "meetings": _dump_items(getattr(response, "meetings", ())),
        "approvals": _dump_items(getattr(response, "approvals", ())),
        "server_operations": _dump_items(getattr(response, "server_operations", ())),
        "warnings": list(getattr(response, "warnings", ())),
        "metadata": dict(getattr(response, "metadata", {}) or {}),
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
        source = str(payload.get("source") or "all")
        depth = str(payload.get("depth") or "normal")
        if depth == "deep":
            response = context.agentic.agentic_search.search(
                AgenticSearchRequest(query=question, source_filter=source, access=access)
            )
        else:
            response = context.retrieval.ask.ask(
                RetrievalQuery(
                    text=question,
                    source_filter=source,
                    mode=str(payload.get("mode") or "answer"),
                    depth=depth,
                    access=access,
                )
            )
        return {
            "text": response.text,
            "detail_markdown": response.detail_markdown,
            "citations": _dump_items(getattr(response, "citations", ())),
            "confidence": getattr(response, "confidence", "low"),
            "warnings": list(getattr(response, "warnings", ())),
            "metadata": dict(getattr(response, "metadata", {}) or {}),
        }

    @app.post("/work")
    def work(payload: dict[str, object]) -> dict[str, object]:
        response = context.workflow.workflow.run(
            WorkRequest(
                work_type=str(payload.get("type") or payload.get("work_type") or ""),
                instruction=str(payload.get("instruction") or ""),
                target=str(payload.get("target") or ""),
                output_format=str(payload.get("format") or payload.get("output_format") or "markdown"),
                access=_access(payload),
            )
        )
        return _workflow_payload(response)

    @app.post("/approval")
    def approval(payload: dict[str, object]) -> dict[str, object]:
        response = context.workflow.workflow.approval(
            action=str(payload.get("action") or ""),
            target_type=str(payload.get("type") or payload.get("target_type") or "task"),
            target_id=str(payload.get("target_id") or ""),
            comment=str(payload.get("comment") or ""),
            access=_access(payload),
        )
        return _workflow_payload(response)

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
                "metadata": {},
            }
        if action == "cost_report":
            return context.automation.readiness.cost_report()
        if action == "eval":
            return {"action": "eval", "mode": "local_harness", "metadata": {}}
        raise HTTPException(status_code=400, detail="unsupported admin action")

    return app


__all__ = ["create_app"]
