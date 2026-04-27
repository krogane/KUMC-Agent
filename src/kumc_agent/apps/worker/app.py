from __future__ import annotations

import logging
from pathlib import Path
import asyncio

from kumc_agent.apps.automation import build_automation_app_context
from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.apps.ingestion import build_ingestion_app_context
from kumc_agent.apps.workflow import build_workflow_app_context
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.models.source import BackfillScope
from kumc_agent.domain.models.workflow import WorkRequest
from kumc_agent.utils.logging import configure_logging, default_execution_log_path

logger = logging.getLogger(__name__)


def run_once(
    *,
    base_dir: Path | None = None,
    job_type: str = "worker.health",
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    context = build_foundation_app_context(base_dir=base_dir)
    automation = build_automation_app_context(base_dir=base_dir)
    job = context.jobs.start(job_type)
    try:
        result = _dispatch_job(base_dir=base_dir, job_type=job_type, payload=payload or {})
        report = context.health.check(actor_id="worker", actor_type="service")
        context.jobs.complete(
            job,
            metadata={
                "health_status": report.status,
                "result": result,
            },
        )
        payload = report.as_dict()
        payload["job_type"] = job_type
        payload["result"] = result
        return payload
    except Exception as exc:
        context.jobs.fail(job, str(exc))
        raise


def _dispatch_job(
    *,
    base_dir: Path | None,
    job_type: str,
    payload: dict[str, object],
) -> dict[str, object]:
    if job_type == "worker.health":
        automation = build_automation_app_context(base_dir=base_dir)
        readiness = automation.readiness.report()
        rules = automation.automation.seed_defaults()
        return {
            "readiness_status": readiness.status,
            "automation_rules": len(rules),
            "side_effects": "none",
        }
    if job_type == "ingest_backfill":
        ingestion = build_ingestion_app_context(base_dir=base_dir)
        source = str(payload.get("source") or "").strip()
        results = asyncio.run(
            ingestion.service.backfill_many(
                source_kinds=(source,) if source else tuple(),
                scope=BackfillScope(
                    limit=int(payload["limit"]) if payload.get("limit") is not None else None,
                    force=bool(payload.get("force", False)),
                ),
            )
        )
        return {"results": [result.__dict__ for result in results], "side_effects": "indexing_only"}
    if job_type == "member_profiles_rebuild":
        workflow = build_workflow_app_context(base_dir=base_dir)
        guild_id = str(payload.get("guild_id") or "").strip()
        if guild_id:
            guild_ids = [guild_id]
        else:
            foundation = build_foundation_app_context(base_dir=base_dir)
            guild_ids = [str(value) for value in foundation.config.security.discord_guild_allow_list]
        results = [
            workflow.member_profile_builder.rebuild_guild(guild_id=value).__dict__
            for value in guild_ids
            if workflow.member_profile_builder is not None
        ]
        return {
            "results": results,
            "guild_ids": guild_ids,
            "side_effects": "member_profile_indexing",
        }
    if job_type == "weekly_summary_draft":
        workflow = build_workflow_app_context(base_dir=base_dir)
        response = workflow.workflow.run(
            WorkRequest(
                work_type="announcement_draft",
                instruction=str(payload.get("instruction") or "週次まとめ draft"),
                access=AccessContext(user_id="worker", is_admin=True),
            )
        )
        return {"text": response.text, "metadata": response.metadata, "side_effects": "draft_only"}
    if job_type == "task_due_reminder":
        workflow = build_workflow_app_context(base_dir=base_dir)
        response = workflow.workflow.run(
            WorkRequest(
                work_type="task_notify_due",
                instruction=f"days: {payload.get('days', 1)}",
                access=AccessContext(user_id="worker", is_admin=True),
            )
        )
        return {
            "notified_tasks": len(response.tasks),
            "metadata": response.metadata,
            "side_effects": "notification_state_recorded",
        }
    if job_type == "task_approval_batch":
        workflow = build_workflow_app_context(base_dir=base_dir)
        response = workflow.workflow.run(
            WorkRequest(
                work_type="task_batch_approval",
                instruction=str(payload.get("instruction") or ""),
                access=AccessContext(user_id="worker", is_admin=True),
            )
        )
        return {
            "candidate_count": len(response.task_candidates),
            "change_candidate_count": len(response.task_change_candidates),
            "batch_count": len(response.task_approval_batches),
            "metadata": response.metadata,
            "side_effects": "approval_batch_recorded",
        }
    if job_type == "event_reminder":
        workflow = build_workflow_app_context(base_dir=base_dir)
        kind = str(payload.get("kind") or "before")
        response = workflow.workflow.run(
            WorkRequest(
                work_type="event_notify",
                instruction=f"days: {payload.get('days', 1)} kind: {kind}",
                access=AccessContext(user_id="worker", is_admin=True),
            )
        )
        return {
            "notified_events": len(response.events),
            "metadata": response.metadata,
            "side_effects": "notification_state_recorded",
        }
    if job_type == "event_approval_batch":
        workflow = build_workflow_app_context(base_dir=base_dir)
        response = workflow.workflow.run(
            WorkRequest(
                work_type="event_batch_approval",
                instruction=str(payload.get("instruction") or ""),
                access=AccessContext(user_id="worker", is_admin=True),
            )
        )
        return {
            "candidate_count": len(response.event_candidates),
            "change_candidate_count": len(response.event_change_candidates),
            "batch_count": len(response.event_approval_batches),
            "metadata": response.metadata,
            "side_effects": "approval_batch_recorded",
        }
    if job_type == "workflow_prepare":
        workflow = build_workflow_app_context(base_dir=base_dir)
        response = workflow.workflow.run(
            WorkRequest(
                work_type=str(payload.get("work_type") or "meeting_prepare"),
                instruction=str(payload.get("instruction") or ""),
                target=str(payload.get("target") or ""),
                access=AccessContext(user_id="worker", is_admin=True),
            )
        )
        return {"text": response.text, "metadata": response.metadata, "side_effects": "draft_or_candidate_only"}
    return {"status": "skipped", "reason": f"unsupported job_type: {job_type}", "side_effects": "none"}


def main(*, base_dir: Path | None = None) -> None:
    context = build_foundation_app_context(base_dir=base_dir)
    configure_logging(
        context.config.app.log_level,
        file_path=default_execution_log_path(base_dir=context.config.base_dir),
    )
    result = run_once(base_dir=base_dir)
    logger.info("Wave 1 worker skeleton completed. result=%s", result)
