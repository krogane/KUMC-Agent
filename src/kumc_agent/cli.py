from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

from kumc_agent.frontends.console.repl import run_repl
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.runtime.container import build_runtime_context
from kumc_agent.usecases.chat.answer import ChatRequest
from kumc_agent.usecases.chat.entry import ChatEntryRequest
from kumc_agent.usecases.eval.ragas import EvaluateRagasRequest
from kumc_agent.usecases.indexing.auto_update import AutoIndexUpdateRequest
from kumc_agent.usecases.indexing.build import BuildIndexRequest
from kumc_agent.utils.logging import configure_logging, default_execution_log_path

logger = logging.getLogger(__name__)


def _build_tool_rag_payload(answer: object) -> dict[str, object]:
    metadata = dict(getattr(answer, "metadata", {}) or {})
    for key in ("contexts", "llm_prompt", "raw"):
        metadata.pop(key, None)
    return {
        "answer": getattr(answer, "text", ""),
        "route": getattr(answer, "route", ""),
        "sources": [
            {
                "id": source.id,
                "label": source.label,
                "uri": source.uri,
            }
            for source in getattr(answer, "sources", [])
        ],
        "metadata": metadata,
    }


def _workflow_response_payload(response: object) -> dict[str, object]:
    def _dump_items(items: object) -> list[dict[str, object]]:
        return [
            _dump_workflow_item(item)
            for item in items or []
        ]

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


def _dump_workflow_item(item: object) -> dict[str, object]:
    payload = {
        key: value.isoformat() if hasattr(value, "isoformat") else value
        for key, value in getattr(item, "__dict__", {}).items()
    }
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
    metadata = dict(value or {}) if isinstance(value, dict) else {}
    for key in ("contexts", "context", "llm_prompt", "raw", "secret"):
        metadata.pop(key, None)
    for key, item in list(metadata.items()):
        if isinstance(item, str):
            metadata[key] = _compact_payload_text(_mask_payload_secret(item), 1200)
    return metadata


def _compact_payload_text(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _mask_payload_secret(text: str) -> str:
    return re.sub(
        r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )


def _automation_response_payload(response: object) -> dict[str, object]:
    def _dump_items(items: object) -> list[dict[str, object]]:
        return [
            {
                key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in getattr(item, "__dict__", {}).items()
            }
            for item in items or []
        ]

    return {
        "text": getattr(response, "text", ""),
        "detail_markdown": getattr(response, "detail_markdown", ""),
        "rules": _dump_items(getattr(response, "rules", ())),
        "runs": _dump_items(getattr(response, "runs", ())),
        "warnings": list(getattr(response, "warnings", ())),
        "metadata": dict(getattr(response, "metadata", {}) or {}),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kumc-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("repl", help="Run console REPL")

    chat_parser = subparsers.add_parser("chat", help="Run one chat query")
    chat_parser.add_argument("--query", required=True)

    tool_parser = subparsers.add_parser("tool", help="Tool bridge commands")
    tool_sub = tool_parser.add_subparsers(dest="tool_command", required=True)
    tool_rag_parser = tool_sub.add_parser("rag", help="Run local RAG tool payload")
    tool_rag_parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="RAG query text. Specify multiple times to run multiple queries.",
    )
    tool_rag_parser.add_argument("--question-author", default=None)
    tool_rag_parser.add_argument("--history-scope", default=None)
    tool_rag_parser.add_argument(
        "--scope",
        default="all",
        choices=("all", "minecraft_wiki"),
    )
    tool_rag_parser.add_argument("--force-fast-mode", action="store_true")
    tool_rag_parser.add_argument("--user-id", default="")
    tool_rag_parser.add_argument("--guild-id", default="")
    tool_rag_parser.add_argument("--role-id", action="append", default=None)
    tool_rag_parser.add_argument("--admin", action="store_true")

    index_parser = subparsers.add_parser("index", help="Index operations")
    index_sub = index_parser.add_subparsers(dest="index_command", required=True)
    build_parser = index_sub.add_parser("build")
    build_parser.add_argument("--no-refresh-sources", action="store_true")
    build_parser.add_argument("--full-rebuild", action="store_true")
    build_parser.add_argument("--stage", action="append", dest="stages", default=None)
    update_parser = index_sub.add_parser("update")
    update_parser.add_argument("--no-refresh-sources", action="store_true")
    update_parser.add_argument("--full-rebuild", action="store_true")
    update_parser.add_argument("--stage", action="append", dest="stages", default=None)

    eval_parser = subparsers.add_parser("eval", help="Evaluation operations")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    ragas_parser = eval_sub.add_parser("ragas")
    ragas_parser.add_argument("--eval-file", type=Path, default=None)
    ragas_parser.add_argument("--limit", type=int, default=None)
    ragas_parser.add_argument("--result-path", type=Path, default=None)
    ragas_parser.add_argument("--ragas-batch-size", type=int, default=None)
    ragas_parser.add_argument("--ragas-max-workers", type=int, default=None)
    ragas_parser.add_argument("--ragas-timeout-seconds", type=float, default=None)
    ragas_parser.add_argument("--ragas-max-retries", type=int, default=None)
    ragas_parser.add_argument("--answer-cache-path", type=Path, default=None)
    ragas_parser.add_argument("--disable-answer-cache", action="store_true")
    ragas_parser.add_argument("--refresh-answer-cache", action="store_true")
    ragas_parser.add_argument("--disable-history-for-eval", action="store_true")

    subparsers.add_parser("bot", help="Run Wave 1 Discord slash-command bot")

    api_parser = subparsers.add_parser("api", help="Run Wave 1 API app")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)

    worker_parser = subparsers.add_parser("worker", help="Run worker job once")
    worker_parser.add_argument("--job-type", default="worker.health")
    worker_parser.add_argument("--payload-json", default="{}")

    admin_parser = subparsers.add_parser("admin", help="Admin actions")
    admin_parser.add_argument(
        "--action",
        choices=(
            "health",
            "readiness",
            "sync",
            "eval",
            "feature_flags",
            "permissions",
            "reindex",
            "cost_report",
            "member_profiles",
        ),
        required=True,
    )
    admin_parser.add_argument("--scope", default="")
    admin_parser.add_argument("--limit", type=int, default=None)
    admin_parser.add_argument("--force", action="store_true")

    db_parser = subparsers.add_parser("db", help="Database operations")
    db_sub = db_parser.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("migrate", help="Apply PostgreSQL migrations")

    ingest_parser = subparsers.add_parser("ingest", help="Wave 2 ingestion operations")
    ingest_sub = ingest_parser.add_subparsers(dest="ingest_command", required=True)
    backfill_parser = ingest_sub.add_parser("backfill")
    backfill_parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source connector to backfill. Use multiple times or omit for all enabled.",
    )
    backfill_parser.add_argument("--limit", type=int, default=None)
    backfill_parser.add_argument("--force", action="store_true")

    ask_parser = subparsers.add_parser("ask", help="Wave 3 integrated ask route")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument(
        "--source",
        default="all",
        choices=(
            "all",
            "drive",
            "discord",
            "notion",
            "hatena",
            "x",
            "crafters_colony",
            "minecraft_wiki",
            "image",
            "member",
            "task",
            "event",
        ),
    )
    ask_parser.add_argument("--mode", default="answer", choices=("answer", "search_only", "fast", "careful"))
    ask_parser.add_argument("--depth", default="normal", choices=("light", "normal", "deep"))
    ask_parser.add_argument("--user-id", default="")
    ask_parser.add_argument("--guild-id", default="")
    ask_parser.add_argument("--role-id", action="append", default=None)
    ask_parser.add_argument("--admin", action="store_true")

    work_parser = subparsers.add_parser("work", help="Wave 4 workflow operations")
    work_parser.add_argument(
        "--type",
        required=True,
        choices=(
            "meeting_prepare",
            "meeting_minutes_draft",
            "task_extract",
            "task_add",
            "task_list",
            "task_done",
            "task_update",
            "task_delete",
            "task_notify_due",
            "task_batch_approval",
            "event_add",
            "event_extract",
            "event_list",
            "event_brief",
            "event_update",
            "event_delete",
            "event_notify",
            "event_batch_approval",
            "event_complete",
            "schedule_add",
            "schedule_list",
            "doc_draft",
            "x_draft",
            "announcement_draft",
            "mc_status",
            "mc_request",
            "image_search",
            "member_search",
        ),
    )
    work_parser.add_argument("--instruction", default="")
    work_parser.add_argument("--target", default="")
    work_parser.add_argument("--format", default="markdown", choices=("compact", "markdown"))
    work_parser.add_argument("--user-id", default="")
    work_parser.add_argument("--guild-id", default="")
    work_parser.add_argument("--role-id", action="append", default=None)
    work_parser.add_argument("--admin", action="store_true")

    approval_parser = subparsers.add_parser("approval", help="Wave 4 approval operations")
    approval_parser.add_argument(
        "--type",
        default="task",
        choices=(
            "task",
            "event",
            "schedule",
            "announcement",
            "automation_rule",
            "server_operation",
            "finance_record",
            "member_assignment",
            "other",
        ),
    )
    approval_parser.add_argument("--action", required=True, choices=("list", "show", "approve", "reject", "edit"))
    approval_parser.add_argument("--target-id", default="")
    approval_parser.add_argument("--comment", default="")
    approval_parser.add_argument("--user-id", default="")
    approval_parser.add_argument("--guild-id", default="")
    approval_parser.add_argument("--role-id", action="append", default=None)
    approval_parser.add_argument("--admin", action="store_true")

    automation_parser = subparsers.add_parser("automation", help="Wave 7 automation operations")
    automation_parser.add_argument(
        "--action",
        required=True,
        choices=("list", "show", "dry_run", "run", "enable", "disable", "set_mode"),
    )
    automation_parser.add_argument("--rule-id", default="")
    automation_parser.add_argument(
        "--mode",
        default="dry_run",
        choices=("dry_run", "approval_required", "auto_run"),
    )
    automation_parser.add_argument("--trigger-key", default="manual")
    automation_parser.add_argument("--idempotency-key", default="")
    automation_parser.add_argument("--user-id", default="")
    automation_parser.add_argument("--guild-id", default="")
    automation_parser.add_argument("--role-id", action="append", default=None)
    automation_parser.add_argument("--admin", action="store_true")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "bot":
        from kumc_agent.apps.bot.app import main as run_bot

        run_bot()
        return

    if args.command == "api":
        from kumc_agent.apps.api.app import main as run_api

        run_api(host=args.host, port=args.port)
        return

    if args.command == "worker":
        from kumc_agent.apps.worker.app import run_once

        result = run_once(
            job_type=args.job_type,
            payload=json.loads(args.payload_json or "{}"),
        )
        print(json.dumps(result, ensure_ascii=False, default=str))
        return

    if args.command == "admin":
        from kumc_agent.apps.foundation import build_foundation_app_context

        foundation = build_foundation_app_context()
        configure_logging(
            foundation.config.app.log_level,
            file_path=default_execution_log_path(base_dir=foundation.config.base_dir),
        )
        if args.action == "health":
            report = foundation.health.check(actor_id="cli", actor_type="service")
            print(json.dumps(report.as_dict(), ensure_ascii=False))
            return
        if args.action == "readiness":
            from kumc_agent.apps.automation import build_automation_app_context

            automation = build_automation_app_context(seed_defaults=False)
            report = automation.readiness.report()
            print(json.dumps(report.as_dict(), ensure_ascii=False))
            return
        if args.action in {"sync", "reindex"}:
            if args.scope == "member_profiles":
                from kumc_agent.apps.workflow import build_workflow_app_context

                workflow = build_workflow_app_context()
                guild_ids = [
                    str(value)
                    for value in (
                        foundation.config.security.discord_guild_allow_list
                        if not args.limit
                        else foundation.config.security.discord_guild_allow_list[: args.limit]
                    )
                ]
                if not guild_ids:
                    print(
                        json.dumps(
                            {
                                "action": args.action,
                                "source_kind": "member_profiles",
                                "results": [],
                                "metadata": {"reason": "no_guild_ids_configured"},
                            },
                            ensure_ascii=False,
                        )
                    )
                    return
                results = [
                    workflow.member_profile_builder.rebuild_guild(guild_id=guild_id).__dict__
                    for guild_id in guild_ids
                    if workflow.member_profile_builder is not None
                ]
                print(
                    json.dumps(
                        {
                            "action": args.action,
                            "source_kind": "member_profiles",
                            "results": results,
                            "metadata": {},
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                )
                return
            context = build_runtime_context()
            result = context.auto_index_update.execute(
                AutoIndexUpdateRequest(
                    trigger="admin",
                    source_filter=(args.scope,) if args.scope else tuple(),
                    force=bool(args.force or args.action == "reindex"),
                    full_rebuild=bool(args.action == "reindex"),
                )
            )
            payload = result.as_payload()
            payload["action"] = args.action
            print(json.dumps(payload, ensure_ascii=False, default=str))
            return
        if args.action == "eval":
            from kumc_agent.apps.automation import build_automation_app_context

            automation = build_automation_app_context(seed_defaults=False)
            report = automation.readiness.report()
            print(
                json.dumps(
                    {
                        "action": "eval",
                        "mode": "local_harness",
                        "readiness": report.as_dict(),
                    },
                    ensure_ascii=False,
                )
            )
            return
        if args.action == "feature_flags":
            print(json.dumps(foundation.feature_flags.modes(), ensure_ascii=False))
            return
        if args.action == "permissions":
            print(
                json.dumps(
                    {
                        "maintenance_command_author_ids": foundation.config.security.maintenance_command_author_ids,
                        "discord_guild_allow_list": foundation.config.security.discord_guild_allow_list,
                        "admin_configured": bool(foundation.config.security.maintenance_command_author_ids),
                        "guild_allow_list_configured": bool(foundation.config.security.discord_guild_allow_list),
                    },
                    ensure_ascii=False,
                )
            )
            return
        if args.action == "cost_report":
            from kumc_agent.apps.automation import build_automation_app_context

            automation = build_automation_app_context(seed_defaults=False)
            print(json.dumps(automation.readiness.cost_report(), ensure_ascii=False))
            return
        if args.action == "member_profiles":
            from kumc_agent.apps.workflow import build_workflow_app_context

            workflow = build_workflow_app_context()
            configured_guild_ids = [
                str(value) for value in foundation.config.security.discord_guild_allow_list
            ]
            guild_ids = [args.scope] if args.scope else configured_guild_ids
            results = [
                workflow.member_profile_builder.rebuild_guild(guild_id=str(guild_id)).__dict__
                for guild_id in guild_ids
                if workflow.member_profile_builder is not None
            ]
            print(
                json.dumps(
                    {
                        "action": "member_profiles",
                        "results": results,
                        "metadata": {"guild_ids": guild_ids},
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )
            return

    if args.command == "db":
        from kumc_agent.apps.foundation import build_foundation_app_context

        foundation = build_foundation_app_context()
        configure_logging(
            foundation.config.app.log_level,
            file_path=default_execution_log_path(base_dir=foundation.config.base_dir),
        )
        if args.db_command == "migrate":
            result = foundation.migrations.apply()
            print(
                json.dumps(
                    {
                        "applied": list(result.applied),
                        "skipped": list(result.skipped),
                    },
                    ensure_ascii=False,
                )
            )
            return

    if args.command == "ingest":
        from kumc_agent.apps.ingestion import build_ingestion_app_context
        from kumc_agent.domain.models.source import BackfillScope

        ingestion = build_ingestion_app_context()
        if args.ingest_command == "backfill":
            results = asyncio.run(
                ingestion.service.backfill_many(
                    source_kinds=tuple(args.source or ()),
                    scope=BackfillScope(limit=args.limit, force=bool(args.force)),
                )
            )
            print(
                json.dumps(
                    [result.__dict__ for result in results],
                    ensure_ascii=False,
                )
            )
            return

    if args.command == "ask":
        if args.source == "member":
            from kumc_agent.apps.workflow import build_workflow_app_context
            from kumc_agent.domain.models.workflow import WorkRequest

            workflow = build_workflow_app_context()
            response = workflow.workflow.run(
                WorkRequest(
                    work_type="member_search",
                    instruction=args.question,
                    access=AccessContext(
                        user_id=args.user_id,
                        guild_id=args.guild_id,
                        role_ids=tuple(args.role_id or ()),
                        is_admin=bool(args.admin),
                    ),
                )
            )
            print(json.dumps(_workflow_response_payload(response), ensure_ascii=False, default=str))
            return
        if args.depth == "deep":
            from kumc_agent.apps.agentic import build_agentic_app_context
            from kumc_agent.domain.models.agentic import AgenticSearchRequest

            agentic = build_agentic_app_context()
            response = agentic.agentic_search.search(
                AgenticSearchRequest(
                    query=args.question,
                    source_filter=args.source,
                    access=AccessContext(
                        user_id=args.user_id,
                        guild_id=args.guild_id,
                        role_ids=tuple(args.role_id or ()),
                        is_admin=bool(args.admin),
                    ),
                )
            )
            print(
                json.dumps(
                    {
                        "text": response.text,
                        "detail_markdown": response.detail_markdown,
                        "confidence": response.confidence,
                        "warnings": list(response.warnings),
                        "citations": [citation.__dict__ for citation in response.citations],
                        "agent_run_id": response.run.id,
                    },
                    ensure_ascii=False,
                )
            )
            return
        from kumc_agent.apps.retrieval import build_retrieval_app_context
        from kumc_agent.domain.models.retrieval import RetrievalQuery

        retrieval = build_retrieval_app_context()
        response = retrieval.ask.ask(
            RetrievalQuery(
                text=args.question,
                source_filter=args.source,
                mode=args.mode,
                depth=args.depth,
                access=AccessContext(
                    user_id=args.user_id,
                    guild_id=args.guild_id,
                    role_ids=tuple(args.role_id or ()),
                    is_admin=bool(args.admin),
                ),
            )
        )
        print(
            json.dumps(
                {
                    "text": response.text,
                    "detail_markdown": response.detail_markdown,
                    "confidence": response.confidence,
                    "warnings": list(response.warnings),
                    "citations": [citation.__dict__ for citation in response.citations],
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "work":
        from kumc_agent.apps.workflow import build_workflow_app_context
        from kumc_agent.domain.models.workflow import WorkRequest

        workflow = build_workflow_app_context()
        response = workflow.workflow.run(
            WorkRequest(
                work_type=args.type,
                instruction=args.instruction,
                target=args.target,
                output_format=args.format,
                access=AccessContext(
                    user_id=args.user_id,
                    guild_id=args.guild_id,
                    role_ids=tuple(args.role_id or ()),
                    is_admin=bool(args.admin),
                ),
            )
        )
        print(json.dumps(_workflow_response_payload(response), ensure_ascii=False, default=str))
        return

    if args.command == "approval":
        from kumc_agent.apps.workflow import build_workflow_app_context

        workflow = build_workflow_app_context()
        response = workflow.workflow.approval(
            action=args.action,
            target_type=args.type,
            target_id=args.target_id,
            comment=args.comment,
            access=AccessContext(
                user_id=args.user_id,
                guild_id=args.guild_id,
                role_ids=tuple(args.role_id or ()),
                is_admin=bool(args.admin),
            ),
        )
        print(json.dumps(_workflow_response_payload(response), ensure_ascii=False, default=str))
        return

    if args.command == "automation":
        from kumc_agent.apps.automation import build_automation_app_context

        automation = build_automation_app_context()
        access = AccessContext(
            user_id=args.user_id,
            guild_id=args.guild_id,
            role_ids=tuple(args.role_id or ()),
            is_admin=bool(args.admin),
        )
        if args.action == "list":
            response = automation.automation.list_rules()
        elif args.action == "show":
            response = automation.automation.show(rule_id=args.rule_id)
        elif args.action == "enable":
            response = automation.automation.enable(rule_id=args.rule_id, access=access)
        elif args.action == "disable":
            response = automation.automation.disable(rule_id=args.rule_id, access=access)
        elif args.action == "set_mode":
            response = automation.automation.set_mode(
                rule_id=args.rule_id,
                mode=args.mode,
                access=access,
            )
        elif args.action == "dry_run":
            response = automation.automation.dry_run(
                rule_id=args.rule_id,
                trigger_key=args.trigger_key,
                idempotency_key=args.idempotency_key,
                access=access,
            )
        else:
            response = automation.automation.run(
                rule_id=args.rule_id,
                trigger_key=args.trigger_key,
                idempotency_key=args.idempotency_key,
                access=access,
            )
        print(json.dumps(_automation_response_payload(response), ensure_ascii=False, default=str))
        return

    context = build_runtime_context()
    configure_logging(
        context.config.app.log_level,
        file_path=default_execution_log_path(base_dir=context.config.base_dir),
    )
    logger.info("CLI command started: %s", args.command)

    if args.command == "repl":
        logger.info("Starting REPL session")
        run_repl(context)
        return

    if args.command == "chat":
        logger.info("Running chat query. length=%d", len(args.query or ""))
        answer = context.chat_entry.execute(ChatEntryRequest(query=args.query))
        print(answer.text)
        logger.info("Chat query completed")
        return

    if args.command == "tool" and args.tool_command == "rag":
        raw_queries = args.query if isinstance(args.query, list) else [args.query]
        queries = [str(value or "") for value in raw_queries]
        logger.info(
            "Running local RAG tool. query_count=%d total_query_length=%d",
            len(queries),
            sum(len(query) for query in queries),
        )

        if len(queries) == 1:
            access_context = AccessContext(
                user_id=str(args.user_id or ""),
                guild_id=str(args.guild_id or ""),
                role_ids=tuple(args.role_id or ()),
                is_admin=bool(args.admin),
            )
            answer = context.chat_answer.execute(
                ChatRequest(
                    query=queries[0],
                    question_author=args.question_author,
                    history_scope=args.history_scope,
                    force_fast_mode=bool(args.force_fast_mode),
                    disable_history=True,
                    routing_history_override=[],
                    generation_history_override=[],
                    force_disable_additional_memory=True,
                    access_context=access_context,
                    route_override=(
                        "minecraft_wiki" if args.scope == "minecraft_wiki" else None
                    ),
                )
            )
            payload = _build_tool_rag_payload(answer)
            print(json.dumps(payload, ensure_ascii=False))
        else:
            access_context = AccessContext(
                user_id=str(args.user_id or ""),
                guild_id=str(args.guild_id or ""),
                role_ids=tuple(args.role_id or ()),
                is_admin=bool(args.admin),
            )
            results: list[dict[str, object]] = []
            for query in queries:
                answer = context.chat_answer.execute(
                    ChatRequest(
                        query=query,
                        question_author=args.question_author,
                        history_scope=args.history_scope,
                        force_fast_mode=bool(args.force_fast_mode),
                        disable_history=True,
                        routing_history_override=[],
                        generation_history_override=[],
                        force_disable_additional_memory=True,
                        access_context=access_context,
                        route_override=(
                            "minecraft_wiki" if args.scope == "minecraft_wiki" else None
                        ),
                    )
                )
                result = _build_tool_rag_payload(answer)
                result["query"] = query
                results.append(result)
            print(
                json.dumps(
                    {
                        "query_count": len(results),
                        "results": results,
                    },
                    ensure_ascii=False,
                )
            )
        logger.info("Local RAG tool completed")
        return

    if args.command == "index":
        if args.index_command == "build":
            logger.info("Running index build")
            result = context.build_index.execute(
                BuildIndexRequest(
                    refresh_sources=not args.no_refresh_sources,
                    full_rebuild=bool(args.full_rebuild),
                    stage_selection=tuple(args.stages or ()) or None,
                )
            )
        else:
            logger.info("Running index update")
            auto_result = context.auto_index_update.execute(
                AutoIndexUpdateRequest(
                    trigger="manual",
                    refresh_sources=not args.no_refresh_sources,
                    force=bool(args.full_rebuild),
                    full_rebuild=bool(args.full_rebuild),
                    stage_selection=tuple(args.stages or ()) or None,
                )
            )
            logger.info("Index update completed. status=%s run_id=%s", auto_result.status, auto_result.run_id)
            print(json.dumps(auto_result.as_payload(), ensure_ascii=False, default=str))
            return
        logger.info(
            "Index command completed. loaded_sources=%d documents=%d chunks=%d",
            result.loaded_sources,
            result.documents,
            result.chunks,
        )
        print(
            json.dumps(
                {
                    "loaded_sources": result.loaded_sources,
                    "documents": result.documents,
                    "chunks": result.chunks,
                    "index_dir": str(result.index_dir),
                    "metadata": {},
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "eval" and args.eval_command == "ragas":
        eval_file = (
            args.eval_file
            if args.eval_file is not None
            else context.config.app.eval_dir / "ragas.jsonl"
        )
        logger.info("Running ragas eval. eval_file=%s", eval_file)
        result = context.eval_ragas.execute(
            EvaluateRagasRequest(
                eval_file=eval_file,
                limit=args.limit,
                result_path=args.result_path,
                ragas_batch_size=args.ragas_batch_size,
                ragas_max_workers=args.ragas_max_workers,
                ragas_timeout_seconds=args.ragas_timeout_seconds,
                ragas_max_retries=args.ragas_max_retries,
                answer_cache_path=args.answer_cache_path,
                answer_cache_enabled=False if args.disable_answer_cache else None,
                refresh_answer_cache=bool(args.refresh_answer_cache),
                disable_history_for_eval=True if args.disable_history_for_eval else None,
            )
        )
        logger.info(
            "Ragas eval completed. total=%d exact_match=%.3f token_overlap=%.3f "
            "metrics=%s metadata=%s",
            result.total,
            result.exact_match,
            result.token_overlap,
            result.ragas_metrics,
            result.ragas_metadata,
        )
        print(
            json.dumps(
                {
                    "total": result.total,
                    "exact_match": result.exact_match,
                    "token_overlap": result.token_overlap,
                    "ragas_metrics": result.ragas_metrics,
                    "ragas_metadata": result.ragas_metadata,
                },
                ensure_ascii=False,
            )
        )
        return


if __name__ == "__main__":
    main()
