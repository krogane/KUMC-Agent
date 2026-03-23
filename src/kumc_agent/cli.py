from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from kumc_agent.frontends.console.repl import run_repl
from kumc_agent.frontends.discord.app import main as run_discord
from kumc_agent.frontends.http.app import main as run_http
from kumc_agent.runtime.container import build_runtime_context
from kumc_agent.usecases.chat.answer import ChatRequest
from kumc_agent.usecases.chat.entry import ChatEntryRequest
from kumc_agent.usecases.eval.ragas import EvaluateRagasRequest
from kumc_agent.usecases.indexing.build import BuildIndexRequest
from kumc_agent.usecases.indexing.update import UpdateIndexRequest
from kumc_agent.utils.logging import configure_logging, default_execution_log_path

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kumc-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("repl", help="Run console REPL")

    chat_parser = subparsers.add_parser("chat", help="Run one chat query")
    chat_parser.add_argument("--query", required=True)

    tool_parser = subparsers.add_parser("tool", help="Tool bridge commands")
    tool_sub = tool_parser.add_subparsers(dest="tool_command", required=True)
    tool_rag_parser = tool_sub.add_parser("rag", help="Run local RAG tool payload")
    tool_rag_parser.add_argument("--query", required=True)
    tool_rag_parser.add_argument("--question-author", default=None)
    tool_rag_parser.add_argument("--history-scope", default=None)
    tool_rag_parser.add_argument("--force-fast-mode", action="store_true")

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

    subparsers.add_parser("discord", help="Run Discord frontend")
    subparsers.add_parser("http", help="Run HTTP stub frontend")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "discord":
        run_discord()
        return

    if args.command == "http":
        run_http()
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
        logger.info("Running local RAG tool. query_length=%d", len(args.query or ""))
        answer = context.chat_answer.execute(
            ChatRequest(
                query=args.query,
                question_author=args.question_author,
                history_scope=args.history_scope,
                force_fast_mode=bool(args.force_fast_mode),
                disable_history=True,
                routing_history_override=[],
                generation_history_override=[],
                force_disable_additional_memory=True,
            )
        )
        tool_metadata = dict(answer.metadata)
        tool_metadata.pop("contexts", None)
        payload = {
            "answer": answer.text,
            "route": answer.route,
            "sources": [
                {
                    "id": source.id,
                    "label": source.label,
                    "uri": source.uri,
                }
                for source in answer.sources
            ],
            "routing_decision": answer.metadata.get("routing_decision"),
            "fast_mode": bool(answer.metadata.get("fast_mode", False)),
            "metadata": tool_metadata,
        }
        print(json.dumps(payload, ensure_ascii=False))
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
            result = context.update_index.execute(
                UpdateIndexRequest(
                    refresh_sources=not args.no_refresh_sources,
                    full_rebuild=bool(args.full_rebuild),
                    stage_selection=tuple(args.stages or ()) or None,
                )
            )
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
