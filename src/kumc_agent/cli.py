from __future__ import annotations

import argparse
import json
from pathlib import Path

from kumc_agent.frontends.console.repl import run_repl
from kumc_agent.frontends.discord.app import main as run_discord
from kumc_agent.frontends.http.app import main as run_http
from kumc_agent.runtime.container import build_runtime_context
from kumc_agent.usecases.chat.answer import ChatRequest
from kumc_agent.usecases.eval.ragas import EvaluateRagasRequest
from kumc_agent.usecases.indexing.build import BuildIndexRequest
from kumc_agent.usecases.indexing.update import UpdateIndexRequest
from kumc_agent.utils.logging import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kumc-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("repl", help="Run console REPL")

    chat_parser = subparsers.add_parser("chat", help="Run one chat query")
    chat_parser.add_argument("--query", required=True)

    index_parser = subparsers.add_parser("index", help="Index operations")
    index_sub = index_parser.add_subparsers(dest="index_command", required=True)
    build_parser = index_sub.add_parser("build")
    build_parser.add_argument("--no-refresh-sources", action="store_true")
    update_parser = index_sub.add_parser("update")
    update_parser.add_argument("--no-refresh-sources", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluation operations")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    ragas_parser = eval_sub.add_parser("ragas")
    ragas_parser.add_argument("--eval-file", type=Path, default=None)
    ragas_parser.add_argument("--limit", type=int, default=None)
    ragas_parser.add_argument("--result-path", type=Path, default=None)

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
    configure_logging(context.config.app.log_level)

    if args.command == "repl":
        run_repl(context)
        return

    if args.command == "chat":
        answer = context.chat_answer.execute(ChatRequest(query=args.query))
        print(answer.text)
        return

    if args.command == "index":
        if args.index_command == "build":
            result = context.build_index.execute(
                BuildIndexRequest(refresh_sources=not args.no_refresh_sources)
            )
        else:
            result = context.update_index.execute(
                UpdateIndexRequest(refresh_sources=not args.no_refresh_sources)
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
        result = context.eval_ragas.execute(
            EvaluateRagasRequest(
                eval_file=eval_file,
                limit=args.limit,
                result_path=args.result_path,
            )
        )
        print(
            json.dumps(
                {
                    "total": result.total,
                    "exact_match": result.exact_match,
                    "token_overlap": result.token_overlap,
                },
                ensure_ascii=False,
            )
        )
        return


if __name__ == "__main__":
    main()
