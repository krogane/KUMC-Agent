from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_SRC = Path(__file__).resolve().parents[1]
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from config import AppConfig, EmbeddingFactory
from eval.constants import (
    DEFAULT_EVAL_FILE,
    DEFAULT_RESULT_PATH,
    DEFAULT_SAVE_DATASET,
    LOG_LEVEL,
)
from eval.dataset import load_eval_items
from eval.records import build_records, save_dataset
from eval.ragas_runner import run_ragas
from pipeline.rag_pipeline import RagPipeline

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RAG quality with ragas.")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help="JSONL file with question + ground_truth(s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation rows (0 = no limit).",
    )
    parser.add_argument(
        "--save-dataset",
        type=Path,
        default=DEFAULT_SAVE_DATASET,
        help="Optional path to save generated dataset as JSONL.",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help="Optional path to save evaluation results (CSV or text).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="",
        help="Optional Gemini model name for ragas evaluation.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _build_arg_parser().parse_args()

    base_dir = Path(__file__).resolve().parents[3]
    load_dotenv(base_dir / ".env")
    cfg = AppConfig.from_here(base_dir=base_dir)
    llm_api_key = cfg.gemini_api_key

    rag_pipeline = RagPipeline(
        index_dir=cfg.index_dir,
        embedding_factory=EmbeddingFactory(cfg.embedding_model),
        llm_api_key=llm_api_key,
        config=cfg,
    )

    eval_path = args.eval_file
    if not eval_path.is_absolute():
        eval_path = cfg.base_dir / eval_path

    items = load_eval_items(eval_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    records = build_records(items=items, rag_pipeline=rag_pipeline)

    save_dataset_path = args.save_dataset
    if save_dataset_path and not save_dataset_path.is_absolute():
        save_dataset_path = cfg.base_dir / save_dataset_path

    result_path = args.result_path
    if result_path and not result_path.is_absolute():
        result_path = cfg.base_dir / result_path

    if save_dataset_path:
        save_dataset(records, save_dataset_path)

    judge_model = args.judge_model or cfg.genai_model

    run_ragas(
        records,
        rag_pipeline=rag_pipeline,
        llm_api_key=llm_api_key,
        judge_model=judge_model,
        result_path=result_path,
    )


if __name__ == "__main__":
    main()
