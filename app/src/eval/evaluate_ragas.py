from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_SRC = Path(__file__).resolve().parents[1]
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from config import AppConfig, EmbeddingFactory
from eval.constants import (
    APP_DIR,
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
METRICS_PREFIX = "EVAL_METRICS_JSON:"


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "app":
        return base_dir / path
    if parts and parts[0] == "data":
        return APP_DIR / path
    if parts and parts[0] == "eval":
        return APP_DIR / "data" / path
    return base_dir / path


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


def _coerce_numeric_metrics(values: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = float(value)
    return metrics


def _extract_summary_metrics(result: object) -> dict[str, float]:
    if result is None:
        return {}

    scores = getattr(result, "scores", None)
    if isinstance(scores, dict):
        metrics = _coerce_numeric_metrics(scores)
        if metrics:
            return metrics

    try:
        mapping = dict(result)  # type: ignore[arg-type]
    except Exception:
        mapping = {}

    metrics = _coerce_numeric_metrics(mapping)
    if metrics:
        return metrics

    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        try:
            df = to_pandas()
        except Exception:
            return {}

        if hasattr(df, "select_dtypes"):
            numeric_df = df.select_dtypes(include="number")
        else:
            numeric_df = df

        if hasattr(numeric_df, "mean"):
            try:
                means = numeric_df.mean(numeric_only=True)
            except TypeError:
                means = numeric_df.mean()
            try:
                return {
                    key: float(value)
                    for key, value in means.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            except Exception:
                return {}

    return {}


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

    eval_path = _resolve_path(args.eval_file, base_dir=cfg.base_dir)

    items = load_eval_items(eval_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    records = build_records(items=items, rag_pipeline=rag_pipeline)

    save_dataset_path = (
        _resolve_path(args.save_dataset, base_dir=cfg.base_dir)
        if args.save_dataset
        else None
    )

    result_path = (
        _resolve_path(args.result_path, base_dir=cfg.base_dir)
        if args.result_path
        else None
    )

    if save_dataset_path:
        save_dataset(records, save_dataset_path)

    judge_model = args.judge_model or cfg.genai_model

    result = run_ragas(
        records,
        rag_pipeline=rag_pipeline,
        llm_api_key=llm_api_key,
        judge_model=judge_model,
        result_path=result_path,
    )

    metrics = _extract_summary_metrics(result)
    if metrics:
        print(f"{METRICS_PREFIX}{json.dumps(metrics, ensure_ascii=True)}")


if __name__ == "__main__":
    main()
