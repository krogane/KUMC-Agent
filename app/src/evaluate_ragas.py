from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from rag_pipeline import RagPipeline
from config import AppConfig, EmbeddingFactory, DEFAULT_SYSTEM_RULES

logger = logging.getLogger(__name__)

DEFAULT_EVAL_FILE: Path = Path("app") / "data" / "eval" / "ragas.jsonl"
DEFAULT_RESULT_DIR: Path = Path("app") / "data" / "eval" / "result"
DEFAULT_SAVE_DATASET: Path = DEFAULT_RESULT_DIR / "ragas_dataset.jsonl"
DEFAULT_RESULT_PATH: Path = DEFAULT_RESULT_DIR / "ragas_result.csv"

LOG_LEVEL: str = "INFO"


@dataclass(frozen=True)
class EvalItem:
    question: str
    ground_truths: list[str]


def _load_eval_items(path: Path) -> list[EvalItem]:
    if not path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found: {path}. "
            "Create a JSONL file with {question, ground_truth} or {question, ground_truths}."
        )

    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as fr:
        for line_no, line in enumerate(fr, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path} line {line_no}: {e}") from e

            # Accept both "question" and "query" for compatibility with different JSONL schemas.
            question = obj.get("question", obj.get("query"))
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Missing/invalid 'question' (or 'query') in {path} line {line_no}"
                )

            ground_truths = obj.get("ground_truths", obj.get("ground_truth"))
            ground_truths = _normalize_ground_truths(ground_truths, path, line_no)

            items.append(EvalItem(question=question.strip(), ground_truths=ground_truths))

    return items


def _normalize_ground_truths(value: object, path: Path, line_no: int) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise ValueError(
        f"Missing/invalid 'ground_truths' (string or list of strings) in {path} line {line_no}"
    )


def _build_records(
    *,
    items: Iterable[EvalItem],
    rag_pipeline: RagPipeline,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in items:
        answer, contexts = rag_pipeline.answer_with_contexts(item.question)
        records.append(
            {
                "question": item.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truths": item.ground_truths,
                "ground_truth": item.ground_truths[0],
            }
        )
    return records


def _save_dataset(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fw:
        for record in records:
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_ragas_llm(llm_api_key: str, model: str):
    if not llm_api_key:
        return None

    try:
        from google import genai
    except ImportError:
        logger.info("google-genai not installed; using ragas defaults.")
        return None

    try:
        from ragas.llms import llm_factory
    except ImportError:
        logger.info("ragas not installed; cannot build custom evaluator LLM.")
        return None

    client = genai.Client(api_key=llm_api_key)

    try:
        return llm_factory(model, provider="google", client=client)
    except TypeError:
        try:
            return llm_factory(model, client=client)
        except Exception:
            logger.exception(
                "Failed to build evaluator LLM via llm_factory; falling back to ragas defaults."
            )
            return None


def _run_ragas(
    records: list[dict[str, object]],
    *,
    rag_pipeline: RagPipeline,
    llm_api_key: str,
    judge_model: str,
    result_path: Path | None,
) -> None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        raise RuntimeError("ragas and datasets are required to run evaluation.") from e

    dataset = Dataset.from_list(records)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    llm = _build_ragas_llm(llm_api_key, judge_model)
    eval_kwargs = {"metrics": metrics}
    if llm is not None:
        eval_kwargs["llm"] = llm
        eval_kwargs["embeddings"] = rag_pipeline.embeddings()

    try:
        result = evaluate(dataset, **eval_kwargs)
    except TypeError:
        result = evaluate(dataset, metrics=metrics)

    logger.info("Ragas result: %s", result)

    if result_path:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(result, "to_pandas"):
            result.to_pandas().to_csv(result_path, index=False)
        else:
            result_path.write_text(str(result), encoding="utf-8")


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

    base_dir = Path(__file__).resolve().parents[2]
    load_dotenv(base_dir / ".env")
    cfg = AppConfig.from_here(system_rules=DEFAULT_SYSTEM_RULES, base_dir=base_dir)
    llm_api_key = os.getenv("GEMINI_API_KEY", "")

    rag_pipeline = RagPipeline(
        index_dir=cfg.index_dir,
        embedding_factory=EmbeddingFactory(cfg.embedding_model_name),
        llm_api_key=llm_api_key,
        config=cfg,
    )

    eval_path = args.eval_file
    if not eval_path.is_absolute():
        eval_path = cfg.base_dir / eval_path

    items = _load_eval_items(eval_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    records = _build_records(items=items, rag_pipeline=rag_pipeline)

    save_dataset_path = args.save_dataset
    if save_dataset_path and not save_dataset_path.is_absolute():
        save_dataset_path = cfg.base_dir / save_dataset_path

    result_path = args.result_path
    if result_path and not result_path.is_absolute():
        result_path = cfg.base_dir / result_path

    if save_dataset_path:
        _save_dataset(records, save_dataset_path)

    judge_model = args.judge_model or cfg.genai_model

    _run_ragas(
        records,
        rag_pipeline=rag_pipeline,
        llm_api_key=llm_api_key,
        judge_model=judge_model,
        result_path=result_path,
    )


if __name__ == "__main__":
    main()
