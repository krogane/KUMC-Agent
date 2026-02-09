from __future__ import annotations

import logging
from pathlib import Path

from pipeline.rag_pipeline import RagPipeline

logger = logging.getLogger(__name__)


def build_ragas_llm(llm_api_key: str, model: str):
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


def run_ragas(
    records: list[dict[str, object]],
    *,
    rag_pipeline: RagPipeline,
    llm_api_key: str,
    judge_model: str,
    result_path: Path | None,
    eval_answer_relevancy_enabled: bool,
    eval_faithfulness_enabled: bool,
    eval_context_precision_enabled: bool,
    eval_context_recall_enabled: bool,
) -> object:
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
    metric_options = [
        ("answer_relevancy", answer_relevancy, eval_answer_relevancy_enabled),
        ("faithfulness", faithfulness, eval_faithfulness_enabled),
        ("context_precision", context_precision, eval_context_precision_enabled),
        ("context_recall", context_recall, eval_context_recall_enabled),
    ]
    metrics = [metric for _, metric, enabled in metric_options if enabled]
    metric_names = [name for name, _, enabled in metric_options if enabled]
    if not metrics:
        raise ValueError("At least one RAGAS metric must be enabled.")
    logger.info("Enabled RAGAS metrics: %s", ", ".join(metric_names))

    llm = build_ragas_llm(llm_api_key, judge_model)
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

    return result
