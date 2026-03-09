from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from kumc_agent.usecases.chat.answer import ChatAnswerUsecase, ChatRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluateRagasRequest:
    eval_file: Path
    limit: int | None = None
    result_path: Path | None = None


@dataclass(frozen=True)
class RagasResult:
    total: int
    exact_match: float
    token_overlap: float
    ragas_metrics: dict[str, float] = field(default_factory=dict)


class EvaluateRagasUsecase:
    def __init__(self, *, chat_usecase: ChatAnswerUsecase) -> None:
        self._chat_usecase = chat_usecase

    def execute(self, request: EvaluateRagasRequest) -> RagasResult:
        items = self._load_items(request.eval_file)
        if request.limit and request.limit > 0:
            items = items[: request.limit]

        records: list[dict[str, object]] = []
        exact_count = 0
        overlap_scores: list[float] = []

        for item in items:
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            answer_obj = self._chat_usecase.execute(
                ChatRequest(
                    query=question,
                    append_sources_to_response=False,
                )
            )
            answer = str(answer_obj.text or "").strip()
            contexts = self._contexts_from_metadata(answer_obj.metadata)
            truths = self._ground_truths(item)
            if any(truth and truth in answer for truth in truths):
                exact_count += 1
            overlap_scores.append(self._token_overlap(answer, truths))
            records.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truths": truths,
                    "ground_truth": truths[0] if truths else "",
                }
            )

        total = len(records)
        exact_match = (exact_count / total) if total else 0.0
        token_overlap = (sum(overlap_scores) / total) if total else 0.0
        ragas_metrics = self._run_ragas(records)

        result = RagasResult(
            total=total,
            exact_match=exact_match,
            token_overlap=token_overlap,
            ragas_metrics=ragas_metrics,
        )

        if request.result_path is not None:
            request.result_path.parent.mkdir(parents=True, exist_ok=True)
            request.result_path.write_text(
                json.dumps(
                    {
                        "total": result.total,
                        "exact_match": result.exact_match,
                        "token_overlap": result.token_overlap,
                        "ragas_metrics": result.ragas_metrics,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        return result

    @staticmethod
    def _load_items(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        out: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    @staticmethod
    def _ground_truths(item: dict[str, object]) -> list[str]:
        if "ground_truths" in item and isinstance(item["ground_truths"], list):
            return [str(v).strip() for v in item["ground_truths"] if str(v).strip()]
        value = str(item.get("ground_truth") or "").strip()
        return [value] if value else []

    @staticmethod
    def _token_overlap(answer: str, truths: list[str]) -> float:
        answer_tokens = set((answer or "").lower().split())
        if not answer_tokens or not truths:
            return 0.0
        best = 0.0
        for truth in truths:
            truth_tokens = set((truth or "").lower().split())
            if not truth_tokens:
                continue
            score = len(answer_tokens & truth_tokens) / max(1, len(truth_tokens))
            if score > best:
                best = score
        return best

    @staticmethod
    def _contexts_from_metadata(metadata: dict[str, object]) -> list[str]:
        raw_contexts = metadata.get("contexts")
        if not isinstance(raw_contexts, list):
            return []
        contexts: list[str] = []
        for value in raw_contexts:
            text = str(value or "").strip()
            if text:
                contexts.append(text)
        return contexts

    def _run_ragas(self, records: list[dict[str, object]]) -> dict[str, float]:
        if not records:
            return {}
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError:
            logger.warning(
                "ragas or datasets is not available. Skipping ragas metrics in current eval."
            )
            return {}

        metric_options = [
            (
                "answer_relevancy",
                answer_relevancy,
                _env_bool("EVAL_ANSWER_RELEVANCY_ENABLED", True),
            ),
            (
                "faithfulness",
                faithfulness,
                _env_bool("EVAL_FAITHFULNESS_ENABLED", True),
            ),
            (
                "context_precision",
                context_precision,
                _env_bool("EVAL_CONTEXT_PRECISION_ENABLED", True),
            ),
            (
                "context_recall",
                context_recall,
                _env_bool("EVAL_CONTEXT_RECALL_ENABLED", True),
            ),
        ]
        metrics = [metric for _, metric, enabled in metric_options if enabled]
        metric_names = [name for name, _, enabled in metric_options if enabled]
        if not metrics:
            raise ValueError("At least one RAGAS metric must be enabled.")
        logger.info("Enabled RAGAS metrics (current): %s", ", ".join(metric_names))

        dataset = Dataset.from_list(records)
        result = evaluate(dataset, metrics=metrics)
        return self._extract_summary_metrics(result)

    @staticmethod
    def _extract_summary_metrics(result: object) -> dict[str, float]:
        if result is None:
            return {}
        scores = getattr(result, "scores", None)
        if isinstance(scores, dict):
            normalized = _coerce_numeric_metrics(scores)
            if normalized:
                return normalized
        try:
            mapping = dict(result)  # type: ignore[arg-type]
        except Exception:
            mapping = {}
        normalized = _coerce_numeric_metrics(mapping)
        if normalized:
            return normalized
        to_pandas = getattr(result, "to_pandas", None)
        if not callable(to_pandas):
            return {}
        try:
            frame = to_pandas()
        except Exception:
            return {}
        numeric_frame = (
            frame.select_dtypes(include="number")
            if hasattr(frame, "select_dtypes")
            else frame
        )
        if not hasattr(numeric_frame, "mean"):
            return {}
        try:
            means = numeric_frame.mean(numeric_only=True)
        except TypeError:
            means = numeric_frame.mean()
        try:
            return {
                str(key): float(value)
                for key, value in means.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        except Exception:
            return {}


def _coerce_numeric_metrics(values: dict[str, object]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            normalized[str(key)] = float(value)
    return normalized


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
