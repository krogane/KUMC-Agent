from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from kumc_agent.infra.llm.gemini_rate_limit import (
    ragas_rate_limiter_name,
    wait_for_gemini_rate_limit,
)
from kumc_agent.usecases.chat.answer import ChatAnswerUsecase, ChatRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluateRagasRequest:
    eval_file: Path
    limit: int | None = None
    result_path: Path | None = None
    ragas_batch_size: int | None = None


@dataclass(frozen=True)
class RagasResult:
    total: int
    exact_match: float
    token_overlap: float
    ragas_metrics: dict[str, float] = field(default_factory=dict)


class EvaluateRagasUsecase:
    def __init__(
        self,
        *,
        chat_usecase: ChatAnswerUsecase,
        gemini_api_key: str = "",
        ragas_gemini_model: str = "",
        ragas_gemini_requests_per_minute: int = 0,
        default_ragas_batch_size: int = 0,
        eval_answer_relevancy_enabled: bool | None = None,
        eval_faithfulness_enabled: bool | None = None,
        eval_context_precision_enabled: bool | None = None,
        eval_context_recall_enabled: bool | None = None,
    ) -> None:
        self._chat_usecase = chat_usecase
        self._gemini_api_key = str(gemini_api_key or "").strip()
        self._ragas_gemini_model = str(ragas_gemini_model or "").strip()
        self._ragas_gemini_requests_per_minute = max(
            0,
            int(ragas_gemini_requests_per_minute),
        )
        self._default_ragas_batch_size = max(0, int(default_ragas_batch_size))
        self._eval_answer_relevancy_enabled = _resolve_metric_toggle(
            env_name="EVAL_ANSWER_RELEVANCY_ENABLED",
            config_value=eval_answer_relevancy_enabled,
            default=True,
        )
        self._eval_faithfulness_enabled = _resolve_metric_toggle(
            env_name="EVAL_FAITHFULNESS_ENABLED",
            config_value=eval_faithfulness_enabled,
            default=True,
        )
        self._eval_context_precision_enabled = _resolve_metric_toggle(
            env_name="EVAL_CONTEXT_PRECISION_ENABLED",
            config_value=eval_context_precision_enabled,
            default=True,
        )
        self._eval_context_recall_enabled = _resolve_metric_toggle(
            env_name="EVAL_CONTEXT_RECALL_ENABLED",
            config_value=eval_context_recall_enabled,
            default=True,
        )

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
        ragas_metrics = self._run_ragas(
            records,
            ragas_batch_size=self._resolve_batch_size(request.ragas_batch_size),
        )

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

    def _resolve_batch_size(self, request_batch_size: int | None) -> int | None:
        if request_batch_size is not None:
            size = int(request_batch_size)
            return size if size > 0 else None
        if self._default_ragas_batch_size > 0:
            return self._default_ragas_batch_size
        return None

    def _run_ragas(
        self,
        records: list[dict[str, object]],
        *,
        ragas_batch_size: int | None,
    ) -> dict[str, float]:
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
                self._eval_answer_relevancy_enabled,
            ),
            (
                "faithfulness",
                faithfulness,
                self._eval_faithfulness_enabled,
            ),
            (
                "context_precision",
                context_precision,
                self._eval_context_precision_enabled,
            ),
            (
                "context_recall",
                context_recall,
                self._eval_context_recall_enabled,
            ),
        ]
        metrics = [metric for _, metric, enabled in metric_options if enabled]
        metric_names = [name for name, _, enabled in metric_options if enabled]
        if not metrics:
            raise ValueError("At least one RAGAS metric must be enabled.")
        logger.info("Enabled RAGAS metrics (current): %s", ", ".join(metric_names))

        eval_kwargs: dict[str, object] = {"metrics": metrics}
        llm = self._build_ragas_llm()
        if llm is not None:
            eval_kwargs["llm"] = llm

        batches = _split_batches(records, batch_size=ragas_batch_size)
        if len(batches) > 1:
            logger.info(
                "Running RAGAS evaluation in %d batches (batch_size=%d).",
                len(batches),
                ragas_batch_size,
            )

        weighted_sums: dict[str, float] = {}
        weighted_counts: dict[str, int] = {}

        for batch in batches:
            wait_for_gemini_rate_limit(
                max_requests_per_minute=self._ragas_gemini_requests_per_minute,
                limiter_name=ragas_rate_limiter_name(),
            )
            dataset = Dataset.from_list(batch)
            try:
                result = self._evaluate_ragas(
                    evaluate=evaluate,
                    dataset=dataset,
                    eval_kwargs=eval_kwargs,
                    metrics=metrics,
                )
            except Exception:
                logger.exception("RAGAS evaluation failed. Skipping ragas metrics.")
                return {}
            batch_metrics = self._extract_summary_metrics(result)
            if not batch_metrics:
                continue
            weight = len(batch)
            for metric_name, metric_value in batch_metrics.items():
                weighted_sums[metric_name] = (
                    weighted_sums.get(metric_name, 0.0)
                    + (metric_value * float(weight))
                )
                weighted_counts[metric_name] = weighted_counts.get(metric_name, 0) + weight

        if not weighted_sums:
            return {}

        return {
            metric_name: weighted_sums[metric_name] / float(weighted_counts[metric_name])
            for metric_name in sorted(weighted_sums.keys())
            if weighted_counts.get(metric_name, 0) > 0
        }

    @staticmethod
    def _evaluate_ragas(
        *,
        evaluate,
        dataset: object,
        eval_kwargs: dict[str, object],
        metrics: list[object],
    ) -> object:
        try:
            return evaluate(dataset, **eval_kwargs)
        except TypeError:
            return evaluate(dataset, metrics=metrics)

    def _build_ragas_llm(self):
        if not self._gemini_api_key or not self._ragas_gemini_model:
            return None
        try:
            from google import genai
        except ImportError:
            logger.info("google-genai is not available. Running ragas without custom LLM.")
            return None

        try:
            from ragas.llms import llm_factory
        except ImportError:
            logger.info("ragas.llms is not available. Running ragas without custom LLM.")
            return None

        client = _RateLimitedGeminiClient(
            genai.Client(api_key=self._gemini_api_key),
            max_requests_per_minute=self._ragas_gemini_requests_per_minute,
        )
        try:
            return llm_factory(self._ragas_gemini_model, provider="google", client=client)
        except TypeError:
            try:
                return llm_factory(self._ragas_gemini_model, client=client)
            except Exception:
                logger.exception(
                    "Failed to build custom RAGAS LLM with rate-limited Gemini client."
                )
                return None
        except Exception:
            logger.exception(
                "Failed to build custom RAGAS LLM with rate-limited Gemini client."
            )
            return None

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


def _split_batches(
    records: list[dict[str, object]],
    *,
    batch_size: int | None,
) -> list[list[dict[str, object]]]:
    size = int(batch_size or 0)
    if size <= 0:
        return [records]
    return [records[i : i + size] for i in range(0, len(records), size)]


def _resolve_metric_toggle(
    *,
    env_name: str,
    config_value: bool | None,
    default: bool,
) -> bool:
    if config_value is not None:
        return bool(config_value)
    return _env_bool(env_name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class _RateLimitedGeminiClient:
    def __init__(self, client: object, *, max_requests_per_minute: int) -> None:
        self._client = client
        models = getattr(client, "models", None)
        if models is None:
            self.models = None
        else:
            self.models = _RateLimitedGeminiModelsProxy(
                models,
                max_requests_per_minute=max_requests_per_minute,
            )

    def __getattr__(self, item: str):
        return getattr(self._client, item)


class _RateLimitedGeminiModelsProxy:
    def __init__(self, models: object, *, max_requests_per_minute: int) -> None:
        self._models = models
        self._max_requests_per_minute = max(0, int(max_requests_per_minute))

    def __getattr__(self, item: str):
        target = getattr(self._models, item)
        if not callable(target):
            return target

        def _wrapped(*args, **kwargs):
            wait_for_gemini_rate_limit(
                max_requests_per_minute=self._max_requests_per_minute,
                limiter_name=ragas_rate_limiter_name(),
            )
            return target(*args, **kwargs)

        return _wrapped
