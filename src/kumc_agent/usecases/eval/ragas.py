from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from kumc_agent.infra.llm.gemini_rate_limit import (
    ragas_embedding_rate_limiter_name,
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
    ragas_max_workers: int | None = None
    ragas_timeout_seconds: float | None = None
    ragas_max_retries: int | None = None
    answer_cache_path: Path | None = None
    answer_cache_enabled: bool | None = None
    refresh_answer_cache: bool = False
    disable_history_for_eval: bool | None = None
    cancel_event: threading.Event | None = None


@dataclass(frozen=True)
class RagasResult:
    total: int
    exact_match: float
    token_overlap: float
    ragas_metrics: dict[str, float] = field(default_factory=dict)
    ragas_metadata: dict[str, object] = field(default_factory=dict)
    records: tuple[dict[str, object], ...] = tuple()


@dataclass(frozen=True)
class _RagasRunOutcome:
    metrics: dict[str, float]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _EvaluationResolution:
    result: object | None
    canceled: bool


@dataclass(frozen=True)
class _AnswerGenerationTask:
    batch_order: int
    item_index: int
    question: str
    ground_truths: list[str]
    cache_key: str


@dataclass(frozen=True)
class _GeneratedAnswer:
    task: _AnswerGenerationTask
    answer: str
    contexts: list[str]
    sources: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class EvaluateRagasUsecase:
    def __init__(
        self,
        *,
        chat_usecase: ChatAnswerUsecase,
        gemini_api_key: str = "",
        ragas_gemini_model: str = "",
        ragas_gemini_requests_per_minute: int = 0,
        ragas_gemini_embedding_requests_per_minute: int | None = None,
        default_answer_generation_batch_size: int = 0,
        default_ragas_batch_size: int = 0,
        default_ragas_max_workers: int = 0,
        default_ragas_timeout_seconds: float = 0.0,
        default_ragas_max_retries: int = 0,
        default_answer_cache_enabled: bool = True,
        default_answer_cache_path: Path | None = None,
        default_disable_history_for_eval: bool = True,
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
        if ragas_gemini_embedding_requests_per_minute is None:
            ragas_embedding_requests_per_minute = self._ragas_gemini_requests_per_minute
        else:
            ragas_embedding_requests_per_minute = int(
                ragas_gemini_embedding_requests_per_minute
            )
        self._ragas_gemini_embedding_requests_per_minute = max(
            0,
            ragas_embedding_requests_per_minute,
        )
        self._default_answer_generation_batch_size = max(
            0,
            int(default_answer_generation_batch_size),
        )
        self._default_ragas_batch_size = max(0, int(default_ragas_batch_size))
        self._default_ragas_max_workers = max(0, int(default_ragas_max_workers))
        self._default_ragas_timeout_seconds = max(
            0.0,
            float(default_ragas_timeout_seconds),
        )
        self._default_ragas_max_retries = max(0, int(default_ragas_max_retries))
        self._default_answer_cache_enabled = bool(default_answer_cache_enabled)
        self._default_answer_cache_path = default_answer_cache_path
        self._default_disable_history_for_eval = bool(default_disable_history_for_eval)
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

        answer_generation_batch_size = self._resolve_answer_generation_batch_size()
        ragas_batch_size = self._resolve_batch_size(request.ragas_batch_size)
        ragas_max_workers = self._resolve_ragas_max_workers(request.ragas_max_workers)
        ragas_timeout_seconds = self._resolve_ragas_timeout_seconds(
            request.ragas_timeout_seconds
        )
        ragas_max_retries = self._resolve_ragas_max_retries(request.ragas_max_retries)
        disable_history_for_eval = self._resolve_disable_history_for_eval(
            request.disable_history_for_eval
        )
        answer_cache_enabled = self._resolve_answer_cache_enabled(
            request.answer_cache_enabled
        )
        answer_cache_path = self._resolve_answer_cache_path(request)

        answer_cache: dict[str, dict[str, object]] = {}
        if answer_cache_enabled and not request.refresh_answer_cache:
            answer_cache = self._load_answer_cache(answer_cache_path)

        cache_hits = 0
        cache_misses = 0
        canceled = False
        records: list[dict[str, object]] = []
        answer_records: list[dict[str, object]] = []
        exact_count = 0
        overlap_scores: list[float] = []

        pending_questions: list[dict[str, object]] = []
        for index, item in enumerate(items):
            if _is_cancel_requested(request.cancel_event):
                canceled = True
                logger.info("RAGAS eval canceled while preparing answer records.")
                break
            question = self._question_text(item)
            if not question:
                continue
            truths = self._ground_truths(item)
            pending_questions.append(
                {
                    "item_index": index,
                    "question": question,
                    "ground_truths": truths,
                }
            )

        question_batches = _split_batches(
            pending_questions,
            batch_size=answer_generation_batch_size,
        )
        if len(question_batches) > 1:
            logger.info(
                "Preparing RAGAS answer records in %d batches (batch_size=%d).",
                len(question_batches),
                answer_generation_batch_size,
            )

        answer_generation_max_workers = self._resolve_answer_generation_max_workers(
            task_count=max((len(batch) for batch in question_batches), default=0),
            answer_generation_batch_size=answer_generation_batch_size,
            ragas_max_workers=ragas_max_workers,
        )
        if answer_generation_max_workers > 1:
            logger.info(
                "Running RAGAS answer generation with up to %d workers per batch.",
                answer_generation_max_workers,
            )

        for batch in question_batches:
            if _is_cancel_requested(request.cancel_event):
                canceled = True
                logger.info("RAGAS eval canceled while preparing answer records.")
                break

            batch_cache_entries: list[dict[str, object]] = []
            resolved_batch: dict[int, _GeneratedAnswer] = {}
            pending_tasks: list[_AnswerGenerationTask] = []

            for batch_order, payload in enumerate(batch):
                if _is_cancel_requested(request.cancel_event):
                    canceled = True
                    logger.info("RAGAS eval canceled while preparing answer records.")
                    break

                question = str(payload.get("question") or "").strip()
                if not question:
                    continue
                item_index = int(payload.get("item_index") or 0)
                truths = [
                    str(value).strip()
                    for value in list(payload.get("ground_truths") or [])
                    if str(value).strip()
                ]
                cache_key = _answer_cache_key(question)
                cached = answer_cache.get(cache_key)
                if cached and str(cached.get("question") or "").strip() == question:
                    cache_hits += 1
                    resolved_batch[batch_order] = _GeneratedAnswer(
                        task=_AnswerGenerationTask(
                            batch_order=batch_order,
                            item_index=item_index,
                            question=question,
                            ground_truths=truths,
                            cache_key=cache_key,
                        ),
                        answer=str(cached.get("answer") or "").strip(),
                        contexts=_normalize_contexts(cached.get("contexts")),
                        sources=_normalize_sources(cached.get("sources")),
                        metadata=_normalize_metadata(cached.get("metadata")),
                    )
                    continue

                pending_tasks.append(
                    _AnswerGenerationTask(
                        batch_order=batch_order,
                        item_index=item_index,
                        question=question,
                        ground_truths=truths,
                        cache_key=cache_key,
                    )
                )

            if pending_tasks and not canceled:
                generated_answers, generation_canceled = self._generate_answers_for_tasks(
                    tasks=pending_tasks,
                    disable_history_for_eval=disable_history_for_eval,
                    cancel_event=request.cancel_event,
                    max_workers=self._resolve_answer_generation_max_workers(
                        task_count=len(pending_tasks),
                        answer_generation_batch_size=answer_generation_batch_size,
                        ragas_max_workers=ragas_max_workers,
                    ),
                )
                cache_misses += len(generated_answers)
                for generated in generated_answers:
                    resolved_batch[generated.task.batch_order] = generated
                    if answer_cache_enabled:
                        cache_record = {
                            "question_hash": generated.task.cache_key,
                            "question": generated.task.question,
                            "answer": generated.answer,
                            "contexts": generated.contexts,
                            "sources": generated.sources,
                            "metadata": generated.metadata,
                        }
                        answer_cache[generated.task.cache_key] = cache_record
                        batch_cache_entries.append(cache_record)

                if generation_canceled:
                    canceled = True
                    logger.info("RAGAS eval canceled while preparing answer records.")

            for batch_order in sorted(resolved_batch.keys()):
                generated = resolved_batch[batch_order]
                question = generated.task.question
                answer = generated.answer
                contexts = generated.contexts
                truths = generated.task.ground_truths

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
                answer_records.append(
                    {
                        "question": question,
                        "answer": answer,
                        "context_texts": contexts,
                        "contexts": _context_payloads(contexts, generated.sources),
                        "citations": generated.sources,
                        "sources": generated.sources,
                        "retrieval_trace": _retrieval_trace_from_metadata(
                            generated.metadata,
                            generated.sources,
                        ),
                        "metadata": generated.metadata,
                    }
                )

            if answer_cache_enabled and batch_cache_entries:
                self._append_answer_cache_entries(answer_cache_path, batch_cache_entries)

            if canceled:
                break

        total = len(records)
        exact_match = (exact_count / total) if total else 0.0
        token_overlap = (sum(overlap_scores) / total) if total else 0.0
        ragas_outcome = self._run_ragas(
            records,
            ragas_batch_size=ragas_batch_size,
            ragas_max_workers=ragas_max_workers,
            ragas_timeout_seconds=ragas_timeout_seconds,
            ragas_max_retries=ragas_max_retries,
            cancel_event=request.cancel_event,
        )

        ragas_metadata: dict[str, object] = {
            "answer_cache_enabled": answer_cache_enabled,
            "answer_cache_path": str(answer_cache_path) if answer_cache_enabled else "",
            "answer_cache_refresh": bool(request.refresh_answer_cache),
            "answer_cache_hits": int(cache_hits),
            "answer_cache_misses": int(cache_misses),
            "disable_history_for_eval": disable_history_for_eval,
            "answer_generation_batch_size": int(answer_generation_batch_size or 0),
            "answer_generation_max_workers": int(answer_generation_max_workers),
            "records_prepared": total,
            "canceled": bool(canceled),
        }
        ragas_metadata.update(ragas_outcome.metadata)

        result = RagasResult(
            total=total,
            exact_match=exact_match,
            token_overlap=token_overlap,
            ragas_metrics=ragas_outcome.metrics,
            ragas_metadata=ragas_metadata,
            records=tuple(answer_records),
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
                        "ragas_metadata": result.ragas_metadata,
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
    def _question_text(item: dict[str, object]) -> str:
        # Backward compatibility: some eval datasets use "query" instead of "question".
        for key in ("question", "query"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
        return ""

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

    def _resolve_answer_generation_batch_size(self) -> int | None:
        if self._default_answer_generation_batch_size > 0:
            return self._default_answer_generation_batch_size
        if self._default_ragas_batch_size > 0:
            return self._default_ragas_batch_size
        return None

    @staticmethod
    def _resolve_answer_generation_max_workers(
        *,
        task_count: int,
        answer_generation_batch_size: int | None,
        ragas_max_workers: int | None,
    ) -> int:
        if task_count <= 0:
            return 0
        max_workers = int(ragas_max_workers or 0)
        if max_workers <= 0:
            max_workers = int(answer_generation_batch_size or 0)
        if max_workers <= 0:
            max_workers = 1
        return min(task_count, max_workers)

    def _generate_answer_for_task(
        self,
        *,
        task: _AnswerGenerationTask,
        disable_history_for_eval: bool,
    ) -> _GeneratedAnswer:
        answer_obj = self._chat_usecase.execute(
            self._build_eval_chat_request(
                question=task.question,
                item_index=task.item_index,
                disable_history_for_eval=disable_history_for_eval,
            )
        )
        return _GeneratedAnswer(
            task=task,
            answer=str(answer_obj.text or "").strip(),
            contexts=self._contexts_from_metadata(answer_obj.metadata),
            sources=_sources_to_payload(answer_obj.sources),
            metadata={str(key): value for key, value in answer_obj.metadata.items()},
        )

    def _generate_answers_for_tasks(
        self,
        *,
        tasks: list[_AnswerGenerationTask],
        disable_history_for_eval: bool,
        cancel_event: threading.Event | None,
        max_workers: int,
    ) -> tuple[list[_GeneratedAnswer], bool]:
        if not tasks:
            return [], False
        if max_workers <= 1:
            generated: list[_GeneratedAnswer] = []
            for task in tasks:
                if _is_cancel_requested(cancel_event):
                    return generated, True
                generated.append(
                    self._generate_answer_for_task(
                        task=task,
                        disable_history_for_eval=disable_history_for_eval,
                    )
                )
            return generated, False

        generated: list[_GeneratedAnswer] = []
        pending: set[Future[_GeneratedAnswer]] = set()
        canceled = False
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            for task in tasks:
                if _is_cancel_requested(cancel_event):
                    canceled = True
                    break
                pending.add(
                    executor.submit(
                        self._generate_answer_for_task,
                        task=task,
                        disable_history_for_eval=disable_history_for_eval,
                    )
                )

            while pending and not canceled:
                done, pending = wait(
                    pending,
                    timeout=0.2,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    if _is_cancel_requested(cancel_event):
                        canceled = True
                    continue
                for future in done:
                    generated.append(future.result())
                if _is_cancel_requested(cancel_event):
                    canceled = True
        except Exception:
            canceled = True
            raise
        finally:
            if canceled:
                for future in pending:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

        return generated, canceled

    def _resolve_ragas_max_workers(self, request_value: int | None) -> int | None:
        if request_value is not None:
            value = int(request_value)
            return value if value > 0 else None
        if self._default_ragas_max_workers > 0:
            return self._default_ragas_max_workers
        return None

    def _resolve_ragas_timeout_seconds(self, request_value: float | None) -> float | None:
        if request_value is not None:
            value = float(request_value)
            return value if value > 0 else None
        if self._default_ragas_timeout_seconds > 0:
            return self._default_ragas_timeout_seconds
        return None

    def _resolve_ragas_max_retries(self, request_value: int | None) -> int | None:
        if request_value is not None:
            value = int(request_value)
            return value if value >= 0 else 0
        return self._default_ragas_max_retries

    def _resolve_answer_cache_enabled(self, request_value: bool | None) -> bool:
        if request_value is not None:
            return bool(request_value)
        return self._default_answer_cache_enabled

    def _resolve_disable_history_for_eval(self, request_value: bool | None) -> bool:
        if request_value is not None:
            return bool(request_value)
        return self._default_disable_history_for_eval

    def _resolve_answer_cache_path(self, request: EvaluateRagasRequest) -> Path:
        if request.answer_cache_path is not None:
            return request.answer_cache_path
        if self._default_answer_cache_path is not None:
            return self._default_answer_cache_path
        return request.eval_file.parent / "cache" / "ragas_answers.jsonl"

    def _build_eval_chat_request(
        self,
        *,
        question: str,
        item_index: int,
        disable_history_for_eval: bool,
    ) -> ChatRequest:
        if not disable_history_for_eval:
            return ChatRequest(
                query=question,
                append_sources_to_response=False,
            )
        return ChatRequest(
            query=question,
            append_sources_to_response=False,
            history_scope=f"__eval__:{item_index}",
            routing_history_override=[],
            generation_history_override=[],
            force_disable_additional_memory=True,
        )

    @staticmethod
    def _load_answer_cache(path: Path) -> dict[str, dict[str, object]]:
        if not path.exists():
            return {}
        entries: dict[str, dict[str, object]] = {}
        try:
            with path.open("r", encoding="utf-8") as fr:
                for line in fr:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    if not isinstance(payload, dict):
                        continue
                    key = str(payload.get("question_hash") or "").strip()
                    if not key:
                        continue
                    entries[key] = payload
        except Exception:
            logger.exception("Failed to load RAGAS answer cache: %s", path)
        return entries

    @staticmethod
    def _append_answer_cache_entries(
        path: Path,
        entries: list[dict[str, object]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fw:
            for payload in entries:
                fw.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _run_ragas(
        self,
        records: list[dict[str, object]],
        *,
        ragas_batch_size: int | None,
        ragas_max_workers: int | None,
        ragas_timeout_seconds: float | None,
        ragas_max_retries: int | None,
        cancel_event: threading.Event | None,
    ) -> _RagasRunOutcome:
        if not records:
            return _RagasRunOutcome(metrics={}, metadata={})
        try:
            from datasets import Dataset
            import ragas
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            evaluate = getattr(ragas, "evaluate", None)
            if not callable(evaluate):
                aevaluate = getattr(ragas, "aevaluate", None)
                if callable(aevaluate):
                    import asyncio

                    def _evaluate_sync(dataset, **kwargs):  # type: ignore[no-untyped-def]
                        return asyncio.run(aevaluate(dataset, **kwargs))

                    evaluate = _evaluate_sync
                else:
                    raise ImportError("ragas.evaluate is unavailable")
            try:
                from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM
            except Exception:
                MetricWithEmbeddings = None  # type: ignore[assignment]
                MetricWithLLM = None  # type: ignore[assignment]
        except ImportError:
            logger.warning(
                "ragas or datasets is not available. Skipping ragas metrics in current eval."
            )
            return _RagasRunOutcome(
                metrics={},
                metadata={"skipped_reason": "dependency_missing"},
            )

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
        metric_pairs = [
            (name, metric) for name, metric, enabled in metric_options if enabled
        ]
        if not metric_pairs:
            raise ValueError("At least one RAGAS metric must be enabled.")
        llm = self._build_ragas_llm()
        if llm is None and MetricWithLLM is not None:
            llm_metric_names = [
                name
                for name, metric in metric_pairs
                if isinstance(metric, MetricWithLLM)
            ]
            if llm_metric_names:
                logger.warning(
                    "Skipping LLM-based RAGAS metrics because custom evaluator LLM "
                    "could not be initialized: %s",
                    ", ".join(llm_metric_names),
                )
            metric_pairs = [
                (name, metric)
                for name, metric in metric_pairs
                if not isinstance(metric, MetricWithLLM)
            ]
        answer_relevancy_strictness_forced = self._force_answer_relevancy_strictness_one(
            metric_pairs=metric_pairs,
            llm=llm,
        )

        embeddings = self._build_ragas_embeddings()
        if embeddings is None and MetricWithEmbeddings is not None:
            embedding_metric_names = [
                name
                for name, metric in metric_pairs
                if isinstance(metric, MetricWithEmbeddings)
            ]
            if embedding_metric_names:
                logger.warning(
                    "Skipping embedding-based RAGAS metrics because embeddings "
                    "client could not be initialized: %s",
                    ", ".join(embedding_metric_names),
                )
            metric_pairs = [
                (name, metric)
                for name, metric in metric_pairs
                if not isinstance(metric, MetricWithEmbeddings)
            ]

        if not metric_pairs:
            logger.warning(
                "No runnable RAGAS metrics remain after dependency/client checks."
            )
            return _RagasRunOutcome(
                metrics={},
                metadata={"skipped_reason": "no_runnable_metrics"},
            )

        metric_names = [name for name, _ in metric_pairs]
        metrics = [metric for _, metric in metric_pairs]
        logger.info("Enabled RAGAS metrics (current): %s", ", ".join(metric_names))

        eval_kwargs: dict[str, object] = {"metrics": metrics}
        if llm is not None:
            eval_kwargs["llm"] = llm
        if embeddings is not None:
            eval_kwargs["embeddings"] = embeddings

        run_config = self._build_ragas_run_config(
            max_workers=ragas_max_workers,
            timeout_seconds=ragas_timeout_seconds,
            max_retries=ragas_max_retries,
        )
        dataset = Dataset.from_list(records)
        try:
            result_or_executor = self._evaluate_ragas(
                evaluate=evaluate,
                dataset=dataset,
                eval_kwargs=eval_kwargs,
                metrics=metrics,
                batch_size=ragas_batch_size,
                run_config=run_config,
                return_executor=True,
            )
            resolution = self._resolve_evaluation_result(
                result_or_executor=result_or_executor,
                cancel_event=cancel_event,
            )
            if resolution.canceled:
                return _RagasRunOutcome(
                    metrics={},
                    metadata={
                        "mode": "single_pass",
                        "canceled": True,
                        "batch_size": int(ragas_batch_size or 0),
                        "answer_relevancy_strictness_forced": bool(
                            answer_relevancy_strictness_forced
                        ),
                    },
                )
            summary = self._extract_summary_metrics(resolution.result)
            if not summary:
                summary = self._extract_summary_metrics_from_executor_results(
                    resolution.result,
                    metric_names=metric_names,
                    row_count=len(records),
                )
            if summary:
                return _RagasRunOutcome(
                    metrics=summary,
                    metadata={
                        "mode": "single_pass",
                        "canceled": False,
                        "batch_size": int(ragas_batch_size or 0),
                        "max_workers": int(ragas_max_workers or 0),
                        "timeout_seconds": float(ragas_timeout_seconds or 0.0),
                        "max_retries": int(ragas_max_retries or 0),
                        "failed_batches": 0,
                        "failed_records": 0,
                        "total_batches": 1,
                        "answer_relevancy_strictness_forced": bool(
                            answer_relevancy_strictness_forced
                        ),
                    },
                )
            logger.warning(
                "RAGAS single-pass evaluation returned no numeric metrics. "
                "Falling back to chunked mode."
            )
        except Exception:
            logger.exception(
                "RAGAS single-pass evaluation failed. Falling back to chunked mode."
            )

        batches = _split_batches(records, batch_size=ragas_batch_size)
        if len(batches) > 1:
            logger.info(
                "Running RAGAS fallback evaluation in %d batches (batch_size=%d).",
                len(batches),
                ragas_batch_size,
            )

        weighted_sums: dict[str, float] = {}
        weighted_counts: dict[str, int] = {}
        failed_batches = 0
        failed_records = 0

        for batch in batches:
            if _is_cancel_requested(cancel_event):
                logger.info("RAGAS fallback evaluation canceled.")
                break
            wait_for_gemini_rate_limit(
                max_requests_per_minute=self._ragas_gemini_requests_per_minute,
                limiter_name=ragas_rate_limiter_name(),
            )
            dataset = Dataset.from_list(batch)
            try:
                result_or_executor = self._evaluate_ragas(
                    evaluate=evaluate,
                    dataset=dataset,
                    eval_kwargs=eval_kwargs,
                    metrics=metrics,
                    batch_size=None,
                    run_config=run_config,
                    return_executor=True,
                )
                resolution = self._resolve_evaluation_result(
                    result_or_executor=result_or_executor,
                    cancel_event=cancel_event,
                )
                if resolution.canceled:
                    break
                result = resolution.result
            except Exception:
                failed_batches += 1
                failed_records += len(batch)
                logger.exception(
                    "RAGAS fallback batch failed. Continuing with remaining batches."
                )
                continue
            batch_metrics = self._extract_summary_metrics(result)
            if not batch_metrics:
                batch_metrics = self._extract_summary_metrics_from_executor_results(
                    result,
                    metric_names=metric_names,
                    row_count=len(batch),
                )
            if not batch_metrics:
                continue
            weight = len(batch)
            for metric_name, metric_value in batch_metrics.items():
                weighted_sums[metric_name] = (
                    weighted_sums.get(metric_name, 0.0)
                    + (metric_value * float(weight))
                )
                weighted_counts[metric_name] = weighted_counts.get(metric_name, 0) + weight

        merged_metrics = {
            metric_name: weighted_sums[metric_name] / float(weighted_counts[metric_name])
            for metric_name in sorted(weighted_sums.keys())
            if weighted_counts.get(metric_name, 0) > 0
        }
        return _RagasRunOutcome(
            metrics=merged_metrics,
            metadata={
                "mode": "fallback_batches",
                "canceled": _is_cancel_requested(cancel_event),
                "batch_size": int(ragas_batch_size or 0),
                "max_workers": int(ragas_max_workers or 0),
                "timeout_seconds": float(ragas_timeout_seconds or 0.0),
                "max_retries": int(ragas_max_retries or 0),
                "failed_batches": failed_batches,
                "failed_records": failed_records,
                "total_batches": len(batches),
                "answer_relevancy_strictness_forced": bool(
                    answer_relevancy_strictness_forced
                ),
            },
        )

    @staticmethod
    def _evaluate_ragas(
        *,
        evaluate,
        dataset: object,
        eval_kwargs: dict[str, object],
        metrics: list[object],
        batch_size: int | None,
        run_config: object,
        return_executor: bool,
    ) -> object:
        kwargs: dict[str, object] = dict(eval_kwargs)
        if batch_size is not None and batch_size > 0:
            kwargs["batch_size"] = int(batch_size)
        if run_config is not None:
            kwargs["run_config"] = run_config
        if return_executor:
            kwargs["return_executor"] = True

        optional_drop_order = [
            "return_executor",
            "run_config",
            "batch_size",
            "llm",
            "embeddings",
        ]
        current_kwargs = dict(kwargs)
        while True:
            try:
                return evaluate(dataset, **current_kwargs)
            except TypeError:
                dropped = False
                for key in optional_drop_order:
                    if key in current_kwargs:
                        current_kwargs = dict(current_kwargs)
                        current_kwargs.pop(key, None)
                        dropped = True
                        break
                if dropped:
                    continue
                if current_kwargs != {"metrics": metrics}:
                    current_kwargs = {"metrics": metrics}
                    continue
                raise

    @staticmethod
    def _resolve_evaluation_result(
        *,
        result_or_executor: object,
        cancel_event: threading.Event | None,
    ) -> _EvaluationResolution:
        if result_or_executor is None:
            return _EvaluationResolution(result=None, canceled=False)
        result_getter = getattr(result_or_executor, "results", None)
        if not callable(result_getter):
            result_getter = getattr(result_or_executor, "result", None)
        if not callable(result_getter):
            return _EvaluationResolution(result=result_or_executor, canceled=False)

        result_box: dict[str, object] = {}
        error_box: dict[str, BaseException] = {}
        completed = threading.Event()

        def _collect() -> None:
            try:
                result_box["value"] = result_getter()
            except BaseException as exc:  # pragma: no cover - external runtime behavior
                error_box["error"] = exc
            finally:
                completed.set()

        worker = threading.Thread(target=_collect, daemon=True)
        worker.start()
        while True:
            if completed.wait(timeout=0.2):
                break
            if _is_cancel_requested(cancel_event):
                cancel = getattr(result_or_executor, "cancel", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:  # pragma: no cover - external runtime behavior
                        logger.exception("Failed to cancel RAGAS executor.")
                completed.wait(timeout=2.0)
                return _EvaluationResolution(result=None, canceled=True)

        error = error_box.get("error")
        if error is not None:
            raise error
        return _EvaluationResolution(result=result_box.get("value"), canceled=False)

    @staticmethod
    def _build_ragas_run_config(
        *,
        max_workers: int | None,
        timeout_seconds: float | None,
        max_retries: int | None,
    ) -> object | None:
        try:
            from ragas.run_config import RunConfig
        except Exception:
            return None

        kwargs: dict[str, object] = {}
        if max_workers is not None and max_workers > 0:
            kwargs["max_workers"] = int(max_workers)
        if timeout_seconds is not None and timeout_seconds > 0:
            kwargs["timeout"] = float(timeout_seconds)
        if max_retries is not None and max_retries >= 0:
            kwargs["max_retries"] = int(max_retries)
        try:
            return RunConfig(**kwargs)
        except TypeError:
            filtered = {
                key: value
                for key, value in kwargs.items()
                if _callable_accepts_keyword_argument(RunConfig, key)
            }
            try:
                return RunConfig(**filtered)
            except Exception:
                logger.exception("Failed to build RAGAS run config.")
                return None
        except Exception:
            logger.exception("Failed to build RAGAS run config.")
            return None

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

        if not _callable_accepts_keyword_argument(llm_factory, "provider"):
            logger.warning(
                "Installed ragas.llms.llm_factory does not support 'provider'. "
                "Skipping custom Gemini LLM to avoid incompatible client API."
            )
            return None

        if not _callable_accepts_keyword_argument(llm_factory, "client"):
            logger.warning(
                "Installed ragas.llms.llm_factory does not support 'client'. "
                "Skipping custom Gemini LLM because this ragas version requires"
                " provider-specific client wiring."
            )
            return None

        try:
            client = genai.Client(api_key=self._gemini_api_key)
        except Exception:
            logger.exception("Failed to initialize Gemini client for RAGAS LLM.")
            return None

        llm_kwargs: dict[str, object] = {"provider": "google", "client": client}

        try:
            return llm_factory(self._ragas_gemini_model, **llm_kwargs)
        except TypeError:
            try:
                return llm_factory(self._ragas_gemini_model, client=client)
            except Exception:
                logger.exception("Failed to build custom RAGAS LLM with Gemini provider.")
                return None
        except Exception:
            logger.exception("Failed to build custom RAGAS LLM with Gemini provider.")
            return None

    def _build_ragas_embeddings(self):
        if not self._gemini_api_key:
            return None
        try:
            from google import genai
        except ImportError:
            logger.info(
                "google-genai is not available. Running ragas without custom embeddings."
            )
            return None

        try:
            from ragas.embeddings.google_provider import GoogleEmbeddings
        except ImportError:
            logger.info(
                "ragas.embeddings is not available. Running ragas without custom embeddings."
            )
            return None

        try:
            client = genai.Client(api_key=self._gemini_api_key)
        except Exception:
            logger.exception("Failed to initialize Gemini client for RAGAS embeddings.")
            return None

        try:
            embeddings = GoogleEmbeddings(client=client)
        except Exception:
            logger.exception("Failed to build RAGAS embeddings with Gemini client.")
            return None
        return _RateLimitedEmbeddingsAdapter(
            _as_legacy_embeddings(embeddings),
            max_requests_per_minute=self._ragas_gemini_embedding_requests_per_minute,
            limiter_name=ragas_embedding_rate_limiter_name(),
        )

    @staticmethod
    def _is_instructor_ragas_llm(llm: object) -> bool:
        if llm is None:
            return False
        try:
            from ragas.llms.base import InstructorBaseRagasLLM
        except Exception:
            return False
        return isinstance(llm, InstructorBaseRagasLLM)

    @classmethod
    def _force_answer_relevancy_strictness_one(
        cls,
        *,
        metric_pairs: list[tuple[str, object]],
        llm: object,
    ) -> bool:
        # Instructor-based LLMs in ragas 0.4.x return a single generation for
        # generate_multiple(), so force strictness=1 to avoid warning spam.
        if not cls._is_instructor_ragas_llm(llm):
            return False
        for metric_name, metric in metric_pairs:
            if metric_name != "answer_relevancy":
                continue
            strictness = getattr(metric, "strictness", None)
            try:
                strictness_value = int(strictness)
            except Exception:
                return False
            if strictness_value <= 1:
                return False
            try:
                setattr(metric, "strictness", 1)
            except Exception:
                logger.exception(
                    "Failed to force answer_relevancy.strictness=1 for Instructor LLM."
                )
                return False
            logger.info(
                "Forced answer_relevancy.strictness from %d to 1 because "
                "Instructor-based RAGAS LLM supports only single generation.",
                strictness_value,
            )
            return True
        return False

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

    @staticmethod
    def _extract_summary_metrics_from_executor_results(
        result: object,
        *,
        metric_names: list[str],
        row_count: int,
    ) -> dict[str, float]:
        if not isinstance(result, list):
            return {}
        if row_count <= 0 or not metric_names:
            return {}

        metric_count = len(metric_names)
        sums: dict[str, float] = {name: 0.0 for name in metric_names}
        counts: dict[str, int] = {name: 0 for name in metric_names}

        for row_index in range(row_count):
            base_index = row_index * metric_count
            for metric_index, metric_name in enumerate(metric_names):
                value_index = base_index + metric_index
                if value_index >= len(result):
                    break
                value = result[value_index]
                if isinstance(value, bool):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    continue
                sums[metric_name] += numeric
                counts[metric_name] += 1

        return {
            metric_name: sums[metric_name] / float(counts[metric_name])
            for metric_name in metric_names
            if counts.get(metric_name, 0) > 0
        }


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
    if not records:
        return []
    size = int(batch_size or 0)
    if size <= 0:
        return [records]
    return [records[i : i + size] for i in range(0, len(records), size)]


def _answer_cache_key(question: str) -> str:
    return hashlib.sha256((question or "").encode("utf-8")).hexdigest()


def _normalize_contexts(raw_contexts: object) -> list[str]:
    if not isinstance(raw_contexts, list):
        return []
    contexts: list[str] = []
    for value in raw_contexts:
        if isinstance(value, dict):
            text = str(value.get("text") or value.get("quote") or "").strip()
        else:
            text = str(value or "").strip()
        if text:
            contexts.append(text)
    return contexts


def _normalize_sources(raw_sources: object) -> list[dict[str, object]]:
    if not isinstance(raw_sources, list):
        return []
    sources: list[dict[str, object]] = []
    for value in raw_sources:
        if isinstance(value, dict):
            source_id = str(
                value.get("source_id")
                or value.get("id")
                or value.get("source_item_id")
                or value.get("chunk_id")
                or ""
            ).strip()
            if not source_id:
                continue
            sources.append(
                {
                    "source_id": source_id,
                    "id": source_id,
                    "label": str(value.get("label") or value.get("title") or source_id),
                    "uri": str(value.get("uri") or value.get("url") or ""),
                    "source_kind": str(value.get("source_kind") or value.get("kind") or ""),
                }
            )
            continue
        source_id = str(value or "").strip()
        if source_id:
            sources.append({"source_id": source_id, "id": source_id, "label": source_id})
    return sources


def _normalize_metadata(raw_metadata: object) -> dict[str, object]:
    if not isinstance(raw_metadata, dict):
        return {}
    return {str(key): value for key, value in raw_metadata.items()}


def _sources_to_payload(sources: object) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not isinstance(sources, list):
        sources = list(sources) if isinstance(sources, tuple) else []
    for source in sources:
        source_id = str(getattr(source, "id", "") or "").strip()
        if not source_id:
            continue
        label = str(getattr(source, "label", "") or source_id)
        uri = str(getattr(source, "uri", "") or "")
        out.append(
            {
                "source_id": source_id,
                "id": source_id,
                "label": label,
                "uri": uri,
                "source_kind": _source_kind_from_id(source_id),
            }
        )
    return out


def _context_payloads(
    contexts: list[str],
    sources: list[dict[str, object]],
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for index, text in enumerate(contexts):
        source = sources[index] if index < len(sources) else {}
        source_id = str(source.get("source_id") or source.get("id") or f"context:{index}")
        payloads.append(
            {
                "source_id": source_id,
                "id": source_id,
                "source_kind": str(source.get("source_kind") or _source_kind_from_id(source_id)),
                "text": text,
            }
        )
    return payloads


def _retrieval_trace_from_metadata(
    metadata: dict[str, object],
    sources: list[dict[str, object]],
) -> list[dict[str, object]]:
    raw_trace = metadata.get("retrieval_trace")
    if isinstance(raw_trace, list):
        return _normalize_sources(raw_trace)
    return list(sources)


def _source_kind_from_id(source_id: str) -> str:
    if ":" in source_id:
        return source_id.split(":", 1)[0]
    return ""


def _is_cancel_requested(cancel_event: threading.Event | None) -> bool:
    if cancel_event is None:
        return False
    return bool(cancel_event.is_set())


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


def _callable_accepts_keyword_argument(func: object, keyword: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        # If the signature cannot be inspected, prefer optimistic behavior.
        return True
    if keyword in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _as_legacy_embeddings(embeddings: object) -> object:
    if callable(getattr(embeddings, "embed_query", None)) and callable(
        getattr(embeddings, "embed_documents", None)
    ):
        return embeddings
    return _LegacyEmbeddingsAdapter(embeddings)


class _RateLimitedEmbeddingsAdapter:
    def __init__(
        self,
        embeddings: object,
        *,
        max_requests_per_minute: int,
        limiter_name: str,
    ) -> None:
        self._embeddings = embeddings
        self._max_requests_per_minute = max(0, int(max_requests_per_minute))
        self._limiter_name = (limiter_name or "").strip()

    def __getattr__(self, item: str):
        return getattr(self._embeddings, item)

    def _wait_for_slot(self) -> None:
        wait_for_gemini_rate_limit(
            max_requests_per_minute=self._max_requests_per_minute,
            limiter_name=self._limiter_name,
        )

    def set_run_config(self, run_config: object) -> None:
        setter = getattr(self._embeddings, "set_run_config", None)
        if callable(setter):
            setter(run_config)

    def embed_query(self, text: str) -> list[float]:
        self._wait_for_slot()
        embed_query = getattr(self._embeddings, "embed_query", None)
        if callable(embed_query):
            return list(embed_query(text))
        embed_text = getattr(self._embeddings, "embed_text", None)
        if callable(embed_text):
            return list(embed_text(text))
        raise AttributeError("Underlying embeddings object has no query embedding method.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._wait_for_slot()
        embed_documents = getattr(self._embeddings, "embed_documents", None)
        if callable(embed_documents):
            return [list(v) for v in embed_documents(texts)]
        embed_texts = getattr(self._embeddings, "embed_texts", None)
        if callable(embed_texts):
            return [list(v) for v in embed_texts(texts)]
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        self._wait_for_slot()
        aembed_query = getattr(self._embeddings, "aembed_query", None)
        if callable(aembed_query):
            return list(await aembed_query(text))
        aembed_text = getattr(self._embeddings, "aembed_text", None)
        if callable(aembed_text):
            return list(await aembed_text(text))
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._wait_for_slot()
        aembed_documents = getattr(self._embeddings, "aembed_documents", None)
        if callable(aembed_documents):
            return [list(v) for v in await aembed_documents(texts)]
        aembed_texts = getattr(self._embeddings, "aembed_texts", None)
        if callable(aembed_texts):
            return [list(v) for v in await aembed_texts(texts)]
        return self.embed_documents(texts)


class _LegacyEmbeddingsAdapter:
    def __init__(self, embeddings: object) -> None:
        self._embeddings = embeddings

    def __getattr__(self, item: str):
        return getattr(self._embeddings, item)

    def set_run_config(self, run_config: object) -> None:
        setter = getattr(self._embeddings, "set_run_config", None)
        if callable(setter):
            setter(run_config)

    def embed_query(self, text: str) -> list[float]:
        embed_query = getattr(self._embeddings, "embed_query", None)
        if callable(embed_query):
            return list(embed_query(text))
        embed_text = getattr(self._embeddings, "embed_text", None)
        if callable(embed_text):
            return list(embed_text(text))
        raise AttributeError("Underlying embeddings object has no query embedding method.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embed_documents = getattr(self._embeddings, "embed_documents", None)
        if callable(embed_documents):
            return [list(v) for v in embed_documents(texts)]
        embed_texts = getattr(self._embeddings, "embed_texts", None)
        if callable(embed_texts):
            return [list(v) for v in embed_texts(texts)]
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        aembed_query = getattr(self._embeddings, "aembed_query", None)
        if callable(aembed_query):
            return list(await aembed_query(text))
        aembed_text = getattr(self._embeddings, "aembed_text", None)
        if callable(aembed_text):
            return list(await aembed_text(text))
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        aembed_documents = getattr(self._embeddings, "aembed_documents", None)
        if callable(aembed_documents):
            return [list(v) for v in await aembed_documents(texts)]
        aembed_texts = getattr(self._embeddings, "aembed_texts", None)
        if callable(aembed_texts):
            return [list(v) for v in await aembed_texts(texts)]
        return self.embed_documents(texts)
