from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any
from uuid import uuid4

from kumc_agent.domain.models.operations import EvalRun
from kumc_agent.infra.operations.repository import OperationsRepository
from kumc_agent.usecases.eval.adapters import (
    ContractEvalAdapter,
    EvalAdapter,
    build_default_adapter_registry,
)
from kumc_agent.usecases.eval.assertions import AssertionEngine
from kumc_agent.usecases.eval.dataset import load_eval_set
from kumc_agent.usecases.eval.ragas import EvaluateRagasUsecase
from kumc_agent.usecases.eval.safety import SafetyAssertionEngine, sanitize_report_payload
from kumc_agent.usecases.eval.schema import (
    AssertionOutcome,
    EvalBatchResult,
    EvalCaseResult,
    EvalRunResult,
    utc_now_iso,
)


@dataclass(frozen=True)
class EvaluateRequest:
    target: str
    suite: str = "smoke"
    eval_set_path: Path | None = None
    limit: int | None = None
    mode: str = "deterministic"
    result_path: Path | None = None
    cancel_event: Any = None
    fail_on_critical: bool = True
    safety_zero_tolerance: bool | None = None
    ragas_batch_size: int | None = None
    ragas_max_workers: int | None = None
    ragas_timeout_seconds: float | None = None
    ragas_max_retries: int | None = None
    missing_eval_set_policy: str | None = None
    min_cases: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluateBatchRequest:
    suite: str
    mode: str
    targets: tuple[str, ...]
    result_path: Path | None = None
    limit: int | None = None
    fail_on_critical: bool = True
    missing_eval_set_policy: str = "fail"
    min_cases: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalRunner:
    def __init__(
        self,
        *,
        eval_sets_dir: Path,
        results_dir: Path,
        operations_repository: OperationsRepository | None = None,
        ragas_usecase: EvaluateRagasUsecase | None = None,
        adapter_registry: dict[str, EvalAdapter] | None = None,
        thresholds: dict[str, Any] | None = None,
        safety_zero_tolerance: bool = True,
    ) -> None:
        self._eval_sets_dir = eval_sets_dir
        self._results_dir = results_dir
        self._operations_repository = operations_repository
        self._adapters = adapter_registry or build_default_adapter_registry(
            ragas_usecase=ragas_usecase,
        )
        self._thresholds = thresholds or {}
        self._safety_zero_tolerance = bool(safety_zero_tolerance)
        self._assertions = AssertionEngine()

    def execute(self, request: EvaluateRequest) -> EvalRunResult:
        run_id = f"eval-{utc_now_iso().replace(':', '').replace('+', 'Z')}-{uuid4().hex[:8]}"
        eval_set_id = f"{request.target}:{request.suite}"
        eval_set_path = request.eval_set_path or (
            self._eval_sets_dir / request.target / f"{request.suite}.jsonl"
        )
        zero_tolerance = (
            self._safety_zero_tolerance
            if request.safety_zero_tolerance is None
            else bool(request.safety_zero_tolerance)
        )
        safety = SafetyAssertionEngine(zero_tolerance=zero_tolerance)

        if _is_cancel_requested(request.cancel_event):
            return self._finalize(
                run_id=run_id,
                eval_set_id=eval_set_id,
                target=request.target,
                suite=request.suite,
                status="canceled",
                case_results=[],
                request=request,
                metadata={
                    "eval_set_path": str(eval_set_path),
                    "canceled_before_start": True,
                },
            )

        missing_policy = _resolve_missing_eval_set_policy(request)
        if eval_set_path.exists():
            cases = load_eval_set(
                eval_set_path,
                target=request.target,
                suite=request.suite,
                limit=request.limit,
            )
            if len(cases) < max(0, int(request.min_cases)):
                return self._finalize(
                    run_id=run_id,
                    eval_set_id=eval_set_id,
                    target=request.target,
                    suite=request.suite,
                    status="failed",
                    case_results=[
                        EvalCaseResult(
                            case_id="__eval_set_min_cases__",
                            target=request.target,
                            suite=request.suite,
                            status="failed",
                            passed=False,
                            latency_ms=0.0,
                            severity="critical",
                            failure_reason=(
                                f"EvalSet has {len(cases)} cases; "
                                f"minimum required is {request.min_cases}"
                            ),
                        )
                    ],
                    request=request,
                    metadata={
                        "eval_set_path": str(eval_set_path),
                        "empty_eval_set": not cases,
                        "mode": request.mode,
                        "safety_zero_tolerance": zero_tolerance,
                        "min_cases": request.min_cases,
                    },
                )
        else:
            if missing_policy == "fail":
                return self._finalize(
                    run_id=run_id,
                    eval_set_id=eval_set_id,
                    target=request.target,
                    suite=request.suite,
                    status="failed",
                    case_results=[
                        EvalCaseResult(
                            case_id="__eval_set_missing__",
                            target=request.target,
                            suite=request.suite,
                            status="failed",
                            passed=False,
                            latency_ms=0.0,
                            severity="critical",
                            failure_reason=f"EvalSet not found: {eval_set_path}",
                        )
                    ],
                    request=request,
                    metadata={
                        "eval_set_path": str(eval_set_path),
                        "empty_eval_set": True,
                        "mode": request.mode,
                        "safety_zero_tolerance": zero_tolerance,
                        "missing_eval_set_policy": missing_policy,
                    },
                )
            cases = []

        case_results: list[EvalCaseResult] = []
        for case in cases:
            if _is_cancel_requested(request.cancel_event):
                case_results.append(
                    EvalCaseResult(
                        case_id=case.id,
                        target=case.target,
                        suite=case.suite,
                        status="canceled",
                        passed=False,
                        latency_ms=0.0,
                        severity=case.severity,
                        failure_reason="canceled",
                    )
                )
                break
            started = time.perf_counter()
            adapter = self._adapters.get(case.target) or ContractEvalAdapter(target=case.target)
            try:
                adapter_result = adapter.run_case(case=case, request=request)
                actual = adapter_result.actual
                assertion_results = self._assertions.evaluate(case=case, actual=actual)
                safety_result = safety.evaluate(case=case, actual=actual)
                passed = _case_passed(
                    assertion_results=assertion_results,
                    safety_result=safety_result,
                    zero_tolerance=zero_tolerance,
                )
                status = "passed" if passed else "failed"
                failure_reason = _failure_reason(assertion_results, safety_result)
                latency_ms = (time.perf_counter() - started) * 1000.0
                case_results.append(
                    EvalCaseResult(
                        case_id=case.id,
                        target=case.target,
                        suite=case.suite,
                        status=status,
                        passed=passed,
                        latency_ms=latency_ms,
                        severity=case.severity,
                        assertion_results=assertion_results,
                        metrics=adapter_result.metrics,
                        safety=sanitize_report_payload(safety_result),
                        failure_reason=failure_reason,
                        output=sanitize_report_payload(actual),
                        metadata=sanitize_report_payload(
                            {
                                "adapter_status": adapter_result.status,
                                "adapter": adapter_result.metadata,
                                "case": case.metadata,
                            }
                        ),
                    )
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000.0
                case_results.append(
                    EvalCaseResult(
                        case_id=case.id,
                        target=case.target,
                        suite=case.suite,
                        status="failed",
                        passed=False,
                        latency_ms=latency_ms,
                        severity=case.severity,
                        failure_reason=f"adapter error: {exc}",
                        metadata={"error_type": type(exc).__name__},
                    )
                )

        return self._finalize(
            run_id=run_id,
            eval_set_id=eval_set_id,
            target=request.target,
            suite=request.suite,
            status=_run_status(case_results, request=request, thresholds=self._thresholds),
            case_results=case_results,
            request=request,
            metadata={
                "eval_set_path": str(eval_set_path),
                "empty_eval_set": not cases,
                "mode": request.mode,
                "safety_zero_tolerance": zero_tolerance,
            },
        )

    def execute_batch(self, request: EvaluateBatchRequest) -> EvalBatchResult:
        batch_id = f"eval-batch-{utc_now_iso().replace(':', '').replace('+', 'Z')}-{uuid4().hex[:8]}"
        runs: list[EvalRunResult] = []
        for target in request.targets:
            runs.append(
                self.execute(
                    EvaluateRequest(
                        target=target,
                        suite=request.suite,
                        limit=request.limit,
                        mode=request.mode,
                        fail_on_critical=request.fail_on_critical,
                        missing_eval_set_policy=request.missing_eval_set_policy,
                        min_cases=request.min_cases,
                    )
                )
            )

        total = sum(run.total for run in runs)
        passed = sum(run.passed for run in runs)
        failed = sum(run.failed for run in runs)
        failures = tuple(
            {
                **failure,
                "run_id": run.run_id,
                "eval_set_id": run.eval_set_id,
            }
            for run in runs
            for failure in run.failures
        )
        status = _batch_status(runs)
        metrics = _aggregate_batch_metrics(runs)
        result_path = request.result_path or (self._results_dir / f"{batch_id}.json")
        metadata = sanitize_report_payload(
            {
                **request.metadata,
                "artifact_path": str(result_path),
                "targets": list(request.targets),
                "suite": request.suite,
                "mode": request.mode,
                "missing_eval_set_policy": request.missing_eval_set_policy,
                "min_cases": request.min_cases,
            }
        )
        result = EvalBatchResult(
            run_id=batch_id,
            suite=request.suite,
            mode=request.mode,
            status=status,
            total=total,
            passed=passed,
            failed=failed,
            metrics=metrics,
            failures=failures,
            runs=tuple(runs),
            metadata=dict(metadata),
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result.to_artifact_payload(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return result

    def _finalize(
        self,
        *,
        run_id: str,
        eval_set_id: str,
        target: str,
        suite: str,
        status: str,
        case_results: list[EvalCaseResult],
        request: EvaluateRequest,
        metadata: dict[str, Any],
    ) -> EvalRunResult:
        metrics = MetricsAggregator().aggregate(case_results)
        failures = tuple(_failure_summary(result) for result in case_results if not result.passed)
        result_path = request.result_path or (self._results_dir / f"{run_id}.json")
        result_metadata = sanitize_report_payload(
            {
                **metadata,
                "artifact_path": str(result_path),
                "request": {
                    "target": request.target,
                    "suite": request.suite,
                    "limit": request.limit,
                    "mode": request.mode,
                    "fail_on_critical": request.fail_on_critical,
                },
            }
        )
        result = EvalRunResult(
            run_id=run_id,
            eval_set_id=eval_set_id,
            target=target,
            suite=suite,
            status=status,
            total=len(case_results),
            passed=sum(1 for item in case_results if item.passed),
            failed=sum(1 for item in case_results if not item.passed),
            metrics=metrics,
            failures=failures,
            case_results=tuple(case_results),
            metadata=dict(result_metadata),
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result.to_artifact_payload(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if self._operations_repository is not None:
            self._operations_repository.save_eval_run(
                EvalRun(
                    id=run_id,
                    eval_set_id=eval_set_id,
                    status=status,
                    metrics=metrics,
                    metadata=result.metadata,
                )
            )
        return result


class MetricsAggregator:
    def aggregate(self, results: list[EvalCaseResult]) -> dict[str, Any]:
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        failed = total - passed
        latencies = sorted(result.latency_ms for result in results)
        severity_failures: dict[str, int] = {}
        assertion_counts: dict[str, int] = {}
        assertion_passed: dict[str, int] = {}
        scored_assertions: dict[str, list[float]] = {}
        safety_counts = {
            "acl_violation_count": 0,
            "secret_leak_count": 0,
            "side_effect_violation_count": 0,
            "arbitrary_shell_violation_count": 0,
            "metadata_policy_violation_count": 0,
        }
        estimated_cost = 0.0
        adapter_metric_values: dict[str, list[float]] = {}
        for result in results:
            if not result.passed:
                severity_failures[result.severity] = severity_failures.get(result.severity, 0) + 1
            for assertion in result.assertion_results:
                assertion_counts[assertion.type] = assertion_counts.get(assertion.type, 0) + 1
                if assertion.passed:
                    assertion_passed[assertion.type] = assertion_passed.get(assertion.type, 0) + 1
                score = assertion.metadata.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    scored_assertions.setdefault(assertion.type, []).append(float(score))
            for key in safety_counts:
                value = result.safety.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    safety_counts[key] += int(value)
            cost = result.metrics.get("estimated_cost")
            if isinstance(cost, (int, float)) and math.isfinite(float(cost)):
                estimated_cost += float(cost)
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    adapter_metric_values.setdefault(key, []).append(float(value))
        assertion_rates = {
            f"{name}_pass_rate": assertion_passed.get(name, 0) / count
            for name, count in assertion_counts.items()
            if count > 0
        }
        scored_averages = {
            f"{name}_score": sum(values) / len(values)
            for name, values in scored_assertions.items()
            if values
        }
        adapter_metrics = _aggregate_numeric_metrics(adapter_metric_values)
        metrics = {
            "case_pass_rate": (passed / total) if total else 1.0,
            "total": total,
            "passed": passed,
            "failed": failed,
            "critical_failure_count": severity_failures.get("critical", 0)
            + severity_failures.get("blocker", 0),
            "severity_failures": severity_failures,
            "latency_p50_ms": median(latencies) if latencies else 0.0,
            "latency_p95_ms": _percentile(latencies, 0.95),
            "estimated_cost": estimated_cost,
            **safety_counts,
            **assertion_rates,
            **scored_averages,
            **adapter_metrics,
        }
        metrics.setdefault("schema_valid_rate", metrics.get("schema_has_keys_pass_rate", 1.0))
        metrics.setdefault("metadata_policy_pass_rate", metrics.get("metadata_policy_pass_rate", 1.0))
        metrics.setdefault("approval_boundary_pass_rate", metrics.get("approval_required_pass_rate", 1.0))
        metrics.setdefault("side_effect_boundary_pass_rate", metrics.get("no_side_effect_pass_rate", 1.0))
        metrics.setdefault("top_k_hit_rate", metrics.get("top_k_contains_pass_rate", 1.0))
        metrics.setdefault("routing_accuracy", metrics.get("route_equals_pass_rate", 1.0))
        metrics.setdefault("citation_recall", metrics.get("citation_source_recall_score", 1.0))
        metrics.setdefault("retrieval_recall", metrics.get("retrieval_recall_score", 1.0))
        return metrics


def _case_passed(
    *,
    assertion_results: tuple[AssertionOutcome, ...],
    safety_result: dict[str, Any],
    zero_tolerance: bool,
) -> bool:
    if any(not result.passed for result in assertion_results):
        return False
    if zero_tolerance and bool(safety_result.get("zero_tolerance_failed")):
        return False
    return True


def _failure_reason(
    assertion_results: tuple[AssertionOutcome, ...],
    safety_result: dict[str, Any],
) -> str:
    for result in assertion_results:
        if not result.passed:
            return f"{result.type}: {result.message}"
    if bool(safety_result.get("zero_tolerance_failed")):
        return "safety zero tolerance violation"
    return ""


def _failure_summary(result: EvalCaseResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "target": result.target,
        "severity": result.severity,
        "reason": result.failure_reason[:300],
    }


def _run_status(
    results: list[EvalCaseResult],
    *,
    request: EvaluateRequest,
    thresholds: dict[str, Any],
) -> str:
    if any(result.status == "canceled" for result in results):
        return "canceled"
    if not results:
        return "succeeded"
    metrics = MetricsAggregator().aggregate(results)
    min_pass_rate = _threshold_value(
        thresholds,
        target=request.target,
        suite=request.suite,
        key="min_pass_rate",
        default=1.0,
    )
    if float(metrics["case_pass_rate"]) < float(min_pass_rate):
        return "failed"
    if request.fail_on_critical and int(metrics["critical_failure_count"]) > 0:
        return "failed"
    if any(not result.passed for result in results):
        return "failed"
    if any(bool(result.metadata.get("adapter", {}).get("degraded")) for result in results):
        if request.mode == "full":
            return "failed"
        return "degraded"
    return "succeeded"


def _resolve_missing_eval_set_policy(request: EvaluateRequest) -> str:
    policy = str(request.missing_eval_set_policy or "").strip().lower()
    if policy in {"fail", "skip", "succeed"}:
        return policy
    if request.mode in {"full", "safety", "sampled"}:
        return "fail"
    return "succeed"


def _batch_status(runs: list[EvalRunResult]) -> str:
    if any(run.status == "failed" for run in runs):
        return "failed"
    if any(run.status == "canceled" for run in runs):
        return "canceled"
    if any(run.status == "degraded" for run in runs):
        return "degraded"
    return "succeeded"


def _aggregate_batch_metrics(runs: list[EvalRunResult]) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    for run in runs:
        for key, value in run.metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(key, []).append(float(value))
    metrics = _aggregate_numeric_metrics(values)
    total = sum(run.total for run in runs)
    passed = sum(run.passed for run in runs)
    failed = sum(run.failed for run in runs)
    metrics.update(
        {
            "case_pass_rate": (passed / total) if total else 1.0,
            "total": total,
            "passed": passed,
            "failed": failed,
            "run_count": len(runs),
            "failed_run_count": sum(1 for run in runs if run.status == "failed"),
        }
    )
    return metrics


def _aggregate_numeric_metrics(values: dict[str, list[float]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, items in values.items():
        finite = [value for value in items if math.isfinite(value)]
        if not finite:
            continue
        if key.endswith("_count") or key in {"total", "passed", "failed"}:
            metrics[key] = sum(finite)
        else:
            metrics[key] = sum(finite) / len(finite)
    return metrics


def _threshold_value(
    thresholds: dict[str, Any],
    *,
    target: str,
    suite: str,
    key: str,
    default: Any,
) -> Any:
    for name in (f"{target}:{suite}", target, "default"):
        value = thresholds.get(name)
        if isinstance(value, dict) and key in value:
            return value[key]
    return default


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * percentile))))
    return values[index]


def _is_cancel_requested(cancel_event: Any) -> bool:
    if cancel_event is None:
        return False
    is_set = getattr(cancel_event, "is_set", None)
    return bool(is_set()) if callable(is_set) else False
