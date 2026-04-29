from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


SEVERITIES = {"blocker", "critical", "major", "minor"}
RUN_STATUSES = {"running", "succeeded", "failed", "canceled", "degraded"}
CASE_STATUSES = {"passed", "failed", "canceled", "skipped", "pending"}


@dataclass(frozen=True)
class EvalAssertion:
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    severity: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EvalAssertion":
        assertion_type = str(payload.get("type") or "").strip()
        if not assertion_type:
            raise ValueError("assertion.type is required")
        severity = str(payload.get("severity") or "").strip()
        params = {
            key: value
            for key, value in payload.items()
            if key not in {"type", "severity"}
        }
        return cls(type=assertion_type, params=params, severity=severity)

    def to_payload(self) -> dict[str, Any]:
        payload = {"type": self.type, **self.params}
        if self.severity:
            payload["severity"] = self.severity
        return payload


@dataclass(frozen=True)
class EvalCase:
    id: str
    target: str
    suite: str
    input: dict[str, Any]
    expected: dict[str, Any] = field(default_factory=dict)
    assertions: tuple[EvalAssertion, ...] = tuple()
    tags: tuple[str, ...] = tuple()
    severity: str = "major"
    metadata: dict[str, Any] = field(default_factory=dict)

    def eval_set_id(self) -> str:
        return f"{self.target}:{self.suite}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "suite": self.suite,
            "input": self.input,
            "expected": self.expected,
            "assertions": [assertion.to_payload() for assertion in self.assertions],
            "tags": list(self.tags),
            "severity": self.severity,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AssertionOutcome:
    type: str
    passed: bool
    message: str = ""
    severity: str = "major"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    target: str
    suite: str
    status: str
    passed: bool
    latency_ms: float
    severity: str = "major"
    assertion_results: tuple[AssertionOutcome, ...] = tuple()
    metrics: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "target": self.target,
            "suite": self.suite,
            "status": self.status,
            "passed": self.passed,
            "latency_ms": self.latency_ms,
            "severity": self.severity,
            "assertions": [
                assertion.to_payload() for assertion in self.assertion_results
            ],
            "metrics": self.metrics,
            "safety": self.safety,
            "failure_reason": self.failure_reason,
            "output": self.output,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EvalRunResult:
    run_id: str
    eval_set_id: str
    target: str
    suite: str
    status: str
    total: int
    passed: int
    failed: int
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: tuple[dict[str, Any], ...] = tuple()
    case_results: tuple[EvalCaseResult, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "eval_set_id": self.eval_set_id,
            "target": self.target,
            "suite": self.suite,
            "status": self.status,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "metrics": self.metrics,
            "failures": list(self.failures),
            "metadata": self.metadata,
        }

    def to_artifact_payload(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload["cases"] = [case.to_payload() for case in self.case_results]
        return payload


@dataclass(frozen=True)
class EvalBatchResult:
    run_id: str
    suite: str
    mode: str
    status: str
    total: int
    passed: int
    failed: int
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: tuple[dict[str, Any], ...] = tuple()
    runs: tuple[EvalRunResult, ...] = tuple()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "suite": self.suite,
            "mode": self.mode,
            "status": self.status,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "metrics": self.metrics,
            "failures": list(self.failures),
            "metadata": self.metadata,
            "runs": [run.to_payload() for run in self.runs],
        }

    def to_artifact_payload(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload["runs"] = [run.to_artifact_payload() for run in self.runs]
        return payload


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
