from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

from kumc_agent.domain.models.agentic import AgentRun, AgentStep
from kumc_agent.infra.database.postgres import PostgresClient


class AgentTraceRepository(Protocol):
    def save_run(self, run: AgentRun) -> AgentRun:
        ...

    def save_step(self, step: AgentStep) -> AgentStep:
        ...


@dataclass(frozen=True)
class FileAgentTraceRepository:
    root_dir: Path

    def save_run(self, run: AgentRun) -> AgentRun:
        stored = replace(
            run,
            created_at=run.created_at or datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        _append_jsonl(self.root_dir / "agent_runs.jsonl", _run_payload(stored))
        return stored

    def save_step(self, step: AgentStep) -> AgentStep:
        stored = replace(step, created_at=step.created_at or datetime.now(UTC))
        _append_jsonl(self.root_dir / "agent_steps.jsonl", _step_payload(stored))
        return stored


@dataclass(frozen=True)
class PostgresAgentTraceRepository:
    postgres: PostgresClient

    def save_run(self, run: AgentRun) -> AgentRun:
        stored = replace(
            run,
            created_at=run.created_at or datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        payload = _run_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into agent_runs (
                      id, query, status, access_context, budget, citations,
                      answer, confidence, metadata, created_at, updated_at
                    )
                    values (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb, %s, %s)
                    on conflict (id) do update set
                      status = excluded.status,
                      citations = excluded.citations,
                      answer = excluded.answer,
                      confidence = excluded.confidence,
                      metadata = excluded.metadata,
                      updated_at = excluded.updated_at
                    """,
                    (
                        payload["id"],
                        payload["query"],
                        payload["status"],
                        json.dumps(payload["access"], ensure_ascii=False, default=str),
                        json.dumps(payload["budget"], ensure_ascii=False, default=str),
                        json.dumps(payload["citations"], ensure_ascii=False, default=str),
                        payload["answer"],
                        payload["confidence"],
                        json.dumps(payload["metadata"], ensure_ascii=False, default=str),
                        payload["created_at"],
                        payload["updated_at"],
                    ),
                )
            conn.commit()
        return stored

    def save_step(self, step: AgentStep) -> AgentStep:
        stored = replace(step, created_at=step.created_at or datetime.now(UTC))
        payload = _step_payload(stored)
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into agent_steps (
                      id, run_id, state, input_payload, output_payload,
                      status, cost_usd, created_at
                    )
                    values (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
                    on conflict (id) do nothing
                    """,
                    (
                        payload["id"],
                        payload["run_id"],
                        payload["state"],
                        json.dumps(payload["input"], ensure_ascii=False, default=str),
                        json.dumps(payload["output"], ensure_ascii=False, default=str),
                        payload["status"],
                        payload["cost_usd"],
                        payload["created_at"],
                    ),
                )
            conn.commit()
        return stored


def build_agent_trace_repository(
    *,
    postgres: PostgresClient,
    fallback_dir: Path,
) -> AgentTraceRepository:
    if postgres.is_configured():
        return PostgresAgentTraceRepository(postgres=postgres)
    return FileAgentTraceRepository(root_dir=fallback_dir)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fw:
        fw.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _run_payload(run: AgentRun) -> dict[str, object]:
    return {
        "id": run.id,
        "query": run.query,
        "status": run.status,
        "access": asdict(run.access),
        "budget": asdict(run.budget),
        "citations": [citation.__dict__ for citation in run.citations],
        "answer": run.answer,
        "confidence": run.confidence,
        "metadata": dict(run.metadata),
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def _step_payload(step: AgentStep) -> dict[str, object]:
    return {
        "id": step.id,
        "run_id": step.run_id,
        "state": step.state,
        "input": dict(step.input),
        "output": dict(step.output),
        "status": step.status,
        "cost_usd": step.cost_usd,
        "created_at": step.created_at,
    }
