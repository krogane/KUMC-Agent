from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

from kumc_agent.domain.models.agentic import AgentBudget, AgentRun, AgentStep
from kumc_agent.domain.models.retrieval import AccessContext, Citation
from kumc_agent.infra.database.postgres import PostgresClient


class AgentTraceRepository(Protocol):
    def save_run(self, run: AgentRun) -> AgentRun:
        ...

    def save_step(self, step: AgentStep) -> AgentStep:
        ...

    def get_run(self, run_id: str) -> AgentRun | None:
        ...

    def list_steps(self, run_id: str) -> list[AgentStep]:
        ...

    def latest_runs(self, *, limit: int = 20) -> list[AgentRun]:
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

    def get_run(self, run_id: str) -> AgentRun | None:
        latest: AgentRun | None = None
        for payload in _read_jsonl(self.root_dir / "agent_runs.jsonl"):
            if str(payload.get("id") or "") == run_id:
                latest = _run_from_payload(payload)
        return latest

    def list_steps(self, run_id: str) -> list[AgentStep]:
        return [
            _step_from_payload(payload)
            for payload in _read_jsonl(self.root_dir / "agent_steps.jsonl")
            if str(payload.get("run_id") or "") == run_id
        ]

    def latest_runs(self, *, limit: int = 20) -> list[AgentRun]:
        by_id: dict[str, AgentRun] = {}
        for payload in _read_jsonl(self.root_dir / "agent_runs.jsonl"):
            run = _run_from_payload(payload)
            by_id[run.id] = run
        return sorted(
            by_id.values(),
            key=lambda run: str(run.updated_at or run.created_at or ""),
            reverse=True,
        )[: max(1, int(limit))]


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

    def get_run(self, run_id: str) -> AgentRun | None:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, query, status, access_context, budget, citations,
                           answer, confidence, metadata, created_at, updated_at
                    from agent_runs
                    where id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
        return _run_from_row(row) if row else None

    def list_steps(self, run_id: str) -> list[AgentStep]:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, run_id, state, input_payload, output_payload,
                           status, cost_usd, created_at
                    from agent_steps
                    where run_id = %s
                    order by created_at asc
                    """,
                    (run_id,),
                )
                rows = cur.fetchall()
        return [_step_from_row(row) for row in rows]

    def latest_runs(self, *, limit: int = 20) -> list[AgentRun]:
        with self.postgres.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, query, status, access_context, budget, citations,
                           answer, confidence, metadata, created_at, updated_at
                    from agent_runs
                    order by created_at desc
                    limit %s
                    """,
                    (max(1, int(limit)),),
                )
                rows = cur.fetchall()
        return [_run_from_row(row) for row in rows]


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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payloads: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


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


def _run_from_payload(payload: dict[str, object]) -> AgentRun:
    access = payload.get("access") if isinstance(payload.get("access"), dict) else {}
    budget = payload.get("budget") if isinstance(payload.get("budget"), dict) else {}
    citations = payload.get("citations") if isinstance(payload.get("citations"), list) else []
    return AgentRun(
        id=str(payload.get("id") or ""),
        query=str(payload.get("query") or ""),
        status=str(payload.get("status") or ""),
        access=AccessContext(
            user_id=str(access.get("user_id") or ""),
            guild_id=str(access.get("guild_id") or ""),
            role_ids=tuple(str(item) for item in access.get("role_ids") or ()),
            is_admin=bool(access.get("is_admin")),
        ),
        budget=AgentBudget(
            **{
                key: value
                for key, value in budget.items()
                if key in AgentBudget.__dataclass_fields__
            }
        ),
        citations=tuple(_citation_from_payload(item) for item in citations if isinstance(item, dict)),
        answer=str(payload.get("answer") or ""),
        confidence=str(payload.get("confidence") or "low"),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
        created_at=payload.get("created_at"),  # type: ignore[arg-type]
        updated_at=payload.get("updated_at"),  # type: ignore[arg-type]
    )


def _step_from_payload(payload: dict[str, object]) -> AgentStep:
    return AgentStep(
        id=str(payload.get("id") or ""),
        run_id=str(payload.get("run_id") or ""),
        state=str(payload.get("state") or ""),
        input=dict(payload.get("input") or {}) if isinstance(payload.get("input"), dict) else {},
        output=dict(payload.get("output") or {}) if isinstance(payload.get("output"), dict) else {},
        status=str(payload.get("status") or "succeeded"),
        cost_usd=float(payload.get("cost_usd") or 0.0),
        created_at=payload.get("created_at"),  # type: ignore[arg-type]
    )


def _citation_from_payload(payload: dict[str, object]) -> Citation:
    return Citation(
        source_item_id=str(payload.get("source_item_id") or ""),
        chunk_id=str(payload.get("chunk_id") or ""),
        label=str(payload.get("label") or ""),
        url=str(payload.get("url") or ""),
        quote=str(payload.get("quote") or ""),
        score=payload.get("score") if isinstance(payload.get("score"), float) else None,
    )


def _run_from_row(row: object) -> AgentRun:
    (
        run_id,
        query,
        status,
        access,
        budget,
        citations,
        answer,
        confidence,
        metadata,
        created_at,
        updated_at,
    ) = row
    return _run_from_payload(
        {
            "id": run_id,
            "query": query,
            "status": status,
            "access": access or {},
            "budget": budget or {},
            "citations": citations or [],
            "answer": answer,
            "confidence": confidence,
            "metadata": metadata or {},
            "created_at": created_at,
            "updated_at": updated_at,
        }
    )


def _step_from_row(row: object) -> AgentStep:
    step_id, run_id, state, input_payload, output_payload, status, cost_usd, created_at = row
    return AgentStep(
        id=str(step_id),
        run_id=str(run_id),
        state=str(state),
        input=dict(input_payload or {}),
        output=dict(output_payload or {}),
        status=str(status),
        cost_usd=float(cost_usd or 0.0),
        created_at=created_at,
    )
