from __future__ import annotations

from dataclasses import replace
from time import monotonic
from uuid import uuid4

from kumc_agent.domain.models.agentic import (
    AgentBudget,
    AgentRun,
    AgentStep,
    AgenticSearchRequest,
    AgenticSearchResponse,
)
from kumc_agent.domain.models.retrieval import Citation, RetrievalQuery
from kumc_agent.infra.agentic.repository import AgentTraceRepository
from kumc_agent.utils.hashing import stable_hash


class AgenticSearchService:
    def __init__(
        self,
        *,
        ask_service: object,
        repository: AgentTraceRepository,
    ) -> None:
        self.ask_service = ask_service
        self.repository = repository

    def search(self, request: AgenticSearchRequest) -> AgenticSearchResponse:
        started = monotonic()
        run = AgentRun(
            id=str(uuid4()),
            query=request.query,
            status="running",
            access=request.access,
            budget=request.budget,
        )
        run = self.repository.save_run(run)
        steps: list[AgentStep] = []

        def add_step(
            state: str,
            *,
            input_payload: dict[str, object],
            output_payload: dict[str, object],
            status: str = "succeeded",
            cost_usd: float = 0.0,
        ) -> AgentStep:
            step = self.repository.save_step(
                AgentStep(
                    id=str(uuid4()),
                    run_id=run.id,
                    state=state,
                    input=input_payload,
                    output=output_payload,
                    status=status,
                    cost_usd=cost_usd,
                )
            )
            steps.append(step)
            return step

        subqueries = self._plan(request.query, request.budget)
        add_step(
            "PLAN",
            input_payload={"query": request.query},
            output_payload={"subqueries": subqueries, "success_criteria": _success_criteria(request.query)},
        )

        notes: list[str] = []
        citations: list[Citation] = []
        warnings: list[str] = []
        search_calls = 0
        total_cost = 0.0
        for subquery in subqueries:
            if not self._within_budget(
                request.budget,
                steps=len(steps) + 2,
                search_calls=search_calls + 1,
                cost_usd=total_cost + 0.01,
                elapsed=monotonic() - started,
            ):
                warnings.append("Agentic Search budget reached before all subqueries were searched.")
                break
            response = self.ask_service.ask(
                RetrievalQuery(
                    text=subquery,
                    source_filter=request.source_filter,
                    mode="search_only",
                    depth="normal",
                    access=request.access,
                )
            )
            search_calls += 1
            total_cost += 0.01
            add_step(
                "SEARCH",
                input_payload={"query": subquery, "source_filter": request.source_filter},
                output_payload={
                    "citation_count": len(response.citations),
                    "confidence": response.confidence,
                },
                cost_usd=0.01,
            )
            notes.append(self._note_from_response(subquery, response.detail_markdown or response.text))
            citations.extend(response.citations)
            add_step(
                "READ",
                input_payload={"query": subquery},
                output_payload={"note": notes[-1][:500]},
            )

        unique_citations = _unique_citations(citations)[: request.budget.max_read_chunks]
        missing = self._missing_info(request.query, unique_citations, notes, request.budget)
        verified = not missing
        add_step(
            "VERIFY",
            input_payload={"query": request.query, "citation_count": len(unique_citations)},
            output_payload={"verified": verified, "missing": missing},
            status="succeeded" if verified else "needs_more_evidence",
        )

        if verified:
            answer = self._answer(request.query, notes, unique_citations)
            confidence = "high" if len(unique_citations) >= 2 else "medium"
            status = "succeeded"
        else:
            answer = self._insufficient_answer(missing)
            confidence = "low"
            status = "insufficient_evidence"
        add_step(
            "ANSWER",
            input_payload={"query": request.query},
            output_payload={"confidence": confidence, "status": status},
        )

        final_run = replace(
            run,
            status=status,
            steps=tuple(steps),
            citations=tuple(unique_citations),
            answer=answer,
            confidence=confidence,
            metadata={
                "search_calls": search_calls,
                "cost_usd": round(total_cost, 4),
                "elapsed_seconds": round(monotonic() - started, 3),
            },
        )
        final_run = self.repository.save_run(final_run)
        detail = self._detail_markdown(final_run, notes, missing)
        return AgenticSearchResponse(
            text=answer,
            detail_markdown=detail,
            citations=tuple(unique_citations),
            confidence=confidence,
            run=final_run,
            warnings=tuple(warnings),
        )

    def _plan(self, query: str, budget: AgentBudget) -> list[str]:
        parts = [
            part.strip()
            for part in query.replace("？", "?").replace("。", "?").split("?")
            if part.strip()
        ]
        subqueries = [query.strip()]
        for part in parts:
            if part not in subqueries:
                subqueries.append(part)
        for prefix in ("根拠", "関連資料", "未決事項"):
            candidate = f"{prefix} {query}".strip()
            if candidate not in subqueries:
                subqueries.append(candidate)
        return subqueries[: max(1, budget.max_search_calls)]

    def _within_budget(
        self,
        budget: AgentBudget,
        *,
        steps: int,
        search_calls: int,
        cost_usd: float,
        elapsed: float,
    ) -> bool:
        return (
            steps <= budget.max_steps
            and search_calls <= budget.max_search_calls
            and cost_usd <= budget.max_cost_usd
            and elapsed <= budget.max_latency_seconds
        )

    def _note_from_response(self, query: str, text: str) -> str:
        lines = [line.strip("-* \n") for line in text.splitlines() if line.strip()]
        excerpt = " / ".join(lines[:4])
        return f"{query}: {excerpt[:1000]}"

    def _missing_info(
        self,
        query: str,
        citations: list[Citation],
        notes: list[str],
        budget: AgentBudget,
    ) -> list[str]:
        missing: list[str] = []
        if budget.require_citations and not citations:
            missing.append("引用可能な根拠")
        if not any(note.strip() for note in notes):
            missing.append("本文として読める検索結果")
        if len(query.strip()) < 3:
            missing.append("具体的な質問")
        return missing

    def _answer(self, query: str, notes: list[str], citations: list[Citation]) -> str:
        bullets = []
        for note in notes[:4]:
            if note:
                bullets.append(f"- {note}")
        source_line = f"根拠: {len(citations)}件"
        return "\n".join(["Agentic Search の結果です。", *bullets, source_line])

    def _insufficient_answer(self, missing: list[str]) -> str:
        return "\n".join(
            [
                "十分な根拠を見つけられませんでした。",
                "",
                "不足している情報:",
                *[f"- {item}" for item in missing],
            ]
        )

    def _detail_markdown(
        self,
        run: AgentRun,
        notes: list[str],
        missing: list[str],
    ) -> str:
        citation_lines = [
            f"- {citation.label or citation.chunk_id} {citation.url}".strip()
            for citation in run.citations
        ] or ["- No citations."]
        return "\n".join(
            [
                "# Agentic Search Trace",
                "",
                f"- run_id: `{run.id}`",
                f"- status: `{run.status}`",
                f"- confidence: `{run.confidence}`",
                f"- search_calls: {run.metadata.get('search_calls', 0)}",
                "",
                "## Notes",
                *([f"- {note}" for note in notes] or ["- No notes."]),
                "",
                "## Verification",
                *([f"- missing: {item}" for item in missing] or ["- verified"]),
                "",
                "## Citations",
                *citation_lines,
            ]
        )


def _success_criteria(query: str) -> list[str]:
    criteria = ["引用根拠がある", "質問に直接関係する"]
    if any(word in query for word in ("比較", "違い", "どちら")):
        criteria.append("比較対象が両方ある")
    if any(word in query for word in ("いつ", "日時", "期限")):
        criteria.append("日付または時刻がある")
    return criteria


def _unique_citations(citations: list[Citation]) -> list[Citation]:
    seen: set[str] = set()
    out: list[Citation] = []
    for citation in citations:
        key = stable_hash(f"{citation.source_item_id}:{citation.chunk_id}:{citation.url}")
        if key in seen:
            continue
        seen.add(key)
        out.append(citation)
    return out
