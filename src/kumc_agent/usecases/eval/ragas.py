from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kumc_agent.usecases.chat.answer import ChatAnswerUsecase, ChatRequest


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


class EvaluateRagasUsecase:
    def __init__(self, *, chat_usecase: ChatAnswerUsecase) -> None:
        self._chat_usecase = chat_usecase

    def execute(self, request: EvaluateRagasRequest) -> RagasResult:
        items = self._load_items(request.eval_file)
        if request.limit and request.limit > 0:
            items = items[: request.limit]

        exact_count = 0
        overlap_scores: list[float] = []

        for item in items:
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            answer = self._chat_usecase.execute(ChatRequest(query=question)).text
            truths = self._ground_truths(item)
            if any(truth and truth in answer for truth in truths):
                exact_count += 1
            overlap_scores.append(self._token_overlap(answer, truths))

        total = len(items)
        exact_match = (exact_count / total) if total else 0.0
        token_overlap = (sum(overlap_scores) / total) if total else 0.0

        result = RagasResult(total=total, exact_match=exact_match, token_overlap=token_overlap)

        if request.result_path is not None:
            request.result_path.parent.mkdir(parents=True, exist_ok=True)
            request.result_path.write_text(
                json.dumps(
                    {
                        "total": result.total,
                        "exact_match": result.exact_match,
                        "token_overlap": result.token_overlap,
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
