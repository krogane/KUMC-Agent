from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re

from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.domain.ports.prompts import PromptRepositoryPort

logger = logging.getLogger(__name__)

_LOCAL_REFUSAL_RE = re.compile(
    r"(api[_ -]?key|password|passwd|secret|token|住所|電話番号|口座|認証情報)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerFilterResult:
    action: str
    reason_code: str = ""
    raw: str = ""
    fallback: bool = False

    @property
    def refused(self) -> bool:
        return self.action == "refuse"


class AnswerFilterComponent:
    def __init__(
        self,
        *,
        llm: LLMPort,
        prompts: PromptRepositoryPort,
        filter_prompt_name: str = "answer_filter",
        refusal_prompt_name: str = "answer_refusal",
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        max_retries: int = 1,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._filter_prompt_name = str(filter_prompt_name or "answer_filter").strip()
        self._refusal_prompt_name = str(refusal_prompt_name or "answer_refusal").strip()
        self._temperature = float(temperature)
        self._max_output_tokens = max(1, int(max_output_tokens))
        self._max_retries = max(0, int(max_retries))

    def evaluate(self, *, answer_text: str) -> AnswerFilterResult:
        text = str(answer_text or "").strip()
        if not text:
            return AnswerFilterResult(action="allow")
        if _LOCAL_REFUSAL_RE.search(text):
            local = self._evaluate_with_llm(answer_text=text)
            if local.action != "allow":
                return local
            return AnswerFilterResult(action="refuse", reason_code="local_sensitive_pattern")
        return self._evaluate_with_llm(answer_text=text)

    def generate_refusal(self, *, query: str) -> str:
        system_prompt = self._prompt(
            self._refusal_prompt_name,
            fallback=(
                "あなたは安全な回答拒否文を作るアシスタントです。"
                "入力質問だけを見て、機密情報や個人情報は出せないことを簡潔に伝えてください。"
            ),
        )
        user_prompt = json.dumps({"query": str(query or "").strip()}, ensure_ascii=False)
        try:
            raw = self._llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
            )
        except Exception:
            logger.exception("Refusal answer generation failed.")
            return "その内容には機密情報または個人情報が含まれる可能性があるため回答できません。"
        parsed = self._parse_refusal(raw)
        return parsed or "その内容には機密情報または個人情報が含まれる可能性があるため回答できません。"

    def _evaluate_with_llm(self, *, answer_text: str) -> AnswerFilterResult:
        system_prompt = self._prompt(
            self._filter_prompt_name,
            fallback=(
                "あなたは回答安全性フィルタです。生成済み回答だけを見て、"
                "機密情報、認証情報、個人情報、権限外資料の内容が含まれる場合はrefuse、"
                "問題なければallowをJSONで返してください。"
            ),
        )
        user_prompt = json.dumps(
            {
                "answer": answer_text,
                "output_format": {"action": "allow|refuse", "reason_code": "string"},
            },
            ensure_ascii=False,
        )
        last_raw = ""
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=self._temperature,
                    max_output_tokens=self._max_output_tokens,
                )
            except Exception:
                logger.exception("Answer filter generation failed.")
                break
            last_raw = raw
            parsed = self._parse_filter_payload(raw)
            if parsed is not None:
                return AnswerFilterResult(
                    action=parsed[0],
                    reason_code=parsed[1],
                    raw=last_raw,
                )
            if attempt < self._max_retries:
                continue
        return AnswerFilterResult(
            action="refuse",
            reason_code="filter_fallback_refuse",
            raw=last_raw,
            fallback=True,
        )

    def _prompt(self, name: str, *, fallback: str) -> str:
        try:
            value = self._prompts.get(name)
        except Exception:
            value = ""
        return value or fallback

    @staticmethod
    def _parse_filter_payload(text: str) -> tuple[str, str] | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        cleaned = AnswerFilterComponent._strip_code_fence(raw)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        action = str(payload.get("action") or payload.get("decision") or "").strip().lower()
        if action not in {"allow", "refuse"}:
            return None
        reason_code = str(payload.get("reason_code") or "").strip()
        return action, reason_code

    @staticmethod
    def _parse_refusal(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        cleaned = AnswerFilterComponent._strip_code_fence(raw)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return raw
        if isinstance(payload, dict):
            return str(payload.get("answer") or payload.get("text") or "").strip()
        if isinstance(payload, str):
            return payload.strip()
        return ""

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = (text or "").strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) < 2 or not lines[-1].strip().startswith("```"):
            return stripped
        return "\n".join(lines[1:-1]).strip()
