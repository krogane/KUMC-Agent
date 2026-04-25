from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import json
import logging
import re
import time

from kumc_agent.domain.models.entry_routing import EntryRoutingDecision
from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit

logger = logging.getLogger(__name__)

_ROUTE_VALUES = {"direct_rag", "openclaw"}
_RETRYABLE_GENERATION_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_GENERATION_ERROR_KEYWORDS = {
    "unavailable",
    "high demand",
    "rate limit",
    "resource exhausted",
    "overloaded",
    "timeout",
    "timed out",
    "connection reset",
}

_DEFAULT_PROMPT_NAME = "routing_openclaw_gate"
_DEFAULT_SYSTEM_PROMPT = (
    "あなたは、クエリ入口の経路判定エンジンです。JSONのみを返してください。\n"
    "以下の2つの route のどちらかを必ず選んでください。\n"
    "- direct_rag: サークル関連の事実照会、または資料名指定系の質問\n"
    "- openclaw: 複雑な質問、サークル関連以外の質問、ツール実行依頼、文章生成依頼\n\n"
    "出力形式:\n"
    '{"route":"direct_rag|openclaw","reason":"判定理由"}\n'
    "- reason は1文で簡潔に書く。"
)


class EntryQueryRouter:
    def __init__(
        self,
        *,
        provider: str,
        gemini_model: str,
        temperature: float,
        max_new_tokens: int,
        max_retries: int,
        gemini_api_key: str,
        gemini_requests_per_minute: int,
        prompt_name: str = _DEFAULT_PROMPT_NAME,
        log_enabled: bool = False,
    ) -> None:
        self._provider = str(provider or "").strip()
        self._gemini_model = str(gemini_model or "").strip()
        self._temperature = float(temperature)
        self._max_new_tokens = max(1, int(max_new_tokens))
        self._max_retries = max(0, int(max_retries))
        self._gemini_api_key = str(gemini_api_key or "").strip()
        self._gemini_requests_per_minute = max(0, int(gemini_requests_per_minute))
        self._prompt_name = str(prompt_name or "").strip() or _DEFAULT_PROMPT_NAME
        self._log_enabled = bool(log_enabled)
        self._prompt_cache: dict[str, str] = {}

    def decide(self, query: str) -> EntryRoutingDecision:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return EntryRoutingDecision(route="direct_rag", reason="empty_query")

        last_raw = ""
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._generate_payload(cleaned_query)
            except Exception as exc:
                if attempt >= self._max_retries or not self._is_retryable_generation_error(
                    exc
                ):
                    logger.exception("Entry route generation failed.")
                    break
                wait_seconds = self._retry_backoff_seconds(attempt=attempt)
                logger.warning(
                    "Entry route transient generation error. retry=%d/%d wait=%.2fs error=%s",
                    attempt + 1,
                    self._max_retries,
                    wait_seconds,
                    str(exc),
                )
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                continue

            last_raw = raw
            decision = self._parse_payload(raw)
            if decision is not None:
                if self._log_enabled:
                    logger.info("Entry route decision: %s", asdict(decision))
                return decision
            if attempt < self._max_retries:
                logger.info(
                    "Invalid entry routing output. retry=%d/%d",
                    attempt + 1,
                    self._max_retries,
                )
        return EntryRoutingDecision(
            route="openclaw",
            reason="fallback:classification_failed",
            payload={"raw": last_raw} if last_raw else {},
        )

    @property
    def model_label(self) -> str:
        provider = self._provider.strip().lower().replace(".", "_")
        if provider == "gemini":
            return f"gemini:{self._gemini_model}"
        return provider or "unknown"

    def _generate_payload(self, query: str) -> str:
        provider = self._provider.strip().lower().replace(".", "_")
        if provider == "gemini":
            return self._generate_payload_gemini(query)
        raise ValueError(f"Unsupported routing provider: {self._provider}")

    def _generate_payload_gemini(self, query: str) -> str:
        if not self._gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        if not self._gemini_model:
            raise RuntimeError("Gemini model is not set for entry routing.")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is required for Gemini access.") from exc

        client = _genai_client(self._gemini_api_key)
        request_config: dict[str, object] = {
            "temperature": self._temperature,
            "max_output_tokens": self._max_new_tokens,
            "response_mime_type": "application/json",
            "system_instruction": self._load_prompt_template(self._prompt_name),
        }
        wait_for_gemini_rate_limit(
            max_requests_per_minute=self._gemini_requests_per_minute
        )
        response = client.models.generate_content(
            model=self._gemini_model,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "次のクエリを route=direct_rag か route=openclaw のどちらかで判定し、"
                                "JSON objectのみで返してください。\n\n"
                                f"query:\n{query}"
                            )
                        }
                    ],
                }
            ],
            config=genai.types.GenerateContentConfig(**request_config),
        )
        return (response.text or "").strip()

    def _schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "route": {
                    "type": "string",
                    "enum": ["direct_rag", "openclaw"],
                },
                "reason": {"type": "string"},
            },
            "required": ["route", "reason"],
            "additionalProperties": True,
        }

    def _load_prompt_template(self, prompt_name: str) -> str:
        key = str(prompt_name or _DEFAULT_PROMPT_NAME).strip() or _DEFAULT_PROMPT_NAME
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached

        value = ""
        path = self._prompt_path(key)
        if path is not None and path.exists():
            try:
                value = path.read_text(encoding="utf-8").strip()
            except OSError:
                logger.exception("Failed to load entry routing prompt file: %s", path)
                value = ""
        if not value:
            value = _DEFAULT_SYSTEM_PROMPT
        self._prompt_cache[key] = value
        return value

    @staticmethod
    def _prompt_path(prompt_name: str) -> Path | None:
        resolved = Path(__file__).resolve()
        if len(resolved.parents) <= 5:
            return None
        prompts_dir = resolved.parents[5] / "assets" / "prompts"
        file_name = str(prompt_name or _DEFAULT_PROMPT_NAME).strip() or _DEFAULT_PROMPT_NAME
        candidate = Path(file_name)
        if candidate.suffix != ".md":
            candidate = candidate.with_suffix(".md")
        if candidate.is_absolute():
            return candidate
        return prompts_dir / candidate

    def _parse_payload(self, text: str) -> EntryRoutingDecision | None:
        payload = self._load_json_payload((text or "").strip())
        if not isinstance(payload, dict):
            return None
        route = str(payload.get("route") or "").strip().lower()
        if route not in _ROUTE_VALUES:
            return None
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            reason = "classified"
        return EntryRoutingDecision(
            route=route,  # type: ignore[arg-type]
            reason=reason,
            payload=dict(payload),
        )

    @staticmethod
    def _load_json_payload(text: str) -> dict[str, object] | None:
        cleaned = EntryQueryRouter._strip_code_fence(text).strip()
        if not cleaned:
            return None
        parsed = EntryQueryRouter._load_json_object(cleaned)
        if parsed is not None:
            return parsed
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        return EntryQueryRouter._load_json_object(cleaned[start : end + 1])

    @staticmethod
    def _load_json_object(text: str) -> dict[str, object] | None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
        return None

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return text
        lines = stripped.splitlines()
        if len(lines) < 2:
            return text
        if not lines[-1].strip().startswith("```"):
            return text
        return "\n".join(lines[1:-1]).strip()

    @staticmethod
    def _retry_backoff_seconds(*, attempt: int) -> float:
        return min(8.0, float(2 ** max(0, int(attempt))))

    @staticmethod
    def _is_retryable_generation_error(exc: Exception) -> bool:
        for item in EntryQueryRouter._exception_chain(exc):
            status_code = EntryQueryRouter._extract_status_code(item)
            if status_code in _RETRYABLE_GENERATION_STATUS_CODES:
                return True
        message = " ".join(
            str(item).lower() for item in EntryQueryRouter._exception_chain(exc)
        )
        return any(token in message for token in _RETRYABLE_GENERATION_ERROR_KEYWORDS)

    @staticmethod
    def _exception_chain(exc: Exception) -> list[BaseException]:
        chain: list[BaseException] = [exc]
        current: BaseException | None = exc
        while current is not None and len(chain) < 8:
            next_exc = current.__cause__ or current.__context__
            if next_exc is None or next_exc in chain:
                break
            chain.append(next_exc)
            current = next_exc
        return chain

    @staticmethod
    def _extract_status_code(exc: BaseException) -> int | None:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        if isinstance(status, str) and status.isdigit():
            return int(status)
        message = str(exc)
        match = re.search(r"\b([1-5]\d{2})\b", message)
        if not match:
            return None
        return int(match.group(1))


@lru_cache(maxsize=4)
def _genai_client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is required for Gemini access.") from exc
    return genai.Client(api_key=api_key)
