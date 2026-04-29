from __future__ import annotations

import json
import logging
from typing import Any
from urllib import request

from kumc_agent.domain.ports.llms import LLMPort

logger = logging.getLogger(__name__)


class OpenAILLM(LLMPort):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = float(timeout_seconds)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        if not self._api_key:
            return "OpenAI APIキーが未設定のため、ローカルフォールバック回答です。"
        payload = {
            "model": self._model,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        try:
            req = request.Request(
                "https://api.openai.com/v1/responses",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=self._timeout_seconds) as response:  # noqa: S310 - configured API endpoint.
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            text = str(parsed.get("output_text") or "").strip()
            if text:
                return text
            return _extract_response_text(parsed)
        except Exception:  # pragma: no cover - network/model dependent
            logger.exception("OpenAI generation failed")
            return "OpenAIでの回答生成に失敗しました。"


def _extract_response_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()

