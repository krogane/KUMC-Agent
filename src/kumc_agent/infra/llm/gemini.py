from __future__ import annotations

import logging
from typing import Any

from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit
from kumc_agent.infra.llm.gemini_thinking import run_with_optional_thinking

logger = logging.getLogger(__name__)


class GeminiLLM(LLMPort):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        requests_per_minute: int = 60,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._requests_per_minute = max(0, int(requests_per_minute))

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        thinking_level: str,
    ) -> str:
        if not self._api_key:
            return "Gemini APIキーが未設定のため、ローカルフォールバック回答です。"
        try:
            from google import genai

            client = genai.Client(api_key=self._api_key)
            contents = [{"role": "user", "parts": [{"text": user_prompt}]}]

            def _request(include_thinking: bool):
                config_kwargs: dict[str, Any] = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
                if system_prompt:
                    config_kwargs["system_instruction"] = system_prompt
                if include_thinking:
                    config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
                        thinking_level=thinking_level
                    )
                wait_for_gemini_rate_limit(
                    max_requests_per_minute=self._requests_per_minute
                )
                return client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(**config_kwargs),
                )

            response = run_with_optional_thinking(
                model_name=self._model,
                request_label="Gemini generation",
                run_request=_request,
            )
            return (response.text or "").strip()
        except Exception:  # pragma: no cover - network/model dependent
            logger.exception("Gemini generation failed")
            return "Geminiでの回答生成に失敗しました。"
