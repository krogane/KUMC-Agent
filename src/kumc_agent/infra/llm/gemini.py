from __future__ import annotations

import logging
from typing import Any

from kumc_agent.domain.ports.llms import LLMPort
from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit

logger = logging.getLogger(__name__)


def _merge_system_prompt_into_user_prompt(*, system_prompt: str, user_prompt: str) -> str:
    normalized_system_prompt = (system_prompt or "").strip()
    normalized_user_prompt = (user_prompt or "").strip()
    if not normalized_system_prompt:
        return normalized_user_prompt
    if not normalized_user_prompt:
        return normalized_system_prompt
    return (
        "[System Instructions]\n"
        f"{normalized_system_prompt}\n\n"
        "[User Prompt]\n"
        f"{normalized_user_prompt}"
    )


def _is_developer_instruction_disabled_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "developer instruction is not enabled" in message


class GeminiLLM(LLMPort):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        requests_per_minute: int = 60,
        limiter_name: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._requests_per_minute = max(0, int(requests_per_minute))
        self._limiter_name = (limiter_name or "").strip()

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        if not self._api_key:
            return "Gemini APIキーが未設定のため、ローカルフォールバック回答です。"
        try:
            from google import genai

            client = genai.Client(api_key=self._api_key)
            contents = [{"role": "user", "parts": [{"text": user_prompt}]}]
            config_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            try:
                response = self._generate_content(
                    client=client,
                    genai=genai,
                    contents=contents,
                    config_kwargs=config_kwargs,
                )
            except Exception as exc:
                if not system_prompt or not _is_developer_instruction_disabled_error(exc):
                    raise
                logger.warning(
                    "Gemini model does not support system instruction. "
                    "Retrying with merged prompt."
                )
                fallback_contents = [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": _merge_system_prompt_into_user_prompt(
                                    system_prompt=system_prompt,
                                    user_prompt=user_prompt,
                                )
                            }
                        ],
                    }
                ]
                fallback_config_kwargs: dict[str, Any] = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
                response = self._generate_content(
                    client=client,
                    genai=genai,
                    contents=fallback_contents,
                    config_kwargs=fallback_config_kwargs,
                )
            return (response.text or "").strip()
        except Exception:  # pragma: no cover - network/model dependent
            logger.exception("Gemini generation failed")
            return "Geminiでの回答生成に失敗しました。"

    def _generate_content(
        self,
        *,
        client,
        genai,
        contents: list[dict[str, Any]],
        config_kwargs: dict[str, Any],
    ):
        wait_for_gemini_rate_limit(
            max_requests_per_minute=self._requests_per_minute,
            limiter_name=self._limiter_name,
        )
        return client.models.generate_content(
            model=self._model,
            contents=contents,
            config=genai.types.GenerateContentConfig(**config_kwargs),
        )
