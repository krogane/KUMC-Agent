from __future__ import annotations

import logging

from kumc_agent.domain.ports.llms import LLMPort

logger = logging.getLogger(__name__)


class GeminiLLM(LLMPort):
    def __init__(self, *, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

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
            response = client.models.generate_content(
                model=self._model,
                contents=[
                    {"role": "system", "parts": [{"text": system_prompt}]},
                    {"role": "user", "parts": [{"text": user_prompt}]},
                ],
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_level=thinking_level
                    ),
                ),
            )
            return (response.text or "").strip()
        except Exception:  # pragma: no cover - network/model dependent
            logger.exception("Gemini generation failed")
            return "Geminiでの回答生成に失敗しました。"
