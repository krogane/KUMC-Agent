from __future__ import annotations

import logging

from kumc_agent.domain.ports.llms import LLMPort

logger = logging.getLogger(__name__)


class LlamaCppLLM(LLMPort):
    def __init__(self, *, model_path: str) -> None:
        self._model_path = model_path

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        if not self._model_path:
            return "llama.cppモデルが未設定のため、ローカルフォールバック回答です。"
        try:
            from llama_cpp import Llama

            llm = Llama(model_path=self._model_path)
            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_output_tokens,
                temperature=temperature,
            )
            return (
                (result.get("choices", [{}])[0].get("message", {}) or {}).get("content")
                or ""
            ).strip()
        except Exception:  # pragma: no cover - model/runtime dependent
            logger.exception("llama.cpp generation failed")
            return "llama.cppでの回答生成に失敗しました。"
