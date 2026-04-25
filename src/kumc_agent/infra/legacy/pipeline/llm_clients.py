from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from google import genai

from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit
from kumc_agent.infra.legacy.config import AppConfig


def generate_with_gemini(*, api_key: str, prompt: str, config: AppConfig) -> str:
    return generate_with_gemini_config(
        api_key=api_key,
        prompt=prompt,
        system_rules=config.system_rules,
        model=config.genai_model,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        thinking_level=config.thinking_level,
        requests_per_minute=getattr(config, "gemini_requests_per_minute", 60),
    )


def generate_with_gemini_config(
    *,
    api_key: str,
    prompt: str,
    system_rules: Sequence[str],
    model: str,
    temperature: float,
    max_output_tokens: int,
    thinking_level: str,
    requests_per_minute: int = 60,
) -> str:
    client = _genai_client(api_key)
    wait_for_gemini_rate_limit(max_requests_per_minute=requests_per_minute)
    response = client.models.generate_content(
        model=model,
        contents=[
            {
                "role": "system",
                "parts": [{"text": "\n".join(system_rules)}],
            },
            {
                "role": "user",
                "parts": [{"text": prompt}],
            },
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


@lru_cache(maxsize=1)
def _genai_client(api_key: str) -> genai.Client:
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Please set it in .env")
    return genai.Client(api_key=api_key)
