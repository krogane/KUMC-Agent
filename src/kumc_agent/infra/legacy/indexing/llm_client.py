from __future__ import annotations

from functools import lru_cache
from typing import Any

from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit


def generate_text(
    *,
    provider: str,
    api_key: str,
    prompt: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_output_tokens: int,
    thinking_level: str,
    response_mime_type: str | None = None,
    gemini_requests_per_minute: int = 60,
    gemini_rate_limiter_name: str = "",
) -> str:
    provider = (provider or "").lower()
    if provider == "gemini":
        return _generate_with_gemini(
            api_key=api_key,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
            response_mime_type=response_mime_type,
            requests_per_minute=gemini_requests_per_minute,
            limiter_name=gemini_rate_limiter_name,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}. Use 'gemini'.")


def _generate_with_gemini(
    *,
    api_key: str,
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    thinking_level: str,
    response_mime_type: str | None,
    requests_per_minute: int,
    limiter_name: str,
) -> str:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for Gemini access."
        ) from exc

    client = _genai_client(api_key)
    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "thinking_config": genai.types.ThinkingConfig(thinking_level=thinking_level),
    }
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type

    wait_for_gemini_rate_limit(
        max_requests_per_minute=requests_per_minute,
        limiter_name=limiter_name,
    )
    response = client.models.generate_content(
        model=model,
        contents=[
            {"role": "system", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ],
        config=genai.types.GenerateContentConfig(**config_kwargs),
    )
    return (response.text or "").strip()


@lru_cache(maxsize=1)
def _genai_client(api_key: str):
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Please set it in .env")
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for Gemini access."
        ) from exc
    return genai.Client(api_key=api_key)
