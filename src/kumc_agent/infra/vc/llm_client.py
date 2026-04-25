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
    response_mime_type: str | None = None,
    gemini_requests_per_minute: int = 60,
) -> str:
    normalized_provider = (provider or "").lower()
    if normalized_provider == "gemini":
        return _generate_with_gemini(
            api_key=api_key,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
            requests_per_minute=gemini_requests_per_minute,
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
    response_mime_type: str | None,
    requests_per_minute: int,
) -> str:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("google-genai is required for Gemini access.") from exc

    client = _genai_client(api_key)
    contents = [{"role": "user", "parts": [{"text": prompt}]}]

    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    wait_for_gemini_rate_limit(max_requests_per_minute=requests_per_minute)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=genai.types.GenerateContentConfig(**config_kwargs),
    )
    return (response.text or "").strip()


@lru_cache(maxsize=1)
def _genai_client(api_key: str):
    if not api_key:
        raise RuntimeError("KUMC_GEMINI_API_KEY is not set. Please set it in .env")
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("google-genai is required for Gemini access.") from exc
    return genai.Client(api_key=api_key)
