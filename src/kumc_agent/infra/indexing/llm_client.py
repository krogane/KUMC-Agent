from __future__ import annotations

from functools import lru_cache
from typing import Any

from kumc_agent.infra.llm.gemini_rate_limit import wait_for_gemini_rate_limit
from kumc_agent.infra.indexing.llama_lock import LLAMA_LOCK, reset_llama_cache


def generate_text(
    *,
    provider: str,
    api_key: str,
    prompt: str,
    model: str,
    system_prompt: str,
    llama_model_path: str,
    llama_ctx_size: int,
    temperature: float,
    max_output_tokens: int,
    llama_threads: int,
    llama_gpu_layers: int,
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
            response_mime_type=response_mime_type,
            requests_per_minute=gemini_requests_per_minute,
            limiter_name=gemini_rate_limiter_name,
        )
    if provider == "llama":
        return _generate_with_llama(
            prompt=prompt,
            system_prompt=system_prompt,
            model_path=llama_model_path,
            ctx_size=llama_ctx_size,
            threads=llama_threads,
            gpu_layers=llama_gpu_layers,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(
        f"Unsupported LLM provider: {provider}. Use 'gemini' or 'llama'."
    )


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
    limiter_name: str,
) -> str:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for Gemini access."
        ) from exc

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
    wait_for_gemini_rate_limit(
        max_requests_per_minute=requests_per_minute,
        limiter_name=limiter_name,
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=genai.types.GenerateContentConfig(**config_kwargs),
    )
    return (response.text or "").strip()


def _generate_with_llama(
    *,
    prompt: str,
    system_prompt: str,
    model_path: str,
    ctx_size: int,
    threads: int,
    gpu_layers: int,
    temperature: float,
    max_output_tokens: int,
) -> str:
    llama = _llama_client(
        model_path,
        ctx_size,
        threads,
        gpu_layers,
    )
    with LLAMA_LOCK:
        reset_llama_cache(llama)
        result = llama.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
    return (
        (result.get("choices", [{}])[0].get("message", {}) or {}).get("content")
        or ""
    ).strip()


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


@lru_cache(maxsize=1)
def _llama_client(
    model_path: str,
    ctx_size: int,
    threads: int,
    gpu_layers: int,
):
    if not model_path:
        raise RuntimeError(
            "LLAMA_MODEL is not set. Please set LLAMA_MODEL (and LLM_MODEL_DIR) in .env"
        )

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Please install it to use llama.cpp."
        ) from exc

    return Llama(
        model_path=model_path,
        n_ctx=ctx_size,
        n_threads=threads,
        n_gpu_layers=gpu_layers,
    )
