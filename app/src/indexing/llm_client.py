from __future__ import annotations

from functools import lru_cache
from typing import Any


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
    thinking_level: str,
    llama_threads: int,
    llama_gpu_layers: int,
    response_mime_type: str | None = None,
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
    thinking_level: str,
    response_mime_type: str | None,
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

    response = client.models.generate_content(
        model=model,
        contents=[
            {"role": "system", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ],
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
            "LLAMA_MODEL_PATH is not set. Please set it in .env"
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
