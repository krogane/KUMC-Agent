from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from google import genai

from config import AppConfig
from pipeline.llama_lock import LLAMA_LOCK, reset_llama_cache


def generate_with_gemini(*, api_key: str, prompt: str, config: AppConfig) -> str:
    return generate_with_gemini_config(
        api_key=api_key,
        prompt=prompt,
        system_rules=config.system_rules,
        model=config.genai_model,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        thinking_level=config.thinking_level,
    )


def generate_with_llama(
    *,
    messages: list[dict[str, str]],
    config: AppConfig,
) -> str:
    return generate_with_llama_config(
        messages=messages,
        model_path=config.llama_model_path,
        ctx_size=config.llama_ctx_size,
        threads=config.llama_threads,
        gpu_layers=config.llama_gpu_layers,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        stop=["\n---"],
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
) -> str:
    client = _genai_client(api_key)
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


def generate_with_llama_config(
    *,
    messages: list[dict[str, str]],
    model_path: str,
    ctx_size: int,
    threads: int,
    gpu_layers: int,
    temperature: float,
    max_output_tokens: int,
    stop: list[str] | None = None,
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
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
            stop=stop,
        )
    return (
        (result.get("choices", [{}])[0].get("message", {}) or {}).get("content")
        or ""
    ).strip()


@lru_cache(maxsize=1)
def _genai_client(api_key: str) -> genai.Client:
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Please set it in .env")
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
    except ImportError as e:
        raise RuntimeError(
            "llama-cpp-python is not installed. Please install it to use llama.cpp."
        ) from e

    return Llama(
        model_path=model_path,
        n_ctx=ctx_size,
        n_threads=threads,
        n_gpu_layers=gpu_layers,
    )
