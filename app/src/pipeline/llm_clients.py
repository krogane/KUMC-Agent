from __future__ import annotations

from functools import lru_cache

from google import genai

from config import AppConfig
from pipeline.llama_lock import LLAMA_LOCK, reset_llama_cache


def generate_with_gemini(*, api_key: str, prompt: str, config: AppConfig) -> str:
    client = _genai_client(api_key)
    response = client.models.generate_content(
        model=config.genai_model,
        contents=[
            {
                "role": "system",
                "parts": [{"text": "\n".join(config.system_rules)}],
            },
            {
                "role": "user",
                "parts": [{"text": prompt}],
            },
        ],
        config=genai.types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            thinking_config=genai.types.ThinkingConfig(
                thinking_level=config.thinking_level
            ),
        ),
    )
    return (response.text or "").strip()


def generate_with_llama(
    *,
    messages: list[dict[str, str]],
    config: AppConfig,
) -> str:
    llama = _llama_client(
        config.llama_model_path,
        config.llama_ctx_size,
        config.llama_threads,
        config.llama_gpu_layers,
    )
    with LLAMA_LOCK:
        reset_llama_cache(llama)
        result = llama.create_chat_completion(
            messages=messages,
            max_tokens=config.max_output_tokens,
            temperature=config.temperature,
            stop=["\n---"],
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
        raise RuntimeError("LLAMA_MODEL_PATH is not set. Please set it in .env")

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
