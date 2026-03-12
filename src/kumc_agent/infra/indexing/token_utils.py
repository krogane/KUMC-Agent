from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def estimate_tokens(*, text: str, model_name: str) -> int:
    if not text:
        return 0
    if _is_gguf_model(model_name):
        tokens = _llama_tokenize(text=text, model_path=model_name)
        return max(1, len(tokens))
    tokenizer = _get_tokenizer(model_name)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return max(1, len(tokens))


@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    if not model_name:
        raise RuntimeError("Tokenizer model name is required for token estimation.")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for tokenizer-based token estimation."
        ) from exc
    return AutoTokenizer.from_pretrained(model_name, local_files_only=True)


def _is_gguf_model(model_name: str) -> bool:
    return Path(model_name).suffix.lower() == ".gguf"


def _llama_tokenize(*, text: str, model_path: str) -> list[int]:
    llama = _get_llama_model(model_path)
    return llama.tokenize(text.encode("utf-8"), add_bos=False, special=False)


@lru_cache(maxsize=2)
def _get_llama_model(model_path: str):
    if not model_path:
        raise RuntimeError("Llama model path is required for token estimation.")
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is required for llama.cpp token estimation."
        ) from exc
    return Llama(model_path=model_path, embedding=True)
