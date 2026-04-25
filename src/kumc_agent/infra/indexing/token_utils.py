from __future__ import annotations

from functools import lru_cache


def estimate_tokens(*, text: str, model_name: str) -> int:
    if not text:
        return 0
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
