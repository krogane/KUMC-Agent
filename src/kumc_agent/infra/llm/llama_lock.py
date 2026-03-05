from __future__ import annotations

import threading

LLAMA_LOCK = threading.Lock()


def reset_llama_cache(llama: object) -> None:
    reset = getattr(llama, "reset", None)
    if callable(reset):
        reset()
