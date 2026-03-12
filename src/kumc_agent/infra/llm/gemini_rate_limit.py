from __future__ import annotations

from collections import deque
import logging
import threading
import time

_WINDOW_SECONDS = 60.0
_DEFAULT_LIMITER_NAME = "default"
_RAGAS_LIMITER_NAME = "ragas_eval"
_INDEX_SUMMARY_LIMITER_NAME = "index_summary"
_EMBEDDING_LIMITER_NAME = "embedding"

logger = logging.getLogger(__name__)


class _GeminiRateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def wait_for_slot(self, *, max_requests_per_minute: int) -> None:
        limit = max(0, int(max_requests_per_minute))
        if limit == 0:
            return

        while True:
            wait_seconds = 0.0
            with self._lock:
                now = time.monotonic()
                window_start = now - _WINDOW_SECONDS
                while self._timestamps and self._timestamps[0] <= window_start:
                    self._timestamps.popleft()

                if len(self._timestamps) < limit:
                    self._timestamps.append(now)
                    return

                wait_seconds = max(0.0, self._timestamps[0] + _WINDOW_SECONDS - now)

            if wait_seconds <= 0:
                continue
            logger.info(
                "Gemini API rate limit reached (%s requests/min). Waiting %.2f seconds.",
                limit,
                wait_seconds,
            )
            time.sleep(wait_seconds)


_GEMINI_RATE_LIMITERS: dict[str, _GeminiRateLimiter] = {}
_GEMINI_RATE_LIMITERS_LOCK = threading.Lock()


def wait_for_gemini_rate_limit(
    *,
    max_requests_per_minute: int,
    limiter_name: str = _DEFAULT_LIMITER_NAME,
) -> None:
    name = (limiter_name or "").strip() or _DEFAULT_LIMITER_NAME
    with _GEMINI_RATE_LIMITERS_LOCK:
        limiter = _GEMINI_RATE_LIMITERS.get(name)
        if limiter is None:
            limiter = _GeminiRateLimiter()
            _GEMINI_RATE_LIMITERS[name] = limiter
    limiter.wait_for_slot(max_requests_per_minute=max_requests_per_minute)


def ragas_rate_limiter_name() -> str:
    return _RAGAS_LIMITER_NAME


def index_summary_rate_limiter_name() -> str:
    return _INDEX_SUMMARY_LIMITER_NAME


def embedding_rate_limiter_name() -> str:
    return _EMBEDDING_LIMITER_NAME
