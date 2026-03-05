from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def elapsed_timer() -> Iterator[callable[[], float]]:
    start = time.perf_counter()

    def _elapsed() -> float:
        return time.perf_counter() - start

    yield _elapsed
