from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable


class AutoIndexScheduler:
    def __init__(self, *, time_hhmm: str, weekdays: list[int]) -> None:
        self._time_hhmm = time_hhmm
        self._weekdays = set(weekdays)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, job: Callable[[], None]) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        def _loop() -> None:
            last_run_date = ""
            while not self._stop_event.is_set():
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")
                if (
                    now.weekday() in self._weekdays
                    and now.strftime("%H:%M") == self._time_hhmm
                    and today != last_run_date
                ):
                    job()
                    last_run_date = today
                time.sleep(30)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
