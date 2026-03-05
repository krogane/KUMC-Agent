from __future__ import annotations

from typing import Callable, Protocol


class SchedulerPort(Protocol):
    def start(self, job: Callable[[], None]) -> None:
        ...

    def stop(self) -> None:
        ...
