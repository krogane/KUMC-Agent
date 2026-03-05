from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Lifecycle:
    starters: list[Callable[[], None]]
    stoppers: list[Callable[[], None]]

    def start(self) -> None:
        for starter in self.starters:
            starter()

    def stop(self) -> None:
        for stopper in reversed(self.stoppers):
            stopper()
