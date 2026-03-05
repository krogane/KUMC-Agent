from __future__ import annotations

from typing import Protocol


class LoaderPort(Protocol):
    def load(self) -> int:
        """Load external source data into local raw storage. Returns loaded item count."""
