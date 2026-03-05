from __future__ import annotations

from datetime import datetime


def recency_weight(*, updated_at: datetime | None, mode: str) -> float:
    if updated_at is None:
        return 0.0
    mode_normalized = (mode or "off").strip().lower()
    if mode_normalized == "hard":
        return 1.0
    if mode_normalized == "soft":
        return 0.3
    return 0.0
