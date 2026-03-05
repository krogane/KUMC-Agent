from __future__ import annotations


def should_refuse(*, query: str, keywords: list[str]) -> bool:
    lowered = (query or "").lower()
    return any(keyword.lower() in lowered for keyword in keywords)
