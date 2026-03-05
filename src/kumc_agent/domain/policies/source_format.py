from __future__ import annotations

from kumc_agent.domain.models.source import Source


def format_sources(sources: list[Source]) -> str:
    if not sources:
        return ""
    lines = ["\nSources:"]
    for i, source in enumerate(sources, start=1):
        if source.uri:
            lines.append(f"{i}. {source.label} ({source.uri})")
        else:
            lines.append(f"{i}. {source.label}")
    return "\n".join(lines)
