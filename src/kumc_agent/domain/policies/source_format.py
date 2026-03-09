from __future__ import annotations

from kumc_agent.domain.models.source import Source


def format_sources(sources: list[Source]) -> str:
    if not sources:
        return ""
    refs: list[str] = []
    seen: set[str] = set()
    for source in sources:
        label = str(source.label or "").strip()
        uri = str(source.uri or "").strip()
        ref = label or uri
        if not ref:
            continue
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    if not refs:
        return ""
    sources_text = "\n".join(f"- {ref}" for ref in refs)
    return (
        "\n\n"
        "※回答は必ずしも正しいとは限りません。重要な情報は確認するようにしてください。\n"
        f"主な情報源:\n{sources_text}"
    )
