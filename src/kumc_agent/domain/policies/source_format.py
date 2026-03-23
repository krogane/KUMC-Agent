from __future__ import annotations

from kumc_agent.domain.models.source import Source

SOURCE_DISCLAIMER_TEXT = (
    "※回答は必ずしも正しいとは限りません。重要な情報は確認するようにしてください。"
)


def format_sources(sources: list[Source], *, include_disclaimer: bool = True) -> str:
    if not sources:
        return ""
    refs: list[str] = []
    seen: set[str] = set()
    for source in sources:
        uri = str(source.uri or "").strip()
        if not uri:
            continue
        if uri in seen:
            continue
        seen.add(uri)
        refs.append(uri)
    if not refs:
        return ""
    sources_text = "\n".join(f"- {ref}" for ref in refs)
    disclaimer_line = f"{SOURCE_DISCLAIMER_TEXT}\n" if include_disclaimer else ""
    return f"\n\n{disclaimer_line}主な情報源:\n{sources_text}"
