from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedCommand:
    kind: str
    payload: str = ""
    force_fast_mode: bool = False
    invocation: str = "none"


def _strip_fast_prefix(query: str) -> tuple[str, bool]:
    normalized = (query or "").strip()
    if not normalized:
        return "", False
    parts = normalized.split(maxsplit=1)
    if parts[0].strip().lower() != "fast":
        return normalized, False
    if len(parts) == 1:
        return "", True
    return parts[1].strip(), True


def parse_command(*, content: str, prefix: str, index_command_prefix: str) -> ParsedCommand:
    normalized = (content or "").strip()
    if not normalized:
        return ParsedCommand(kind="none")

    prefix_value = prefix.strip().lower()
    lowered = normalized.lower()
    if lowered.startswith(index_command_prefix.lower()):
        return ParsedCommand(kind="build_index", invocation="maintenance")
    if lowered.startswith(f"{prefix_value} eval"):
        return ParsedCommand(kind="eval", invocation="maintenance")
    if lowered.startswith(f"{prefix_value} stop"):
        return ParsedCommand(kind="stop", invocation="maintenance")
    if lowered == f"{prefix_value} join":
        return ParsedCommand(kind="join_vc", invocation="maintenance")
    if lowered == f"{prefix_value} quit":
        return ParsedCommand(kind="quit_vc", invocation="maintenance")

    if lowered.startswith(prefix.lower()):
        query = normalized[len(prefix) :].strip()
        query, fast = _strip_fast_prefix(query)
        return ParsedCommand(
            kind="chat",
            payload=query,
            force_fast_mode=fast,
            invocation="prefix",
        )

    return ParsedCommand(kind="none")
