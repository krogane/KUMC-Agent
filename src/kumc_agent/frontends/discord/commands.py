from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedCommand:
    kind: str
    payload: str = ""


def parse_command(*, content: str, prefix: str, index_command_prefix: str) -> ParsedCommand:
    normalized = (content or "").strip()
    if not normalized:
        return ParsedCommand(kind="none")

    prefix_value = prefix.strip().lower()
    lowered = normalized.lower()
    if lowered.startswith(index_command_prefix.lower()):
        return ParsedCommand(kind="build_index")
    if lowered.startswith(f"{prefix_value} eval"):
        return ParsedCommand(kind="eval")
    if lowered.startswith(f"{prefix_value} stop"):
        return ParsedCommand(kind="stop")
    if lowered == f"{prefix_value} join":
        return ParsedCommand(kind="join_vc")
    if lowered == f"{prefix_value} quit":
        return ParsedCommand(kind="quit_vc")

    if lowered.startswith(prefix.lower()):
        query = normalized[len(prefix) :].strip()
        return ParsedCommand(kind="chat", payload=query)

    return ParsedCommand(kind="none")
