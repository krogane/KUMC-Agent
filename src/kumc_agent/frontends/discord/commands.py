from __future__ import annotations

from collections.abc import Mapping
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


def _normalize_subcommand(value: str) -> str:
    normalized = (value or "").strip().lower()
    return normalized.replace("_", "-")


def _as_option_nodes(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return tuple()
    nodes: list[Mapping[str, object]] = []
    for item in value:
        if isinstance(item, Mapping):
            nodes.append(item)
    return tuple(nodes)


def _extract_subcommand_node(
    options: tuple[Mapping[str, object], ...],
) -> tuple[str, tuple[Mapping[str, object], ...]]:
    current_options = options
    latest_name = ""
    latest_options: tuple[Mapping[str, object], ...] = tuple()
    while current_options:
        subcommand_node: Mapping[str, object] | None = None
        for node in current_options:
            option_type = node.get("type")
            if option_type in (1, 2):  # SUB_COMMAND / SUB_COMMAND_GROUP
                subcommand_node = node
                break
        if subcommand_node is None:
            break
        latest_name = str(subcommand_node.get("name") or "").strip()
        latest_options = _as_option_nodes(subcommand_node.get("options"))
        current_options = latest_options
    return latest_name, latest_options


def _extract_query_options(
    options: tuple[Mapping[str, object], ...],
) -> tuple[str, bool]:
    query_value = ""
    fast = False
    for node in options:
        name = str(node.get("name") or "").strip().lower()
        value = node.get("value")
        if name in {"query", "text", "message"} and isinstance(value, str):
            query_value = value.strip()
        elif name == "fast":
            if isinstance(value, bool):
                fast = value
            elif isinstance(value, str):
                fast = value.strip().lower() in {"1", "true", "yes", "on"}
    query_value, prefixed_fast = _strip_fast_prefix(query_value)
    return query_value, (fast or prefixed_fast)


def parse_interaction_command(*, name: str, options: object) -> ParsedCommand:
    if (name or "").strip().lower() != "ai":
        return ParsedCommand(kind="none")

    option_nodes = _as_option_nodes(options)
    subcommand_name, subcommand_options = _extract_subcommand_node(option_nodes)
    normalized_subcommand = _normalize_subcommand(subcommand_name)

    if not normalized_subcommand:
        for node in option_nodes:
            option_name = str(node.get("name") or "").strip().lower()
            option_value = node.get("value")
            if option_name in {"command", "action"} and isinstance(option_value, str):
                normalized_subcommand = _normalize_subcommand(option_value)
                break

    if normalized_subcommand == "build-index":
        return ParsedCommand(kind="build_index", invocation="maintenance")
    if normalized_subcommand == "eval":
        return ParsedCommand(kind="eval", invocation="maintenance")
    if normalized_subcommand == "stop":
        return ParsedCommand(kind="stop", invocation="maintenance")
    if normalized_subcommand == "join":
        return ParsedCommand(kind="join_vc", invocation="maintenance")
    if normalized_subcommand == "quit":
        return ParsedCommand(kind="quit_vc", invocation="maintenance")

    query, force_fast_mode = _extract_query_options(subcommand_options or option_nodes)
    if query:
        return ParsedCommand(
            kind="chat",
            payload=query,
            force_fast_mode=force_fast_mode,
            invocation="slash",
        )
    return ParsedCommand(kind="none")
