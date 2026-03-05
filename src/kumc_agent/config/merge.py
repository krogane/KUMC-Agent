from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


class MergeError(ValueError):
    pass


def deep_merge(
    base: dict[str, Any],
    override: Mapping[str, Any],
    *,
    allow_new_keys: bool,
    path: str = "",
) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        key_path = f"{path}.{key}" if path else key
        if key not in merged:
            if not allow_new_keys:
                raise MergeError(f"Unknown config key: {key_path}")
            merged[key] = deepcopy(value)
            continue

        current = merged[key]
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(
                current,
                value,
                allow_new_keys=allow_new_keys,
                path=key_path,
            )
            continue

        if isinstance(current, list) and isinstance(value, list):
            merged[key] = deepcopy(value)
            continue

        merged[key] = deepcopy(value)
    return merged
