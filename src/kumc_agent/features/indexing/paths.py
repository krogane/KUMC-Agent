from __future__ import annotations

import json
from pathlib import Path


def resolve_current_index_dir(index_dir: Path) -> Path:
    root = Path(index_dir)
    pointer = root / "current.json"
    if not pointer.exists():
        return root
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except Exception:
        return root
    if not isinstance(payload, dict):
        return root
    candidates: list[Path] = []
    raw_path = str(payload.get("path") or "").strip()
    if raw_path:
        candidate = Path(raw_path)
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
    snapshot_id = str(payload.get("snapshot_id") or "").strip()
    if snapshot_id:
        candidates.extend(
            (
                root / "releases" / snapshot_id,
                root / "previous" / snapshot_id,
                root / "staging" / snapshot_id,
            )
        )
    root_resolved = _safe_resolve(root)
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if not _is_within(_safe_resolve(candidate), root_resolved):
            continue
        return candidate
    return root


def resolve_feature_index_dir(index_dir: Path) -> Path:
    path = Path(index_dir)
    parent = path.parent
    active_parent = resolve_current_index_dir(parent)
    if active_parent == parent:
        return path
    candidate = active_parent / path.name
    return candidate if candidate.exists() else path


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
