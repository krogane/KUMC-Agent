from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
import shutil
from pathlib import Path


@dataclass(frozen=True)
class SnapshotPublishResult:
    snapshot_id: str
    previous_snapshot_id: str
    current_pointer: Path
    previous_pointer: Path
    release_dir: Path


class IndexSnapshotPublisher:
    def __init__(self, *, index_dir: Path, keep_snapshots: int = 3) -> None:
        self._index_dir = index_dir
        self._staging_root = index_dir / "staging"
        self._previous_root = index_dir / "previous"
        self._releases_root = index_dir / "releases"
        self._keep_snapshots = max(1, int(keep_snapshots))

    def staging_dir(self, run_id: str) -> Path:
        safe_run_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_id)
        return self._staging_root / safe_run_id

    def publish(self, *, run_id: str, staging_dir: Path) -> SnapshotPublishResult:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        snapshot_id = staging_dir.name
        previous_snapshot_id = self._snapshot_id("current")
        release_dir = self._releases_root / snapshot_id
        self._replace_tree(release_dir, staging_dir, skip_names=set())
        current_pointer = self._index_dir / "current.json"
        previous_pointer = self._index_dir / "previous.json"
        self._write_pointer_atomic(
            previous_pointer,
            {
                "snapshot_id": previous_snapshot_id,
                "path": str(self._release_dir(previous_snapshot_id)) if previous_snapshot_id else "",
            },
        )
        self._write_pointer_atomic(
            current_pointer,
            {
                "snapshot_id": snapshot_id,
                "path": str(release_dir),
                "previous_snapshot_id": previous_snapshot_id,
                "published_at": datetime.now(UTC).isoformat(),
            },
        )
        self._prune_releases()
        return SnapshotPublishResult(
            snapshot_id=snapshot_id,
            previous_snapshot_id=previous_snapshot_id,
            current_pointer=current_pointer,
            previous_pointer=previous_pointer,
            release_dir=release_dir,
        )

    def rollback(self, *, previous_snapshot_id: str) -> dict[str, object]:
        previous_dir = self._release_dir(previous_snapshot_id)
        if not previous_snapshot_id or not previous_dir.exists():
            return {
                "status": "failed",
                "reason": "previous_snapshot_not_found",
                "previous_snapshot_id": previous_snapshot_id,
            }
        current_snapshot_id = self._snapshot_id("current")
        self._write_pointer_atomic(
            self._index_dir / "previous.json",
            {
                "snapshot_id": current_snapshot_id,
                "path": str(self._release_dir(current_snapshot_id)) if current_snapshot_id else "",
            },
        )
        self._write_pointer_atomic(
            self._index_dir / "current.json",
            {
                "snapshot_id": previous_snapshot_id,
                "path": str(previous_dir),
                "rolled_back_from_snapshot_id": current_snapshot_id,
                "published_at": datetime.now(UTC).isoformat(),
            },
        )
        return {
            "status": "succeeded",
            "previous_snapshot_id": previous_snapshot_id,
        }

    def rollback_to_latest_previous(self) -> dict[str, object]:
        previous_snapshot_id = self._snapshot_id("previous")
        if not previous_snapshot_id and self._releases_root.exists():
            current_snapshot_id = self._snapshot_id("current")
            snapshots = sorted(
                (
                    path
                    for path in self._releases_root.iterdir()
                    if path.is_dir() and path.name != current_snapshot_id
                ),
                key=lambda value: value.stat().st_mtime,
                reverse=True,
            )
            previous_snapshot_id = snapshots[0].name if snapshots else ""
        return self.rollback(previous_snapshot_id=previous_snapshot_id)

    def _replace_tree(self, target_dir: Path, source_dir: Path, *, skip_names: set[str]) -> None:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            if item.name in skip_names:
                continue
            target = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    def _prune_releases(self) -> None:
        if not self._releases_root.exists():
            return
        protected = {self._snapshot_id("current"), self._snapshot_id("previous")}
        snapshots = sorted(
            (path for path in self._releases_root.iterdir() if path.is_dir()),
            key=lambda value: value.stat().st_mtime,
            reverse=True,
        )
        kept = 0
        for path in snapshots:
            if path.name in protected:
                continue
            kept += 1
            if kept <= self._keep_snapshots:
                continue
            shutil.rmtree(path, ignore_errors=True)

    def _release_dir(self, snapshot_id: str) -> Path:
        return self._releases_root / snapshot_id

    def _snapshot_id(self, pointer_name: str) -> str:
        pointer = self._index_dir / f"{pointer_name}.json"
        if not pointer.exists():
            return ""
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(payload.get("snapshot_id") or "") if isinstance(payload, dict) else ""

    @staticmethod
    def _write_pointer_atomic(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp.replace(path)
