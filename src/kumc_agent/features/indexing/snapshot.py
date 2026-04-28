from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import shutil
from pathlib import Path


@dataclass(frozen=True)
class SnapshotPublishResult:
    snapshot_id: str
    previous_snapshot_id: str
    current_pointer: Path
    previous_pointer: Path


class IndexSnapshotPublisher:
    def __init__(self, *, index_dir: Path, keep_snapshots: int = 3) -> None:
        self._index_dir = index_dir
        self._staging_root = index_dir / "staging"
        self._previous_root = index_dir / "previous"
        self._keep_snapshots = max(1, int(keep_snapshots))

    def staging_dir(self, run_id: str) -> Path:
        safe_run_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_id)
        return self._staging_root / safe_run_id

    def publish(self, *, run_id: str, staging_dir: Path) -> SnapshotPublishResult:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        snapshot_id = staging_dir.name
        previous_snapshot_id = self._snapshot_id("previous")
        if self._has_root_artifacts():
            previous_snapshot_id = f"previous-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
            previous_dir = self._previous_root / previous_snapshot_id
            self._replace_tree(
                previous_dir,
                self._index_dir,
                skip_names={"staging", "previous", "current.json", "previous.json", ".auto_index.lock"},
            )
        try:
            self._copy_root_artifacts(staging_dir)
        except Exception:
            if previous_snapshot_id:
                self.rollback(previous_snapshot_id=previous_snapshot_id)
            raise
        current_pointer = self._index_dir / "current.json"
        previous_pointer = self._index_dir / "previous.json"
        current_pointer.write_text(
            json.dumps({"snapshot_id": snapshot_id, "path": str(staging_dir)}, ensure_ascii=False),
            encoding="utf-8",
        )
        previous_pointer.write_text(
            json.dumps({"snapshot_id": previous_snapshot_id}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._prune_previous_snapshots()
        return SnapshotPublishResult(
            snapshot_id=snapshot_id,
            previous_snapshot_id=previous_snapshot_id,
            current_pointer=current_pointer,
            previous_pointer=previous_pointer,
        )

    def rollback(self, *, previous_snapshot_id: str) -> dict[str, object]:
        previous_dir = self._previous_root / previous_snapshot_id
        if not previous_snapshot_id or not previous_dir.exists():
            return {
                "status": "failed",
                "reason": "previous_snapshot_not_found",
                "previous_snapshot_id": previous_snapshot_id,
            }
        self._copy_root_artifacts(previous_dir)
        return {
            "status": "succeeded",
            "previous_snapshot_id": previous_snapshot_id,
        }

    def rollback_to_latest_previous(self) -> dict[str, object]:
        previous_snapshot_id = self._snapshot_id("previous")
        if not previous_snapshot_id and self._previous_root.exists():
            snapshots = sorted(
                (path for path in self._previous_root.iterdir() if path.is_dir()),
                key=lambda value: value.stat().st_mtime,
                reverse=True,
            )
            previous_snapshot_id = snapshots[0].name if snapshots else ""
        return self.rollback(previous_snapshot_id=previous_snapshot_id)

    def _has_root_artifacts(self) -> bool:
        return any(
            path.is_file()
            for path in self._index_dir.iterdir()
            if path.name not in {"current.json", "previous.json", ".auto_index.lock"}
        ) if self._index_dir.exists() else False

    def _copy_root_artifacts(self, source_dir: Path) -> None:
        for path in self._index_dir.iterdir():
            if path.name in {"staging", "previous"}:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        for item in source_dir.iterdir():
            target = self._index_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

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

    def _prune_previous_snapshots(self) -> None:
        if not self._previous_root.exists():
            return
        snapshots = sorted(
            (path for path in self._previous_root.iterdir() if path.is_dir()),
            key=lambda value: value.stat().st_mtime,
            reverse=True,
        )
        for path in snapshots[self._keep_snapshots :]:
            shutil.rmtree(path, ignore_errors=True)

    def _snapshot_id(self, pointer_name: str) -> str:
        pointer = self._index_dir / f"{pointer_name}.json"
        if not pointer.exists():
            return ""
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(payload.get("snapshot_id") or "") if isinstance(payload, dict) else ""
