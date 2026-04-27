from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from kumc_agent.domain.models.source import SourceRawItem

ChangeKind = Literal["new", "updated", "permission_changed", "deleted", "permission_lost", "skipped"]


@dataclass(frozen=True)
class SourceItemState:
    source_kind: str
    external_id: str
    checksum: str = ""
    revision: str = ""
    acl_hash: str = ""
    index_status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceChange:
    source_kind: str
    external_id: str
    change_kind: ChangeKind
    reason: str = ""


def detect_source_change(
    *,
    item: SourceRawItem,
    previous: SourceItemState | None,
    force: bool = False,
) -> SourceChange:
    if previous is None:
        return SourceChange(item.source_kind, item.external_id, "new", "not_seen_before")
    if force:
        return SourceChange(item.source_kind, item.external_id, "updated", "force")
    if previous.index_status in {"deleted", "permission_lost", "quarantined"}:
        return SourceChange(item.source_kind, item.external_id, "updated", "previously_inactive")

    current_revision = _revision(item.metadata)
    current_acl_hash = item.access_scope.source_acl_hash or ""
    if item.checksum and previous.checksum and item.checksum != previous.checksum:
        return SourceChange(item.source_kind, item.external_id, "updated", "checksum_changed")
    if current_revision and previous.revision and current_revision != previous.revision:
        return SourceChange(item.source_kind, item.external_id, "updated", "revision_changed")
    if current_acl_hash != previous.acl_hash:
        return SourceChange(
            item.source_kind,
            item.external_id,
            "permission_changed",
            "acl_hash_changed",
        )
    return SourceChange(item.source_kind, item.external_id, "skipped", "unchanged")


def state_from_source_item_payload(payload: dict[str, object]) -> SourceItemState:
    metadata = _dict(payload.get("metadata"))
    access_scope = _dict(payload.get("access_scope"))
    return SourceItemState(
        source_kind=str(payload.get("source_kind") or ""),
        external_id=str(payload.get("external_id") or ""),
        checksum=str(payload.get("checksum") or ""),
        revision=_revision(metadata),
        acl_hash=str(access_scope.get("source_acl_hash") or ""),
        index_status=str(payload.get("index_status") or "active"),
        metadata=metadata,
    )


def _revision(metadata: dict[str, Any]) -> str:
    for key in ("revision", "revision_id", "modified_time", "last_edited_time", "updated_at"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
