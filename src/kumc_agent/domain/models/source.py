from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class Source:
    id: str
    label: str
    uri: str = ""


@dataclass(frozen=True)
class AccessScope:
    visibility: Literal["public", "guild", "role", "private", "admin"] = "admin"
    guild_id: str | None = None
    role_ids: tuple[str, ...] = tuple()
    user_ids: tuple[str, ...] = tuple()
    source_acl_hash: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "visibility": self.visibility,
            "guild_id": self.guild_id,
            "role_ids": list(self.role_ids),
            "user_ids": list(self.user_ids),
            "source_acl_hash": self.source_acl_hash,
        }


@dataclass(frozen=True)
class SourceAccount:
    id: str
    kind: str
    display_name: str
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceRawItem:
    source_kind: str
    external_id: str
    title: str
    text: str
    canonical_url: str = ""
    author_id: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    access_scope: AccessScope = field(default_factory=AccessScope)
    raw_path: str = ""
    checksum: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceDeleteItem:
    source_kind: str
    external_id: str
    reason: str = "deleted"


@dataclass(frozen=True)
class BackfillScope:
    limit: int | None = None
    source_ids: tuple[str, ...] = tuple()
    force: bool = False


@dataclass(frozen=True)
class SyncCursor:
    source_kind: str
    cursor: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedDocument:
    id: str
    source_item_id: str
    source_kind: str
    external_id: str
    version: int
    title: str
    normalized_text: str
    normalized_format: str
    language: str | None
    access_scope: AccessScope
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)
