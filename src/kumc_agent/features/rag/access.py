from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.retrieval import AccessContext

_PROTECTED_SOURCE_TYPES = frozenset(
    {
        "docs",
        "sheets",
        "google_drive",
        "drive",
        "messages",
        "discord_message",
        "discord",
    }
)
_PUBLIC_SOURCE_TYPES = frozenset(
    {"hatenablog", "crafters_colony", "x_posts", "notion", "vc_transcript"}
)
_DENIED_INDEX_STATUSES = frozenset({"deleted", "quarantined", "permission_lost"})


@dataclass(frozen=True)
class RagAccessFilter:
    allowed_guild_ids: tuple[str, ...] = tuple()
    admin_user_ids: tuple[str, ...] = tuple()

    def allow_chunk(
        self,
        chunk: Chunk,
        *,
        access: AccessContext | None,
    ) -> bool:
        return self.allow_metadata(chunk.metadata or {}, access=access)

    def allow_metadata(
        self,
        metadata: dict[str, object],
        *,
        access: AccessContext | None,
    ) -> bool:
        if self._is_hard_denied(metadata):
            return False

        source_type = str(metadata.get("source_type") or "").strip().lower()
        access_scope = metadata.get("access_scope")
        if isinstance(access_scope, dict):
            return self._allow_access_scope(access_scope, access=access)
        if source_type in _PUBLIC_SOURCE_TYPES:
            return True
        if source_type not in _PROTECTED_SOURCE_TYPES:
            return True

        return self._allow_protected(metadata=metadata, access=access)

    def filter_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        access: AccessContext | None,
    ) -> list[Chunk]:
        return [chunk for chunk in chunks if self.allow_chunk(chunk, access=access)]

    @staticmethod
    def _is_hard_denied(metadata: dict[str, object]) -> bool:
        redaction_policy = str(metadata.get("redaction_policy") or "").strip().lower()
        if redaction_policy == "deny":
            return True
        index_status = str(metadata.get("index_status") or "").strip().lower()
        return index_status in _DENIED_INDEX_STATUSES

    def _allow_protected(
        self,
        *,
        metadata: dict[str, object],
        access: AccessContext | None,
    ) -> bool:
        allowed_guilds = {value for value in self.allowed_guild_ids if value}
        admin_users = {value for value in self.admin_user_ids if value}

        if not allowed_guilds and not admin_users:
            return True
        if access is None:
            return False

        request_guild_id = str(access.guild_id or "").strip()
        if allowed_guilds and request_guild_id in allowed_guilds:
            scope_guild = str(metadata.get("guild_id") or "").strip()
            if not scope_guild or scope_guild == request_guild_id:
                return True

        if request_guild_id:
            return False

        request_user_id = str(access.user_id or "").strip()
        if bool(access.is_admin) and (not admin_users or request_user_id in admin_users):
            return True
        return bool(request_user_id and request_user_id in admin_users)

    @staticmethod
    def _allow_access_scope(
        scope: dict[str, object],
        *,
        access: AccessContext | None,
    ) -> bool:
        visibility = str(scope.get("visibility") or "admin").strip().lower()
        if visibility == "public":
            return True
        if access is None:
            return False
        if access.is_admin:
            return True
        if visibility == "admin":
            return False
        if visibility == "guild":
            scope_guild = str(scope.get("guild_id") or "").strip()
            return bool(scope_guild and access.guild_id and scope_guild == access.guild_id)
        if visibility == "role":
            allowed_roles = {str(value) for value in scope.get("role_ids") or []}
            return bool(allowed_roles & set(access.role_ids))
        if visibility == "private":
            allowed_users = {str(value) for value in scope.get("user_ids") or []}
            return bool(access.user_id and access.user_id in allowed_users)
        return False
