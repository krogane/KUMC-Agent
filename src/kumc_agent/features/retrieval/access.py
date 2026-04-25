from __future__ import annotations

from kumc_agent.domain.models.chunk import Chunk
from kumc_agent.domain.models.retrieval import AccessContext
from kumc_agent.domain.policies.chunk_visibility import is_chunk_allowed_for_answer_context


def is_chunk_visible(chunk: Chunk, access: AccessContext) -> bool:
    if not is_chunk_allowed_for_answer_context(chunk):
        return False
    metadata = dict(chunk.metadata or {})
    scope = metadata.get("access_scope")
    if not isinstance(scope, dict):
        return access.is_admin
    visibility = str(scope.get("visibility") or "admin").strip().lower()
    if visibility == "public":
        return True
    if access.is_admin:
        return True
    if visibility == "admin":
        return False
    if visibility == "guild":
        guild_id = str(scope.get("guild_id") or "").strip()
        return bool(guild_id and access.guild_id and guild_id == access.guild_id)
    if visibility == "role":
        allowed = {str(value) for value in scope.get("role_ids") or []}
        return bool(allowed & set(access.role_ids))
    if visibility == "private":
        allowed_users = {str(value) for value in scope.get("user_ids") or []}
        return bool(access.user_id and access.user_id in allowed_users)
    return False
