from __future__ import annotations

from dataclasses import dataclass

from kumc_agent.domain.models.retrieval import AccessContext


@dataclass(frozen=True)
class ServerManagementAccessPolicy:
    admin_user_ids: tuple[str, ...] = tuple()

    def is_admin(self, access: AccessContext) -> bool:
        if access.is_admin:
            return True
        return bool(access.user_id) and access.user_id in set(self.admin_user_ids)

    def forbidden_text(self) -> str:
        return "権限がありません。サーバー管理情報の有無は表示しません。"

    def forbidden_metadata(self) -> dict[str, object]:
        return {"authorized": False, "policy_decision": "denied"}
