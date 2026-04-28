from __future__ import annotations

from kumc_agent.domain.models.minecraft import ActionSpec


class MinecraftActionSpecRegistry:
    def __init__(self) -> None:
        specs = (
            ActionSpec(
                operation="status",
                description="Minecraft support safety status.",
                risk_level="low",
                approval_policy="self",
                read_only=True,
                executor_name="status",
            ),
            ActionSpec(
                operation="docker_ps",
                description="Inspect Minecraft-related containers.",
                risk_level="low",
                approval_policy="admin",
                optional_args=("service_name",),
                read_only=True,
                executor_name="docker_ps",
            ),
            ActionSpec(
                operation="file_search",
                description="Search configured server files by path/query.",
                risk_level="medium",
                approval_policy="admin_dry_run",
                required_args=("path", "query"),
                read_only=True,
                executor_name="file_search",
            ),
            ActionSpec(
                operation="compose_up",
                description="Plan docker compose up for a configured service.",
                risk_level="high",
                approval_policy="admin_approval",
                required_args=("service_name",),
                executor_name="compose",
            ),
            ActionSpec(
                operation="compose_restart",
                description="Plan docker compose restart for a configured service.",
                risk_level="high",
                approval_policy="admin_approval",
                required_args=("service_name",),
                executor_name="compose",
            ),
            ActionSpec(
                operation="restart",
                description="Plan Minecraft server restart.",
                risk_level="high",
                approval_policy="admin_approval",
                optional_args=("service_name",),
                executor_name="compose",
            ),
            ActionSpec(
                operation="whitelist_update",
                description="Plan whitelist add/remove operation.",
                risk_level="high",
                approval_policy="admin_approval",
                required_args=("player_name",),
                optional_args=("whitelist_action", "service_name"),
                executor_name="whitelist",
            ),
            ActionSpec(
                operation="backup_create",
                description="Create a configured server backup archive.",
                risk_level="high",
                approval_policy="admin_approval",
                executor_name="backup",
            ),
            ActionSpec(
                operation="compose_down",
                description="Plan docker compose down.",
                risk_level="critical",
                approval_policy="two_person_or_disabled",
                optional_args=("service_name",),
                executor_name="compose",
            ),
        )
        self._specs = {spec.operation: spec for spec in specs}

    def get(self, operation: str) -> ActionSpec:
        return self._specs[_normalize_operation(operation)]

    def list(self) -> tuple[ActionSpec, ...]:
        return tuple(self._specs.values())

    def has(self, operation: str) -> bool:
        return _normalize_operation(operation) in self._specs


def _normalize_operation(value: str) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    aliases = {
        "mc_status": "status",
        "docker": "docker_ps",
        "ps": "docker_ps",
        "compose": "compose_up",
        "up": "compose_up",
        "down": "compose_down",
        "restart_mc_server": "restart",
        "server_restart": "restart",
        "whitelist": "whitelist_update",
        "backup": "backup_create",
        "backup_create_server": "backup_create",
    }
    return aliases.get(normalized, normalized)
