from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from kumc_agent.domain.models.audit import AuditEvent
from kumc_agent.domain.models.health import ComponentHealth, HealthReport
from kumc_agent.features.foundation.feature_flags import FeatureFlagService
from kumc_agent.features.foundation.tracing import current_trace_id
from kumc_agent.infra.audit.repository import AuditLogRepository
from kumc_agent.infra.cache.redis_client import RedisClient
from kumc_agent.infra.database.postgres import PostgresClient
from kumc_agent.infra.object_storage.s3 import S3ObjectStorageClient


@dataclass(frozen=True)
class FoundationHealthService:
    postgres: PostgresClient
    redis: RedisClient
    object_storage: S3ObjectStorageClient
    audit_log: AuditLogRepository
    feature_flags: FeatureFlagService

    def check(
        self,
        *,
        actor_id: str = "system",
        actor_type: str = "system",
    ) -> HealthReport:
        components: list[ComponentHealth] = [
            self.postgres.check(),
            self.redis.check(),
            self.object_storage.check(),
            self._feature_flag_health(),
        ]
        report = self._build_report(components)
        try:
            self.audit_log.append(
                AuditEvent(
                    action="admin.health",
                    actor_id=actor_id,
                    actor_type=actor_type,
                    target="foundation",
                    outcome=report.status,
                    risk_level="low",
                    trace_id=current_trace_id(),
                    metadata=report.as_dict(),
                )
            )
            components.append(ComponentHealth(name="audit_log", status="healthy"))
        except Exception as exc:
            components.append(
                ComponentHealth(name="audit_log", status="unhealthy", detail=str(exc))
            )
        return self._build_report(components)

    def _feature_flag_health(self) -> ComponentHealth:
        modes = self.feature_flags.modes()
        disabled = [
            name
            for name, mode in modes.items()
            if mode in {"disabled", "approval_required"}
        ]
        return ComponentHealth(
            name="feature_flags",
            status="healthy",
            detail=", ".join(disabled),
        )

    @staticmethod
    def _build_report(components: list[ComponentHealth]) -> HealthReport:
        if any(component.status == "unhealthy" for component in components):
            status = "unhealthy"
        elif any(component.status == "not_configured" for component in components):
            status = "degraded"
        else:
            status = "healthy"
        return HealthReport(
            status=status,
            checked_at=datetime.now(UTC),
            components=tuple(components),
        )
