from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.config.load import load_runtime_config
from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.features.foundation.feature_flags import FeatureFlagService
from kumc_agent.features.foundation.health import FoundationHealthService
from kumc_agent.infra.audit.repository import AuditLogRepository, build_audit_repository
from kumc_agent.infra.cache.redis_client import RedisClient
from kumc_agent.infra.database.postgres import PostgresClient
from kumc_agent.infra.jobs.lifecycle import (
    JobLifecycleRepository,
    build_job_lifecycle_repository,
)
from kumc_agent.infra.migrations.runner import PostgresMigrationRunner
from kumc_agent.infra.object_storage.s3 import S3ObjectStorageClient


@dataclass(frozen=True)
class FoundationAppContext:
    config: RuntimeConfig
    postgres: PostgresClient
    redis: RedisClient
    object_storage: S3ObjectStorageClient
    audit_log: AuditLogRepository
    jobs: JobLifecycleRepository
    migrations: PostgresMigrationRunner
    health: FoundationHealthService
    feature_flags: FeatureFlagService


def build_foundation_app_context(*, base_dir: Path | None = None) -> FoundationAppContext:
    config = load_runtime_config(base_dir=base_dir)
    postgres = PostgresClient(config.infrastructure.database)
    redis = RedisClient(config.infrastructure.redis)
    object_storage = S3ObjectStorageClient(config.infrastructure.object_storage)
    audit_log = build_audit_repository(
        postgres=postgres,
        fallback_path=config.base_dir / "logs" / "audit.jsonl",
    )
    jobs = build_job_lifecycle_repository(
        postgres=postgres,
        redis_client=redis,
        fallback_path=config.base_dir / "logs" / "jobs.jsonl",
    )
    migrations = PostgresMigrationRunner(
        client=postgres,
        config=config.infrastructure.migrations,
    )
    feature_flags = FeatureFlagService(config.features.risk_flags)
    health = FoundationHealthService(
        postgres=postgres,
        redis=redis,
        object_storage=object_storage,
        audit_log=audit_log,
        feature_flags=feature_flags,
    )
    return FoundationAppContext(
        config=config,
        postgres=postgres,
        redis=redis,
        object_storage=object_storage,
        audit_log=audit_log,
        jobs=jobs,
        migrations=migrations,
        health=health,
        feature_flags=feature_flags,
    )
