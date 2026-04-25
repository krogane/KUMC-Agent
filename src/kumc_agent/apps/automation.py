from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.features.automation import AutomationService
from kumc_agent.features.hardening import ProductionReadinessService
from kumc_agent.infra.automation import build_automation_repository
from kumc_agent.infra.operations import build_operations_repository


@dataclass(frozen=True)
class AutomationAppContext:
    automation: AutomationService
    readiness: ProductionReadinessService


def build_automation_app_context(
    *,
    base_dir: Path | None = None,
    seed_defaults: bool = True,
) -> AutomationAppContext:
    foundation = build_foundation_app_context(base_dir=base_dir)
    repository = build_automation_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "automation",
    )
    operations = build_operations_repository(
        postgres=foundation.postgres,
        fallback_dir=foundation.config.base_dir / "data" / "operations",
    )
    automation = AutomationService(
        repository=repository,
        feature_flags=foundation.feature_flags,
        audit_log=foundation.audit_log,
        operations=operations,
    )
    if seed_defaults:
        automation.seed_defaults()
    readiness = ProductionReadinessService(
        config=foundation.config,
        feature_flags=foundation.feature_flags,
        runbook_dir=foundation.config.base_dir / "docs" / "runbooks",
    )
    return AutomationAppContext(automation=automation, readiness=readiness)
