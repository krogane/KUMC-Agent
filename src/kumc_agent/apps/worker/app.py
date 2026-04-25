from __future__ import annotations

import logging
from pathlib import Path

from kumc_agent.apps.automation import build_automation_app_context
from kumc_agent.apps.foundation import build_foundation_app_context
from kumc_agent.utils.logging import configure_logging, default_execution_log_path

logger = logging.getLogger(__name__)


def run_once(*, base_dir: Path | None = None) -> dict[str, object]:
    context = build_foundation_app_context(base_dir=base_dir)
    automation = build_automation_app_context(base_dir=base_dir)
    job = context.jobs.start("worker.health")
    try:
        report = context.health.check(actor_id="worker", actor_type="service")
        readiness = automation.readiness.report()
        rules = automation.automation.seed_defaults()
        context.jobs.complete(
            job,
            metadata={
                "health_status": report.status,
                "readiness_status": readiness.status,
                "automation_rules": len(rules),
            },
        )
        payload = report.as_dict()
        payload["readiness_status"] = readiness.status
        payload["automation_rules"] = len(rules)
        return payload
    except Exception as exc:
        context.jobs.fail(job, str(exc))
        raise


def main(*, base_dir: Path | None = None) -> None:
    context = build_foundation_app_context(base_dir=base_dir)
    configure_logging(
        context.config.app.log_level,
        file_path=default_execution_log_path(base_dir=context.config.base_dir),
    )
    result = run_once(base_dir=base_dir)
    logger.info("Wave 1 worker skeleton completed. result=%s", result)
