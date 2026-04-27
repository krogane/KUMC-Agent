from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

from kumc_agent.config.schema import RuntimeConfig
from kumc_agent.domain.models.hardening import (
    ProductionReadinessCheck,
    ProductionReadinessReport,
)
from kumc_agent.features.foundation.feature_flags import FeatureFlagService
from kumc_agent.features.hardening.cost_cap import CostCapPolicy
from kumc_agent.features.hardening.prompt_injection import PromptInjectionRedTeam

_REQUIRED_COMMANDS = ("/ask", "/work", "/approval", "/automation", "/admin")
_REQUIRED_RUNBOOKS = (
    "backup_restore.md",
    "incident_response.md",
    "nf_day.md",
    "minecraft_operation_rollback.md",
    "staged_rollout.md",
)


@dataclass(frozen=True)
class ProductionReadinessService:
    config: RuntimeConfig
    feature_flags: FeatureFlagService
    runbook_dir: Path
    registered_commands: tuple[str, ...] = _REQUIRED_COMMANDS
    cost_cap: CostCapPolicy = CostCapPolicy()

    def report(self) -> ProductionReadinessReport:
        checks = (
            self._command_check(),
            self._risk_flags_check(),
            self._migration_check(),
            self._runbook_check(),
            self._prompt_injection_check(),
            self._cost_cap_check(),
            self._backup_restore_harness_check(),
            self._staged_rollout_check(),
        )
        status = _overall_status(checks)
        return ProductionReadinessReport(
            status=status,
            checked_at=datetime.now(UTC),
            checks=checks,
            summary=_summary(status, checks),
            metadata={
                "rollback_demo": "runbook_harness_ready",
                "load_test": "harness_ready",
                "backup_restore_test": "harness_ready",
            },
        )

    def cost_report(self) -> dict[str, object]:
        agent_steps_path = self.config.base_dir / "data" / "agentic" / "agent_steps.jsonl"
        agent_runs_path = self.config.base_dir / "data" / "agentic" / "agent_runs.jsonl"
        total_cost = 0.0
        step_count = 0
        tool_counts: dict[str, int] = {}
        tool_failures: dict[str, int] = {}
        if agent_steps_path.exists():
            for line in agent_steps_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                total_cost += float(payload.get("cost_usd") or 0.0)
                step_count += 1
                output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
                tool_name = str(output.get("tool_name") or "")
                if tool_name:
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                    if str(payload.get("status") or "") in {"failed", "insufficient_input"}:
                        tool_failures[tool_name] = tool_failures.get(tool_name, 0) + 1
        latest_runs: dict[str, dict[str, object]] = {}
        if agent_runs_path.exists():
            for line in agent_runs_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict) and payload.get("id"):
                    latest_runs[str(payload["id"])] = payload
        status_counts: dict[str, int] = {}
        for payload in latest_runs.values():
            status = str(payload.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        warnings = self.cost_cap.check(
            projected_daily_usd=total_cost,
            projected_run_usd=0.0,
        )
        return {
            "total_agentic_cost_usd": round(total_cost, 4),
            "agent_step_count": step_count,
            "agent_run_count": len(latest_runs),
            "agent_run_status_counts": status_counts,
            "agent_tool_counts": tool_counts,
            "agent_tool_failure_counts": tool_failures,
            "daily_cap_usd": self.cost_cap.daily_usd_cap,
            "per_run_cap_usd": self.cost_cap.per_run_usd_cap,
            "warnings": list(warnings),
            "source": str(agent_steps_path),
        }

    def _command_check(self) -> ProductionReadinessCheck:
        missing = [command for command in _REQUIRED_COMMANDS if command not in self.registered_commands]
        return ProductionReadinessCheck(
            id="slash_command_registry",
            title="Slash command registry",
            status="pass" if not missing else "fail",
            detail="required commands registered" if not missing else f"missing: {', '.join(missing)}",
            metadata={"registered": list(self.registered_commands)},
        )

    def _risk_flags_check(self) -> ProductionReadinessCheck:
        modes = self.feature_flags.modes()
        unsafe = [
            name
            for name in (
                "external_posting",
                "minecraft_server_ops",
                "accounting_finalize",
                "auto_reply",
                "image_generation",
                "automation_auto_run",
            )
            if modes.get(name) == "enabled"
        ]
        return ProductionReadinessCheck(
            id="risk_flags",
            title="Risk flags are conservative",
            status="pass" if not unsafe else "fail",
            detail="high-impact actions require approval or are disabled"
            if not unsafe
            else f"unsafe enabled flags: {', '.join(unsafe)}",
            metadata=modes,
        )

    def _migration_check(self) -> ProductionReadinessCheck:
        path = self.config.infrastructure.migrations.directory / "007_automation_rules_runs.sql"
        return ProductionReadinessCheck(
            id="automation_migration",
            title="Automation migration exists",
            status="pass" if path.exists() else "fail",
            detail=str(path),
        )

    def _runbook_check(self) -> ProductionReadinessCheck:
        missing = [name for name in _REQUIRED_RUNBOOKS if not (self.runbook_dir / name).exists()]
        return ProductionReadinessCheck(
            id="runbooks",
            title="Runbooks are present",
            status="pass" if not missing else "fail",
            detail="all required runbooks exist" if not missing else f"missing: {', '.join(missing)}",
            metadata={"required": list(_REQUIRED_RUNBOOKS)},
        )

    def _prompt_injection_check(self) -> ProductionReadinessCheck:
        findings = PromptInjectionRedTeam().run_default_eval()
        critical_failures = [
            finding
            for finding in findings
            if finding.severity == "high" and not finding.pattern
        ]
        return ProductionReadinessCheck(
            id="prompt_injection_redteam",
            title="Prompt injection red-team cases are detected",
            status="pass" if not critical_failures and len(findings) >= 4 else "fail",
            detail=f"detected findings: {len(findings)}",
            metadata={"findings": [finding.__dict__ for finding in findings]},
        )

    def _cost_cap_check(self) -> ProductionReadinessCheck:
        warnings = self.cost_cap.check(projected_daily_usd=0.0, projected_run_usd=0.0)
        return ProductionReadinessCheck(
            id="cost_cap",
            title="Cost cap policy is configured",
            status="pass" if not warnings else "fail",
            detail=(
                f"daily={self.cost_cap.daily_usd_cap} USD, "
                f"per_run={self.cost_cap.per_run_usd_cap} USD"
            ),
            metadata={"warnings": list(warnings)},
        )

    @staticmethod
    def _backup_restore_harness_check() -> ProductionReadinessCheck:
        return ProductionReadinessCheck(
            id="backup_restore_harness",
            title="Backup / restore test harness",
            status="manual_gate",
            detail="runbook checklist is ready; no live restore is executed by this harness",
        )

    @staticmethod
    def _staged_rollout_check() -> ProductionReadinessCheck:
        return ProductionReadinessCheck(
            id="staged_rollout",
            title="Staged rollout plan",
            status="manual_gate",
            detail="production guild rollout requires manual approval after smoke checks",
        )


def _overall_status(checks: tuple[ProductionReadinessCheck, ...]) -> str:
    if any(check.status == "fail" for check in checks):
        return "not_ready"
    if any(check.status == "manual_gate" for check in checks):
        return "ready_with_manual_gates"
    return "ready"


def _summary(status: str, checks: tuple[ProductionReadinessCheck, ...]) -> str:
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return f"{status}: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
