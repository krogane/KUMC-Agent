# Incident Response Runbook

## First Actions

1. Disable high-impact automation rules with `/automation action:disable`.
2. Set risky feature flags to `disabled` or `approval_required`.
3. For server operation incidents, set `features.risk_flags.minecraft_server_ops` to `disabled` in `configs/ops/features.yaml` and redeploy/restart the process.
4. Capture health and readiness output.
5. Preserve audit logs before making further changes.

## Triage

1. Identify affected guild, command, rule, user, and trace ID.
2. Check audit log entries for action runs and automation runs.
3. For server operations, inspect `workflow.server_operation.*` audit entries and `server_operations` status transitions.
4. Check prompt-injection findings for retrieved or connected content.
5. Decide whether rollback, config-only mitigation, or code rollback is needed.

## Closeout

1. Document root cause and affected users.
2. Add regression tests or red-team cases.
3. Re-enable automation one rule at a time after smoke checks.
