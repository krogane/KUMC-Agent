# Staged Rollout Runbook

## Stage 0: Local

1. Run unit tests for Wave 1 through Wave 7.
2. Run CLI smoke checks for `admin`, `automation`, `ask`, and `work`.
3. Confirm readiness status is `ready_with_manual_gates` or `ready`.

## Stage 1: Staging Guild

1. Register slash commands in a staging guild.
2. Keep automation auto-run disabled.
3. Run dry-run automation checks.
4. Validate audit log entries.

## Stage 2: Production Guild

1. Announce rollout window.
2. Enable only low-risk rules first.
3. Monitor audit logs and cost caps.
4. Roll back by disabling rules and reverting to the previous deployment.
