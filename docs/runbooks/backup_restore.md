# Backup / Restore Runbook

## Scope

- PostgreSQL schema and data
- JSONL fallback stores under `data/`
- audit and job logs under `logs/`
- generated artifacts under object storage prefix

## Backup Checklist

1. Confirm automation write actions are disabled or approval-required.
2. Export PostgreSQL with a timestamped snapshot.
3. Copy JSONL fallback stores and logs to the backup location.
4. Verify snapshot checksums.
5. Record snapshot ID in the incident log.

## Restore Test Harness

1. Restore into a staging database or isolated fallback directory.
2. Run migrations.
3. Run `kumc-agent admin --action readiness`.
4. Run smoke checks for `/ask`, `/work`, `/approval`, `/automation`, and `/admin`.
5. Keep production traffic paused until smoke checks pass.

This runbook is a checklist harness. It does not execute a live restore by itself.
