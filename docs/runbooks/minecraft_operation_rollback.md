# Minecraft Operation Rollback Runbook

## Scope

Minecraft support is dry-run-first. Write operations require approval and should have a rollback note before execution.

## Before Any Write

1. Capture current server status.
2. Confirm the requested operation, server, service, and operator.
3. Confirm two approvers for high-risk operations.
4. Verify backup or snapshot availability.

## Rollback Checklist

1. Stop new write requests.
2. Restore the last known-good configuration or snapshot.
3. Restart only the affected service.
4. Verify player access, whitelist state, and server logs.
5. Record the rollback in audit notes.
