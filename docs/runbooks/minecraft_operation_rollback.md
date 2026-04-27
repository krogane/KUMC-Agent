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

## Operation Notes

- `docker_ps`: read-only. No rollback is needed; preserve the result if it was used during incident triage.
- `compose_up`: if startup causes issues, approve and execute the matching stop/down operation for the same configured service.
- `compose_restart` / `restart`: inspect logs first, then repeat restart only if the service is stuck. Restore the latest known-good backup if data corruption is suspected.
- `compose_down`: critical. Keep disabled unless two-person approval and an explicit recovery path are available. Roll forward with `compose_up` for the same configured service.
- `whitelist_update`: rollback is the inverse operation for the same player name and server.

Do not write secrets, internal IP addresses, network keys, PINs, or unlock steps into rollback notes.
