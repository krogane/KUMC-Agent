# NF Day Operations Runbook

## Before Opening

1. Run `kumc-agent db migrate`.
2. Run `kumc-agent admin --action readiness`.
3. Confirm `/ask`, `/work`, `/approval`, `/automation`, and `/admin` are registered.
4. Keep external posting, auto-reply, image generation, and Minecraft writes approval-required or disabled.

## During Operations

1. Use `/automation action:list` to inspect enabled rules.
2. Use dry-run before any manual automation run.
3. Route external announcements through approval.
4. Record production changes in the incident or operations log.

## After Closing

1. Export audit logs and automation run logs.
2. Disable temporary rules.
3. Record follow-up tasks with `/work type:task_extract`.
