# OpenClaw Migration Report

## Scope

- Entry points: Discord / CLI / REPL
- OpenClaw integration style: thin CLI wrapper (`openclaw` command)
- RAG invocation: `kumc-agent tool rag`

## Before / After

### Query entry (CLI / REPL)

- Before: local `ChatAnswerUsecase` directly
- After: OpenClaw-first path via wrapper; local RAG fallback on failure

### Discord

- Before: KUMC Discord frontend handled text + maintenance + VC
- After: OpenClaw handles text/standard commands; KUMC Discord frontend remains as VC sidecar

### Chat history

- Before: `RagService` in-process history bucket
- After: OpenClaw-managed conversation history in OpenClaw mode; local history recording disabled for OpenClaw tool path

### Auto index scheduler

- Before: Discord frontend internal `auto_index_loop`
- After: OpenClaw cron as primary; internal auto loop disabled in OpenClaw mode

## Compatibility

- Maintained:
  - Existing RAG pipeline and outputs
  - Existing answer logging format
  - VC functionality via sidecar mode
- Changed:
  - Discord command UX is expected to follow OpenClaw standard commands

## Known Differences / Operational Notes

- If `openclaw` command is missing or fails, CLI/REPL automatically fall back to local RAG.
- Discord text fallback is operationally controlled by toggling `KUMC_OPENCLAW_ENABLED`.
- OpenClaw installation, agent setup, Discord adapter setup, and cron registration are external operational tasks.
