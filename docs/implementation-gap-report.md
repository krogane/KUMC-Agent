# KUMC-Agent Full Implementation Gap Report

作成日: 2026-04-25

対象: `docs/kumc-agent-redesign-v4.md` の Wave 1-7、GitHub Issues 初期バックログ、本番公開条件。

## Summary

Wave 1-7 は基盤と主要経路を実装済みだが、設計書が求める「フルプロダクト」には未到達である。現時点の実装は、外部副作用を抑えた安全なローカル実装、dry-run、approval gate、file/PostgreSQL repository の土台が中心である。

今回追加した実装:

- `/work type:task_add|task_list|task_done`
- `/approval action:show|edit` for task targets
- `/admin action:sync|eval|feature_flags|permissions|reindex|cost_report`
- ingestion metadata の `terms_review_status` / `external_reuse_allowed`
- `KUMC_FEATURE_AUTOMATION_AUTO_RUN_MODE`

## Wave Status

| Wave | Status | 主な実装済み | 主な未実装 / partial |
|---|---|---|---|
| Wave 1 Foundation | Partial | `apps/bot`, `apps/api`, `apps/worker`, config, feature flags, audit, migrations, health | OpenTelemetry 本格連携、Redis queue worker、CI migration gate、production Secret Manager |
| Wave 2 Connector / Ingestion / SecretFinding | Partial | connector interface、file-backed Drive/Discord/Notion/Hatena/X/Crafters/Minecraft Wiki、raw snapshot、chunking、secret finding、deleted marker | Drive changes API、Discord incremental capture、外部 source の利用規約確認、fixture integration tests、権限変更の自動反映 |
| Wave 3 Retrieval / Ask | Partial | dense/sparse/RRF/MMR/context/citation、ACL filter、`/ask` | pgvector 本番検索、外部 embedding/rerank adapter の本番設定、eval gate、Minecraft edition/version aware answer の強化 |
| Wave 4 Workflow | Partial | meeting/task/event/schedule、TaskCandidate approval、今回 task_add/list/done と show/edit 追加 | Event/Schedule candidate approval、Google Calendar 連携、重複検出の高度化、汎用 WorkflowCandidate |
| Wave 5 Agentic / DocGen / Announcement | Partial | state-machine Agentic Search、budget、DocGen Markdown、Announcement draft、fact check | pairwise judge、本格 template/exporter、Agentic eval 基準、外部公開 safety eval の網羅 |
| Wave 6 Minecraft | Partial | ActionSpec registry、dry-run、ServerOperation repository、approval-required 保存 | isolated executor、二者承認の実行、実サーバー連携、ログ収集、backup 実行 |
| Wave 7 Automation / Hardening | Partial | automation rule/run、enable/disable/mode、prompt injection red-team harness、cost cap policy、runbooks、readiness report | load test 実行、backup/restore 実演、1週間 staging 運用、production guild 実公開確認 |

## Command Coverage

| Command | Implemented | Remaining |
|---|---|---|
| `/ask` | `source`, `mode`, `depth`; deep routes to Agentic Search | image/member/task/event source routing; Minecraft version/edition specialized answer |
| `/work` | meeting, task_extract, task_add/list/done, event, schedule, doc/x/announcement, mc | image_search, image_usage_request, member_search, task candidate duplicate review UI |
| `/approval` | task list/show/edit/approve/reject | event, announcement, automation_rule, asset_usage, server_operation, finance_record, member_assignment |
| `/automation` | list/show/dry_run/run/enable/disable/set_mode | scheduler trigger runner, approval integration for automation candidates, external posting executor |
| `/admin` | health/readiness/sync/eval/feature_flags/permissions/reindex/cost_report | role-id based production policy editor, cancellation, full CI eval gate, live cost backend |

## Data Model Gaps

- `WorkflowCandidate` is not implemented.
- `Asset`, `AssetUsageRequest`, `member_profiles`, `finance_records`, `indexing_runs`, `action_runs`, and full `workflow_runs` tables are not implemented.
- `approval_records` exists, but current workflow only handles task approvals.
- `agent_runs` / `agent_steps` exist; action and automation trace integration is partial.
- `terms_review_status` is now stored in source/document/chunk metadata, but review workflow is not implemented.

## Security / Safety Gaps

- SecretFinding exists and deny chunks are excluded, but full `secret_redaction` eval set and CI gate are not implemented.
- Prompt injection detection exists as a red-team harness, but all retrieved contexts are not yet centrally wrapped with explicit untrusted delimiters.
- External-public safety gate exists in DocGen/Announcement in lightweight form only.
- Production secrets still support `.env`; Secret Manager migration is operational work.
- Role permission resolution is config-based; full Discord role policy editor is not implemented.

## Retrieval / Agentic Gaps

- Retrieval currently uses file/PostgreSQL chunk stores and hashed/local fallback embeddings when external provider is absent.
- pgvector-native search and external rerank production paths are not complete.
- Agentic Search is deterministic and bounded, but does not yet run a full search/read/verify tool catalog with pairwise judge.
- Cost is tracked for Agentic steps in a simple local cost model, not a provider-billed ledger.

## Workflow Gaps

- Task add/list/done are now implemented.
- Event and Schedule are stored directly, not as approval-gated candidates.
- Meeting preparation uses heuristic extraction plus retrieval; no LLM schema extraction gate.
- Member search and image/asset workflow are not implemented.
- Google Calendar and external posting integrations require external setup.

## Automation Gaps

- Automation rules and run records exist.
- `auto_run` is disabled by default and only records executed-equivalent with no side effects.
- Scheduler/webhook trigger runner is not implemented.
- Automation approval flow for creating new rule candidates is not implemented.
- External posting, role change, server operation, and accounting finalize remain approval-required or blocked.

## Production Hardening Gaps

- Readiness report exists, but returns `ready_with_manual_gates` until manual gates are completed.
- Load test is documented as a harness/runbook, not executed.
- Backup/restore is documented as a harness/runbook, not executed.
- Rollback has runbooks, but no live demonstration record.
- Staging one-week operation and quality review require user/team operation.

