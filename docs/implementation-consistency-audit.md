# Implementation Consistency Audit

Date: 2026-04-25

This audit compares the current codebase with the implemented Wave 1-7 surface and records unclear, inconsistent, inefficient, or unused implementation details. `src/kumc_agent/infra/legacy` was excluded from deletion review by project instruction.

## Scope

- Reviewed `README.md`, `docs/kumc-agent-redesign-v4.md`, `src/kumc_agent`, and `tests`.
- Checked static imports from `src/kumc_agent` and `tests`.
- Treated code as removable when it was unreachable from current CLI/API/tests, or when a legacy compatibility entrypoint had been superseded by the unified `bot` / `api` app entrypoints.

## Updated Documentation

- `README.md` was updated to describe the current Wave 1-7 implementation.
- `docs/current-system-detailed-design.md` was marked as pre-Wave-1-7 design material so it is not mistaken for the current runtime surface.
- The old statement that HTTP and DocGen are globally stub-only was corrected:
  - `kumc_agent.cli api` is the implemented API app entrypoint.
  - `frontends.http.app` is the HTTP route adapter used by `apps/api`.
  - request-based `DocGenService.run(request)` is implemented, while parameterless `DocGenService().run()` still raises `NotImplementedError` for compatibility with existing stub tests.

## Removed Dead Code

The following files had no static references from `src/kumc_agent` or `tests`, were outside `infra/legacy`, and were not current app entrypoints:

- `src/kumc_agent/domain/policies/recency.py`
- `src/kumc_agent/domain/ports/chunkers.py`
- `src/kumc_agent/domain/ports/loaders.py`
- `src/kumc_agent/domain/ports/parsers.py`
- `src/kumc_agent/domain/ports/rerankers.py`
- `src/kumc_agent/domain/ports/retrievers.py`
- `src/kumc_agent/domain/ports/scheduler.py`
- `src/kumc_agent/features/docgen/config.py`
- `src/kumc_agent/features/indexing/config.py`
- `src/kumc_agent/infra/parsing/office_openxml.py`
- `src/kumc_agent/infra/parsing/pdf_ocr.py`
- `src/kumc_agent/infra/scheduler/auto_index.py`
- `src/kumc_agent/infra/storage/sqlite.py`
- `src/kumc_agent/runtime/lifecycle.py`
- `src/kumc_agent/usecases/docgen/run.py`
- `src/kumc_agent/utils/timing.py`

Deletion rationale:

- The domain port files were prototype Protocols not used by current services. Current implementations use concrete repositories, loaders, and service constructors.
- `infra/parsing/*` contained placeholder parsing behavior. Google Drive / PDF / Office parsing is implemented in the loader path instead.
- `infra/scheduler/auto_index.py` duplicated scheduler behavior that is currently implemented in the Discord compatibility frontend and Wave 7 automation service.
- `infra/storage/sqlite.py` was not wired. Current Wave persistence uses file repositories and optional PostgreSQL repositories.
- `usecases/docgen/run.py` was bypassed by the Wave 5 workflow path, which calls `features.docgen.service.DocGenService`.

## Unified App Entrypoints

The runtime surface is unified around the app entrypoints:

- `kumc_agent.cli bot`: Discord slash-command process.
- `kumc_agent.cli api`: HTTP API process.
- `kumc_agent.cli worker`: worker process.

The old `discord` and `http` CLI entrypoints were removed. `frontends.discord` and `frontends.http` remain as protocol adapters that receive contexts from `apps/*`; they no longer build runtime/app contexts themselves.

## Compatibility Surfaces Kept

The following old behavior is still intentionally kept:

- `DocGenService().run()` without a request
  - Kept raising `NotImplementedError` because an existing unit test checks the parameterless call.
  - `DocGenService().run(DocGenRequest(...))` is implemented and covered by Wave 5 tests.

## Unclear Or Inconsistent Specifications

### DocGen Stub Naming

`tests/unit/test_stubs.py` still calls the requestless DocGen behavior a stub. This is now only true for `DocGenService().run()` without a request. Request-based draft generation is implemented through Wave 5 and workflow commands.

### Admin Versus Legacy CLI Commands

Several functions have both local/legacy CLI paths and newer operational paths:

- `index build/update` and `admin --action sync/reindex`
- `eval ragas` and `admin --action eval`
- `chat/tool rag` and `ask`

This keeps existing scripts working, but can confuse operators. New production operation should prefer `admin`, `ask`, and app entrypoints.

### Approval Type Scope

`approval --type` now accepts `task`, `event`, `schedule`, and future approval target types such as `announcement`, `automation_rule`, `asset_usage`, `server_operation`, `finance_record`, `member_assignment`, and `other`. `task` / `event` / `schedule` can be merged into 正本. Future target types currently record a safe no-side-effect approval record; executor-specific merge/apply behavior is still intentionally deferred.

### Asset And Member Workflow Scope

`image_search`, `image_usage_request`, and `member_search` now exist as safe minimum workflows. They rely on local/PostgreSQL repositories and do not yet include real image vector indexing, automated OCR/caption ingestion, profile synchronization, or external publishing.

### Automation Auto-Run Safety

Wave 7 includes `set_mode` with `dry_run`, `approval_required`, and `auto_run`, plus `KUMC_FEATURE_AUTOMATION_AUTO_RUN_MODE`. External side effects still need connector-specific production wiring and audit validation before enabling `auto_run`.

### External Reuse And Terms Metadata

Ingestion now records terms review metadata such as `terms_review_status` and `external_reuse_allowed`. This is local metadata only. Actual policy decisions for web reuse, X reuse, Minecraft Wiki content, and generated announcements still need owner review.

## Inefficient Or Risky Implementation Details

### File Repository Scans

Several local repositories use JSONL append/read patterns and derive current state by scanning files. This is acceptable for local development and tests, but can become O(n) per operation as data grows. Production should use PostgreSQL migrations and indexed tables for workflow, ingestion, retrieval, automation, audit, and cost records.

### Repeated Context Construction

Many CLI commands build app contexts independently. This is simple and reliable for one-shot commands, but inefficient for repeated admin operations. A long-running worker/API process should own shared context and repository instances.

### Bot Context Eagerness

The bot app builds broad app contexts up front. This reduces command latency but may initialize components that are not needed for a given command. If startup time or credentials become problematic, context creation should move to lazy per-command factories.

### Retrieval Fallback Quality

The local embedding fallback and file-backed retrieval path are useful for deterministic tests, but production answer quality depends on real embedding, reranking, and indexed retrieval stores. This should be validated with the eval command before deployment.

### Full Unit Discovery

Targeted Wave tests run without network. Full `unittest discover tests/unit` still includes pre-existing tests that may require external APIs, local models, or DNS. CI should split deterministic unit tests from integration/environment tests.

## Items Requiring User Or External Service Action

These are not safe to complete purely in code:

- Provision PostgreSQL and run `kumc_agent.cli db migrate`.
- Configure Discord application commands and bot token.
- Configure Google Drive / Notion / X / Minecraft / S3 credentials.
- Decide the policy for external content reuse before enabling announcement publication from scraped sources.
- Validate Minecraft operations on a staging server before enabling non-dry-run actions.
- Decide when local compatibility commands such as `index/eval/chat` can be removed or hidden.

## Follow-Up Recommendations

- Add an explicit integration-test marker or test naming convention for network/model-dependent tests.
- Promote JSONL repositories to PostgreSQL for production workloads.
- Extend approval records beyond `task` before enabling external side effects.
- Add a deployment document for the unified `bot` / `api` / `worker` processes.
