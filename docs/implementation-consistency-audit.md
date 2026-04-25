# Implementation Consistency Audit

Date: 2026-04-25

This audit compares the current codebase with the implemented Wave 1-7 surface and records unclear, inconsistent, inefficient, or unused implementation details. `src/kumc_agent/infra/legacy` was excluded from deletion review by project instruction.

## Scope

- Reviewed `README.md`, `docs/kumc-agent-redesign-v4.md`, `src/kumc_agent`, and `tests`.
- Checked static imports from `src/kumc_agent` and `tests`.
- Treated code as removable only when it was unreachable from current CLI/API/tests, not a legacy compatibility entrypoint, and not part of `infra/legacy`.

## Updated Documentation

- `README.md` was updated to describe the current Wave 1-7 implementation.
- `docs/current-system-detailed-design.md` was marked as pre-Wave-1-7 design material so it is not mistaken for the current runtime surface.
- The old statement that HTTP and DocGen are globally stub-only was corrected:
  - `kumc_agent.cli api` is the implemented API app entrypoint.
  - `kumc_agent.cli http` / `frontends.http.app` remains a compatibility stub.
  - request-based `DocGenService.run(request)` is implemented, while parameterless `DocGenService().run()` still raises `NotImplementedError` for compatibility with existing stub tests.

## Removed Dead Code

The following files had no static references from `src/kumc_agent` or `tests`, were outside `infra/legacy`, and were not current app entrypoints:

- `src/kumc_agent/domain/policies/recency.py`
- `src/kumc_agent/domain/policies/refusal.py`
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

## Compatibility Surfaces Kept

The following files look stub-like or old, but were not deleted because they are still reachable or tested:

- `src/kumc_agent/frontends/http/app.py`
  - Kept because `kumc_agent.cli http` imports it and `tests/unit/test_stubs.py` asserts the compatibility stub behavior.
  - The implemented API app is `src/kumc_agent/apps/api/app.py`.
- `src/kumc_agent/frontends/discord/*`
  - Kept because `kumc_agent.cli discord`, legacy `/ai` commands, and `tests/unit/test_discord_commands.py` still use it.
  - The Wave app entrypoint is `kumc_agent.cli bot`.
- `DocGenService().run()` without a request
  - Kept raising `NotImplementedError` because an existing unit test checks the parameterless call.
  - `DocGenService().run(DocGenRequest(...))` is implemented and covered by Wave 5 tests.

## Unclear Or Inconsistent Specifications

### Dual API Entrypoints

There are two HTTP-related surfaces:

- `kumc_agent.cli api`: implemented Wave 1 API app.
- `kumc_agent.cli http`: compatibility stub.

This is intentional for compatibility, but the command naming is easy to misread. The README now directs new usage to `api`. A later cleanup can remove `http` only after tests and external scripts no longer depend on it.

### Dual Discord Entrypoints

There are two Discord surfaces:

- `kumc_agent.cli bot`: Wave slash-command app context.
- `kumc_agent.cli discord`: legacy-compatible `/ai` frontend.

The split is useful during migration, but it should be treated as temporary. Operational docs should name which process is deployed in production.

### DocGen Stub Naming

`tests/unit/test_stubs.py` still calls the requestless DocGen behavior a stub. This is now only true for `DocGenService().run()` without a request. Request-based draft generation is implemented through Wave 5 and workflow commands.

### Admin Versus Legacy CLI Commands

Several functions now have two command paths:

- `index build/update` and `admin --action sync/reindex`
- `eval ragas` and `admin --action eval`
- `chat/tool rag` and `ask`

This keeps existing scripts working, but can confuse operators. New production operation should prefer `admin`, `ask`, and app entrypoints. Legacy-compatible commands should remain until migration consumers are known.

### Approval Type Scope

`approval --type` currently accepts only `task`. The redesign includes broader approval concepts for announcements, external posting, automation, and Minecraft operations. Current code routes those as draft / dry-run first, but approval object coverage is not yet uniform.

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

- Confirm which Discord process should be deployed: `bot` or legacy-compatible `discord`.
- Provision PostgreSQL and run `kumc_agent.cli db migrate`.
- Configure Discord application commands and bot token.
- Configure Google Drive / Notion / X / Minecraft / S3 credentials.
- Decide the policy for external content reuse before enabling announcement publication from scraped sources.
- Validate Minecraft operations on a staging server before enabling non-dry-run actions.
- Decide when compatibility commands (`http`, `discord`, legacy `index/eval/chat`) can be removed.

## Follow-Up Recommendations

- Rename or hide `http` once no deployment script depends on it.
- Add an explicit integration-test marker or test naming convention for network/model-dependent tests.
- Promote JSONL repositories to PostgreSQL for production workloads.
- Extend approval records beyond `task` before enabling external side effects.
- Add a deployment document that chooses one production entrypoint per process.
