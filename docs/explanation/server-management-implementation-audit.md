# サーバー管理 実装再調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/kumc-agent.md` の「9. サーバー管理」
- `docs/design/server-management.md`
- `docs/plan/server-management.md`

調査対象:

- `src/kumc_agent/domain/models/minecraft.py`
- `src/kumc_agent/features/minecraft/*`
- `src/kumc_agent/infra/minecraft/*`
- `src/kumc_agent/features/workflow/service.py`
- `src/kumc_agent/apps/workflow.py`
- `src/kumc_agent/config/*`
- `configs/main/server_management.yaml`
- `assets/prompts/server_operation_planner.md`
- `src/kumc_agent/cli.py`
- `src/kumc_agent/frontends/discord/app.py`
- `src/kumc_agent/frontends/http/app.py`
- `tests/unit/test_minecraft_support.py`
- `tests/unit/test_server_management.py`
- `tests/unit/test_server_operation_executor.py`
- `tests/unit/test_cli_server_management_payload.py`
- `tests/unit/test_config_loading.py`
- `tests/unit/test_database_migrations.py`

`src/kumc_agent/infra/legacy` は、プロジェクト方針に従い調査対象から除外した。

## 結論

サーバー管理は、更新後の仕様に対して完全実装済みと判断する。

初期ActionSpecには `status`, `docker_ps`, `file_search`, `compose_up`, `compose_restart`, `restart`, `whitelist_update`, `backup_create`, `compose_down` が含まれる。自然言語Plannerは専用LLMを前提とし、LLM出力をJSON validationとActionSpec照合に通す。ラベル付き入力のdeterministic parserはCLI互換とテスト補助に限定される。

任意shell文字列を実行する経路はなく、executorは登録済みoperationだけを固定argv、`shell=False`、設定済みserver/service/path allow listで実行する。副作用操作は承認前に実行されず、high riskはadmin approval、criticalは二者承認またはdisabledになる。`file_search` はread-onlyだが `admin_dry_run` としてapprove必須である。

## 実装済み要素

| 項目 | 状態 | 主な実装箇所 |
| --- | --- | --- |
| 専用LLM Planner | 実装済み。自然言語はLLM JSONを検証し、unsupportedは候補保存なし | `features/minecraft/planner.py`, `assets/prompts/server_operation_planner.md` |
| 初期ActionSpec | `backup_create` を含む9操作を登録 | `features/minecraft/actions.py` |
| dry-run前validation | configured server、compose_dir、service allow list、file search root、player nameを検証 | `features/minecraft/service.py` |
| 承認ポリシー | `admin_dry_run` はapprove必須。highはadmin approval、criticalは二者承認またはdisabled | `features/minecraft/service.py`, `features/workflow/service.py` |
| `compose_down` | 設定済み `compose_dir` で `docker compose down` を実行 | `infra/minecraft/executor.py` |
| backup作成 | tar.gz archive作成、保存先設定、世代管理、path maskを実装 | `infra/minecraft/executor.py`, `features/minecraft/config.py` |
| executor失敗処理 | 例外/timeoutを `failed` として保存し、`running` 放置を防止 | `features/minecraft/service.py` |
| 状態snapshot | compose/whitelist実行前後に `docker compose ps --format json` を取得 | `infra/minecraft/executor.py` |
| docker ps出力 | parse後にmask/truncateし、service labelとsummaryを保持 | `infra/minecraft/executor.py` |
| 監査ログ | Workflow公開経路でoperation id、risk、approver、stdout/stderr、state snapshotをAuditEvent metadataへ保存 | `features/workflow/service.py` |
| sanitize | secret、内部IP、network key、PIN、unlock steps、絶対path、backup pathを外部payload/detailから抑止 | `features/foundation/payload_sanitizer.py`, `features/minecraft/service.py`, `infra/minecraft/repository.py` |
| 設定 | `server_management.backup` を含めconfigs配下で管理 | `configs/main/server_management.yaml`, `config/schema.py`, `config/load.py` |

## 仕様との差分再調査

| 旧差分 | 再調査結果 |
| --- | --- |
| executor例外やtimeoutが `failed` として保存されない | 解消。`execute()` が例外を捕捉し、failed resultを保存する |
| 監査ログがstdout/stderr/state/承認者を満たさない | 解消。Workflowのserver_operation各経路がAuditEvent metadataへ保存する |
| 専用LLM Plannerが未実装 | 解消。自然言語Plannerは専用LLM前提。deterministic parserはラベル付き入力専用 |
| unsupported自然文が `docker_ps` にfallbackする | 解消。候補を保存せず `対応操作を確認してください。` を返す |
| schema validationがdry-run保存前に不十分 | 解消。write/file/whitelist/backup操作は保存前に設定とallow listを検証する |
| `compose_down` の意味とexecutorが不一致 | 解消。`compose_down` は `docker compose down` と定義し、executorも一致 |
| status表示がdry-run-onlyのまま | 解消。executor接続状態とapproval gated writesを表示する |
| `file_search` のapproval policyが曖昧 | 解消。`admin_dry_run` はapprove必須として実装 |
| container/server state snapshotが実状態ではない | 解消。compose state snapshotを実コマンドで取得する |
| `docker_ps` 出力項目が少ない | 解消。service labelを含め、固定formatter summaryを保存する |
| `docker_ps` がtruncate後にparseする | 解消。raw stdoutをparseしてからsanitize/truncateする |
| file_search allow pathの基準が曖昧 | 解消。configs base dir基準または絶対pathを許可し、依頼pathも許可root配下なら相対/絶対の両方を許可 |
| direct service利用時はApprovalRecord/auditが残らない | 仕様整理済み。外部公開経路は `WorkflowService` 経由に限定し、audit/ApprovalRecordはWorkflowで保存する |
| 実executorの直接unit testがない | 解消。DockerPs/Compose/FileSearch/Backup executor testを追加 |
| backup作成ActionSpecが未実装 | 解消。`backup_create` を初期ActionSpecとして実装 |

残差分はない。

## 仕様改善点の反映

| # | 改善点 | 反映内容 |
| --- | --- | --- |
| 1 | backup作成の範囲固定 | `backup_create` を初期ActionSpecに含め、configとexecutorを追加 |
| 2 | Planner仕様を専用LLM前提に明文化 | 設計/計画/実装を専用LLM前提へ更新 |
| 3 | `admin_dry_run` の意味を具体化 | approve必須として実装し、`file_search` で検証 |
| 4 | `compose_down` の意味を定義 | `docker compose down` と定義し、service argをargvへ渡さない |
| 5 | server allow list未設定時の安全動作 | `status`/`docker_ps` 以外はconfigured server必須 |
| 6 | file search path基準 | configs base dir基準と絶対path許可を仕様化し実装 |
| 7 | unsupported/ambiguous応答 | `対応操作を確認してください。` を固定応答にした |
| 8 | executor失敗時transaction方針 | `running` 後の例外も `failed` resultとして保存 |
| 9 | auditとmetadataの責務分離 | Workflow公開経路でaudit、ServerOperation metadataで詳細保持、payloadではsanitize |
| 10 | dry-run/detail sanitize | command previewとdetailで絶対path/secretをマスク |
| 11 | docker output parse順 | parse前truncateを廃止 |
| 12 | テスト完了条件の具体化 | executor、approval、unknown自然文、失敗保存、absolute path、payload maskをunit test化 |
| 13 | runbookと承認人数の整合 | highはadmin approval、criticalは二者承認に統一 |

## 検証

```bash
app/.venv/bin/python -m unittest tests.unit.test_minecraft_support tests.unit.test_server_management tests.unit.test_server_operation_executor tests.unit.test_cli_server_management_payload tests.unit.test_database_migrations tests.unit.test_config_loading tests.unit.test_foundation_services tests.unit.test_automation_hardening
```

結果:

```text
Ran 53 tests in 0.105s
OK
```

追加でunit全体を実行した。

```bash
app/.venv/bin/python -m unittest discover tests/unit
```

結果:

```text
Ran 262 tests in 96.923s
OK
```
