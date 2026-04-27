# サーバー管理 実装計画

## 1. 方針
`docs/design/kumc-agent.md` と `docs/design/server-management.md` に従い、サーバー管理を実装する。

実装では `src/kumc_agent/infra/legacy` を参照・依存しない。既存の共通部品は `domain.models.minecraft`、`features.minecraft.service.MinecraftSupportService`、`features.minecraft.actions.MinecraftActionSpecRegistry`、`infra.minecraft.repository`、`features.workflow.service.WorkflowService`、`domain.models.retrieval.AccessContext`、`infra.audit`、`FeatureFlagService` を優先して使う。現行実装と設計が矛盾する場合は `kumc-agent.md` を優先する。

初期実装では、Minecraftサーバー運用を対象にする。汎用サーバー管理への拡張はActionSpecと設定を増やして対応できる形にする。

## 2. 完了条件
- サーバー管理操作はadminだけが受付できる。
- 非adminへの拒否応答にserver名、container名、pending件数、operation idなどの内部情報が含まれない。
- 自然言語入力から1件以上の操作計画を抽出できる。
- 操作計画は定義済みActionSpecに限定され、任意shell commandを実行しない。
- required args不足時は候補を保存せず、不足項目を質問できる。
- 対象server、compose directory、service name、path、player nameをschema validationできる。
- `docker_ps` はadmin限定かつ事前承認不要のread-only操作として実行できる。
- `compose_up`、`compose_restart`、`restart`、`compose_down` はdry-runで影響範囲、想定停止時間、rollback方針を提示できる。
- 副作用操作は承認前に実行されない。
- high risk以上はadmin承認必須になる。
- critical操作は二者承認またはdisabledになる。
- `ServerOperation` のstatus、承認者、実行結果をJSONL/Postgres双方へ保存できる。
- 承認後に定義済みexecutorだけが実行される。
- stdout/stderr、server state、container state、実行者、承認者が監査ログへ保存される。
- 一般回答、CLI payload、外部連携payloadからsecret、内部IP、ネットワークキー、PIN、解錠手順が除外・マスクされる。
- CLIや外部連携payloadの診断情報が `metadata` 配下に入る。
- 主要動作を既存unittest方式で検証できる。

## 3. 実装ステップ
### Phase 1: 現行dry-run動作の仕様固定
1. `tests/unit/test_minecraft_support.py` を拡張し、現行のActionSpec registry、dry-run保存、disabled時の保存、required args不足を固定する。
2. registry外operationとshell断片が拒否されることを検証する。
3. `WorkResponse.server_operations` とCLI payloadの形を固定する。
4. `metadata` sanitizationで大きな本文やsecretが出ないことを検証する。

検証:
- `python -m unittest tests.unit.test_minecraft_support`
- `python -m unittest tests.unit.test_database_migrations`

### Phase 2: 権限管理
1. `ServerManagementAccessPolicy` を追加する。
2. `AccessContext.is_admin` と `security.maintenance_command_author_ids` を使ってadmin判定する。
3. `MinecraftSupportService.status()` と `request()` にadmin checkを追加する。
4. `WorkflowService.mc_status()` と `mc_request()` でも同じpolicyを通す。
5. `approval --type server_operation` の `list/show/approve/reject/edit` にadmin checkを追加する。
6. 非admin拒否応答では内部情報を返さない。
7. Discord/HTTP frontendでも同じAccessContextを渡せることを確認する。

検証:
- 非adminは `mc_status` と `mc_request` を拒否されること。
- 非admin拒否応答にpending件数やoperation idが含まれないこと。
- adminはstatus確認とdry-run作成ができること。
- `maintenance_command_author_ids` に含まれるuser idはadmin扱いになること。

### Phase 3: 設定追加
1. `configs/ops/server_management.yaml` を追加する。
2. `RuntimeConfig` に `server_management` sectionを追加する。
3. 設定にはserver name、compose_dir、service allow list、file search allow path、timeout、stdout/stderr上限を持たせる。
4. `load_runtime_config()` でconfigs配下の設定を読み込む。
5. `.env` / `.env.example` にはトークンやAPIキー以外のパラメータを追加しない。
6. 新たに環境変数を追加する場合は `.env` と `.env.example` の両方を更新する。

検証:
- 設定未作成でも安全な空設定で起動できること。
- allow list外のserver/service/pathが拒否されること。
- 既存config読み込みが壊れないこと。

### Phase 4: Planner導入
1. `ServerOperationPlan` modelを追加する。
2. 既存 `_parse_request()` を `ServerOperationPlanner` に移す。
3. ラベル付き入力のdeterministic parserを維持する。
4. 複数操作を抽出できるようにする。
5. 専用LLM plannerを任意で追加し、JSON schema validationを必須にする。
6. LLM出力にshell commandが含まれても直接採用しない。
7. unsupported operationは候補保存せず拒否する。
8. required args不足時は不足項目を返し、候補を保存しない。

検証:
- `operation: compose_restart server: survival service: minecraft` を抽出できること。
- 「再起動してwhitelistにSteveを追加」から2件抽出できること。
- `rm -rf /` や任意shellはunsupportedになること。
- required args不足でrepositoryに保存されないこと。

### Phase 5: ActionSpec拡張
1. `ActionSpec` に必要であれば `executor_name`、`timeout_seconds`、`output_policy` を追加する。
2. 後方互換を維持し、既存registryのActionSpecを移行する。
3. `docker_ps` をadmin限定read-only、事前承認不要として扱う。
4. `compose_up`、`compose_restart`、`restart`、`whitelist_update` をhigh riskに維持する。
5. `compose_down` はcriticalかつ二者承認またはdisabledにする。
6. backup作成ActionSpecを追加する場合はrisk、approval、rollbackを必須にする。

検証:
- registryに許可操作だけが含まれること。
- alias正規化が既存入力を壊さないこと。
- `docker_ps` の承認要否が設計通りになること。
- critical操作が一者承認で実行可能にならないこと。

### Phase 6: Dry-run強化
1. `MinecraftDryRun` または汎用 `ServerOperationDryRun` にvalidation結果を保持する。
2. impact、expected downtime、rollbackをActionSpec別に整備する。
3. `command_preview` から内部pathやsecretを除外する。
4. `execution_allowed` はdry-run時に必ずfalseにする。
5. 複数操作時は `metadata.sequence_index` と `metadata.depends_on` を保存する。
6. feature flagが `disabled` の場合は `status=disabled` として保存する。

検証:
- 副作用操作のdry-runにimpact、downtime、rollbackが含まれること。
- dry-runだけではexecutorが呼ばれないこと。
- disabled時は実行できないstatusになること。
- command previewにsecretらしき値が含まれないこと。

### Phase 7: Repository拡張
1. `ServerOperationRepository` に状態更新、承認者追加、実行結果保存メソッドを追加する。
2. JSONL repositoryで追記型更新を維持し、最新状態を復元できるようにする。
3. Postgres repositoryで `approved_by_user_ids`、`status`、`action_run_id`、`metadata` を更新できるようにする。
4. `list_pending_for_approval()` を追加する。
5. 既存 `save/get/list` の後方互換を維持する。

検証:
- JSONLで同じidの更新後に最新状態だけが返ること。
- Postgresのupsertで承認者とstatusが更新されること。
- status別一覧が作成日時順になること。

### Phase 8: ServerOperation専用承認
1. `WorkflowService._generic_approval()` から `server_operation` を分岐し、専用handlerを追加する。
2. `approval list` は `ServerOperationRepository.list(status="waiting_approval")` を使う。
3. `approval show` はdry-run、risk、承認者、rollbackを表示する。
4. `approval approve` はpolicy確認後に `approved_by_user_ids` を更新する。
5. high riskはadmin承認を必須にする。
6. criticalは二者承認が揃うまで `waiting_approval` を維持する。
7. `approval reject` は `status=rejected` にする。
8. `ApprovalRecord` とaudit logを保存する。

検証:
- 承認で `ApprovalRecord` と `ServerOperation` の両方が更新されること。
- high riskが非admin承認で進まないこと。
- criticalが一者承認で `approved` にならないこと。
- reject後に実行できないこと。

### Phase 9: Read-only executor
1. `ServerOperationExecutor` interfaceを追加する。
2. `DockerPsExecutor` を追加する。
3. `docker ps -a` 相当を `shell=False` で実行する。
4. 出力から内部IP、network、mount、env、secretを除外する。
5. 結果をLLMまたはformatterに渡してadmin向けに要約する。
6. read-only実行でもaudit logとServerOperation metadataを保存する。

検証:
- `docker_ps` はadminなら事前承認なしに実行されること。
- 非adminは実行できないこと。
- executorは任意shell文字列を受け付けないこと。
- 出力にsecretや内部IPが含まれないこと。

### Phase 10: 副作用executor
1. `ComposeExecutor` を追加する。
2. `compose_up`、`compose_restart`、`restart`、`compose_down` を固定引数で実装する。
3. cwdを許可済みcompose_dirに限定する。
4. service nameをallow listで検証する。
5. timeoutを設定する。
6. 実行前後のcontainer stateを取得する。
7. stdout/stderrをマスク・短縮して保存する。
8. 失敗時は `status=failed` とし、rollback方針を返す。
9. 成功時は `status=succeeded` とする。

検証:
- 承認前にexecutorが呼ばれないこと。
- allow list外serviceが拒否されること。
- `shell=False` で実行されること。
- timeout時にfailedになり、stderr抜粋がマスクされること。

### Phase 11: whitelist / file search / backup
1. `WhitelistExecutor` を追加する。
2. add/removeの意図をplannerで抽出する。
3. player_name validationを追加する。
4. rollbackとして逆操作を提示する。
5. `FileSearchExecutor` を追加し、許可済みpathだけを検索する。
6. secretらしき行は全文を返さない。
7. backup作成を追加する場合はbackup保存先、世代管理、容量上限、rollback runbookを設定する。

検証:
- 不正なplayer_nameを拒否すること。
- whitelist削除がhigh risk承認を必要とすること。
- file searchがallow path外を読まないこと。
- secretを含む行がマスクされること。

### Phase 12: 監査ログと出力整形
1. `workflow.server_operation.*` のaudit eventを追加する。
2. stdout/stderr、server state、container state、実行者、承認者を保存する。
3. 一般回答とCLI payload向けにmetadata sanitizationを強化する。
4. `server_operations` のitem metadataもsanitizeする。
5. Discord表示では長いdetailをattachmentまたは短縮表示に逃がす。
6. 失敗時admin通知payloadを作る。

検証:
- audit logにoperation id、actor、approver、statusが残ること。
- CLI payloadのトップレベルにdiagnostic fieldが増えないこと。
- `routing_decision`、`policy_decision`、`trace_id` 相当がmetadata配下に入ること。
- secretや内部IPが外部payloadに出ないこと。

### Phase 13: CLI / Discord / HTTP統合
1. CLI `work --type mc_status/mc_request` の後方互換を維持する。
2. CLI `approval --type server_operation` を専用handlerへ接続する。
3. 必要なら `work --type server_operation_execute` を追加する。
4. Discord `/work` choicesと `/approval` choicesの表示を確認する。
5. HTTP responseの `server_operations` payloadを確認する。
6. docs/explanation/cli.md に利用例を追記する。

検証:
- CLIでdry-run作成、承認、表示、却下ができること。
- Discord/HTTPでpayload shapeが壊れないこと。
- 既存task/event approvalが壊れないこと。

### Phase 14: Runbook整備
1. `docs/runbooks/minecraft_operation_rollback.md` を更新する。
2. docker ps、compose restart、compose down、whitelist変更のrollback手順を記載する。
3. `docs/runbooks/incident_response.md` にサーバー操作事故時のfeature flag停止手順を追記する。
4. critical操作を有効化する条件を明記する。

検証:
- 各high/critical ActionSpecに対応するrollback説明があること。
- runbookにsecretや内部接続情報を書かないこと。

## 4. 変更対象
| 種別 | 主なファイル |
| --- | --- |
| domain | `src/kumc_agent/domain/models/minecraft.py` |
| feature | `src/kumc_agent/features/minecraft/service.py` |
| action registry | `src/kumc_agent/features/minecraft/actions.py` |
| executor | `src/kumc_agent/infra/minecraft/` 配下に新規 |
| repository | `src/kumc_agent/infra/minecraft/repository.py` |
| workflow | `src/kumc_agent/features/workflow/service.py` |
| config schema | `src/kumc_agent/config/schema.py`, `src/kumc_agent/config/load.py` |
| config | `configs/ops/features.yaml`, `configs/ops/security.yaml`, `configs/ops/server_management.yaml` 新規 |
| CLI | `src/kumc_agent/cli.py`, `docs/explanation/cli.md` |
| frontend | `src/kumc_agent/frontends/discord/app.py`, `src/kumc_agent/frontends/http/app.py` |
| DB | `infrastructure/migrations/006_minecraft_server_operations.sql`、必要なら追加migration |
| tests | `tests/unit/test_minecraft_support.py`、CLI/config/repository/executor用の新規unittest |
| runbook | `docs/runbooks/minecraft_operation_rollback.md`, `docs/runbooks/incident_response.md` |

## 5. リスクと対策
| リスク | 対策 |
| --- | --- |
| 任意shell実行につながる | ActionSpecとexecutor固定実装だけを許可し、planner出力文字列をshellに渡さない |
| 非adminが内部状態を知る | 権限チェックを最初に行い、拒否応答に内部情報を含めない |
| LLMが危険操作を誤分類する | JSON schema validation、ActionSpec照合、risk policyをdeterministicに適用する |
| critical操作が誤実行される | 二者承認またはdisabledを必須にし、feature flagでも止められるようにする |
| stdout/stderrからsecretが漏れる | 実行結果保存前とpayload出力前にmask/sanitizeする |
| compose対象を間違える | server/service/path allow listをconfigsで管理し、schema validationする |
| dry-runと実行内容がずれる | executor argsはvalidated planから生成し、command_previewは表示専用にする |
| JSONLとPostgresで挙動がずれる | repository contract testを両実装に適用する |
| 既存workflow approvalを壊す | `server_operation` だけ専用handlerに分岐し、task/event/scheduleの既存経路は触らない |

## 6. テスト一覧
- `tests/unit/test_minecraft_support.py`
- `tests/unit/test_server_management_access.py`
- `tests/unit/test_server_operation_planner.py`
- `tests/unit/test_server_operation_repository.py`
- `tests/unit/test_server_operation_approval.py`
- `tests/unit/test_server_operation_executor.py`
- `tests/unit/test_server_management_config.py`
- `tests/unit/test_cli_server_management_payload.py`
- `tests/unit/test_database_migrations.py`

## 7. 実装順序
1. 現行dry-run動作の仕様固定
2. 権限管理
3. 設定追加
4. Planner導入
5. ActionSpec拡張
6. Dry-run強化
7. Repository拡張
8. ServerOperation専用承認
9. Read-only executor
10. 副作用executor
11. whitelist / file search / backup
12. 監査ログと出力整形
13. CLI / Discord / HTTP統合
14. Runbook整備

