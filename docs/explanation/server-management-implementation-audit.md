# サーバー管理 実装調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/server-management.md`
- `docs/plan/server-management.md`
- `docs/design/kumc-agent.md` の「9. サーバー管理」

調査対象:

- `src/kumc_agent/domain/models/minecraft.py`
- `src/kumc_agent/features/minecraft/*`
- `src/kumc_agent/infra/minecraft/*`
- `src/kumc_agent/features/workflow/service.py`
- `src/kumc_agent/apps/workflow.py`
- `src/kumc_agent/config/*`
- `src/kumc_agent/cli.py`
- `src/kumc_agent/frontends/discord/app.py`
- `src/kumc_agent/frontends/http/app.py`
- `tests/unit/test_minecraft_support.py`
- `tests/unit/test_server_management.py`
- `tests/unit/test_cli_server_management_payload.py`
- `tests/unit/test_database_migrations.py`

`src/kumc_agent/infra/legacy` は、プロジェクト方針に従い調査対象から除外した。

## 結論

現行実装は、初期dry-run実装だけではなく、ActionSpec、admin制御、複数操作planner、JSONL/Postgres repository更新、server_operation専用承認、read-only/compose/whitelist/file_search executor、CLI/Discord/HTTP接続、payload sanitizationまで実装されている。

ただし、`docs/design/server-management.md` と `docs/plan/server-management.md` の完了条件に照らすと、現時点では「仕様通りの完全実装」とは判断できない。主な未達は、専用LLM Planner未実装、監査ログの情報量、executor失敗時のstatus遷移、schema validationの実行前徹底、unsupported依頼の扱い、`compose_down` の意味の不一致、file_searchのapproval/allow path仕様、実executorのテスト不足である。

安全面では、任意shell文字列をそのまま実行する経路は見当たらず、副作用executorも `shell=False` と固定argvで実行される。一方で、実行時例外やtimeoutが `failed` として保存されない場合があり、運用監査と復旧性の観点で完全仕様には届いていない。

## 実装済みの主な要素

| 仕様項目 | 状態 | 主な実装箇所 |
| --- | --- | --- |
| ActionSpec registry | 実装済み。`status`, `docker_ps`, `file_search`, `compose_up`, `compose_restart`, `restart`, `whitelist_update`, `compose_down` を定義 | `features/minecraft/actions.py` |
| domain model | `ActionSpec`, `MinecraftDryRun`, `ServerOperationPlan`, `ServerOperationExecutionResult`, `ServerOperation` を定義 | `domain/models/minecraft.py` |
| admin限定受付 | 実装済み。`is_admin` または `maintenance_command_author_ids` 相当のuser idで判定 | `features/minecraft/access.py`, `apps/workflow.py` |
| 非admin拒否 | 実装済み。候補作成・一覧表示せず、拒否文とmetadataのみ返す | `features/minecraft/service.py` |
| Planner | 部分実装。現行はdeterministic parserであり、仕様前提の専用LLM Plannerではない | `features/minecraft/planner.py` |
| shell断片拒否 | 部分実装済み。`rm -rf`, shell metacharacter, `sh -c`, `bash -c` を拒否 | `features/minecraft/service.py` |
| dry-run保存 | 実装済み。impact、downtime、rollback、command preview、warningsを保存 | `features/minecraft/service.py` |
| read-only docker_ps | 実装済み。adminであれば `docker ps -a --format {{json .}}` を事前承認なしで実行 | `infra/minecraft/executor.py` |
| 副作用executor | 部分実装済み。compose、whitelistを固定argv、`shell=False`、timeout指定で実行 | `infra/minecraft/executor.py` |
| file_search executor | 実装済み。許可root配下だけをPythonで検索し、secret/IPをマスク | `infra/minecraft/executor.py` |
| repository拡張 | 実装済み。JSONL/Postgres双方にstatus更新、承認者追加、実行結果保存を実装 | `infra/minecraft/repository.py` |
| server_operation approval | 実装済み。generic approvalから分岐し、`ServerOperation` と `ApprovalRecord` を更新 | `features/workflow/service.py` |
| critical二者承認 | 部分実装済み。同一userは重複せず、2 userでapproved。server設定でcritical disabledも可能 | `features/minecraft/service.py` |
| config | 実装済み。`configs/main/server_management.yaml` と `RuntimeConfig.server_management` を追加 | `config/load.py`, `config/schema.py` |
| CLI/HTTP/Discord | 実装済み。`mc_status`, `mc_request`, `server_operation_execute`, `approval --type server_operation` 経路あり | `cli.py`, `frontends/*` |
| payload sanitization | 実装済み。metadataのsecret、内部IP、server/container stateなどをpayload出力前に除外・マスク | `features/foundation/payload_sanitizer.py` |
| runbook | 実装済み。rollbackとincident responseにサーバー管理項目あり | `docs/runbooks/*` |

## 完了条件別の判定

| 完了条件 | 判定 | 差分 |
| --- | --- | --- |
| サーバー管理操作はadminだけが受付できる | OK | `MinecraftSupportService` の入口でadmin判定している |
| 非admin拒否応答に内部情報を含めない | OK | 拒否文は固定で、operation/listは返さない |
| 自然言語入力から1件以上の操作計画を抽出できる | NG | 現行はdeterministic parserのみ。仕様前提の専用LLM Planner、JSON schema validation、LLM出力のunsupported処理が未実装 |
| 定義済みActionSpecに限定し、任意shellを実行しない | 部分OK | shell断片拒否とregistry照合はあるが、unsupported自然文の明示拒否が弱い |
| required args不足時は候補を保存しない | OK | `ValueError` で保存前に止まる |
| server/compose directory/service/path/playerをschema validationする | 部分NG | server allow list未設定時は任意server/serviceのdry-runが作れる。file_search pathもdry-run時は許可rootとの照合が不十分 |
| `docker_ps` はadmin限定かつ事前承認不要で実行できる | OK | executor接続済み。ただしservice label出力やLLM要約はない |
| compose系dry-runで影響、停止時間、rollbackを提示 | OK | 固定文で提示している |
| 副作用操作は承認前に実行されない | OK | write操作は `approved` でなければexecute不可 |
| high risk以上はadmin承認必須 | OK | admin以外はapprove不可 |
| criticalは二者承認またはdisabled | 部分OK | 2 user承認とserver設定disabledはあるが、server未設定時のpolicyが曖昧 |
| `ServerOperation` のstatus、承認者、実行結果をJSONL/Postgresへ保存 | OK | repository contractは実装済み |
| 承認後に定義済みexecutorだけが実行される | OK | executor registryでoperation別に固定実装へ分岐 |
| stdout/stderr、server/container state、実行者、承認者が監査ログへ保存される | NG | `ServerOperation.metadata` には保存されるが、`AuditEvent` はaction/actor/outcome/target/risk程度で、stdout/stderr/state/approverは入らない |
| payloadからsecret、内部IP、ネットワークキー、PIN、解錠手順を除外・マスク | 部分OK | metadata/payloadは対応。`detail_markdown` やdry-run `command_preview` に対する明示sanitizeは弱い |
| CLIや外部連携payloadの診断情報がmetadata配下に入る | OK | `server_operations` は主結果、診断系はmetadata配下 |
| 主要動作をunittest方式で検証できる | 部分OK | サービス・approval・payloadはあるが、実executor、Postgres repository contract、config server allow listの直接テストが不足 |

## 仕様との差分

| 優先度 | 差分 | 影響 | 根拠 |
| --- | --- | --- | --- |
| Critical | executor例外やtimeoutが `failed` として保存されない | `execute()` が `running` に更新した後、executorが例外を投げると `save_execution_result()` まで到達せず、operationが `running` のまま残る可能性がある | `MinecraftSupportService.execute()` はexecutor例外をcatchしない |
| Critical | 監査ログが仕様の保存対象を満たさない | stdout/stderr、server state、container state、承認者、executor resultが `AuditEvent` に残らない。operation metadataには残るが、仕様は監査ログ保存を要求している | `WorkflowService._audit()` はaction/actor/outcome/target/riskのみ |
| Critical | 専用LLM Plannerが未実装 | 仕様は自然言語依頼を専用LLMで抽出し、JSON schema validationを通す前提だが、現行はキーワード・ラベルベースのdeterministic parserのみ。複雑な依頼、条件付き依頼、曖昧な引数確認、unsupported分類の精度が仕様に届かない | `ServerOperationPlanner` はLLMPortやprompt/schemaを持たない |
| Critical | unsupported自然文が明示拒否されず `docker_ps` にfallbackする | 「バックアップして」「nginx reload」など未対応依頼がunsupportedではなくread-only container確認として処理され得る。危険な副作用には直結しないが、仕様の「対応しない操作は拒否」と一致しない | `ServerOperationPlanner.plan()` のfallback |
| High | schema validationがdry-run保存前に徹底されていない | allow list未設定時に任意server/serviceでwrite dry-runを作れる。file_searchもserverにallow pathが1つでもあれば任意相対pathをdry-run保存し、実行時までroot照合しない | `MinecraftSupportService._validated_args()` |
| High | `compose_down` の仕様とexecutor動作が不一致 | dry-run previewは `docker compose down <service>` 相当だが、executorは `docker compose stop <service>` を実行する。停止範囲、volume/network影響、rollback説明がずれる | `infra/minecraft/executor.py` の `_compose_command()` |
| High | status表示が実装状態とずれている | `mc_status` が `execution=dry-run-only` / `disabled in Wave 6` と表示するが、実際にはread-only executorと承認後executorが接続済み | `MinecraftSupportService.status()` |
| High | file_searchのapproval policyが曖昧に実装されている | `file_search` は `read_only=True` かつ `admin_dry_run` だが、`server_operation_execute` ではread_onlyなら `waiting_approval` のまま実行できる。別途admin確認を必須にする仕様なら未達 | `MinecraftSupportService.execute()` |
| High | container/server state snapshotが実状態ではない | compose/whitelist executorのsnapshotは `compose_dir` の存在とservice名だけで、実container stateやhealth checkではない | `_state_snapshot()` |
| Medium | `docker_ps` 出力項目が仕様より少ない | service labelが返らず、LLM要約もない。実用上はformatter要約として最低限だが、仕様記述とは差がある | `DockerPsExecutor.execute()` |
| Medium | `docker_ps` はtruncate後にJSON parseしている | stdout上限でJSONLが途中切れになると、本来あるcontainer行をparseできない可能性がある | `DockerPsExecutor.execute()` |
| Medium | file_search allow pathの基準が曖昧 | 設定例ではserver配下の相対pathに見えるが、実装ではbase_dir解決済みPathをrootとして扱う。compose_dir配下なのかbase_dir配下なのか仕様に明記が必要 | `config/load.py`, `FileSearchExecutor` |
| Medium | direct service利用時はApprovalRecord/auditが残らない | 標準経路のWorkflow approvalでは記録されるが、`MinecraftSupportService.approve()` を直接呼ぶとServerOperationだけが更新される | `features/minecraft/service.py` |
| Medium | 実executorの直接unit testがない | serviceにfake executorを挿すテストはあるが、DockerPs/Compose/Whitelist/FileSearch executor自体のargv、allow list、timeout、maskingは未検証 | `tests/unit` |
| Medium | backup作成ActionSpecが未実装 | 計画ではbackup追加は条件付きだが、設計の対象範囲にはバックアップ作成が含まれるため、完全実装の範囲が曖昧 | `features/minecraft/actions.py` |

## 仕様改善点

1. 完全実装の範囲をActionSpecごとに固定する。特にbackup作成は「初期ActionSpecに含める」のか「将来拡張」なのかを明確にする。
2. Planner仕様を専用LLM前提として明文化する。deterministic parserはラベル付き入力やテスト用の補助経路に限定し、通常の自然言語依頼は専用LLM Planner、JSON schema validation、ActionSpec照合、unsupported分類を必須にする。
3. 承認ポリシー表をoperation別に具体化する。`file_search` の `admin_dry_run` が「approve必須」なのか「adminがexecuteを明示すれば可」なのかを明確にする。
4. `compose_down` の意味を再定義する。service単位停止ならoperation名を `compose_stop` に変えるか、criticalな `compose_down` はproject全体downとして別executorに分ける。
5. server allow list未設定時の安全動作を明記する。現実装はdry-run作成を許すが実行時にserver未設定で失敗する。仕様上はwrite候補作成も拒否する方が運用上わかりやすい。
6. `allow_file_search_paths` の基準を明記する。`compose_dir` からの相対pathなのか、repo `base_dir` からのpathなのか、絶対pathを許可するのかを仕様で固定する。
7. unsupported/ambiguous入力の応答方針を追加する。未知の自然文は `docker_ps` fallbackではなく、候補を保存せず「対応操作を確認してください」と返す方が仕様に合う。
8. executor失敗時のtransaction方針を追加する。`running` 更新後の例外、timeout、validation failureを必ず `failed` に遷移し、stderr/rollback/通知payloadを保存することを完了条件に入れる。
9. 監査ログとServerOperation metadataの責務を分ける。監査ログに保存すべき最小key、metadataにだけ残すkey、payloadで除外するkeyを表で定義する。
10. dry-run/detailのsanitize方針を追加する。CLI/HTTP payload metadataだけでなく、`detail_markdown`、`command_preview`、warningsにもsecret/internal pathを出さないルールを明記する。
11. docker output処理をparse前truncate禁止にする。raw stdoutは保存せず、stream/行単位でparseしてからfield単位でmask/limitする仕様が安全。
12. テスト完了条件を具体化する。executor contract、Postgres/File repository同一遷移、timeout、secret masking、allow list拒否、critical二者承認を個別test名として列挙する。
13. runbookと設計の承認人数を揃える。現runbookはhigh-riskで二者承認を確認すると書いているが、詳細設計ではhighはadmin承認、criticalが二者承認である。

## 推奨修正順

1. `MinecraftSupportService.execute()` でexecutor例外とtimeoutをcatchし、operationを必ず `failed` に更新する。`running` のまま残らない回帰テストを追加する。
2. `WorkflowService._audit()` またはserver_operation専用auditで、operation id、risk、executor、actor、approvers、stdout/stderr抜粋、state snapshotをマスク済みで保存する。
3. 専用LLM Plannerを実装する。Planner promptは `assets/prompts` に置き、出力は1件以上の `ServerOperationPlan` JSONに限定し、JSON schema validation後にActionSpecへ正規化する。
4. Plannerのfallbackを変更し、未知・未対応自然文はunsupportedとして候補保存しない。backup要求も明示的にunsupportedまたは実装済みActionSpecへ振り分ける。
5. dry-run前validationを強化する。write操作はconfigured server必須、service/pathはallow list照合必須にする。
6. `compose_down` を仕様に合わせる。`stop` を使うならoperation名・risk・rollback説明も `compose_stop` に寄せる。
7. DockerPs/Compose/Whitelist/FileSearch executorのunit testを追加し、argv、cwd、shell=False、allow list、timeout、masking、state snapshotを検証する。
8. `mc_status` の表示を現行executor状態に更新する。
9. file_searchのapproval policyを仕様として確定し、実装とテストを合わせる。

## 検証

最初にsystem Pythonで実行したところ、`python` コマンドが存在せず失敗した。

```bash
python -m unittest tests.unit.test_minecraft_support tests.unit.test_server_management tests.unit.test_cli_server_management_payload tests.unit.test_database_migrations
```

結果:

```text
zsh:1: command not found: python
```

次に `python3` で実行したところ、グローバル環境に `discord` がなく、CLI payload testのimportで失敗した。

```bash
python3 -m unittest tests.unit.test_minecraft_support tests.unit.test_server_management tests.unit.test_cli_server_management_payload tests.unit.test_database_migrations
```

結果:

```text
Ran 29 tests
FAILED (errors=1)
ModuleNotFoundError: No module named 'discord'
```

プロジェクトのvenvでは関連テストが成功した。

```bash
app/.venv/bin/python -m unittest tests.unit.test_minecraft_support tests.unit.test_server_management tests.unit.test_cli_server_management_payload tests.unit.test_database_migrations
```

結果:

```text
Ran 29 tests in 0.017s
OK
```
