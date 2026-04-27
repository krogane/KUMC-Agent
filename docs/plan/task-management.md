# タスク管理 実装計画

## 1. 方針
`docs/design/kumc-agent.md` と `docs/design/task-management.md` に従い、タスク管理を実装する。

実装では `src/kumc_agent/infra/legacy` を参照・依存しない。既存の共通部品は `domain.models.workflow`、`features.workflow.service.WorkflowService`、`infra.workflow.repository`、`domain.models.retrieval.AccessContext`、`infra.audit`、`OperationsRepository` を優先して使う。現行実装と設計が矛盾する場合は `kumc-agent.md` を優先する。

## 2. 完了条件
- ユーザー入力またはRAGデータ差分から `TaskCandidate` を抽出できる。
- 自動登録の抽出は専用LLMが行い、失敗時は候補を作成せず診断情報を残せる。
- 手動登録で必要情報を抽出できない場合は、候補作成前にユーザーへ質問できる。
- 手動登録でも正本登録前に `TaskCandidate` を作成し、承認前に `Task` を作らない。
- 既存候補・既存Taskとの重複を検出し、候補metadataと承認UIに反映できる。
- 候補作成者、担当者、adminの承認操作範囲を初期実装から分けられる。
- `TaskCandidate` を承認、修正、却下できる。
- 承認後だけ `Task` 正本へ昇格できる。
- 正本Taskの変更・削除も候補作成と承認を通せる。
- 期限、担当者、状態、関連イベント、優先度で表示を絞り込める。
- Discord Componentで候補の承認、修正、却下を操作できる。
- 期限通知、期限超過通知、完了確認通知を送れる。
- 監査ログ、承認履歴、workflow runに副作用操作を記録できる。
- CLIや外部連携payloadの診断情報が `metadata` 配下に入る。
- 主要動作を既存テスト方式で検証できる。

## 3. 実装ステップ
### Phase 1: 仕様固定と現行テスト補強
1. `TaskCandidate`、`Task`、`ApprovalRecord` の現行動作をテストで固定する。
2. `task_extract` と `task_add` が承認前に `Task` を作らないことを明示的に検証する。
3. `approve` 後に `TaskCandidate.status=merged` になることを検証する。
4. `edit`、`reject`、`show` の履歴保存を検証する。
5. CLI payloadで診断情報がトップレベルへ出ないことを検証する。

検証:
- `tests/unit/test_workflow_service.py`
- `tests/unit/test_cli_tool_rag.py` またはCLI用の新規unittest
- migration確認用の既存 `tests/unit/test_database_migrations.py`

### Phase 2: 権限管理
1. タスク管理用のAccessPolicyを追加する。
2. `task_extract`、`task_add`、`task_list`、`task_done` に権限確認を入れる。
3. `approval --type task` の `list`、`show`、`edit`、`approve`、`reject` に操作別policyを入れる。
4. Discord `/work type:task_*` と `/approval type:task` でも同じpolicyを通す。
5. 候補作成者は自分の候補の表示、修正、取り下げを行えるようにする。
6. 担当者は自分が担当候補または担当者になっている候補・Taskの表示、完了確認、状態更新申請を行えるようにする。
7. adminは候補作成、表示、修正、承認、却下、正本変更、削除、通知設定を行えるようにする。
8. 権限外では候補数やTask存在有無を返さない。
9. 必要なadmin user idやrole idは `configs` 配下に置く。トークンやAPIキーを追加する場合のみ `.env` / `.env.example` の両方を更新する。

検証:
- adminは操作できること。
- 候補作成者は自分の候補だけ修正・取り下げできること。
- 担当者は自分のTaskだけ完了確認・状態更新申請できること。
- その他の非adminはtask管理操作を拒否されること。
- 拒否応答に候補数、Task ID、類似候補が含まれないこと。

### Phase 3: Repository検索条件拡張
1. `WorkflowRepository.list_tasks()` に `assignee_user_id`、`due_from`、`due_to`、`priority` を追加する。
2. JSONL repositoryとPostgres repositoryの両方に実装する。
3. `tasks` tableの検索に必要なindex追加を検討する。
4. `TaskCandidate` 一覧にも `created_by`、`related_event_id`、`confidence` など必要な絞り込みを追加する。
5. 既存呼び出しの後方互換を維持する。

検証:
- status、担当者、期限範囲、関連イベント、優先度で絞り込めること。
- JSONLとPostgresで同じ結果順になること。
- 既存 `task_list` が壊れないこと。

### Phase 4: タスク表示強化
1. `task_list` の自然言語instructionから状態、担当者、期限、関連イベント、優先度を抽出する。
2. 抽出した条件をrepository queryへ渡す。
3. 正本Taskと承認待ち候補を分けて表示する。
4. 大量結果の場合は件数制限とページング情報を返す。
5. Discordでは長い結果をattachmentに逃がす既存実装を維持する。
6. 内部の抽出条件やlimitは `metadata` 配下に入れる。

検証:
- `状態: todo`
- `担当: alice`
- `期限: 2026-05-01まで`
- `event: <id>`
- `priority: high`

### Phase 5: 専用LLM抽出
1. `features/task_management` または `features/workflow/task_extraction.py` を新設する。
2. `assets/prompts/task_extraction.md` を追加する。
3. 自動登録経路では入力本文、RAG差分、根拠を専用LLMへ渡し、JSON schemaで `TaskCandidate` 候補を返す。
4. title、description、担当者、期限、関連イベント、優先度、根拠、confidenceを生成する。
5. LLM失敗時、schema不正時、根拠不足時は自動登録候補を作成せず、`metadata.degraded=true` と理由を保存する。
6. 現行ルールベース抽出は自動登録のfallbackにしない。手動登録の入力補助やテスト用helperとして残す場合も、自動候補作成へ接続しない。
7. 抽出モデル名、prompt version、degraded理由は `metadata` 配下に保存する。

検証:
- 担当、期限、状態、優先度を抽出できること。
- 未決事項や質問をタスクとして誤登録しないこと。
- LLM失敗時に候補を作らず、`metadata.degraded=true` を残すこと。

### Phase 5.5: 手動登録の不足情報確認
1. `task_add` の入力からtitle、担当者、期限、関連イベント、優先度、備考を抽出する。
2. 必須情報であるtitleを抽出できない場合は、候補を保存せずユーザーへ質問を返す。
3. 担当者や期限を指定しているように見えるが解釈できない場合は、曖昧な項目を明示して質問する。
4. Discordではmodalまたはfollow-upで不足情報を受け取り、CLIでは不足項目を明示したエラーまたは対話可能な質問文を返す。
5. 不足情報が補完された後に `TaskCandidate(status="proposed", created_by="user")` を保存する。

検証:
- titleが空または曖昧な場合に `TaskCandidate` が保存されないこと。
- 期限らしき入力が解釈不能な場合に期限確認の質問を返すこと。
- 補完後に候補が作成され、承認前に `Task` は作成されないこと。

### Phase 6: 重複検出
1. `DuplicateTaskDetector` を追加する。
2. title正規化、担当者、期限、関連イベント、根拠sourceを使って候補同士・候補とTaskを比較する。
3. 類似度が高い場合は `metadata.duplicate_candidates` に保存する。
4. 既存Taskと同一と判断できる場合は、正本新規登録ではなく変更候補へ誘導する。
5. 承認UIと `task_list` に重複警告を表示する。

検証:
- 同一title、同一担当、同一期限の重複が検出されること。
- 表記ゆれがあるtitleでも高類似として扱えること。
- 重複疑いがあっても候補そのものは監査可能に保存されること。

### Phase 7: 正本変更・削除候補
1. `TaskChangeCandidate` を追加するか、汎用 `WorkflowCandidate(candidate_type="task")` に専用schemaを定義する。
2. 操作種別 `update` / `delete`、変更前payload、変更後payload、理由、根拠を保持する。
3. `task_update`、`task_delete` 相当のwork typeを追加するか、`task_add` 系のinstructionから変更意図をrouteする。
4. 承認前に `tasks` を更新・削除しない。
5. 承認後に正本を更新し、`ApprovalRecord` とaudit logを保存する。
6. 削除は物理削除ではなく論理削除を基本にする。

検証:
- 期限変更候補が承認前に正本へ反映されないこと。
- 承認後に `Task.updated_at` とmetadataが更新されること。
- 削除は既定のlistから除外され、履歴は残ること。

### Phase 8: 承認処理のtransaction化
1. Postgres repositoryで `Task` 作成、`TaskCandidate.status` 更新、`ApprovalRecord` 保存を同一transactionにまとめるAPIを追加する。
2. JSONL repositoryでは既存append-only方式を維持しつつ、失敗時の不整合を検出できるようにする。
3. 二重承認を防ぐため、承認対象statusの再確認をtransaction内で行う。
4. `merged` や `rejected` の再承認を拒否する。
5. 失敗時は候補状態を壊さず、利用者向けに再試行可能な文言を返す。

検証:
- 同じ候補を2回approveしてもTaskが重複作成されないこと。
- `merged` 候補のedit/approveが拒否されること。
- DB失敗時に半端な `Task` だけが残らないこと。

### Phase 9: Discord Component承認UI
1. タスク候補表示用のDiscord view/componentを追加する。
2. approve、reject、edit、show evidence、duplicate detailsを実装する。
3. editはmodalまたはfollow-upで自然言語修正を受け付ける。
4. Component custom idに `target_type=task`、`target_id`、`action`、`batch_id`、nonceを含める。
5. custom idに長文、secret、根拠本文を含めない。
6. Component操作時もAccessPolicyを再確認する。
7. 操作後に最新状態を再取得して表示する。

検証:
- approve buttonでTask正本が作成されること。
- edit後に候補が更新され、再承認できること。
- reject後に再承認できないこと。
- 権限外ユーザーがbuttonを押しても拒否されること。

### Phase 10: まとめ承認
1. まとめ承認batchモデルを追加する。
2. `n` 日ごとに `status=proposed` の自動抽出候補を集約するjobを追加する。
3. batch id、対象期間、候補ID、通知先、送信message idを保存する。
4. Discordへ候補一覧とComponentを送信する。
5. batch単位で一括approve、個別edit、個別rejectを扱えるようにする。
6. 通知済み候補の再送をidempotency keyで抑止する。

検証:
- 対象期間内の候補だけがbatchに含まれること。
- 一度通知した候補を同じbatchで重複通知しないこと。
- batch内の一部候補だけ承認・却下できること。

### Phase 11: 通知・完了確認
1. 期限前通知、期限超過通知、担当者未設定通知、blocked確認、完了確認のschedulerを追加する。
2. 通知対象は `Task.status in todo/doing/blocked` かつ `due_at` 条件で抽出する。
3. 通知済み情報を `Task.metadata.notifications` に保存する。
4. 完了確認Componentから `task_done` 相当の処理を実行する。
5. `done_by`、`done_comment`、通知message idをmetadataへ保存する。
6. audit logと操作履歴を残す。

検証:
- 期限前Taskだけ通知されること。
- 通知済みTaskが同じタイミングで再通知されないこと。
- 完了確認後にstatusが `done` になること。
- 完了済みTaskは通知対象から外れること。

### Phase 12: 自動登録差分連携
1. 自動インデックス更新またはRAG差分検出からタスク抽出を呼び出すadapterを追加する。
2. Discord、Drive、Notionなどの差分sourceを `Citation` として候補に付与する。
3. 大きな本文断片やsecretを含むcontextは候補payloadへ保存しない。
4. 自律エージェントは正本更新ではなく候補作成と通知までに限定する。
5. workflow runに抽出件数、候補件数、重複件数、通知batch idを保存する。

検証:
- RAG差分から候補が作られること。
- 候補に根拠が付くこと。
- 承認前に正本Taskが増えないこと。
- secretらしき文字列が外部payloadへ出ないこと。

### Phase 13: CLI・HTTP・Discord出力整備
1. CLI `work` と `approval` のtask系出力を安定化する。
2. HTTP endpointがある場合はtask系payloadを同じschemaで返す。
3. Discordではephemeral応答とattachment出力を使い分ける。
4. `routing_decision`、`selected_handler`、`trace_id`、抽出条件、重複スコアは `metadata` 配下に入れる。
5. 大きな検索contextやsecretをmetadataから除外・マスクする。
6. `docs/explanation/cli.md` にtask系コマンド例を追記する。

検証:
- トップレベルには `task_candidates`、`tasks`、`approvals` など安定結果だけが出ること。
- 診断情報が `metadata` 配下にあること。
- Discord attachmentにsecretや巨大contextが含まれないこと。

### Phase 14: 評価セット
1. `task_extraction` 評価ケースを追加する。
2. 担当、期限、状態、重複検出、候補止まりの確認を評価項目にする。
3. 未決事項、質問、イベント告知などタスクでない文をnegative caseにする。
4. 承認前にTask正本へ入らないことを評価する。
5. 権限違反、prompt injection、secret混入を安全性caseに入れる。

検証:
- 既存のunittest方式で最低限の評価を実行できること。
- pytest未導入前提でもCIで走ること。
- 将来pytest導入時に移行しやすい構成にすること。

## 4. 推奨ファイル変更範囲
想定される主な変更範囲は次の通り。

| 領域 | ファイル候補 |
| --- | --- |
| domain model | `src/kumc_agent/domain/models/workflow.py` |
| workflow service | `src/kumc_agent/features/workflow/service.py` |
| task feature | `src/kumc_agent/features/task_management/` 新規候補 |
| repository | `src/kumc_agent/infra/workflow/repository.py` |
| migration | `infrastructure/migrations/016_task_management_hardening.sql` 新規候補 |
| prompts | `assets/prompts/task_extraction.md` 新規候補 |
| config | `configs/workflow/task_management.yaml` または `configs/main/task_management.yaml` 新規候補 |
| CLI | `src/kumc_agent/cli.py` |
| Discord | `src/kumc_agent/frontends/discord/app.py` |
| HTTP | `src/kumc_agent/frontends/http/app.py` 存在する場合 |
| audit | `src/kumc_agent/infra/audit/` |
| automation | `src/kumc_agent/apps/automation.py`、`src/kumc_agent/features/automation/` 存在する範囲 |
| docs | `docs/explanation/cli.md`、関連runbook |
| tests | `tests/unit/test_workflow_service.py`、`tests/unit/test_task_management_*.py` 新規候補 |

`.env` または `.env.example` に設定項目を追加する場合は、必ず他方にも反映する。ただし、抽出閾値、通知間隔、承認batch周期、通知先などのパラメータは `.env` ではなく `configs` 配下へ保存する。

## 5. リスクと対策
| リスク | 対策 |
| --- | --- |
| 承認前にTask正本へ入る | 候補作成と正本登録を別APIにし、テストで固定する |
| 二重承認でTaskが重複する | transaction内でstatus再確認し、Task IDをcandidate ID由来にする |
| 操作者種別ごとの権限が混ざる | 候補作成者、担当者、adminのpolicyを操作別に分け、Component操作時も再確認する |
| LLMがタスクでない文を候補化する | JSON schema、negative case、confidence、承認フローで抑制する |
| 重複候補が大量発生する | DuplicateTaskDetectorとまとめ承認UIの警告で扱う |
| 通知が何度も送られる | idempotency keyと `metadata.notifications` を使う |
| 削除で履歴が失われる | 論理削除を基本にし、ApprovalRecordとaudit logを残す |
| Component custom idに情報漏洩する | IDとnonceだけを入れ、本文やsecretを含めない |
| 外部payloadに内部判断が漏れる | 診断情報は `metadata` 配下へ置き、出力前にマスクする |
| legacy依存が混入する | import検査または静的テストで `infra.legacy` 参照を禁止する |

## 6. テスト計画
pytestは未導入前提のため、既存方式に合わせて `unittest` で追加する。

追加候補:

- `tests/unit/test_task_management_access_policy.py`
- `tests/unit/test_task_extraction.py`
- `tests/unit/test_task_duplicate_detector.py`
- `tests/unit/test_task_repository_filters.py`
- `tests/unit/test_task_approval_transactions.py`
- `tests/unit/test_task_notifications.py`
- `tests/unit/test_discord_task_components.py`
- `tests/unit/test_cli_task_payload.py`

重点テスト:

- `task_extract` は `TaskCandidate` のみ作り、`Task` を作らない。
- `task_add` も承認前に `Task` を作らない。
- `approve` 後だけ `Task` が作成される。
- `edit` は `merged` / `rejected` 候補に対して失敗する。
- `reject` 後に `approve` できない。
- 候補作成者、担当者、adminで許可操作が分かれる。
- その他の非adminはtask操作を拒否される。
- 拒否応答は存在有無を漏らさない。
- status、担当、期限、関連イベント、優先度で絞り込める。
- 通知はidempotentである。
- payloadの内部判断は `metadata` 配下にある。

## 7. 実装順序
推奨順序は次の通り。

1. 権限管理と現行テスト補強
2. Repository検索条件拡張と表示強化
3. 重複検出
4. 専用LLM抽出と手動登録の不足情報確認
5. 正本変更・削除候補
6. 承認処理transaction化
7. Discord Component承認UI
8. まとめ承認
9. 通知・完了確認
10. RAG差分・自律エージェント連携
11. CLI/HTTP/Discord payload整備
12. 評価セットとdocs更新

この順序にすると、先に「承認前に正本へ入らない」「操作者種別ごとの権限」「表示絞り込み」を固め、その上にLLM抽出や通知のような副作用の大きい機能を載せられる。
