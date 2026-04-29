# 自律エージェント 実装計画

## 1. 方針
`docs/design/kumc-agent.md` の「12. 自律エージェント」と `docs/design/autonomous-agent.md` に従い、自律エージェントを実装する。

実装では `src/kumc_agent/infra/legacy` を参照・依存しない。既存の共通部品は `domain.models.agentic.AgentRun`、`AgentStep`、`AgentBudget`、`infra.agentic.repository.AgentTraceRepository`、`domain.models.automation.AutomationRun`、`features.automation.service.AutomationService`、`features.workflow.service.WorkflowService`、`domain.models.workflow.WorkResponse`、`domain.models.audit.AuditEvent` を優先して使う。現行実装と設計が矛盾する場合は `kumc-agent.md` を優先する。

初期実装では、自律エージェントを「定期的に状況を確認して、候補・通知案・承認申請・ログを作る機能」として実装する。外部投稿、サーバー操作、タスク/イベント正本更新、オートメーション正本更新は承認前に実行しない。

## 2. 完了条件
- 1日にn回の起動時刻を `configs/main/autonomous_agent.yaml` で設定できる。
- 手動dry-runとworker/scheduler経由のrunで同じ `AutonomousAgentService` を呼べる。
- `idempotency_key` により同一日・同一slot・同一scopeの二重実行を防げる。
- PLANでタスク、イベント、RAG差分、サーバー運用、オートメーションの確認項目を決定できる。
- PLAN/VERIFYで、それぞれ独自のGemini/OpenAIモデルを使う専用LLMを設定でき、決定的guardとfallbackを通せる。
- TOOLで統合入力受付へクエリを送り、通常の権限・安全性ルールで処理できる。
- 統合入力受付が未完成の範囲では、限定adapterで既存Workflow/Automation/Retrieval serviceを安全に呼べる。
- VERIFYで「再検索」「何もしない」「通知候補」「許可申請」「タスク/イベント/オートメーション候補」を選べる。
- 自律エージェントの出力が提案・通知候補・承認申請・ログに限定される。
- 承認前に外部投稿、サーバー操作、タスク/イベント正本更新が行われない。
- `AgentRun` / `AgentStep` に PLAN / TOOL / VERIFY traceが保存される。
- `AutomationRun` または専用run記録で定期起動履歴と `idempotency_key` を確認できる。
- run開始時に `idempotency_key` を予約し、同時起動・途中クラッシュ時も重複runを抑制できる。
- `AuditEvent` に判断理由、参照対象、候補ID、通知候補IDが保存される。
- CLI/worker payloadの診断情報が `metadata` 配下に入る。
- 大きなcontext、secret、権限外情報を外部payloadやtraceに出さない。
- `dry_run=None|true|false` の三値、system actor権限設定、構造化副作用契約、budgetのsearch/replan/latency記録を検証できる。
- 主要動作を既存のunittest方式で検証できる。

## 3. 実装ステップ
### Phase 1: 既存部品と依存範囲の確定
1. `AgentRun`、`AgentStep`、`AgentBudget`、`AgentTraceRepository` の再利用範囲を確定する。
2. `AutomationRun.idempotency_key` と `AutomationRepository.get_run_by_idempotency_key()` の利用可否を確認する。
3. `WorkflowService` のTask/Event候補作成、承認、一覧取得の既存work typeを洗い出す。
4. Minecraft server operationのwaiting approval取得経路を確認する。
5. RAG差分を参照できる既存repositoryまたはindexing run情報を洗い出す。
6. `AgenticSearchService` には依存しない方針を確認する。
7. `src/kumc_agent/infra/legacy` に依存しないことを確認する。

検証:
- `rg "AgenticSearch|Agentic Search|AutonomousAgent|autonomous_agent"` で既存参照と新規追加範囲が把握できていること。
- 自律エージェントから直接書き込み実行される危険なservice呼び出しがないこと。

### Phase 2: 設定追加
1. `configs/main/autonomous_agent.yaml` を追加する。
2. `enabled`、`schedule_times`、`timezone`、`scopes`、`notification_channel_id`、`dry_run`、`lookahead_days`、`duplicate_suppression_hours`、`budget` を定義する。
3. `config/schema.py` にAutonomousAgent設定sectionを追加する。
4. `config/load.py` または既存設定merge経路に読み込みを追加する。
5. `.env` / `.env.example` には設定値を追加しない。secretが必要になった場合のみ両方に反映する。

検証:
- 設定未作成または一部欠落時に安全なdefaultで起動できること。
- `configs` 配下の値だけで起動時刻と対象scopeを変更できること。

### Phase 3: domain model追加
1. `src/kumc_agent/domain/models/autonomous_agent.py` を追加する。
2. `AutonomousAgentRequest` を定義する。
3. `AutonomousAgentSnapshot` を定義する。
4. `AutonomousCheck` または `AutonomousPlan` を定義する。
5. `AutonomousDecision` を定義する。
6. `NotificationProposal` と `ApprovalRequestProposal` を定義する。
7. `AutonomousAgentResponse` を定義する。
8. 外部payloadのトップレベルは主結果だけにし、診断情報は `metadata` 配下に置く。

検証:
- dataclassのdefaultがミュータブル共有になっていないこと。
- `idempotency_key`、`trace_id`、内部判断がトップレベルに出ないこと。

### Phase 4: idempotency管理
1. `features/autonomous_agent/idempotency.py` を追加する。
2. `autonomous-agent:{date}:{slot}:{scope_hash}` 形式でkeyを生成する。
3. timezoneは設定値を使う。
4. scope、guild、channel、lookahead設定をhashに含める。
5. 既存run確認APIを用意する。
6. 初期実装では `AutomationRepository.get_run_by_idempotency_key()` を利用するか、`AgentRun.metadata.idempotency_key` を検索できるrepository APIを追加する。
7. 将来Postgresで効率よく検索できるよう、必要ならmigrationでindexを追加する。

検証:
- 同じdate/slot/scopeで同一keyになること。
- scopeやlookaheadが変わるとkeyが変わること。
- duplicate時にPLAN/TOOLが実行されないこと。

### Phase 5: Snapshot collector実装
1. `features/autonomous_agent/snapshot.py` を追加する。
2. Task collectorで期限が近いTask、期限超過Task、blocked/doing滞留Taskを取得する。
3. Task candidate collectorでproposed候補とapproval batchを取得する。
4. Event collectorで直近n日のEvent、日時未定Event、関連Task不足を取得する。
5. Event candidate collectorでproposed候補とapproval batchを取得する。
6. RAG delta collectorで当日のsource/indexing差分を取得する。未実装の場合は空snapshotとwarningにする。
7. Server operation collectorでwaiting_approvalのServerOperationを取得する。
8. Automation collectorでwaiting_approval/blockedのAutomationRunと次回実行予定を取得する。
9. Recent run collectorで重複通知抑制用の直近run metadataを取得する。
10. 大きな本文や検索contextは保持せず、ID、短いsummary、件数、citationだけにする。

検証:
- 各collectorが失敗しても他scopeは継続できること。
- snapshotにsecretや巨大contextが含まれないこと。
- lookahead設定が反映されること。

### Phase 6: Planner実装
1. `features/autonomous_agent/planner.py` を追加する。
2. 初期実装では決定的ルールでPLANを生成する。
3. 期限が近い未完了Taskから通知候補checkを作る。
4. 期限超過Taskから完了確認checkを作る。
5. 直近Eventの準備Task不足からタスク候補作成queryを作る。
6. 日時・場所未定Eventから確認通知または変更候補queryを作る。
7. RAG差分からタスク/イベント抽出queryを作る。
8. waiting_approvalのServerOperationから承認依頼checkを作る。
9. blocked/waiting AutomationRunから確認checkを作る。
10. 直近runと同じ対象は重複通知抑制する。
11. PLAN outputに `checks`、`required_queries`、`target_refs`、`success_criteria`、`risk`、`side_effect_boundary`、`notification_policy` を含める。

検証:
- 期限内Taskがあると通知checkが作られること。
- 直近Eventに準備Taskがないと統合入力受付queryが作られること。
- 重複対象は抑制されること。
- `side_effect_boundary` が `candidate_only` または `approval_required` になること。

### Phase 7: 統合入力受付adapter実装
1. `features/autonomous_agent/integrated_input.py` を追加する。
2. 統合入力受付のserviceが存在する場合は標準経路として呼び出す。
3. 統合入力受付が未完成の範囲では限定adapterを実装する。
4. `WorkflowService.run()` を呼ぶadapterは候補作成または一覧取得のwork typeだけに限定する。
5. `AutomationService.dry_run()` を呼ぶadapterは外部副作用なしのdry-runだけに限定する。
6. Retrieval adapterはcitation付き要約だけを返す。
7. Server operation adapterはwaiting approvalまたはdry-run候補だけを返す。
8. すべてのadapterに `AccessContext` を渡す。
9. adapter resultを正規化し、candidate ids、approval ids、warnings、citationsを抽出する。

検証:
- adapterが正本更新や外部送信を行わないこと。
- `AccessContext` が失われないこと。
- 失敗時もservice全体をクラッシュさせず、TOOL result `failed` として扱えること。

### Phase 8: AutonomousAgentService実装
1. `features/autonomous_agent/service.py` を追加する。
2. `run(request: AutonomousAgentRequest) -> AutonomousAgentResponse` を公開する。
3. run開始前にidempotencyを確認する。
4. duplicateの場合は既存run情報をmetadataに入れて終了する。
5. `AgentRun(status="running")` を保存する。
6. Snapshotを収集する。
7. PLAN stepを保存する。
8. budget内でTOOL stepを実行する。
9. VERIFY stepを実行する。
10. 必要に応じて `max_replans` まで再検索する。
11. 最終run statusを `succeeded`、`noop`、`needs_approval`、`blocked`、`insufficient_evidence`、`failed` のいずれかにする。
12. `AutonomousAgentResponse` を生成する。

検証:
- PLAN / TOOL / VERIFY の順でstepが保存されること。
- duplicate時にstepが増えないこと。
- budget超過時にwarningを返して停止すること。
- 候補や承認申請がある場合に `needs_approval` になること。

### Phase 9: VERIFY実装
1. `features/autonomous_agent/verifier.py` を追加する。
2. PLANのsuccess criteriaを検査する。
3. 通知候補の重複を検査する。
4. 通知先チャンネル設定を検査する。
5. 候補作成にcitationが必要な場合はcitation不足を検出する。
6. 正本更新済み、外部投稿済み、サーバー操作実行済みを示すpayloadを検出したら失敗にする。
7. secret、内部IP、token、招待URL、個人連絡先の混入を検査する。
8. VERIFY decisionを `retry_search`、`noop`、`notify`、`request_approval`、`create_candidates` から選ぶ。

検証:
- citation不足時に候補作成しないこと。
- 外部副作用済みpayloadを失敗扱いにすること。
- 通知先未設定時は通知候補だけ作りwarningを返すこと。

### Phase 10: 通知候補と承認申請
1. `NotificationProposal` 保存先を決める。初期実装では `AgentRun.metadata.notification_proposals` に保存してよい。
2. 将来の承認UIに備え、通知候補ID、target channel、body、target refs、riskを持たせる。
3. 実際のDiscord送信はこのPhaseでは行わない。
4. `ApprovalRequestProposal` は既存 `ApprovalRecord` またはWorkflow approval targetへ紐付ける。
5. Task/Event候補は既存WorkflowService経由で保存する。
6. ServerOperationは既存Minecraft serviceのdry-run/waiting_approval候補を参照する。
7. Automationの承認待ちは `AutomationRun(status="waiting_approval")` またはproposalとして保存する。

検証:
- 通知候補が送信済みと誤解されるstatusにならないこと。
- 承認申請にtarget type、target id、理由が含まれること。
- 正本テーブルが承認前に変更されないこと。

### Phase 11: app context配線
1. `src/kumc_agent/apps/autonomous_agent.py` を追加する。
2. foundation context、agent trace repository、workflow service、automation service、audit logを組み立てる。
3. 必要に応じてretrieval、minecraft、indexing repositoryを注入する。
4. 循環importが発生する場合はadapter境界を分ける。
5. Postgres未設定時はJSONL repositoryを使う。

検証:
- `build_autonomous_agent_app_context()` が単体で呼べること。
- Postgres有効/無効の両方でrepositoryが解決されること。
- AgenticSearchServiceを組み立てないこと。

### Phase 12: worker/scheduler連携
1. `apps/worker/app.py` に `job_type="autonomous_agent_run"` を追加する。
2. payloadからtrigger、slot、scopes、dry_runを受け取る。
3. `configs/main/autonomous_agent.yaml` の `schedule_times` から、自動インデックス更新と同じ形式の `AutomationRule(trigger=schedule_cron)` を生成する。
4. `action_type="autonomous_agent_run"` をauto-run allowlistに追加し、worker jobへ接続する。
5. `enabled=false` の場合はAutomation ruleをdisabledにし、schedule/automation triggerの直接runは `blocked` として記録する。

検証:
- workerから自律エージェントrunを実行できること。
- 同じslotを2回実行してもduplicateになること。
- scheduler無効時は自動起動しないこと。

### Phase 13: CLI配線
1. `cli.py` に `autonomous` または `agent run-autonomous` 相当のcommandを追加する。
2. `--dry-run`、`--slot`、`--scope`、`--idempotency-key` を受け取れるようにする。
3. JSON出力ではトップレベルに主結果だけを置く。
4. 診断情報、trace id、idempotency keyは `metadata` 配下に置く。
5. detail markdownを表示できるようにする。

検証:
- CLI dry-runで候補保存や通知送信が行われないこと。
- JSON payloadが安定schemaになること。
- duplicate時にmetadataで既存runを確認できること。

### Phase 14: 監査・trace read API
1. `AgentTraceRepository` に必要ならrun/step取得APIを追加する。
2. JSONL repositoryでidempotency keyからrunを検索できるようにするか、AutomationRunで代替する。
3. Postgres repositoryでrun id検索、step一覧取得、idempotency検索を実装する。
4. `AuditEvent` に `autonomous_agent.plan`、`autonomous_agent.tool`、`autonomous_agent.verify`、`autonomous_agent.proposal` を記録する。
5. readiness/monitoringでrun成功率、noop率、needs_approval数、duplicate数を集計できるようにする。

検証:
- run idから全stepを復元できること。
- idempotency keyからduplicate判定できること。
- 監査ログに判断理由と対象IDが残ること。

### Phase 15: sanitizerと安全性
1. `features/autonomous_agent/sanitizer.py` を追加するか、総合エージェント共通sanitizerを再利用する。
2. tool入力、tool出力、snapshot、trace、最終payloadに長さ制限を適用する。
3. secret検出とマスクを適用する。
4. RAG context全文を外部payloadへ出さず、citation idと短い要約にする。
5. `metadata` も外部出力前にマスクする。
6. 高risk対象は候補作成または承認申請に固定する。

検証:
- tokenらしき文字列が最終回答、detail、traceに出ないこと。
- 巨大contextがstep payloadに保存されないこと。
- high/critical riskで自動実行が起きないこと。

### Phase 16: テスト追加
1. `tests/unit/test_autonomous_agent.py` を追加する。
2. idempotency key生成をテストする。
3. duplicate runをテストする。
4. 期限が近いTaskから通知候補が作られることをテストする。
5. 直近Eventの準備不足からTask候補queryが作られることをテストする。
6. RAG差分collector失敗時もrunが継続することをテストする。
7. VERIFYが副作用済みpayloadを拒否することをテストする。
8. dry-runで候補保存や通知送信が行われないことをテストする。
9. payload metadata方針をテストする。
10. schedule slotからAutomationRuleが生成されることをテストする。
11. RAG差分collectorがingestion active chunksからcitation付きsnapshotを作ることをテストする。
12. 専用LLM Planner / Verifierのschema validation、決定的guard、fallbackをテストする。
13. `dry_run=None` がconfig値に従うことをテストする。
14. architecture testにlegacy非依存を追加する必要があれば更新する。

検証:
- `python -m unittest tests/unit/test_autonomous_agent.py`
- 関連既存テスト
- `rg "src/kumc_agent/infra/legacy" src/kumc_agent/features/autonomous_agent src/kumc_agent/apps/autonomous_agent.py`

### Phase 17: ドキュメント更新
1. `docs/explanation/cli.md` に自律エージェントdry-run例を追加する。
2. `docs/runbooks/` に自律エージェントrun確認、duplicate、blocked、needs_approval対応手順を追加する。
3. `docs/design/evaluation-platform.md` に自律エージェント評価観点を追記する。
4. `docs/design/integrated-input.md` に自律エージェントからの呼び出し方を追記する。
5. `docs/plan/integrated-input.md` に自律エージェント連携タスクを追記する。
6. 設定ファイルを追加した場合は、運用説明に `configs/main/autonomous_agent.yaml` を記載する。

検証:
- 自律エージェントの出力が「提案・通知・承認申請・ログ」に限定される説明になっていること。
- `.env` に通常パラメータを追加していないこと。

## 4. 実装順序
推奨順序は次の通り。

1. 設定とdomain modelを追加する。
2. idempotencyとsnapshot collectorを実装する。
3. PlannerとVerifierを決定的ルールで実装する。
4. 限定adapter経由でTOOLを実行する。
5. `AutonomousAgentService` を組み立てる。
6. app context、worker、CLIを配線する。
7. trace read、audit、sanitizerを固める。
8. 統合入力受付完成後、限定adapterを標準経路から外す。

## 5. リスクと対応
| リスク | 対応 |
| --- | --- |
| 自律runが外部投稿を実行してしまう | 初期実装はNotificationProposalのみ。送信は別承認フローに分離する |
| Automationのauto_run allowlistと混同する | 自律エージェントではauto_run allowlistを外部副作用許可として使わない |
| 統合入力受付が未完成でTOOLが詰まる | 限定adapterを一時実装し、統合入力受付完成後に移行する |
| idempotency検索がAgentRunだけでは難しい | 初期はAutomationRunを併用し、必要に応じてAgentRun metadata indexを追加する |
| snapshotが巨大化する | ID、件数、短いsummary、citationだけを保持し、本文は保存しない |
| 通知重複が増える | recent run metadataとduplicate_suppression_hoursで抑制する |
| AgenticSearchの古い実装に引きずられる | 自律エージェントはAgenticSearchServiceに依存せず、汎用AgentRun/Stepだけを再利用する |

## 6. 初期マイルストーン
### M1: Dry-run MVP
- 設定、domain model、idempotency、snapshot、planner、verifierを実装する。
- CLI dry-runで提案と通知候補を表示する。
- 候補保存や通知送信は行わない。

### M2: 候補作成連携
- WorkflowService経由でTask/Event候補を作成する。
- 既存承認フローに候補IDを渡す。
- 正本更新が起きないことをテストする。

### M3: Scheduler/Worker連携
- worker jobからrunできるようにする。
- schedule_timesからslotを決める。
- duplicate抑制とauditを確認する。

### M4: 統合入力受付経由へ移行
- 自律エージェントTOOLを統合入力受付へ接続する。
- 限定adapterはfallbackまたはテスト用に縮小する。
- 通常ルーティングと同じ権限・安全性ルールで処理されることを確認する。
