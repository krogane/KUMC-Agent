# 自律エージェント 実装調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/autonomous-agent.md`
- `docs/plan/autonomous-agent.md`

調査対象:

- `src/kumc_agent/domain/models/autonomous_agent.py`
- `src/kumc_agent/features/autonomous_agent/*`
- `src/kumc_agent/apps/autonomous_agent.py`
- `src/kumc_agent/apps/worker/app.py`
- `src/kumc_agent/cli.py`
- `configs/main/autonomous_agent.yaml`
- `src/kumc_agent/config/schema.py`
- `src/kumc_agent/config/load.py`
- `tests/unit/test_autonomous_agent.py`

`src/kumc_agent/infra/legacy` は、実装依存の有無を確認する目的の検索のみ行い、実装評価の対象外とした。

## 結論

現行実装は、自律エージェントの「初期実装に相当する骨格」は存在するが、仕様が求める完全実装には達していない。

実装済みなのは、設定読み込み、domain model、idempotency key生成、`AgentRun` / `AgentStep` trace、タスク・イベント・サーバー運用・Automation runの一部snapshot、決定的Planner、Workflow fallback adapter、決定的Verifier、CLI/workerからの手動run、最低限のunittestである。

未達の主要点は、1日にn回の自動起動、RAG差分collector、統合入力受付を標準経路にしたTOOL実行、Automation/Retrieval adapter、専用LLM Planner / Verifier、VERIFYの実再検索ループ、承認申請・通知候補の永続化契約、詳細な権限/secret/副作用検証、監査/monitoring/readinessの完全性である。

したがって現状は `M1: Dry-run MVP` と `M3: worker連携の一部` までは進んでいるが、`M2`、`M4`、Phase 14以降の監査・安全性・ドキュメント整備を含む完全実装ではない。

## 実装済みの範囲

| 仕様項目 | 実装状況 | 主な実装箇所 |
| --- | --- | --- |
| 設定ファイル | `configs/main/autonomous_agent.yaml` が存在し、`enabled`、`schedule_times`、`timezone`、`scopes`、`dry_run`、`lookahead_days`、`budget` を定義済み | `configs/main/autonomous_agent.yaml:1`, `src/kumc_agent/config/schema.py:94`, `src/kumc_agent/config/load.py:362` |
| domain model | request、snapshot、plan、query、tool result、通知候補、承認申請候補、responseをdataclass化 | `src/kumc_agent/domain/models/autonomous_agent.py:11` |
| idempotency key | `autonomous-agent:{date}:{slot}:{scope_hash}` 形式で、scope/guild/channel/lookaheadをhash化 | `src/kumc_agent/features/autonomous_agent/idempotency.py:22` |
| duplicate判定 | `AutomationRepository.get_run_by_idempotency_key()` で既存runを確認し、duplicate時はPLAN/TOOLを実行しない | `src/kumc_agent/features/autonomous_agent/service.py:84` |
| trace保存 | `AgentRun` を開始し、PLAN / TOOL / VERIFYを `AgentStep` として保存 | `src/kumc_agent/features/autonomous_agent/service.py:97`, `src/kumc_agent/features/autonomous_agent/service.py:165`, `src/kumc_agent/features/autonomous_agent/service.py:184`, `src/kumc_agent/features/autonomous_agent/service.py:208` |
| snapshot収集 | Task、Task候補、Task approval batch、Event、Event候補、Event approval batch、pending ServerOperation、waiting/blocked AutomationRun、recent runを一部収集 | `src/kumc_agent/features/autonomous_agent/snapshot.py:63`, `src/kumc_agent/features/autonomous_agent/snapshot.py:83`, `src/kumc_agent/features/autonomous_agent/snapshot.py:111`, `src/kumc_agent/features/autonomous_agent/snapshot.py:129`, `src/kumc_agent/features/autonomous_agent/snapshot.py:148` |
| deterministic Planner | 期限接近/超過Task、停滞Task、準備Task不足Event、不足情報Event、RAG差分、ServerOperation、AutomationRunからcheck/queryを作成。完全実装で必要な専用LLM Plannerは未実装 | `src/kumc_agent/features/autonomous_agent/planner.py:24` |
| TOOL adapter | dry-run時に候補作成系queryをskipし、通常時は統合入力受付またはWorkflow fallbackを実行 | `src/kumc_agent/features/autonomous_agent/integrated_input.py:44` |
| VERIFY | forbidden side effect marker、tool result不足、candidate citation不足、通知先未設定、通知候補/承認申請候補の作成を判定。完全実装で必要な専用LLM Verifierは未実装 | `src/kumc_agent/features/autonomous_agent/verifier.py:35` |
| CLI | `kumc-agent autonomous` コマンドを追加し、`--dry-run`、`--slot`、`--scope`、`--idempotency-key` を受け取る | `src/kumc_agent/cli.py:382`, `src/kumc_agent/cli.py:668` |
| worker | `job_type="autonomous_agent_run"` を追加し、payloadからtrigger/slot/scopes/dry_runを受け取る | `src/kumc_agent/apps/worker/app.py:101` |
| legacy非依存 | `features/autonomous_agent`、`apps/autonomous_agent.py`、domain model、専用testにlegacy importは見つからない | `rg "legacy" src/kumc_agent/features/autonomous_agent src/kumc_agent/apps/autonomous_agent.py src/kumc_agent/domain/models/autonomous_agent.py tests/unit/test_autonomous_agent.py` |
| 専用テスト | idempotency、duplicate、期限Task通知、dry-run candidate skipを確認 | `tests/unit/test_autonomous_agent.py:65`, `tests/unit/test_autonomous_agent.py:103`, `tests/unit/test_autonomous_agent.py:135` |

## 仕様との差分

| 重要度 | 差分 | 仕様上の期待 | 現行実装 |
| --- | --- | --- | --- |
| High | 1日にn回の自動起動が未実装 | `schedule_times` に従い1日にn回起動し、scheduler/worker経由で同じserviceを呼ぶ | `schedule_times` は設定にあるが、cron生成・Automation default rule・外部scheduler候補生成はない。worker手動jobのみ実装。`enabled` もservice実行可否には使われていない |
| High | RAG差分collectorが未実装 | 当日のRAGデータ差分、更新source、重要そうな新規資料をsnapshotへ入れる | `rag_delta_collector_unimplemented` warningを返すだけで、plannerのRAG差分queryは実データから発火しない |
| High | 統合入力受付が標準経路になっていない | TOOLは統合入力受付へクエリを送り、通常の権限・安全性ルールで処理する | adapterは統合入力受付を呼べるが、app contextでは `integrated_input=None` としてWorkflow fallback固定 |
| High | 専用LLM Planner / Verifierが未実装 | 完全実装では、決定的ルールを安全な下限として残しつつ、専用LLM Plannerで複合状況から計画を作成し、専用LLM Verifierで根拠・重複・権限・副作用境界を検証する | 現行は決定的Plannerとmarker/条件分岐中心のVerifierのみ。LLM出力schema validation、fallback、監査用reason生成は未実装 |
| High | 再検索/再計画ループがない | VERIFYで不足時に `max_replans` 以内で再検索し、改善不可ならnoop/低confidence提案 | verifierは `retry_search` を返せるが、serviceは再検索せず `insufficient_evidence` に変換して終了する |
| High | 承認申請・候補の永続化契約が弱い | Task/Event/Automation候補、ApprovalRequest、通知候補を承認フローに接続する | NotificationProposal/ApprovalRequestProposalは主にresponseとAgentRun metadataに残る。Automation候補作成は未実装。ApprovalRecordやWorkflow approval targetへの明確な接続はない |
| High | 安全性検証が仕様より浅い | 権限外情報、secret、個人情報、大きなcontext、正本更新済みpayloadを検出する | sanitizerは共通処理を使うが、Verifierの副作用検出はmetadata文字列marker依存。内部IP、招待URL、個人連絡先、権限外情報の明示検査は不足 |
| Medium | `AutomationService.dry_run()` / Retrieval / Server adapterが未実装 | 限定adapterとしてWorkflow、Automation、Retrieval、Server pendingを安全に呼べる | Workflow fallbackのみ。Server pendingはsnapshot repositoryから取得するがTOOL adapterではない。Automation/Retrieval adapterは存在しない |
| Medium | duplicate抑制が完全ではない | 同一対象通知を `duplicate_suppression_hours` に従って抑制し、同一idempotencyの二重runを防ぐ | recent runの `notification_target_refs` だけをlimit件数で見る。時間窓は未使用。idempotency履歴はrun完了後のAutomationRun保存なので、同時起動や途中クラッシュには弱い |
| Medium | AuditEventの粒度が不足 | 判断理由、参照対象、候補ID、通知候補ID、duplicate/skipped/noop、elapsed/cost/search call数を保存 | plan/tool/verify/failureのAuditEventはあるが、proposal専用event、duplicate監査、elapsed/cost/search call数は不足 |
| Medium | output schemaが仕様と一部不一致 | `proposals`、`task_candidates`、`event_candidates`、`automation_runs`、`server_operations`、`run` を扱う | `AutonomousAgentResponse.to_payload()` は `notification_proposals`、`approval_requests`、`candidate_refs`、`warnings`、`metadata` が中心。詳細候補オブジェクトやserver/automation参照のtop-levelはない |
| Medium | worker payload方針違反 | 診断情報・副作用情報は `metadata` 配下に置く | workerは `out["side_effects"] = "none"` をtop-levelに追加しており、payload schema方針に反する |
| Medium | `blocked` statusが実質使われない | 設定不備などは `failed` にせず `blocked` | `_status_from_decision()` は `succeeded`、`needs_approval`、`insufficient_evidence`、`noop` を返すが、設定不備blockedの実装経路がない |
| Low | `task_management.due_soon_notice_days` との関係が曖昧 | 期限接近Task判定に `task_management.due_soon_notice_days` を使う | 実装は `autonomous_agent.lookahead_days.tasks` を使う。独立設定として妥当だが、仕様との優先順位が未定義 |
| Low | docs/runbooks等の更新が未完了 | CLI例、run確認、duplicate/blocked/needs_approval対応、評価基盤への追記 | `docs/explanation` 以外の運用docsへの自律エージェント説明は未整備 |

## 完了条件ごとの判定

| 完了条件 | 判定 | コメント |
| --- | --- | --- |
| 1日にn回の起動時刻をconfigで設定 | Partial | 設定はあるが自動起動に接続されていない |
| 手動dry-runとworker/scheduler経由runで同じservice | Partial | CLI/workerは同じservice。scheduler経由は未実装 |
| idempotency_keyで二重実行防止 | Partial | 通常の連続実行は防ぐ。同時起動・途中クラッシュ耐性は不足 |
| PLANでタスク、イベント、RAG差分、サーバー運用、Automation確認 | Partial | RAG差分は未実装。Automationはwaiting/blockedのみ |
| TOOLで統合入力受付へクエリ | Partial | adapterはあるがapp contextはWorkflow fallback固定 |
| 未完成範囲は限定adapterで安全に呼ぶ | Partial | Workflowのみ。Automation/Retrieval/Server adapterは未実装 |
| VERIFYで再検索/noop/通知/許可申請/候補を選択 | Partial | decisionは返すが再検索は実行されない。Automation候補なし |
| 専用LLM Planner / Verifierで高度な判断を行う | No | 現行は決定的ルールのみ。完全実装ではLLM Planner / Verifierを実装し、schema validationと決定的guardを併用する必要がある |
| 出力が提案・通知候補・承認申請・ログに限定 | Mostly | 外部投稿はしない。worker top-level `side_effects` はpayload方針違反 |
| 承認前に外部投稿/サーバー操作/正本更新なし | Mostly | planner由来のwork_typeでは正本更新は起きにくい。Verifierはmarker依存で、将来拡張時の防御は弱い |
| PLAN/TOOL/VERIFY trace保存 | Yes | 専用テストでも確認済み |
| AutomationRunまたは専用run記録で履歴確認 | Yes | `_record_history()` がAutomationRunを保存 |
| AuditEventに判断理由・参照対象・候補ID・通知候補ID | Partial | 基本eventはあるが粒度とproposal eventが不足 |
| CLI/worker payload診断情報metadata配下 | Partial | CLIは概ね準拠。workerの `side_effects` がtop-level |
| 大きなcontext/secret/権限外情報を出さない | Partial | sanitizerはあるが、権限外情報・個人情報・内部IPなどの検証は限定的 |
| 主要動作をunittestで検証 | Partial | 3 testsのみ。仕様で列挙されたRAG、副作用済みpayload拒否、secret、scheduler等は未検証 |

## 仕様改善点

### 1. 「初期実装」と「完全実装」の境界を明示する

現行の計画書は「初期実装では」と書きつつ、完了条件には完全実装相当の項目を含めている。実装完了判定を明確にするため、各Phaseに `MVP必須`、`完全実装必須`、`将来拡張` のラベルを付けるべきである。

### 2. scheduler仕様を具体化する

`schedule_times`、`enabled`、`timezone`、`slot`、`scheduled_at`、missed run、外部cron/AutomationRuleのどれを正とするかが曖昧である。自動インデックス更新と同様に、外部cronまたはAutomation default ruleからworker jobを起動する設計に寄せるなら、cron生成規則、slot決定、`enabled=false` 時の挙動を明記する必要がある。

### 3. RAG差分のデータソースを定義する

RAG差分は仕様上重要だが、どのrepositoryやindexing runから「当日差分」を取るかが未定義である。`IndexingRun`、`source_items`、ingestion cursor、raw snapshot、source_kind別差分のいずれを使うか、差分itemに必要な `source_id`、`citation_id`、短いsummary、ACL情報を定義すべきである。

### 4. dry-runの意味を三値で定義する

現在のrequestは `dry_run: bool = True` で、service内では `True` を「configに従う」意味でも使っている。仕様では `dry_run=true` は候補保存を抑止する意味なので、`request.dry_run: bool | None` のように `None=設定値に従う`、`True=強制dry-run`、`False=候補保存許可` と明確化すると実装と運用が安定する。

### 5. system actorの権限設定を追加する

自律エージェントはsystem actorだが、どのguild/channel/roleで統合入力受付を呼ぶかが設定化されていない。`system_user_id`、`allowed_guild_ids`、`allowed_role_ids`、`default_channel_id`、scope別権限をconfigに持たせ、workerのadmin defaultも明示的にやめるべきである。

### 6. 承認申請と通知候補の保存先を確定する

設計は「AgentRun.metadataでもよい」と「既存ApprovalRecordまたはWorkflow approval targetへ紐付ける」を混在させている。完全実装では、通知候補、ApprovalRequestProposal、Automation proposal、ServerOperation proposalをどのrepositoryのどのstatusで永続化し、承認UI/Discord通知へどう渡すかを固定する必要がある。

### 7. 外部payloadと内部response objectを分ける

設計の `AutonomousAgentResponse` には `run` など内部寄りの項目がtop-levelにある一方、CLI/外部連携payload方針では診断情報をmetadata配下に置く。内部service返却型と外部payload schemaを分け、外部payloadには `run_id` だけを `metadata` に置く方が一貫する。

### 8. 専用LLM Planner / Verifierを完全実装要件として定義する

現行の決定的Planner / Verifierは安全なfallbackとして残しつつ、完全実装では自律エージェント専用のLLM PlannerとLLM Verifierを実装する方針を明記すべきである。LLM Plannerはsnapshot、recent runs、budget、scope、side effect boundaryを入力に、`checks`、`required_queries`、`target_refs`、`success_criteria`、`risk`、`notification_policy` をschema付きで出力する。LLM Verifierはtool result、citation、権限、重複通知履歴、副作用契約を検査し、`retry_search`、`noop`、`notify`、`request_approval`、`create_candidates` をschema付きで選択する。LLM出力は必ずschema validation、決定的guard、監査可能なreason生成、fallback経路を通す必要がある。

### 9. 副作用検証をmarkerではなく構造化契約にする

現在のVerifierはmetadata文字列に `executed` や `sent` が含まれるかを見る。完全実装ではadapter resultに `side_effects: none | candidate_or_approval_only | master_write | external_post | server_execute`、`master_write_count`、`external_delivery_count` などの構造化フィールドを必須化し、禁止値をschema validationで拒否するべきである。

### 10. 再検索/再計画の予算仕様を具体化する

`max_steps`、`max_search_calls`、`max_replans`、`max_latency_seconds` があるが、現在は主に `max_steps` だけ使われている。TOOL種別ごとのカウント方法、retry時のquery変更規則、citation不足時の再検索条件、elapsed/cost記録を仕様化する必要がある。

### 11. duplicate/idempotencyの競合耐性を仕様化する

完全実装では、run完了後に履歴を書く方式では同時起動を防げない。run開始時に `idempotency_key` を予約し、Postgresではunique constraint、File fallbackではlock/atomic writeを使う、クラッシュ時は `running` のTTLで扱う、という仕様が必要である。

### 12. 評価・テストマトリクスを完了条件に直結させる

仕様には評価観点があるが、テスト完了条件に直接落ちていない。最低限、scheduler slot、RAG差分、統合入力受付経由、専用LLM Planner / Verifierのschema validationとfallback、permission denied、secret masking、forbidden side effect、candidate citation不足、notification duplicate suppression、worker payload metadata方針をunittestまたはintegration testで必須化すべきである。

## 推奨対応順

1. `schedule_times` からAutomation default ruleまたはworker起動候補を生成し、`enabled` と `slot` を実行制御に接続する。
2. RAG差分collectorの正データソースを決め、当日差分snapshotを実装する。
3. 専用LLM Planner / Verifierを実装し、schema validation、決定的guard、fallback、監査用reason生成を組み込む。
4. `build_autonomous_agent_app_context()` で統合入力受付を標準経路に接続し、Workflow fallbackは明示fallbackにする。
5. `max_replans` に基づく再検索/再計画ループを `AutonomousAgentService` に実装する。
6. worker payloadの `side_effects` を `metadata.side_effects` へ移動する。
7. NotificationProposal / ApprovalRequestProposal / Automation proposal の永続化先を確定し、承認フローに接続する。
8. Verifierを構造化 `side_effects` 契約、secret/内部IP/招待URL/個人連絡先検査、権限外情報検査へ拡張する。
9. AuditEventにproposal作成、duplicate、blocked/noop理由、elapsed/cost/search countを追加する。
10. 専用テストをPhase 16の列挙項目まで拡張する。
11. runbook、CLI説明、評価基盤docs、integrated-input docsを更新する。

## 検証

実行した検証:

```bash
python -m unittest tests.unit.test_autonomous_agent
python3 -m unittest tests.unit.test_autonomous_agent
python3 -m unittest tests.unit.test_autonomous_agent tests.unit.test_config_loading
rg "legacy" src/kumc_agent/features/autonomous_agent src/kumc_agent/apps/autonomous_agent.py src/kumc_agent/domain/models/autonomous_agent.py tests/unit/test_autonomous_agent.py
```

結果:

- `python -m unittest tests.unit.test_autonomous_agent`: この環境では `python` コマンドが存在せず未実行。
- `python3 -m unittest tests.unit.test_autonomous_agent`: 3 tests / OK。
- `python3 -m unittest tests.unit.test_autonomous_agent tests.unit.test_config_loading`: 9 tests / OK。
- legacy import検索: 該当なし。
