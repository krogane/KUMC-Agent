# イベント管理 実装調査

調査日: 2026-04-28

対象仕様:
- `docs/design/event-management.md`
- `docs/plan/event-management.md`
- 上位仕様 `docs/design/kumc-agent.md` の「7. イベント管理」

## 結論
イベント管理は、初期実装ではなく、候補作成、承認、正本昇格、変更・削除候補、重複検出、表示フィルタ、通知状態記録、Discord Component、CLI/HTTP payload整形まで実装されている。

ただし、仕様上の「完全実装」としては未達である。特に、RAGデータ差分からの自動抽出連携、既存Eventの自動変更・削除検出、n日ごとのDiscordまとめ承認送信、実通知送信と完了確認Component、設定ファイルに基づくadmin role制御、曖昧な対象Eventの確認フローが不足している。

## 実装済みの範囲
| 仕様項目 | 実装状況 | 根拠 |
| --- | --- | --- |
| Event / EventCandidate / EventChangeCandidate / EventApprovalBatch | 実装済み。Event正本、候補、変更候補、batchモデルがある。 | `src/kumc_agent/domain/models/workflow.py:82`, `src/kumc_agent/domain/models/workflow.py:97`, `src/kumc_agent/domain/models/workflow.py:116`, `src/kumc_agent/domain/models/workflow.py:132` |
| DB schema | `events`、`event_candidates`、`event_change_candidates`、`event_approval_batches` と検索indexがある。 | `infrastructure/migrations/004_workflow_events_tasks_approvals.sql:1`, `infrastructure/migrations/008_workflow_event_schedule_candidates.sql:1`, `infrastructure/migrations/017_event_management_hardening.sql:1` |
| 手動登録で承認前に正本を作らない | `event_add` は `EventCandidate(status="proposed")` を保存し、Eventは作らない。title / starts_at が不足すれば候補を作らない。 | `src/kumc_agent/features/workflow/service.py:633` |
| 専用LLM抽出 | `event_extract` が `EventExtractionService` を呼び、LLM未設定・parse失敗時は `degraded` metadataを返して候補を保存しない。 | `src/kumc_agent/features/workflow/service.py:686`, `src/kumc_agent/features/event_management/service.py:51`, `src/kumc_agent/features/event_management/service.py:104` |
| 重複検出 | title、日時、場所で候補・既存Eventとの重複を検出し、`metadata.duplicate_candidates` に保存する。 | `src/kumc_agent/features/event_management/service.py:169` |
| 承認・修正・却下・正本昇格 | `approval --type event` の list/show/edit/approve/reject に対応。approve時にEventを作成し、候補を `merged` にする。 | `src/kumc_agent/features/workflow/service.py:1474`, `src/kumc_agent/features/workflow/service.py:1927` |
| 正本変更・削除候補 | `event_update` / `event_delete` は `EventChangeCandidate` を作成し、承認まで正本を変更しない。削除は `canceled` への論理削除。 | `src/kumc_agent/features/workflow/service.py:773`, `src/kumc_agent/features/workflow/service.py:821`, `src/kumc_agent/features/workflow/service.py:2036` |
| Postgres transaction | Postgres repositoryではEvent upsert、候補status更新、ApprovalRecord保存を同一transactionで行う。 | `src/kumc_agent/infra/workflow/repository.py:1147`, `src/kumc_agent/infra/workflow/repository.py:1195` |
| 表示・絞り込み | `event_list` は状態、場所、日時、未完了関連タスクで絞り込む。`event_brief` は `Task.related_event_id` から未完了タスクを表示する。 | `src/kumc_agent/features/workflow/service.py:713`, `src/kumc_agent/features/workflow/service.py:737` |
| 通知状態記録 | `event_notify` は対象Eventを選び、`metadata.notifications` に送信済みkeyを記録して重複を避ける。 | `src/kumc_agent/features/workflow/service.py:848`, `src/kumc_agent/features/event_management/service.py:238` |
| Discord Component | Event用の approve / reject / edit / show 系ボタンとedit modalがある。Component操作時はworkflow approvalを通るため権限再確認される。 | `src/kumc_agent/frontends/discord/app.py:156` |
| payload方針 | CLI / HTTP payloadは `event_candidates`、`events`、`approvals` などをトップレベルに置き、診断情報は `metadata` に寄せてsanitizeする。 | `src/kumc_agent/cli.py:48`, `src/kumc_agent/frontends/http/app.py:67` |
| workflow run / audit | `WorkflowService.run()` は件数をworkflow run metadataへ保存し、副作用操作はaudit logへ記録する。 | `src/kumc_agent/features/workflow/service.py:130`, `src/kumc_agent/features/workflow/service.py:2856` |

## 仕様との差分
| 優先度 | 差分 | 詳細 |
| --- | --- | --- |
| High | RAGデータ差分からの自動登録連携が未完 | `event_extract` はwork typeとして手動起動できるが、auto-index差分、Discord/Drive/Notion差分、議事録差分から定期的に呼ばれるadapter/jobは確認できない。`auto_update` はTask/Event正本indexを再構築するだけで、EventCandidate抽出は行わない。 |
| High | 自動変更・削除検出が未実装 | `EventExtractionService` のschemaは新規 `events` のみで、既存Eventに対する `EventChangeCandidate` をLLM抽出しない。変更・削除候補は手動 `event_update` / `event_delete` のルールベース処理に限定される。 |
| High | 手動登録・変更は専用LLMではない | `event_add`、`event_update`、`event_delete` はラベル抽出と日時parse中心。自然文からのtitle推定、対象Event特定、変更差分抽出は仕様より弱い。 |
| High | 曖昧な対象Eventの確認フローがない | `_resolve_event()` は対象が一意に決まらない場合でも先頭Eventを返すため、変更・削除・完了が意図しないEventに向く可能性がある。仕様では候補保存前に質問が必要。 |
| High | Discordへの実通知・完了確認Componentが未完 | `event_notify` は内部metadataに通知済み情報を残すが、Discord特定チャンネルへ送信するdelivery層や完了確認ボタンは確認できない。`event_complete` は手動work typeで、通知内Componentからの完了選択とは接続されていない。 |
| Medium | まとめ承認はbatch作成まで | `event_batch_approval` は候補を集約して `EventApprovalBatch` を保存するが、n日ごとのscheduler、Discord送信、message id保存、batch状態遷移、期間filter、個別候補の通知済み抑止は不足している。 |
| Medium | Event用Discord Componentの操作粒度が不足 | `Evidence` と `Diff` はどちらも `show` に接続され、duplicate details専用操作はない。batch内の候補一覧に対する個別操作・一括approveも未確認。 |
| Medium | 権限設定がconfigsと接続されていない | `configs/main/event_management.yaml` にadmin role idsがあるが、RuntimeConfigにevent_managementセクションはなく、`EventAccessPolicy` は `admin` / `organizer` / `event_manager` という文字列比較をハードコードしている。Discordのrole id運用とずれやすい。 |
| Medium | 通知の仕様が実装とずれる | 仕様は「n日前通知」と「当日通知」だが、現実装のbefore通知は「今日からn日後まで」の範囲選択で、厳密にn日前だけではない。通知n日や通知先も設定値ではなくinstructionまたは固定defaultに依存する。 |
| Medium | JSONL repositoryはtransaction不可 | Postgresはtransaction化済みだが、JSONLはEvent保存、候補status更新、ApprovalRecord保存が順に実行される。設計上はappend-onlyで許容しつつ不整合検出が必要だが、明示的な補償・検出はない。 |
| Low | 手動登録の曖昧日時チェックが実質重複 | `starts_at is None` の不足判定が先に返るため、その後の「日時らしき入力が解釈不能」分岐は到達しにくい。利用者向けには不足と曖昧を分ける仕様のほうが明確。 |
| Low | 設計書の「現行実装との差分」が古い | `docs/design/event-management.md` の3章は「最小経路」として記述しているが、現在は変更候補、重複検出、batch、Discord Componentなどが実装済み。仕様書自体の鮮度管理が必要。 |

## 完全実装に向けた残作業
1. RAG差分・インデックス更新差分から `event_extract` を呼ぶadapter/jobを追加し、抽出件数、候補件数、重複件数、変更候補件数をworkflow runへ保存する。
2. `EventExtractionService` のschemaを新規候補と変更・削除候補に分け、既存Event一覧を入力に含めて `EventChangeCandidate` を抽出できるようにする。
3. 手動 `event_add` / `event_update` / `event_delete` に専用LLMまたは明示的な抽出器を導入し、対象Eventが0件・複数件のときは候補を保存せず質問を返す。
4. `_resolve_event()` の先頭Event fallbackを廃止し、exact id、title一致、候補一覧提示、確認待ちの順にする。
5. `event_batch_approval` をschedulerとDiscord deliveryへ接続し、`period_start` / `period_end`、`notification_message_id`、batch状態、候補ごとの通知済みkeyを更新する。
6. Event通知をDiscord送信層へ接続し、before/day_of/completionそれぞれのdelivery結果、message id、idempotency keyを `Event.metadata.notifications` に保存する。
7. 完了確認Componentを追加し、押下時に権限確認、最新Event状態確認、`event_complete`、ApprovalRecord、audit log保存を実行する。
8. `configs/main/event_management.yaml` をRuntimeConfigに取り込み、admin user id / role id、approval間隔、通知n日前、通知先channelを実処理で使用する。
9. Discord Componentに `show evidence`、`duplicate details`、`diff details` を別actionとして実装し、custom idのnonce/idempotency検証を追加する。
10. JSONL repositoryでmerge中断時の検出、または再実行時の不整合修復ルールを明文化・実装する。

## 仕様の改善点
1. 「自動登録」の入力契約を明確化する。RAG差分の形式、source cursor、本文上限、Citation必須条件、secret除去、どのjobがいつ呼ぶかを仕様に書く。
2. `EventExtractionService` の出力schemaを `new_events` / `event_changes` / `ignored_items` / `degraded` に分ける。新規候補と変更候補を同じ「events」配列に混ぜない。
3. `approved` と `merged` の状態遷移を整理する。現実装ではapprove時にすぐ `merged` になるため、`approved` を中間状態として使う条件が不明確。
4. 手動登録の必須情報を操作別に分ける。新規作成は `title` / `starts_at`、変更は `event_id` または一意な対象条件 + 変更内容、削除は `event_id` + 理由など、保存前質問の条件を具体化する。
5. 対象Event解決の仕様を追加する。id完全一致、title完全一致、部分一致、日時・場所併用、一意でない場合の質問文、候補数上限を定める。
6. まとめ承認batchの状態機械を定義する。`pending` / `sent` / `partially_processed` / `closed` / `failed`、再送可否、候補追加後の扱い、一括approve時の部分失敗を明記する。
7. 通知仕様を厳密化する。「n日前」は当日からn日後だけか、n日以内の直近イベント全体かを決め、timezoneと日付境界を定義する。
8. Discord Componentのaction語彙を固定する。`approve` / `reject` / `edit` / `show_evidence` / `show_duplicates` / `show_diff` / `complete_done` / `complete_not_done` などを明示する。
9. 権限設定の保存先と型を仕様化する。Discord role id、role name、admin user idのどれを使うか、`security` と `event_management` のどちらを正とするかを決める。
10. deliveryと内部状態更新を分ける。通知対象選定、Discord送信、送信結果記録、完了確認を別責務として定義すると、テスト対象と障害時再試行が明確になる。
11. 実装状況表に「調査日」または「最終確認commit」を持たせる。設計書内の「現行実装との差分」が古くなりやすいため、実装監査結果は `docs/explanation/` に分離し、設計書は要求仕様中心にする。

## 検証
既存のunittestで確認した。

```bash
app/.venv/bin/python -m unittest tests.unit.test_workflow_service tests.unit.test_cli_tool_rag tests.unit.test_database_migrations
```

結果: 33件成功。

```bash
app/.venv/bin/python -m unittest tests.unit.test_integrated_input tests.unit.test_autonomous_agent tests.unit.test_auto_index_update
```

結果: 18件成功。
