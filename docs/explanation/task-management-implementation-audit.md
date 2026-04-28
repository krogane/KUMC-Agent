# タスク管理 実装調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/task-management.md`
- `docs/plan/task-management.md`
- 上位仕様として `docs/design/kumc-agent.md` の「6. タスク管理」

調査対象:

- `src/kumc_agent/domain/models/workflow.py`
- `src/kumc_agent/features/task_management/`
- `src/kumc_agent/features/workflow/service.py`
- `src/kumc_agent/infra/workflow/repository.py`
- `src/kumc_agent/cli.py`
- `src/kumc_agent/frontends/discord/app.py`
- `src/kumc_agent/frontends/http/app.py`
- `src/kumc_agent/apps/worker/app.py`
- `infrastructure/migrations/004_workflow_events_tasks_approvals.sql`
- `infrastructure/migrations/016_task_management_hardening.sql`
- `tests/unit/test_workflow_service.py`
- `tests/unit/test_database_migrations.py`
- `tests/unit/test_integrated_input.py`
- `tests/unit/test_discord_commands.py`

`src/kumc_agent/infra/legacy` はタスク管理実装の調査対象から除外した。検索上、タスク管理の現行経路が `infra.legacy` に直接依存している箇所は確認していない。

## 結論

現行実装は、初期実装よりは進んでおり、TaskCandidate、Task、TaskChangeCandidate、TaskApprovalBatch、承認履歴、CLI/HTTP/Discord slash command入口、File/Postgres repositoryの主要骨格を備えている。

ただし、`docs/design/task-management.md` と `docs/plan/task-management.md` が求める「完全実装」としては未達である。特に次の差分は実運用上のブロッカーになり得る。

- RAGデータ差分や自動インデックス更新から `task_extract` を自動起動する経路がない。
- Discord Componentのtask承認UIは approve/reject/show のみで、edit、evidence、duplicate details、batch、nonce付きcustom idに未対応。
- PostgresのTask承認は `Task` 作成、candidate状態更新、`ApprovalRecord` 保存が同一transactionになっていない。
- `configs/main/task_management.yaml` は存在するが、RuntimeConfigやWorkflowServiceへ接続されておらず、batch周期、通知先、task管理admin設定が実利用されていない。
- 通知はTask metadataへ通知済み状態を記録するだけで、Discord送信、完了確認Component、担当者未設定通知、blocked確認は未実装。
- 手動登録・変更は専用LLMではなくルールベース抽出で、曖昧項目確認、差分表示、secret検証、評価セットが不足している。

## 実装済みの主な要素

| 仕様項目 | 状態 | 主な実装箇所 |
| --- | --- | --- |
| `TaskCandidate` / `Task` / `ApprovalRecord` | 実装済み | `domain/models/workflow.py` |
| `TaskChangeCandidate` / `TaskApprovalBatch` | 実装済み | `domain/models/workflow.py`, `016_task_management_hardening.sql` |
| File/Postgres repository | 部分実装済み | `infra/workflow/repository.py` |
| 自動抽出の専用LLM service | 部分実装済み | `features/task_management/service.py` |
| LLM unavailable時のdegraded | 実装済み | `TaskExtractionService.extract()` |
| 手動 `task_add` は承認前にTaskを作らない | 実装済み | `WorkflowService.task_add()` |
| `approval list/show/edit/approve/reject` | 部分実装済み | `WorkflowService.approval()` |
| 承認後のTask昇格とcandidate `merged` 化 | 実装済み | `_approve_task_candidate()` |
| 正本変更・削除候補 | 部分実装済み | `task_update()`, `task_delete()`, `_approve_task_change_candidate()` |
| Task一覧の絞り込み | 部分実装済み | `_extract_task_list_conditions()`, `list_tasks()` |
| 重複検出 | 部分実装済み | `DuplicateTaskDetector` |
| 権限判定 | 部分実装済み | `TaskAccessPolicy` |
| 通知対象抽出と通知済み記録 | 部分実装済み | `TaskNotificationPlanner`, `task_notify_due()` |
| CLI / HTTP payload整形 | 部分実装済み | `cli.py`, `frontends/http/app.py` |
| Discord slash command | 部分実装済み | `frontends/discord/app.py` |

## 仕様との差分

| 優先度 | 差分 | 影響 | 根拠 |
| --- | --- | --- | --- |
| Critical | RAG差分・自動インデックス更新からのタスク候補抽出が接続されていない | 仕様の「Discord/Drive/NotionなどRAGデータ差分から自動登録」が実運用で動かない | `auto_update.py` はTask/Event index再構築のみ。`AutonomousSnapshotCollector` は `rag_delta_collector_unimplemented` を返す |
| Critical | Task承認がtransaction化されていない | PostgresでTask作成後、candidate更新またはApprovalRecord保存に失敗すると不整合が残る | `_approve_task_candidate()` は `save_task()`、`update_task_candidate_status()`、`save_approval()` を個別に呼ぶ。repositoryにtask用 `merge_*` APIがない |
| Critical | Discord task Componentが仕様不足 | Discord上のまとめ承認、自然言語修正、根拠表示、重複詳細確認ができない | `TaskApprovalView` は Approve / Reject / Show のみ。custom idは `task:approve` など固定でtarget_id、batch_id、nonceを含まない |
| High | `configs/main/task_management.yaml` が実設定として使われていない | batch周期、通知先、admin user/role id、prompt名をconfigで制御できない | config fileは読み込まれるが `RuntimeConfig` に `task_management` sectionがなく、`WorkflowService` へ渡されない |
| High | 通知はDiscord送信ではなく状態記録だけ | 期限通知・期限超過通知・完了確認通知がユーザーに届かない | `task_notify_due()` は `Task.metadata.notifications` を更新するだけ。workerも `side_effects=notification_state_recorded` |
| High | 完了確認フローが未実装 | 仕様上の「完了確認後にdone」ではなく、入口を叩くと即時doneになる | `task_done()` は担当者またはadminなら即 `status="done"` に更新 |
| High | まとめ承認はbatch保存のみ | n日ごとのDiscord承認依頼、period、送信message id、一括/個別操作が満たせない | `task_batch_approval()` は全proposed agent候補を集めてbatchを保存するが、period_startは未設定でDiscord送信もない |
| High | 手動登録・変更が専用LLM抽出ではない | 自然言語の曖昧な登録・変更依頼の解釈力が仕様に届かない | `task_add()` / `task_update()` は `_extract_labeled_value()` などの正規表現ベース |
| High | TaskCandidateのevidenceが空になり得る | 仕様の「根拠Citation付与」を満たさず、承認時の根拠確認が弱い | LLM payloadの `evidence` は `metadata.evidence_refs` に入るが、retrieval citationがない場合 `TaskCandidate.evidence` は空 |
| Medium | 重複検出が簡易的 | 類似理由や既存Task同一時の変更候補誘導が不足し、誤重複・見逃しが起きやすい | `DuplicateTaskDetector` は文字集合Jaccard、担当、期限、eventのscoreのみ。metadataに理由は保存しない |
| Medium | Task一覧の候補側フィルタが弱い | 正本Taskは絞れても、承認待ち候補が条件外まで表示される | `task_list()` はcandidateに `related_event_id` だけ後段filter。担当、期限、priority、confidenceは未適用 |
| Medium | 期限範囲検索が片側のみ | `期限: 2026-05-01まで` は扱えるが、`以降` や範囲指定の自然言語が不足 | `_extract_task_list_conditions()` は `due_to` のみ抽出 |
| Medium | Task変更候補の差分表示が薄い | 承認者が変更前後を確認しづらい | `_format_task_change_candidates()` はreasonのみでbefore/afterを表示しない |
| Medium | reject履歴のbeforeが空 | 承認履歴から却下前状態を復元しづらい | task candidate / task change candidateのreject `ApprovalRecord(before={})` |
| Medium | 権限設定がconfigと分離 | 仕様のadmin user/role管理と実装の `AccessContext.is_admin` / 固定role名が一致しない | `TaskAccessPolicy` は `"organizer"` / `"task_manager"` などrole文字列を見る。task_management configのadmin設定は未使用 |
| Medium | HTTP/Discordのエラー変換が不足 | `KeyError` や `ValueError` が利用者向け文言にならず、HTTPでは500になり得る | `WorkflowService.approval()` は対象なしで `KeyError` を投げ、HTTP `/approval` は捕捉しない |
| Low | 評価セットが仕様より少ない | LLM抽出、negative case、secret、prompt injection、権限違反の回帰検知が弱い | unit testは主要happy path中心。`task_extraction` 評価ケースは未確認 |

## 仕様項目別の判定

| 仕様項目 | 判定 | コメント |
| --- | --- | --- |
| タスク候補の自動抽出 | 部分実装 | LLM serviceはあるが、RAG差分連携と定期抽出が未接続 |
| タスク候補の手動登録 | 部分実装 | 候補止まりは実装済み。専用LLM、曖昧担当者確認、secret検証が不足 |
| タスク候補の承認、修正、却下 | 部分実装 | CLI/HTTP/Discord slash経路はある。Discord Componentの修正・根拠・重複詳細が不足 |
| 承認後のTask正本登録 | 部分実装 | 基本動作は実装済み。transaction化とidempotencyが不足 |
| Task表示・絞り込み | 部分実装 | status、assignee、related_event、priority、due_toは対応。候補側filter、due range、pagingが不足 |
| Task正本の変更・削除承認 | 部分実装 | 変更・削除候補と承認適用はある。差分表示、権限粒度、transactionが不足 |
| 期限通知と完了確認 | 未達 | 通知状態記録のみ。Discord通知と完了確認UIがない |
| Discord Componentまとめ承認 | 未達 | task batchは保存のみ。Componentは単体candidateの簡易approve/reject/showのみ |
| CLI/Discord/HTTP payload整形 | 部分実装 | WorkResponse payloadは概ねmetadata方針。workerの `side_effects` top-levelなど未整理が残る |
| 監査ログ、承認履歴、workflow run | 部分実装 | auditとApprovalRecordはある。approval単体操作のWorkflowRun記録、reject before、詳細auditが不足 |

## 仕様の改善点

1. 権限モデルを一本化する。上位仕様はadmin限定寄り、redesign文書はadminまたは担当者承認を示唆し、詳細設計は候補作成者・担当者・adminの操作分離を求めている。操作別matrixを正とする必要がある。
2. `configs/main/task_management.yaml` のschemaを明文化する。`approval_batch_interval_days`、`due_soon_notice_days`、`notification_channel_id`、`admin_user_ids`、`admin_role_ids`、`prompt_name` をRuntimeConfigへ入れる前提まで仕様に書く。
3. evidenceの必須条件を明確にする。`Citation` が必須なのか、LLMの短い `evidence_refs` でもよいのか、手動登録では何を根拠として保存するのかを定義する。
4. `Task.status` に `deleted` を含めるか、削除はmetadataだけにするかを固定する。現実装は `status="deleted"` を使うが、詳細設計の基本状態には含まれていない。
5. 承認transaction APIを仕様に追加する。`merge_task_candidate()` と `merge_task_change_candidate()` をrepository contractに含め、rowcount再確認、二重承認時の応答、File fallbackの不整合検出も定義する。
6. Discord Component custom idの形式を具体化する。100文字制限を前提に、`target_id` を直接入れるのか、短いnonceからDB参照するのかを決める。
7. 通知の「送信」と「通知済み記録」を分けて定義する。Discord送信成功、送信失敗、再送、完了確認、通知message id保存の状態遷移が必要。
8. 自然言語抽出の責務を整理する。自動抽出、手動登録、変更・削除、一覧filterのどこにLLMを使い、どこをルールベースfallbackにするかを仕様化する。
9. HTTP/CLI/Discord共通のエラーpayloadを定義する。権限なし、not found、承認不能状態、入力不足を500やtracebackにしない契約が必要。
10. worker/automation payloadにもmetadata方針を適用する。`side_effects` や実行判断はtop-levelではなくmetadataへ寄せるか、安定結果フィールドとして扱うかを決める。
11. 評価セットを受入条件に昇格する。`task_extraction` のpositive/negative、重複、secret、prompt injection、承認前正本化禁止、権限違反を最低限のCI対象にする。

## 推奨修正順

1. Task承認をrepository-level transactionに移し、Postgresで `Task` upsert、candidate状態更新、`ApprovalRecord` 保存を同一transactionにする。
2. `RuntimeConfig` に `task_management` sectionを追加し、WorkflowService、worker、Discord通知先、TaskAccessPolicyへ接続する。
3. Discord Task ComponentをEvent Component相当に拡張し、edit modal、evidence、duplicate details、batch_id、nonceを実装する。
4. 自動インデックス更新またはingestion差分から `task_extract` を呼ぶadapterを追加し、`rag_delta_collector_unimplemented` を解消する。
5. 通知delivery layerを追加し、期限前・期限超過・完了確認ComponentをDiscordへ送信する。`task_done` は完了確認経路と直接更新経路を分ける。
6. 手動登録・変更・一覧filterのLLM抽出方針を決め、曖昧項目確認と差分表示を強化する。
7. `TaskChangeCandidate` 表示にbefore/after差分を出し、reject履歴にもbeforeを保存する。
8. `task_extraction` 評価セットとDiscord Component/HTTP error/CLI payloadのテストを追加する。

## 検証

実行した検証:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_workflow_service tests.unit.test_database_migrations tests.unit.test_integrated_input
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_discord_commands
```

結果:

- workflow/database/integrated_input: 37 tests / OK
- discord_commands: 1 test / OK

検証で確認できたこと:

- `task_extract` と `task_add` は承認前にTask正本を作成しない。
- `approval approve` 後にTaskが作成され、TaskCandidateは `merged` になる。
- LLM未設定時の `task_extract` はdegraded metadataを返し、候補を作成しない。
- `task_update` / `task_delete` は承認前に正本を変更しない。
- `task_notify_due` は通知対象Taskへ通知済みmetadataを付け、同条件で再通知対象にしない。
- Discord botのslash command登録は確認されているが、task Componentの仕様充足まではテストされていない。
