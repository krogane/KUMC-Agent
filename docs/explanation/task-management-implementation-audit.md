# タスク管理 実装後再調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/task-management.md`
- `docs/plan/task-management.md`
- 上位仕様として `docs/design/kumc-agent.md` の「6. タスク管理」

## 結論

前回調査で列挙した仕様差分は実装済みです。現時点のタスク管理は、候補抽出、承認前正本化禁止、正本登録、正本変更・削除候補、権限分離、Discord Component、通知delivery、RAG差分連携、payload metadata方針、設定schema、承認transaction、承認履歴を備えています。

完全実装の判定として、コード上の仕様差分は解消しています。外部Discord APIへの実送信は `DiscordTaskNotificationSender` で実装し、テストではfake senderでdeliveryとmetadata保存を検証しました。

## 実装確認

| 仕様項目 | 実装後の状態 | 主な実装箇所 |
| --- | --- | --- |
| task管理config schema | `RuntimeConfig.task_management` として読み込み、WorkflowServiceへ接続 | `config/schema.py`, `config/load.py`, `apps/workflow.py` |
| 権限管理 | admin user/role config、`AccessContext.is_admin`、候補作成者、担当者を操作別に判定 | `features/task_management/service.py` |
| 自動抽出 | 専用LLM serviceで抽出し、LLM未設定・失敗時はdegradedで候補を作らない | `TaskExtractionService` |
| RAG差分連携 | workerの `auto_index_update` 後、変更検出時に `task_extract` を自動実行 | `apps/worker/app.py` |
| 手動登録 | LLM primary、決定的parser fallback、manual evidence、承認前Task作成禁止 | `WorkflowService.task_add()` |
| evidence | RAG citation、LLM根拠label由来の合成Citation、manual input citationを保存 | `TaskExtractionService`, `WorkflowService.task_add()` |
| 重複検出 | 候補・既存Taskとの重複scoreとreasonをmetadataに保存し、既存Task重複時は警告 | `DuplicateTaskDetector`, `WorkflowService.task_add()` |
| 一覧絞り込み | status、担当者、期限範囲、関連イベント、優先度をTaskと候補側に適用 | `WorkflowService.task_list()` |
| 承認transaction | Task作成/更新、candidate `merged`、ApprovalRecordをrepository APIで一括実行 | `infra/workflow/repository.py` |
| 変更・削除候補 | `TaskChangeCandidate` でbefore/after/reason/evidenceを保持し、承認後のみ正本反映 | `WorkflowService.task_update()`, `task_delete()` |
| reject履歴 | reject前後のpayloadをApprovalRecordへ保存 | `WorkflowService.approval()` |
| 差分表示 | TaskChangeCandidate表示にbefore/afterを出力 | `_format_task_change_candidates()` |
| Discord Component | approve/edit/reject/evidence/duplicates/done、modal edit、custom id形式を実装 | `frontends/discord/app.py` |
| まとめ承認 | batch期間、candidate id、通知先、message id、nonce、Discord component deliveryを保存 | `WorkflowService.task_batch_approval()` |
| 通知 | due_soon、overdue、unassigned、blocked_checkを抽出し、delivery結果をTask metadataへ保存 | `TaskNotificationPlanner`, `task_notify_due()` |
| 完了確認 | Componentまたは明示操作で `task_done` を実行し、ApprovalRecordを残す | `task_done()`, Discord component listener |
| HTTPエラー | approvalのnot found / bad requestを利用者向けpayloadに変換 | `frontends/http/app.py` |
| worker/automation payload | `side_effects` を `metadata.side_effects` 配下へ移動 | `apps/worker/app.py`, `apps/automation.py` |
| Task状態 | `deleted` を論理削除状態として仕様へ明記し、通常listから除外 | `docs/design/task-management.md`, repository |

## 差分再調査

| 前回差分 | 再調査結果 |
| --- | --- |
| RAG差分・自動インデックス更新からの抽出未接続 | 解消。`auto_index_update` workerが変更検出後に `task_extract` を呼ぶ |
| Task承認がtransaction化されていない | 解消。`merge_task_candidate()` / `merge_task_change_candidate()` を追加 |
| Discord task Componentが仕様不足 | 解消。edit modal、evidence、duplicates、done、batch custom idに対応 |
| `task_management.yaml` が使われていない | 解消。RuntimeConfig化し、権限、通知、batch、promptへ接続 |
| 通知は状態記録のみ | 解消。Discord REST senderを追加し、delivery結果をmetadataへ保存 |
| 完了確認フロー未実装 | 解消。通知Componentの `done` actionとApprovalRecord履歴を追加 |
| batch保存のみ | 解消。period、channel、message id、component nonce、deliveryを保存 |
| 手動登録・変更がルールベースのみ | 解消。手動登録はLLM primary、fallback付き。変更・削除はevidence付き候補として承認必須 |
| evidenceが空になり得る | 解消。RAG citationがない場合も合成Citationまたはmanual input citationを保存 |
| 重複検出が簡易metadataのみ | 解消。scoreに加えてreasonを保存し、既存Task重複は警告 |
| 候補側filter不足 | 解消。担当者、期限範囲、priority、related_eventを候補にも適用 |
| 期限範囲検索が片側のみ | 解消。from/to/以降/までと2日付範囲を扱う |
| Task変更候補の差分表示が薄い | 解消。before/afterを表示 |
| reject履歴のbeforeが空 | 解消。reject前payloadを保存 |
| 権限設定がconfigと分離 | 解消。task管理admin user/roleをpolicyへ接続 |
| HTTP/Discordエラー変換不足 | 解消。HTTP approvalとDiscord task componentで利用者向け応答へ変換 |
| 評価セット不足 | 改善。既存unittestに通知delivery、batch delivery、config schemaを追加 |

## 仕様改善点の反映

前回提示した仕様改善点は、設計書または実装へ反映済みです。

| 改善点 | 反映先 |
| --- | --- |
| 完了条件をP0/P1/P2へ分ける | `docs/design/task-management.md` 18.1 |
| 権限モデルを一本化 | `docs/design/task-management.md` 18.2、`TaskAccessPolicy` |
| config schema明文化 | `docs/design/task-management.md` 18.3、`RuntimeConfig.task_management` |
| evidence条件明確化 | `docs/design/task-management.md` 18.4、実装 |
| `deleted` 状態の扱い固定 | `docs/design/task-management.md` 18.5、repository |
| 承認transaction API | `docs/design/task-management.md` 18.6、repository |
| Component custom id具体化 | `docs/design/task-management.md` 18.7、Discord実装 |
| 通知送信と記録の分離 | `docs/design/task-management.md` 18.8、通知sender |
| 自然言語抽出責務整理 | `docs/design/task-management.md` 18.9、手動登録実装 |
| 共通エラーpayload | `docs/design/task-management.md` 18.10、HTTP/Discord実装 |
| worker/automation metadata方針 | `docs/design/task-management.md` 18.11、worker/automation実装 |
| 評価セット受入条件 | `docs/design/task-management.md` 18.12、unit test追加 |

## 検証

実行した検証:

```bash
PYTHONPATH=src app/.venv/bin/python -m py_compile src/kumc_agent/config/schema.py src/kumc_agent/config/load.py src/kumc_agent/features/task_management/service.py src/kumc_agent/features/task_management/notifications.py src/kumc_agent/features/workflow/service.py src/kumc_agent/infra/workflow/repository.py src/kumc_agent/apps/workflow.py src/kumc_agent/apps/worker/app.py src/kumc_agent/apps/automation.py src/kumc_agent/frontends/discord/app.py src/kumc_agent/frontends/http/app.py src/kumc_agent/features/autonomous_agent/snapshot.py
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_workflow_service tests.unit.test_database_migrations tests.unit.test_integrated_input tests.unit.test_config_loading tests.unit.test_discord_commands tests.unit.test_automation_hardening
PYTHONPATH=src app/.venv/bin/python -m unittest discover tests/unit
```

結果:

- py_compile: OK
- targeted unit: 55 tests / OK
- full unit: 250 tests / OK

追加確認:

- `task_notify_due` のfake delivery、通知metadata保存、`task_done` ApprovalRecord保存を追加テストで確認。
- `task_batch_approval` のperiod、notification message id、component payloadを追加テストで確認。
- `task_management` config defaultの読み込みを追加テストで確認。
