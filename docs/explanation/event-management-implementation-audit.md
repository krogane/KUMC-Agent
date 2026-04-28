# イベント管理 実装再調査

調査日: 2026-04-28

対象仕様:
- `docs/design/event-management.md`
- `docs/plan/event-management.md`
- 上位仕様 `docs/design/kumc-agent.md` の「7. イベント管理」

## 結論
イベント管理は、初期実装ではなく、仕様上必要な完全実装として同期済みである。

再調査時点で、前回監査の「仕様との差分」および「仕様改善点」はすべて実装済みである。Event正本は承認前に変更されず、新規候補、変更候補、削除候補、重複検出、承認、まとめ承認、Discord通知、完了確認、auto-index差分抽出、設定接続まで実装されている。

実Discordへのライブ送信は単体テストでは実行していないが、送信層は `discord.py` の `discord.Client` / `fetch_channel()` / `channel.send()` を使う実装になっている。テストではfake senderでdelivery・message id・Component payloadを検証した。

## 再調査結果
| 仕様項目 | 状態 | 主な実装 |
| --- | --- | --- |
| Event / EventCandidate / EventChangeCandidate / EventApprovalBatch | 実装済み | `src/kumc_agent/domain/models/workflow.py` |
| 専用LLM抽出 | 実装済み。`new_events` と `event_changes` を分離し、LLM不可時は候補を作らない | `src/kumc_agent/features/event_management/service.py`, `assets/prompts/event_extraction.md` |
| 手動登録 | 実装済み。`event_add` は専用LLMで抽出し、不足時は確認応答 | `src/kumc_agent/features/workflow/service.py` |
| 手動変更・削除 | 実装済み。`event_update` / `event_delete` はLLM抽出で変更候補を作り、曖昧時は保存しない | `src/kumc_agent/features/workflow/service.py` |
| RAG差分・index差分連携 | 実装済み。`auto_index_update` のingestion差分から `event_extract_from_delta` を呼ぶ | `src/kumc_agent/usecases/indexing/auto_update.py`, `src/kumc_agent/runtime/container.py` |
| 重複検出 | 実装済み。候補・正本との類似を `metadata.duplicate_candidates` に記録 | `src/kumc_agent/features/event_management/service.py` |
| 対象Event解決 | 実装済み。先頭Event fallbackを廃止し、一意解決できない場合は確認応答 | `src/kumc_agent/features/workflow/service.py` |
| 承認・修正・却下・正本反映 | 実装済み。承認までEvent正本は作成・更新・削除されない | `src/kumc_agent/features/workflow/service.py`, `src/kumc_agent/infra/workflow/repository.py` |
| まとめ承認 | 実装済み。期間、channel、message id、delivery、Component custom idを保存 | `src/kumc_agent/features/workflow/service.py` |
| Discord通知 | 実装済み。`discord.py` で指定channelへ送信しdeliveryを保存 | `src/kumc_agent/features/event_management/notifications.py` |
| 完了確認Component | 実装済み。Componentから `event_complete` を実行し `done` にする | `src/kumc_agent/frontends/discord/app.py` |
| Discord Component action | 実装済み。approve / edit / reject / evidence / diff / duplicates / complete done / not done | `src/kumc_agent/frontends/discord/app.py` |
| 設定接続 | 実装済み。`configs/main/event_management.yaml` をRuntimeConfigに接続 | `src/kumc_agent/config/load.py`, `src/kumc_agent/config/schema.py` |
| admin権限 | 実装済み。event_management設定とsecurity保守管理者IDを併用 | `src/kumc_agent/apps/workflow.py`, `src/kumc_agent/runtime/container.py` |
| 通知条件 | 実装済み。before通知は厳密にn日前、timezoneは設定値を使う | `src/kumc_agent/features/event_management/service.py` |
| payload方針 | 実装済み。診断情報は `metadata` 配下に保持 | CLI / HTTP / workflow response既存経路 |

## 仕様との差分の実装結果
| 前回差分 | 実装結果 |
| --- | --- |
| RAGデータ差分からの自動登録連携が未完 | `AutoIndexUpdateUsecase` にevent差分抽出portを追加し、ingestion差分sourceのactive chunkを `event_extract_from_delta` へ渡すようにした。 |
| 自動変更・削除検出が未実装 | `EventExtractionService` が既存Event一覧を入力に取り、`EventChangeCandidate(operation=update/delete)` を生成する。 |
| 手動登録・変更は専用LLMではない | `event_add` / `event_update` / `event_delete` を専用LLM抽出へ統一した。 |
| 曖昧な対象Eventの確認フローがない | `_resolve_event()` のfallbackを廃止し、変更・削除・完了で一意に決まらない場合は候補を保存しない。 |
| Discordへの実通知・完了確認Componentが未完 | `DiscordEventNotificationSender` を追加し、`discord.py` で送信する。通知には完了/未完了Componentを付与できる。 |
| まとめ承認はbatch作成まで | `event_batch_approval` がDiscord送信、message id保存、delivery記録、期間保存、Component生成を行う。 |
| Event用Discord Componentの操作粒度が不足 | `evidence`、`diff`、`duplicates` を個別actionとしてcustom id化し、approval表示経路へ接続した。 |
| 権限設定がconfigsと接続されていない | `EventManagementSection` を追加し、admin user id / role id、通知先、承認間隔、timezoneを設定から注入した。 |
| 通知仕様が実装とずれる | before通知を「n日以内」ではなく「厳密にn日前」に変更し、設定値を既定値にした。 |
| JSONL repositoryはtransaction不可 | JSONLはテスト・ローカル用append-onlyとして明文化し、Event merge時は正本保存、承認履歴保存、候補mergedの順でappendして、途中停止時に未merged候補を再実行しやすい補償順序にした。productionはPostgres transactionを正とする。 |
| 手動登録の曖昧日時チェックが実質重複 | 手動登録はLLM抽出失敗時に候補を作らず、確認応答へ統一した。 |
| 設計書の差分記述が古い | `docs/design/event-management.md` と `docs/plan/event-management.md` を実装後状態へ更新した。 |

## 仕様改善点の実装結果
| 改善点 | 実装結果 |
| --- | --- |
| 自動登録の入力契約明確化 | auto-index差分入力、chunk本文上限、Citation付与、secretをmetadataへ残さない方針を実装・設計へ反映。 |
| 抽出schema分離 | promptと抽出器を `new_events` / `event_changes` / `ignored_items` / `degraded` に変更。 |
| `approved` / `merged` 状態遷移整理 | 承認後は正本反映済みを `merged` とし、設計書へ明記。 |
| 手動登録の必須情報整理 | 新規はtitle/starts_at、変更は対象Event+変更内容、削除は対象Eventを必須として実装。 |
| 対象Event解決仕様 | id、title一致、部分一致を使い、一意でない場合は未解決にする仕様へ変更。 |
| batch状態機械 | `pending` / `sent` を実装し、period、delivery、message id、component nonceを保存。 |
| 通知仕様厳密化 | `before:{n}:{YYYY-MM-DD}`、`day_of:{date}`、`completion:{date}` の通知keyとtimezoneを実装。 |
| Component action語彙固定 | event approval / completionのcustom id語彙を固定。 |
| 権限設定の保存先と型 | `configs/main/event_management.yaml` とRuntimeConfigでDiscord role id文字列を扱う。 |
| deliveryと内部状態更新の分離 | `EventNotificationSender` Protocolとdelivery dataclassを追加し、workflowは送信結果をmetadataへ記録。 |
| 実装監査結果の分離 | 実装監査は本ファイルへ分離し、設計書は要求仕様中心へ更新。 |

## 検証
実行コマンド:

```bash
app/.venv/bin/python -m unittest tests.unit.test_workflow_service tests.unit.test_auto_index_update tests.unit.test_config_loading
```

結果: 31件成功。

```bash
app/.venv/bin/python -m unittest tests.unit.test_workflow_service tests.unit.test_auto_index_update tests.unit.test_config_loading tests.unit.test_integrated_input tests.unit.test_autonomous_agent tests.unit.test_cli_tool_rag tests.unit.test_database_migrations
```

結果: 65件成功。

```bash
app/.venv/bin/python -m unittest discover -s tests/unit
```

結果: 252件成功。

追加で、対象変更ファイルのcompile確認も実施済み。
