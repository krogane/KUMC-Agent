# 自動インデックス更新 実装後再調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/auto-index-update.md`
- `docs/plan/auto-index-update.md`

## 結論

前回調査で確認した仕様差分は実装済みです。現時点の自動インデックス更新は、仕様上必要な起動経路、差分検出、削除・権限喪失除外、画像/member_profiles/Task/Event stage、品質確認、rollback、通知記録、payload整理を備えています。

運用上は、workerまたは外部cronから `auto_index_update` を起動する構成です。automation default ruleも `configs/main/scheduler.yaml` の設定から生成され、worker payloadの `trigger` / `scheduled_at` も尊重されます。

## 実装確認

| 仕様項目 | 実装後の状態 | 主な実装箇所 |
| --- | --- | --- |
| scheduler設定に従う日次更新 | `auto_index_enabled/time/weekdays/timezone` を `AutoIndexUpdateUsecase` が判定し、automation cronもconfigから生成 | `src/kumc_agent/usecases/indexing/auto_update.py`, `src/kumc_agent/apps/automation.py` |
| CLI/worker/automation/admin経路の統一 | `index update`、admin `sync/reindex`、worker `auto_index_update`、automation actionが同じusecaseを呼ぶ | `src/kumc_agent/cli.py`, `src/kumc_agent/apps/worker/app.py` |
| 二重起動lock | Postgres、Redis、File fallback lockでskip理由を `IndexingRun.metadata` に保存 | `src/kumc_agent/features/indexing/lock.py` |
| cursor/checksum/revision/ACL hash差分 | `sync_cursors` を読み書きし、cursorがあれば `poll_changes()`、なければbackfill。checksum/revision/ACL hashでskip/update判定 | `src/kumc_agent/features/ingestion/service.py`, `src/kumc_agent/infra/ingestion/repository.py`, `src/kumc_agent/features/indexing/change_detection.py` |
| source単位degraded | source別例外を `IngestionResult(status="failed")` に集約し、他sourceの更新を継続。全source失敗時はrun失敗 | `src/kumc_agent/features/ingestion/service.py`, `src/kumc_agent/usecases/indexing/auto_update.py` |
| raw snapshot保存 | 変更itemを `RawSnapshotStore` に保存し、外部payloadへraw本文を出さない | `src/kumc_agent/features/ingestion/service.py` |
| 削除・権限喪失除外 | Postgresは `source_items/chunks.index_status` 更新、File fallbackは状態ログを反映。自動更新indexはactive chunkだけを正本にする | `src/kumc_agent/infra/ingestion/repository.py`, `src/kumc_agent/features/indexing/service.py` |
| Dense/Sparse/資料名index | 自動更新はingestion repository active chunksを優先し、Dense/BM25をstagingへ構築。互換pipelineはfallbackとして維持 | `src/kumc_agent/features/indexing/service.py` |
| 画像index | caption/OCR/feature vector indexをstagingの `image_search` 配下へ構築。削除候補はAssetを `index_status=deleted` に更新 | `src/kumc_agent/features/image_search/service.py` |
| member_profiles | member fingerprint一致時はskipし、退会・除外profileはinactive化して検索除外 | `src/kumc_agent/features/member_search/service.py` |
| Task/Event正本index | workflow repositoryの承認済み正本から `task_event` indexを作成。削除Task/canceled Eventは除外 | `src/kumc_agent/features/indexing/task_event.py` |
| quality smoke check | Dense/Sparse load、chunk急減、smoke query、禁止status混入、画像/member/task_event index loadを確認 | `src/kumc_agent/features/indexing/quality.py` |
| rollback | publish失敗時にprevious snapshotへrollbackし、`metadata.rollback` に保存 | `src/kumc_agent/features/indexing/snapshot.py`, `src/kumc_agent/usecases/indexing/auto_update.py` |
| admin通知 | 失敗・rollback時の通知payloadを `ActionRun(action_type="indexing_notification")` として記録 | `src/kumc_agent/usecases/indexing/auto_update.py` |
| payload方針 | CLI/worker payloadの診断情報を `metadata` 配下に整理し、raw/context/secret系metadataをマスク | `src/kumc_agent/usecases/indexing/auto_update.py`, `src/kumc_agent/apps/worker/app.py` |

## 差分再調査

前回の未達項目ごとの再確認結果です。

| 前回差分 | 再調査結果 |
| --- | --- |
| 日次スケジュールが仕様通りではない | 解消。`auto_index_timezone` を追加し、workerがpayload triggerを尊重。automation default cronもscheduler configから生成 |
| cursor / sync_cursors 未接続 | 解消。File/Postgres repositoryにcursor read/writeを追加し、`IngestionService` が `poll_changes()` を優先 |
| 削除・権限喪失の検索除外が不完全 | 解消。File fallbackでも状態ログを反映し、index構築はactive chunkだけを使用 |
| build pipelineとingestion stateが分離 | 解消。自動更新時は `prefer_ingestion_repository=True` でingestion active chunksをDense/Sparseの正本にする |
| source単位失敗継続なし | 解消。source別失敗はdegraded metadataへ集約し、他sourceを継続 |
| rollback未接続 | 解消。publish例外時にprevious snapshotへrollbackし、結果をrun metadataに保存 |
| quality checkが狭い | 解消。Dense/Sparse artifact load、BM25 smoke query、feature index loadを追加 |
| 画像index連携が部分実装 | 解消。staging配下へimage indexを作り、削除候補をAsset statusへ反映 |
| member_profilesが全体rebuild寄り | 解消。source_fingerprintでskipし、退会・除外profileをinactive化 |
| Task/Event正本index未実装 | 解消。`TaskEventIndexBuildService` を追加 |
| admin通知はpayloadのみ | 解消。operations repositoryに通知ActionRunを記録 |
| payload top-levelに診断情報 | 解消。worker auto-indexの `side_effects` はmetadata配下へ移動 |

## 残リスク

完全に仕様通りであることを確認しましたが、運用面の残リスクはあります。

- 実際の外部API差分効率はconnector実装に依存します。cursor保存と `poll_changes()` の経路は接続済みですが、loader-backed connectorは現在もbackfill相当のpollです。
- publishは既存検索runtime互換のため `data/index` root成果物を更新します。失敗時rollbackは実装済みですが、検索プロセスがcopy中の一瞬を読むリスクを完全にゼロにするには、検索runtime側をpointer参照へ移行する追加改善が必要です。
- Discord等への実送信通知ではなく、現実装のadmin通知はoperations repositoryへの記録です。設計上許容している「ログまたは将来のnotification repository」経路に該当します。

## 検証

実行した検証:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_auto_index_update tests.unit.test_ingestion_service tests.unit.test_config_loading tests.unit.test_image_search tests.unit.test_automation_hardening
PYTHONPATH=src app/.venv/bin/python -m unittest discover tests/unit
PYTHONPATH=src app/.venv/bin/python -m unittest tests.integration.test_chat_index_eval
```

結果:

- targeted unit: 25 tests / OK
- full unit discovery: 218 tests / OK
- integration `test_chat_index_eval`: 1 test / OK

追加確認:

- `tests.unit.test_auto_index_update` に、File fallback削除除外とTask/Event正本indexの検証を追加。
- `tests.unit.test_ingestion_service` によりchecksum skip、secret quarantine、cursor接続後の再実行を確認。
