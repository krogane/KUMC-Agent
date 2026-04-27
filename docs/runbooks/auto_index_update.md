# 自動インデックス更新 Runbook

## 手動実行

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index update
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli worker --job-type auto_index_update
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action sync
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action reindex --force
```

`index update`、worker `auto_index_update`、admin `sync/reindex` は同じ自動更新 usecase を呼ぶ。
実行結果のトップレベルは `status`、`run_id`、`seen`、`changed`、`skipped`、`deleted` を安定フィールドとし、差分内訳、品質確認、snapshot、skip 理由は `metadata` 配下に出る。

## 定期実行

定期実行条件は `configs/ops/scheduler.yaml` の次の値で管理する。

- `scheduler.auto_index_enabled`
- `scheduler.auto_index_time`
- `scheduler.auto_index_weekdays`
- `scheduler.auto_index_lock_ttl_minutes`
- `scheduler.quality_min_chunk_ratio`
- `scheduler.quality_smoke_queries`
- `scheduler.rollback_keep_snapshots`

automation の既定ルールは `auto_index_daily`。実行前確認は次で行う。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action dry_run --rule-id auto_index_daily --admin
```

## 成果物

自動更新は `data/index/staging/{run_id}` に index を作成し、quality smoke check 後に `data/index` 直下へ公開する。
互換性のため検索側は従来通り `data/index` を読む。公開情報は `data/index/current.json` と `data/index/previous.json` に保存する。

## 障害時

lock 取得不可の場合、run は `status=skipped` になり、理由は `metadata.reason=lock_already_held` に残る。
品質確認失敗時は公開せず `status=failed` とし、`metadata.quality_check.critical_failures` と `metadata.notification` を確認する。

直前 snapshot へ戻す必要がある場合は、`data/index/previous/<snapshot_id>` の内容を `data/index` 直下へ戻す。
`current.json`、`previous.json`、`staging/`、`previous/` は削除せず、復元対象の index artifact だけを戻す。
