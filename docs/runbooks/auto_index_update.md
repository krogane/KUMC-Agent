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

`member_profiles` のメンバー情報取得先 Guild は `security.discord_member_profile_guild_ids` で指定する。未設定時は後方互換として `security.discord_guild_allow_list` を使う。

## 定期実行

定期実行条件は `configs/main/scheduler.yaml` の次の値で管理する。

- `scheduler.auto_index_enabled`
- `scheduler.auto_index_time`
- `scheduler.auto_index_weekdays`
- `scheduler.auto_index_timezone`
- `scheduler.auto_index_max_runtime_minutes`
- `scheduler.auto_index_lock_ttl_minutes`
- `scheduler.quality_min_chunk_ratio`
- `scheduler.quality_smoke_queries`
- `scheduler.rollback_keep_snapshots`

automation の既定ルールは `auto_index_daily`。実行前確認は次で行う。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action dry_run --rule-id auto_index_daily --admin
```

## 成果物

自動更新は `data/index/staging/{run_id}` に index を作成し、quality smoke check 後に `data/index/releases/{run_id}` へ完全な snapshot として公開する。
検索側は `data/index/current.json` が指す release directory を読む。公開情報は `data/index/current.json` と `data/index/previous.json` に保存する。

`image_search` などのfeature indexも release snapshot 配下を読む。画像 asset repository への反映は publish 成功後に staged manifest から commit するため、quality check や publish が失敗した run の画像 asset は正本へ反映されない。

Dense embedding は通常更新では差分cacheを使う。cache本体は `data/cache/index_embeddings/` にあり、公開indexには含めない。公開snapshotには `dense_embedding_manifest.jsonl` が含まれ、chunkごとの embedding text hash、provider、model、dimensions、source参照を確認できる。

`IndexingRun.metadata.stage_results.embedding` で次を確認する。

- `embedded_chunks`: 今回実際に `embed_documents()` に渡したchunk数
- `reused_chunks`: cacheからvectorを再利用したchunk数
- `cache_misses`: cacheがなく再埋め込みしたchunk数
- `cache_invalid`: 壊れたcache行や次元不一致で無視したrecord数
- `cache_compaction`: publish成功後のcache compact結果

`--full-rebuild` / admin `reindex` は既定でcacheをbypassし、全chunkを再埋め込みする。通常の `index update` でcacheを使いたくない場合は `configs/main/indexing.yaml` の `indexing.embedding_cache.enabled` を `false` にする。

## 障害時

lock 取得不可の場合、run は `status=skipped` になり、理由は `metadata.reason=lock_already_held` に残る。File fallback の lock は `data/locks/auto_index.lock` に置くため、publish 対象の `data/index` からは分離される。実行中は heartbeat で lock を更新し、`auto_index_max_runtime_minutes` 超過時は失敗扱いにする。
品質確認失敗時は公開せず `status=failed` とし、`metadata.quality_check.critical_failures` と `metadata.notification` を確認する。

embedding cacheが壊れている場合、壊れたrecordは無視され、対象chunkは再埋め込みされる。`cache_invalid` が増え続ける場合は `data/cache/index_embeddings/` を退避または削除して次回runで再作成する。cacheは最適化用であり、削除しても公開済みindex artifactは維持される。

直前 snapshot へ戻す必要がある場合は、`data/index/previous.json` の `snapshot_id` が指す `data/index/releases/{snapshot_id}` へ `current.json` を戻す。
手作業で root artifact をコピーしない。`current.json`、`previous.json`、`staging/`、`releases/` は削除せず、必要なら `IndexSnapshotPublisher.rollback_to_latest_previous()` と同じ pointer 更新だけを行う。
