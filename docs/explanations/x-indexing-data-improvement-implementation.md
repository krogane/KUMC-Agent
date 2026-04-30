# X Indexing データ改善 実装メモ

作成日: 2026-04-30

## 実装した内容

- `data/ingestion/x/posts.jsonl` を 1 ファイル 1 source item として扱うのをやめ、`posts.jsonl` の 1 行を 1 投稿として `SourceRawItem` 化する `iter_x_posts()` を追加した。
- X connector は `posts.jsonl` だけを allowlist として読む。`data/ingestion/x/data/direct-messages*.js`、account/profile/IP/phone 系 raw archive は scanner の対象外。
- 旧 `external_id=x:posts.jsonl` は backfill 時に `SourceDeleteItem` として削除扱いにする。
- X chunk の `chunk_kind` は `x_post` にし、`source_type=x_posts`、`x_post_id`、`x_post_url`、`message_timestamp` を chunk metadata まで保持する。
- repository-backed build と legacy raw build の両方で X を投稿単位にし、X の summary chunk は初期対応として生成しない。
- X quality gate を index build に追加し、以下を critical failure として止める。
  - X chunk の必須 metadata 欠落
  - `external_id=x:posts.jsonl` の残存
  - X chunk 数が投稿数を下回る状態
- `x_impl.py` の archive 変換で account.js 由来の default handle / account id / display name を補完する。
- URL entity の expanded URL を本文に反映し、media 用および残存 `https://t.co/` は本文から除去する。
- `x_media` に `type`、`remote_url`、`local_relative_path`、`content_hash`、`thumbnail_remote_url` を保持する。
- image-search は `x_media.local_relative_path` の local file を remote URL より優先して asset 化する。

## 更新後の実測

- `data/ingestion/x/posts.jsonl`: 445 投稿。
- `data/ingestion/current_source_items.jsonl`: X source item 445 件。
- `data/ingestion/current_chunks.jsonl`: `source_type=x_posts` の X chunk 445 件。
- `data/chunks/first_rec_chunk/x_posts.jsonl`: 445 件。
- `data/chunks/second_rec_chunk/x_posts.jsonl`: 445 件。
- `data/chunks/sparse_second_rec_chunk/x_posts.jsonl`: 444 件。本文が sparse token 化で空になる投稿が 1 件あるため。
- `data/index/dense_chunks.jsonl` / `data/index/bm25_chunks.jsonl`: `source_type=x_posts` の X chunk 445 件。
- `data/index/staging/auto-index-20260429T161458453928Z-manual/dense_chunks.jsonl` / `bm25_chunks.jsonl`: `source_type=x_posts` の X chunk 445 件。
- `external_id=x:posts.jsonl` は `data/chunks` / `data/index` から消えている。
- X chunk 本文内の `https://t.co/` は 0 件。
- `x_author_handle` 欠損は 0 件。

## 実行した更新コマンド

```bash
PYTHONPATH=src app/.venv/bin/python -c "from pathlib import Path; from kumc_agent.infra.loaders.x_impl import convert_x_tweets_js_to_jsonl; root=Path('data/ingestion/x'); stats=convert_x_tweets_js_to_jsonl(raw_x_dir=root, output_path=root/'posts.jsonl', skip_existing=False, update_existing=False, sync_deleted=True); print(stats)"
```

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest tests/unit/test_x_indexing_improvements.py tests/unit/test_raw_loaders_update_policy.py tests/unit/test_ingestion_service.py tests/unit/test_docs_connector_records.py
```

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest tests/unit/test_x_indexing_improvements.py tests/unit/test_indexing_repository_artifacts.py tests/unit/test_indexing_summary_chunking_llm.py
```

本番相当の `index update` は Gemini embedding / image caption 連携を含むため、検証では `KUMC_EMBEDDING_PROVIDER=local` と `KUMC_INDEXING_SUMMARY_ENABLED=false` を指定し、local hash embedding で repository-backed build を実行した。

## 注意点

- `data/ingestion`、`data/chunks`、`data/index` は `.gitignore` 対象の実データであり、コード差分としては残らない。
- 通常運用で再構築する場合は、X backfill 後に repository-backed の `index update` を実行する。Gemini を使う環境では通常設定のままでよい。
- CLI から大量の file-backed ingestion を実行すると、現状は item ごとの compaction で遅くなる。今回の検証では `auto_compact=False` で backfill し、最後に X だけを 1 回 compact した。
