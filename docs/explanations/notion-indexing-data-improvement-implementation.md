# Notion Indexingデータ改善 実装記録

## 目的

`docs/plans/notion-indexing-data-improvement-plan.md` の Phase 1〜8 を、現行の ingestion repository / index build / auto-index 経路に反映した。

この実装は、Notion raw Markdown が存在していても repository / index に反映されない状態を検出し、Notion固有の品質・公開範囲・階層metadata・添付検出・auto-index実行状態を運用データから追えるようにするためのものである。

## 実装範囲

| Phase | 実装内容 |
| --- | --- |
| Phase 1 | `NotionLoader` の同期結果を metadata 化し、page/database誤分類時は page として再取得できるようにした。`ingest audit --source notion` で raw / repository / object storage coverage を確認できる。 |
| Phase 2 | repository-backed build のNotion互換chunk成果物を `data/chunks/*/notion/{page_id}.jsonl` 形式へ統一し、旧形式 `notion.jsonl` は再生成時に stale file として削除する。 |
| Phase 3 | `indexing.notion_quality` を追加し、短文率、heading/url only率、重複率、repository coverage、index coverageを `stage_results.notion_quality.metadata` へ保存する。 |
| Phase 4 | Notion既定公開範囲を `integrations.notion.default_visibility` に切り出し、初期値を `public` にした。raw sidecar / chunk metadataへ `visibility` と `access_scope.visibility` を伝播する。 |
| Phase 5 | Notion再帰取得時のページ階層を `notion_page_path` / `notion_page_path_parts` に保存し、chunk metadataとmaterial aliasへ伝播する。 |
| Phase 6 | 低情報量ページへ `quality_flags` を付け、設定有効時は `index_status=quarantined` にする。完全一致本文には `duplicate_group_id` / `duplicate_group_size` を付ける。 |
| Phase 7 | Notionの `image`, `file`, `pdf`, `video`, `embed` を本文にprivate URLとして出さず、`notion_asset_count` / `notion_unsupported_block_types` として検出できるようにした。 |
| Phase 8 | auto-index runは新規 `running` 行を永続化せず、成功・失敗・キャンセル・skipの終端状態を保存する。Notion coverage集計はrun metadataにも昇格する。 |

## 主要変更

- 設定
  - `configs/main/indexing.yaml`: `indexing.notion_quality` を追加。
  - `configs/main/integrations.yaml`: `integrations.notion.default_visibility` を追加。
  - `src/kumc_agent/config/schema.py`, `src/kumc_agent/config/load.py`: 上記設定のschema / loaderを追加。
- Notion loader
  - `src/kumc_agent/infra/loaders/notion.py`: `default_visibility` と `sync_metadata()` を追加。
  - `src/kumc_agent/infra/loaders/notion_impl.py`: page/database誤分類のpage fallback、階層path、access scope、asset block検出、同期統計を追加。
- Ingestion / connector
  - `src/kumc_agent/infra/connectors/base.py`: loader同期metadataをconnectorから取得可能にした。
  - `src/kumc_agent/features/ingestion/service.py`: `sync_cursors` metadataに `source_sync` を保存する。
  - `src/kumc_agent/infra/connectors/registry.py`: Notion raw itemに品質flags、隔離状態、重複metadataを付与する。
- 品質監査
  - `src/kumc_agent/usecases/ingestion/notion_audit.py`: Notion raw / repository / index coverage、低情報量、重複、access scope、asset検出、stage layoutを監査する。
  - `src/kumc_agent/cli.py`: `ingest audit --source notion` と backfill結果の `metadata.notion_quality` を追加。
- Index / auto-index
  - `src/kumc_agent/features/indexing/service.py`: Notion品質payloadを `stage_results.notion_quality` に入れ、Notion chunk互換成果物をsource別ディレクトリへ出力する。
  - `src/kumc_agent/usecases/indexing/auto_update.py`: 終端状態のみを保存し、Notion coverage集計をrun metadataへ昇格する。
- 設計書
  - `docs/design/circle-info-rag.md`: Notion access scope、page path、品質ゲート、asset検出、source別chunk成果物を追記。
  - `docs/design/auto-index-update.md`: Notion quality、終端run状態、coverage metadataを追記。
  - `docs/design/kumc-agent.md`: Notion metadata、公開範囲、page path embedding、auto-index logを追記。

## 運用上の注意

今回の実装は、欠落を検出し、次回のNotion backfill / auto-indexで正しいmetadataとchunk成果物を生成できるようにする変更である。既存の `data/ingestion/source_items.jsonl` や `data/chunks/*` の実データを直接書き換えるものではない。

実データを更新する場合は、Notion API接続設定を確認したうえで次の順に実行する。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest backfill --source notion
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest audit --source notion
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index update --source notion --full-rebuild
```

`indexing.notion_quality.policy` は初期値 `warn` である。raw / repository / index coverage が揃ったあと、公開前に止めたい運用では `fail` へ切り替える。

## 検証

追加・更新したunit test:

- `tests/unit/test_notion_audit.py`
- `tests/unit/test_notion_loader_impl.py`
- `tests/unit/test_indexing_repository_artifacts.py`
- `tests/unit/test_config_loading.py`
- `tests/unit/test_auto_index_update.py`

代表検証コマンド:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_notion_audit tests.unit.test_notion_loader_impl tests.unit.test_indexing_repository_artifacts tests.unit.test_config_loading tests.unit.test_auto_index_update
```
