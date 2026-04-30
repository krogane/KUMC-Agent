# Indexing legacy機能と依存関係の調査

調査日: 2026-04-30

## 結論

現在の公開CLI入口は `index update` に統一されており、`index build` は argparse 上の選択肢から削除済みである。実行経路も CLI `index update`、CLI admin `sync/reindex`、worker `auto_index_update`、automation `auto_index_daily` は `AutoIndexUpdateUsecase` を通る。

一方で、Indexing 内部には次の legacy / fallback がまだ残っている。

- `BuildIndexUsecase` / `UpdateIndexUsecase` は内部APIとして残り、既定値では `prefer_ingestion_repository=False` なので raw chunk pipeline に到達できる。
- `IndexingService.build()` は ingestion repository の active chunk が空の場合、`data/ingestion/*` の raw file から旧chunk pipelineを実行する。
- Minecraft Wiki は ingestion repository を使う場合でも、専用の chunk pipeline と旧JSONL stage artifactを使う。これは単なる不要legacyではなく、現行 Minecraft Wiki RAG の実装基盤である。
- material search、keyword sparse retrieval、summary chunk 判定、root artifact fallback は旧形式の stage chunk / catalog / keyword artifact 互換に依存している。

したがって、単純に `src/kumc_agent/infra/indexing` の旧chunk処理を削除すると、初期データが repository に移行されていない環境、Minecraft Wiki RAG、material search、keyword sparse retrieval、古い root index artifact を読む環境が壊れる。

## 調査対象

- CLI / frontend入口: `src/kumc_agent/cli.py`, `src/kumc_agent/apps/worker/app.py`, `src/kumc_agent/features/automation/service.py`, `src/kumc_agent/frontends/http/app.py`, `src/kumc_agent/frontends/discord/app.py`
- Indexing usecase: `src/kumc_agent/usecases/indexing/build.py`, `src/kumc_agent/usecases/indexing/update.py`, `src/kumc_agent/usecases/indexing/auto_update.py`
- Indexing service: `src/kumc_agent/features/indexing/service.py`, `src/kumc_agent/features/indexing/snapshot.py`, `src/kumc_agent/features/indexing/paths.py`
- 旧形式 helper: `src/kumc_agent/infra/indexing/chunking.py`, `src/kumc_agent/infra/indexing/config.py`, `src/kumc_agent/infra/indexing/material_catalog.py`, `src/kumc_agent/infra/indexing/keyword_inverted_index.py`
- ingestion repository: `src/kumc_agent/features/ingestion/service.py`, `src/kumc_agent/infra/ingestion/repository.py`
- retrieval 側依存: `src/kumc_agent/features/rag/components/retrieval.py`, `src/kumc_agent/features/rag/service.py`, `src/kumc_agent/infra/retrieval/faiss.py`, `src/kumc_agent/infra/retrieval/sudachi_bm25.py`
- 設計 / runbook: `docs/design/auto-index-update.md`, `docs/design/minecraft-wiki-rag.md`, `docs/runbooks/auto_index_update.md`

## 現行入口の状態

### CLI

`kumc-agent index` のサブコマンドは `update` のみである。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index --help
```

結果:

```text
usage: kumc-agent index [-h] {update} ...
```

`index build` は失敗する。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index build --help
```

結果:

```text
kumc-agent index: error: argument index_command: invalid choice: 'build' (choose from update)
```

CLI `index update` は `AutoIndexUpdateRequest(trigger="manual")` を作り、`context.auto_index_update.execute()` を呼ぶ。CLI admin `sync/reindex` も同じ usecase を呼び、`reindex` の場合は `force=True`, `full_rebuild=True` になる。

### worker / automation

worker `auto_index_update` は `build_runtime_context()` から `runtime.auto_index_update.execute()` を呼び、payload の `source_filter`, `force`, `full_rebuild`, `quality_check_enabled`, `scheduled_at` を `AutoIndexUpdateRequest` に詰める。

automation の既定ルール `auto_index_daily` は action type `auto_index_update` を発火する。`apps/automation.py` の action executor はこの action を worker job として実行するため、公開系の定期更新は worker 経由で `AutoIndexUpdateUsecase` に入る。

### HTTP / Discord admin の差分

HTTP `/admin/action/{sync|reindex}` と Discord `/admin action sync|reindex` は、現時点では `AutoIndexUpdateUsecase` ではなく `IngestionService.backfill_many()` だけを呼ぶ。つまり raw / repository への取り込みは行うが、snapshot build / quality check / publish は行わない。

これは `docs/runbooks/auto_index_update.md` と `docs/design/auto-index-update.md` の「admin sync/reindex は同じ自動更新 usecase に集約」という説明と差分がある。CLI admin は設計通りだが、HTTP / Discord admin は ingestion-only の旧挙動が残っている。

## Indexing内部の分岐

`IndexingService.build()` の分岐は次の通り。

1. `prefer_ingestion_repository=True` かつ `ingestion_repository.load_active_chunks()` が1件以上返る場合、repository-backed path を使う。
2. repository chunk のうち `source_type/source_kind == minecraft_wiki` は除外される。
3. 除外後の repository chunk から first/second/sparse/summary相当の artifact を再構成する。
4. Minecraft Wiki は repository-backed path でも専用 chunk pipeline を別途走らせる。
5. repository chunk が空、または `prefer_ingestion_repository=False` の場合、raw chunk pipeline を実行する。
6. 最終的に `dense_vectors.*`, `dense_chunks.jsonl`, `bm25_*`, keyword index, material catalog, `data/chunks/chunks.jsonl` を作る。

公開自動更新では `AutoIndexUpdateUsecase` が `BuildIndexRequest(index_dir=staging_dir, prefer_ingestion_repository=True)` を渡す。直接 `BuildIndexUsecase.execute()` / `UpdateIndexUsecase.execute()` を呼ぶ場合は request の既定値が `prefer_ingestion_repository=False` なので、raw chunk pipeline が到達可能である。

## legacy機能一覧

| legacy / fallback | 実体 | 到達条件 | 依存している機能 |
| --- | --- | --- | --- |
| `index build` 公開CLI | 旧公開コマンド名 | 現在は到達不可 | なし。CLIでは削除済み。 |
| `BuildIndexUsecase` 内部API | `src/kumc_agent/usecases/indexing/build.py` | `RuntimeContext.build_index`、テスト、内部直接呼び出し | `AutoIndexUpdateUsecase` の staging build、直接 build API、Minecraft Wiki refresh補助 |
| `UpdateIndexUsecase` | `src/kumc_agent/usecases/indexing/update.py` | `RuntimeContext.update_index` から直接呼ぶ場合 | 現在の公開CLIからは未使用。build wrapperとして残存 |
| legacy loader refresh | `DiscordLoader`, `GoogleDriveLoader`, `HatenaBlogLoader`, `CraftersColonyLoader`, `XPostsLoader`, `NotionLoader` | `BuildIndexRequest.refresh_sources=True` の直接 build | raw sourceを `data/ingestion/*` に作る旧refresh経路 |
| raw chunk pipeline | `IndexingService._run_legacy_chunk_pipeline()` | repository active chunk が空、または `prefer_ingestion_repository=False` | 初期環境、直接 build、raw file only 環境、旧テスト |
| legacy stage chunk loader | `_load_index_chunks_from_legacy_dirs()`, `_load_legacy_chunks_from_dirs()` | raw chunk pipeline 後、Minecraft Wiki artifact 読み込み時 | dense/BM25 build、summary除外判定、旧JSONL chunk互換 |
| legacy `AppConfig` adapter | `infra.indexing.config.AppConfig` | `IndexingService.build()` 毎回 | chunking helper、summary helper、keyword/material helper の設定橋渡し |
| legacy prompt env default | `_ensure_legacy_prompt_env_defaults()` | `IndexingService.build()` 毎回 | `summery_chunk_jsonl_dir()` の旧prompt env要求を満たす |
| material catalog legacy build | `_build_material_catalog_legacy()` と `infra.indexing.material_catalog` | raw chunk path の場合 | RAG material search、`material_names` keyword corpus |
| keyword index legacy build | `_build_keyword_inverted_indexes()` | raw chunk path の場合 | sparse mixed retrieval、material search、Minecraft Wiki keyword retrieval |
| Minecraft Wiki chunk pipeline | `_run_minecraft_wiki_chunk_pipeline()` | `data/ingestion/minecraft_wiki` が存在する場合 | Minecraft Wiki RAG。repository-backed pathでも使用 |
| root index artifact fallback | `resolve_current_index_dir()` が `current.json` 不在時 rootを返す | 旧root artifact環境、直接 root build | dense/BM25/keyword/material artifact reader |

## 主要依存の詳細

### 1. RAG retrieval

RAG の `RetrievalComponent` は dense retrieval で `FaissLikeIndex`、sparse retrieval で `SudachiBM25Retriever` と keyword index を読む。これらは `resolve_current_index_dir()` を通じて `current.json` の release snapshot を読むが、`current.json` がない場合は `data/index` root をそのまま読む。

依存artifact:

- `dense_vectors.npy`
- `dense_vectors.faiss`
- `dense_chunks.jsonl`
- `bm25_tokens.json`
- `bm25_chunks.jsonl`
- `keyword/sparse_second_rec.json`
- `keyword/second_rec_sparse.json`

raw chunk pipelineを削除するだけなら repository-backed path で artifact を生成できるが、root fallback を削除すると旧root artifactしかない環境や直接 build の検索が壊れる。

### 2. material search

`RagService` の material route は `material_catalog.json` と `keyword/material_names.json` を読む。legacy catalog では `raw_path` が base dir 相対で保存されるため、読み込み側には `base_dir / raw_path` を試す互換処理が残っている。

repository-backed path では `IndexingService._build_material_catalog_from_repository_chunks()` が `data/material_raw/*.txt` を作り、その絶対pathを catalog に保存する。raw path では `infra.indexing.material_catalog.build_material_catalog()` が `data/ingestion/docs`, `sheets`, `hatenablog`, `crafters_colony`, `notion`, `messages`, `x`, `vc` を走査して catalog を作る。

したがって material search を維持するには、raw catalog 互換を削除する前に repository-backed catalog だけで全sourceの alias / canonical name / raw text が同等に作れることを確認する必要がある。

### 3. sparse mixed retrieval

`RetrievalComponent._search_sparse_mixed_sources()` は keyword corpus `sparse_second_rec` と `second_rec_sparse` を混ぜて検索する。Minecraft Wiki route では `minecraft_wiki_sparse_second_rec` と `minecraft_wiki_second_rec_sparse` を使う。

repository-backed path は `_build_keyword_inverted_indexes_from_repository_artifacts()` で同じ corpus 名を生成して互換を保っている。raw path は `_build_keyword_inverted_indexes()` が旧stage chunk dirから corpus を生成する。

corpus 名や metadata の `chunk_stage`, `chunk_id`, `source_file_name`, `minecraft_wiki_page_id` を変えると、sparse hit を second recursive chunk に戻す処理が壊れる。

### 4. Minecraft Wiki RAG

Minecraft Wiki は「legacy fallback」ではなく、現行機能として旧chunk helperに依存している。

- 手動 build refresh 時、`BuildIndexUsecase._refresh_minecraft_wiki_source()` が `IngestionService.backfill_many(source_kinds=("minecraft_wiki",))` を呼べる。
- `IndexingService.build()` は repository chunk から Minecraft Wiki chunk を除外し、`_run_minecraft_wiki_chunk_pipeline()` の成果物を index chunk に足す。
- Minecraft Wiki専用の chunking / retrieval 設定は `minecraft_wiki_rag.*` から `LegacyAppConfig` に橋渡しされる。
- quality gate は raw dir `data/ingestion/minecraft_wiki` と index chunk数を使って `stage_results.minecraft_wiki_quality` に残る。

このため、`recursive_chunk_dir`, `recursive_chunk_jsonl_dir`, `sparse_chunk_jsonl_dir`, `load_chunks/write_chunks` を削除するには、Minecraft Wiki 専用の repository-native chunking / summary / sparse artifact 生成を先に実装する必要がある。

### 5. image search

image search の index build は `IndexingService.build()` 後半で `image_asset_builder.build_from_ingestion_sources(index_dir=..., commit_repository=False)` を呼ぶ。これは raw source / ingestion source 配下の画像候補を scan し、publish 成功後に `commit_staged_assets()` で正本へ反映する。

raw chunk pipelineそのものには依存しないが、`data/ingestion` の source cache と staged snapshot publish には依存している。`clear_ingestion_source_data` や `ingestion_dir` の意味を変える場合、画像候補のscan元も同時に確認する必要がある。

### 6. workflow extraction / member_profiles / task_event

`AutoIndexUpdateUsecase` は publish後に workflow delta extraction を走らせる。抽出対象の chunk は `event_delta_chunk_source=ingestion_repository` から `load_active_chunks(source_kinds=...)` で読むため、raw chunk pipelineではなく ingestion repository に依存している。

`member_profiles` と `task_event` は auto-index の追加stageとして `staging_dir` に indexを作る。これらも raw chunk pipelineとは別系統だが、公開snapshotと `current.json` の解決には依存する。

## 削除・移行判断

### 今すぐ削除しやすいもの

- 公開CLI `index build`: すでに削除済みなので追加対応は不要。

### 削除前に置換が必要なもの

- `BuildIndexUsecase` の直接 raw refresh: 直接 build APIを残すなら、既定値を repository preferred に変えるか、直接 build caller を auto-index 経由に寄せる必要がある。
- raw chunk pipeline: repository-empty環境の初期build手段を別途用意する必要がある。少なくとも raw sourceから ingestion repository へ移行する backfill / migration が必要。
- legacy stage chunk JSONL: Minecraft Wiki RAG、summary判定、keyword sparse retrieval、material catalog の置換が必要。
- legacy keyword corpus名: retrieval 側が同じ corpus 名を読んでいるため、名称変更は reader と既存snapshot互換を同時に扱う必要がある。
- root artifact fallback: 既存 `data/index` root artifact の migration、または `current.json` 自動生成が必要。

### 注意が必要な仕様差分

- CLI admin `sync/reindex` は auto-index publish まで行う。
- HTTP / Discord admin `sync/reindex` は ingestion backfill のみで、index publish を行わない。

この差分を仕様として残すなら runbook/design に明記する必要がある。設計通りに統一するなら HTTP / Discord admin も `AutoIndexUpdateUsecase` を呼ぶように変更する。

## 確認コマンド

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index --help
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index build --help
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_indexing_repository_artifacts tests.unit.test_auto_index_update
```

確認結果:

- `index --help`: subcommand は `{update}` のみ。
- `index build --help`: `invalid choice: 'build'` で終了。
- unit test: 16 tests, OK。
