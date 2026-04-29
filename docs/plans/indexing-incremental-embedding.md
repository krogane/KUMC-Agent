# Indexing 差分埋め込み実装計画

## 1. 目的

Indexing 時に毎回すべての chunk を embedding API / embedding model に渡すのではなく、通常更新では新規・本文変更があった chunk だけを埋め込み、未変更 chunk は既存 embedding を再利用する。

この計画の主目的は embedding API 呼び出し回数と実行時間の削減である。現行の `FaissLikeIndex` は全体 artifact を上書きするため、当面は `dense_vectors.npy` / `dense_vectors.faiss` / `dense_chunks.jsonl` は毎 run で完全な成果物として再生成する。ただし、完全成果物を作るための vector 行列は「再利用済み vector + 差分埋め込み vector」を結合して作る。

## 2. 現状調査

### 2.1 差分検出は source / ingestion 層に存在する

- `IngestionService.backfill()` は `sync_cursors` を持つ source では `poll_changes(cursor)` を優先し、そうでない場合は backfill で全件を見たうえで checksum / revision / ACL hash を比較する。
  - 根拠: `src/kumc_agent/features/ingestion/service.py` の `backfill()` は `load_item_states()` と `load_sync_cursor()` を読み、差分なしなら `skipped` にして `_ingest_raw_item()` を呼ばない。
- `detect_source_change()` は `new`、`updated`、`permission_changed`、`skipped` を判定する。
  - 根拠: `src/kumc_agent/features/indexing/change_detection.py`
- File / Postgres の ingestion repository は active chunk だけを返す。
  - Postgres: `load_active_chunks()` が `c.index_status = 'active'` と `si.index_status = 'active'` を条件にする。
  - File fallback: `source_deletes.jsonl` を反映し、active document / active chunk だけを返す。

### 2.2 auto-index は差分がない場合は build しない

- `AutoIndexUpdateUsecase.execute()` は ingestion 結果の `changed` / `deleted` と `force` / `full_rebuild` / member / task_event の予定を見て `has_changes` を判定する。
- `has_changes` が false の場合は `metadata.reason = "no_source_changes"` で `skipped` として終了する。
- 差分がある場合は `BuildIndexUsecase` に `prefer_ingestion_repository=True` を渡し、staging directory に index を構築する。

### 2.3 現在の dense embedding は index_chunks 全件を毎回埋め込む

- `IndexingService.build()` は ingestion repository または legacy pipeline から `index_chunks` を作る。
- その直後に `dense_texts = [...]` で全 chunk の embedding text を作り、`self._embedder.embed_documents(dense_texts)` に全件を渡している。
- その結果を `self._faiss_index.build(chunks=index_chunks, embeddings=embeddings)` に渡す。

該当箇所:

- `src/kumc_agent/features/indexing/service.py`
  - `load_active_chunks()` を読む箇所
  - `dense_texts` 全件生成
  - `embed_documents()` 全件呼び出し
  - `FaissLikeIndex.build()` 呼び出し

### 2.4 FaissLikeIndex / BM25 は全体上書き方式

- `FaissLikeIndex.build()` は受け取った embedding 行列を `dense_vectors.npy` に保存し、FAISS index を新規作成して `dense_vectors.faiss` に書く。
- 同時に chunk 一覧を `dense_chunks.jsonl` に上書きする。
- `SudachiBM25Retriever.build()` も tokens / chunks を全体上書きする。

このため、まずは「vector store の部分 upsert/delete」ではなく、「embedding 計算だけ差分化し、最終 artifact は全体を再生成する」方針が最小変更で安全である。

### 2.5 既存の query-time embedding cache は indexing には使われていない

- `HybridRetrievalService.embed_missing_chunks()` は retrieval repository の `load_embeddings()` / `save_embeddings()` を使って、検索時に不足 chunk だけを lazy embedding する。
- File backend は `data/ingestion/embeddings.jsonl` 相当、Postgres backend は `embeddings` table を使う。
- ただしこれは query-time retrieval 用であり、`IndexingService.build()` の dense index 作成には接続されていない。
- 既存 Postgres `embeddings` table は `chunk_id references chunks(id)` を持つため、indexing pipeline が生成する summary chunk や legacy / Minecraft Wiki stage chunk をそのまま保存する用途には不十分な可能性がある。

### 2.6 既存設計書の位置づけ

`docs/design/auto-index-update.md` は現行の `FaissLikeIndex.build()` / `SudachiBM25Retriever.build()` が全体上書きであるため、初期実装では「差分で処理対象を絞りつつ、公開 index は全体再構築」でよいとしている。

今回の実装はこの初期実装から一段進めて、公開 index の全体再構築は維持しつつ、embedding 計算だけを差分化する。

## 3. 実装方針

### 3.1 差分化の単位

差分判定は source item ではなく、最終的に dense index へ入る `index_chunks` 単位で行う。

理由:

- 1つの source item 更新で生成 chunk 数や chunk 境界が変わる可能性がある。
- summary chunk / Minecraft Wiki chunk は indexing stage で派生するため、source item 単位だけでは最終 embedding 対象と一致しない。
- ACL だけの変更では chunk text / embedding text は変わらない場合があり、その場合は vector を再利用できる。

### 3.2 cache key

embedding 再利用の key は最低限次を含める。

| 項目 | 理由 |
| --- | --- |
| `chunk_id` | chunk の安定識別 |
| `embedding_text_hash` | embedding に渡す実文字列の同一性判定 |
| `provider` | local / gemini など provider 差分 |
| `model` | model 差し替え時の誤再利用防止 |
| `dimensions` | 次元数変更時の誤再利用防止 |

`embedding_text_hash` は `stable_hash(_chunk_embedding_text_for_dense(chunk))` で作る。`chunk.metadata.checksum` だけには依存しない。Minecraft Wiki では title / heading を prefix して embedding text を作るため、実際に embedder へ渡す文字列の hash を保存する必要がある。

### 3.3 cache 保存先

実装は indexing 専用 cache として追加する。

推奨保存先:

- File fallback: `config.app.cache_dir / "index_embeddings" / "{provider}-{safe_model}-{dimensions}.jsonl"`
- 追加 artifact: staging / published index に `dense_embedding_manifest.jsonl`

`data/index` 直下は publish / rollback の対象であり、cache の正本にはしない。cache は optimization であり、欠損しても正しく full embed できる必要があるため、`data/cache` 配下に置く。

Postgres 利用時の共有 cache は Phase 2 で検討する。既存 `embeddings` table は FK 制約上、stage 派生 chunk すべての cache には使いにくいため、流用する場合は慎重に確認する。

### 3.4 full rebuild の扱い

`--full-rebuild` / admin `reindex` は従来の意味を維持する。

- 既定では cache を bypass して全件再埋め込みする。
- 将来必要なら config で `reuse_on_full_rebuild` を追加できるが、初期実装では false にする。

通常の `index update` / scheduled auto-index では cache を使い、未変更 chunk の vector を再利用する。

### 3.5 削除・権限変更の扱い

- deleted / permission_lost / quarantined chunk は `load_active_chunks()` の時点で index_chunks に入らないため、最終 `dense_vectors.npy` から自然に除外される。
- ACL だけ変わった chunk は embedding text が同じなら vector を再利用し、`dense_chunks.jsonl` の metadata だけ新しくなる。
- `permission_changed` で embedding text に影響する metadata が変わった場合は、`embedding_text_hash` が変わるため再埋め込みされる。

## 4. 変更計画

### Phase 1: cache model と adapter を追加

追加候補:

- `src/kumc_agent/features/indexing/embedding_cache.py`

定義するもの:

- `IndexEmbeddingRecord`
  - `chunk_id`
  - `embedding_text_hash`
  - `provider`
  - `model`
  - `dimensions`
  - `vector`
  - `chunk_metadata_hash`
  - `source_kind`
  - `source_item_id`
  - `created_at`
- `IndexEmbeddingCache` protocol
  - `load(provider, model, dimensions) -> dict[tuple[chunk_id, embedding_text_hash], np.ndarray]`
  - `save(records)`
  - `compact(active_keys)`
- `FileIndexEmbeddingCache`

実装上の注意:

- cache には本文そのものを保存しない。
- vector shape が `dimensions` と一致しない record は無視する。
- JSONL は追記でよいが、一定件数または publish 成功後に compact して最新 record だけ残す。

### Phase 2: IndexingService に差分 embedding planner を入れる

`IndexingService.build()` の次の箇所を置き換える。

現状:

```python
dense_texts = [self._chunk_embedding_text_for_dense(chunk) for chunk in index_chunks]
embeddings = self._embedder.embed_documents(dense_texts)
```

変更後の概念:

```python
embedding_result = self._embedding_planner.embed_or_reuse(
    chunks=index_chunks,
    embedding_text_for_chunk=self._chunk_embedding_text_for_dense,
    provider=config.providers.embeddings.provider,
    model=config.providers.embeddings.model,
    dimensions=config.providers.embeddings.dimensions,
    force_reembed=full_rebuild,
)
embeddings = embedding_result.matrix
```

`embedding_result` には次を含める。

- `total_chunks`
- `embedded_chunks`
- `reused_chunks`
- `cache_misses`
- `cache_invalid`
- `provider`
- `model`
- `dimensions`

`IndexBuildResult.stage_results["embedding"]` にこの summary を入れる。payload 方針に合わせ、詳細診断は `metadata.stage_results.embedding` 配下に置く。

### Phase 3: dense artifact 生成は既存方式を維持する

- `FaissLikeIndex.build(chunks=index_chunks, embeddings=embeddings)` は当面そのまま使う。
- `dense_chunks.jsonl`、`dense_vectors.npy`、`dense_vectors.faiss` は staging に完全な状態で出す。
- `IndexQualitySmokeChecker` の既存検査はそのまま通る必要がある。

追加 artifact:

- `dense_embedding_manifest.jsonl`
  - `chunk_id`
  - `embedding_text_hash`
  - `provider`
  - `model`
  - `dimensions`
  - `source_kind`
  - `source_item_id`
  - `reused` は run 固有情報なので manifest ではなく run metadata に置く方が望ましい。

manifest は debugging と将来の rollback / cache 復旧用であり、検索 runtime の必須入力にはしない。

### Phase 4: config を追加する

`.env` ではなく `configs/main/indexing.yaml` と config schema に追加する。

追加候補:

```yaml
indexing:
  embedding_cache:
    enabled: true
    compact_after_publish: true
    force_reembed_on_full_rebuild: true
```

対応箇所:

- `src/kumc_agent/config/schema.py`
- `src/kumc_agent/config/load.py`
- `configs/main/indexing.yaml`
- 必要なら config loading tests

`.env` / `.env.example` には追加しない。これはトークンではなく挙動パラメータのため。

### Phase 5: runtime wiring

`build_runtime_context()` で `FileIndexEmbeddingCache` を作り、`IndexingService` へ注入する。

変更候補:

- `src/kumc_agent/runtime/container.py`
- `src/kumc_agent/features/indexing/service.py`

`IndexingService` の constructor に optional `embedding_cache` を追加し、未注入時は従来どおり全件 embed する。これにより unit test と legacy context の互換性を保つ。

### Phase 6: cache compact / prune

publish 成功後、次の active key だけを残す。

- 今回の `index_chunks` に存在する key
- 必要なら直前 snapshot manifest に存在する key

削除済み source の vector は検索 artifact からは消えるが、cache に残る可能性がある。cache は本文を持たないとはいえ vector は派生情報なので、`compact_after_publish` が true なら active key のみに prune する。

AutoIndexUpdateUsecase 側で publish 成功後に compact する案と、IndexingService build 完了時に compact する案がある。publish 失敗時の再利用性と rollback を考えると、初期実装では IndexingService が保存まで行い、AutoIndexUpdateUsecase が publish 成功後に compact を呼ぶ方が安全である。

### Phase 7: docs 更新

実装時に更新する文書:

- `docs/design/auto-index-update.md`
  - `9.3 embedding / sparse index` を「embedding 計算は差分化、公開 artifact は全体再生成」に更新する。
- `docs/runbooks/auto_index_update.md`
  - run metadata の `stage_results.embedding.embedded_chunks` / `reused_chunks` の見方を追加する。
- 必要なら `docs/explanations/` 配下に実装説明を追加する。

## 5. テスト計画

pytest は未導入前提なので `unittest` で追加する。

追加候補:

- `tests/unit/test_indexing_incremental_embedding.py`
- 既存 `tests/unit/test_auto_index_update.py` への metadata assertion 追加
- 既存 config loading tests への `embedding_cache` config 追加

検証項目:

1. cache が空の場合は全 chunk を `embed_documents()` に渡し、cache に保存する。
2. 2回目の build で chunk id / embedding text hash / model / dimensions が同じ場合、`embed_documents()` に渡す件数が 0 になる。
3. 1 chunk だけ text が変わった場合、その chunk だけが `embed_documents()` に渡され、最終 matrix の行順は `index_chunks` と一致する。
4. ACL metadata だけ変わり embedding text が同じ場合、vector は再利用され、`dense_chunks.jsonl` の metadata は新しい値になる。
5. model または dimensions が変わった場合、cache を使わず全件再埋め込みする。
6. `full_rebuild=True` では `force_reembed_on_full_rebuild=true` に従い全件再埋め込みする。
7. deleted / permission_lost chunk は最終 `dense_chunks.jsonl` と `dense_vectors.npy` に入らない。
8. cache の壊れた record、次元不一致 record、JSON decode 不能行は無視し、build は失敗しない。
9. `metadata.stage_results.embedding` に `embedded_chunks` / `reused_chunks` / `cache_invalid` が入る。
10. cache を無効化した場合は従来どおり全件 embed する。

実行コマンド例:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_indexing_incremental_embedding
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_auto_index_update tests.unit.test_config_loading
```

広めに確認する場合:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest discover tests/unit
```

## 6. リスクと対策

| リスク | 対策 |
| --- | --- |
| 古い vector を誤再利用する | `embedding_text_hash`、provider、model、dimensions を key に含める |
| chunk ID は同じだが embedding text が変わる | `chunk_id` だけで cache hit しない |
| summary chunk が非決定的に変わる | final embedding text から hash を作るため、summary text が変われば再埋め込みされる |
| Postgres `embeddings` table を流用して FK で失敗する | 初期実装では indexing 専用 file cache を使う |
| cache に secret 本文が残る | 本文は保存せず hash と vector のみ保存する |
| cache が肥大化する | publish 成功後 compact / prune を行う |
| full rebuild の意味が弱くなる | 初期実装では full rebuild 時に cache bypass する |
| FAISS artifact の部分更新まで同時にやって複雑化する | 初期実装では artifact は全体再生成のままにする |

## 7. 完了条件

- 通常の `index update` で、未変更 chunk は embedding API / model に渡されない。
- 生成される `dense_vectors.npy`、`dense_vectors.faiss`、`dense_chunks.jsonl` は従来どおり全 active chunk を含む。
- `IndexingRun.metadata.stage_results.embedding` で、実埋め込み件数と再利用件数を確認できる。
- `full_rebuild=True` では既定で全件再埋め込みできる。
- cache がない・壊れている・次元不一致の場合も、正しい full embed に fallback する。
- unit test で cache hit / miss / text change / ACL-only change / model change / deletion を検証する。
