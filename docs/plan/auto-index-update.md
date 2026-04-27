# 自動インデックス更新 実装計画

## 1. 方針
`docs/design/kumc-agent.md` と `docs/design/auto-index-update.md` に従い、自動インデックス更新を実装する。

実装では `src/kumc_agent/infra/legacy` を参照・依存しない。既存の共通部品は `usecases.indexing`、`features.indexing`、`features.ingestion`、`infra.connectors`、`infra.operations.repository`、`apps.worker`、`features.automation`、`domain.models.operations.IndexingRun` を優先して使う。現行実装と設計が矛盾する場合は `kumc-agent.md` を優先する。

初期実装では、差分検出で処理対象を絞りつつ、公開Dense/Sparse indexは全体再構築してよい。部分upsert/deleteは後続改善とする。

## 2. 完了条件
- `scheduler.auto_index_enabled/time/weekdays` に従って日次更新を起動できる。
- CLI、worker、automation、admin手動経路から同じ自動更新usecaseを呼べる。
- 二重起動をlockで防止し、skip理由を `IndexingRun` に保存できる。
- Google Drive、Discord、Hatena、X、Crafters Colony、Notion、Minecraft Wikiの差分を検出できる。
- 画像 caption / OCR / feature vector、`member_profiles`、Task / Event 正本を更新対象に含められる。
- cursor、checksum、revision、ACL hashで新規・更新・削除・権限変更を判定できる。
- raw snapshotを保存し、chunking、embedding、Sparse index、資料名indexを更新できる。
- 削除済み、隔離済み、権限喪失データが検索結果から除外される。
- `indexing_runs` に件数、差分、再index対象、品質結果、rollback情報を保存できる。
- 更新後のsmoke checkを実行し、重大失敗時は直前indexを維持または復元できる。
- 失敗時にadmin通知用payloadを作れる。
- CLIや外部連携payloadの診断情報が `metadata` 配下に入る。
- 主要動作を既存テスト方式で検証できる。

## 3. 実装ステップ
### Phase 1: 実行モデルと設定整理
1. `configs/main/scheduler.yaml` に自動更新の運用パラメータを追加する。
2. `SchedulerSection` と `load_runtime_config()` に追加設定を反映する。
3. `KUMC_AUTO_INDEX_*` 既存env bindingとの整合を確認する。
4. `AutoIndexUpdateRequest` / `AutoIndexUpdateResult` を追加する。
5. `trigger`, `source_filter`, `force`, `full_rebuild`, `quality_check_enabled` をrequestで扱う。

検証:
- scheduler設定を読み込めること。
- `.env` / `.env.example` にパラメータやプロンプトを追加しないこと。
- 既存 `index build/update` が壊れないこと。

### Phase 2: IndexingRun接続
1. `IndexingRunRepository` 相当の薄いportを追加するか、既存 `OperationsRepository.save_indexing_run()` を直接利用する。
2. run開始、source別進捗、成功、失敗、skip、rollbackを保存するhelperを追加する。
3. `IndexBuildResult` を拡張し、source別件数、stage結果、snapshot情報を `metadata` で返せるようにする。
4. 既存CLI payloadではトップレベルを安定フィールドに限定し、詳細は `metadata` に入れる。

検証:
- 成功/失敗/skippedの `IndexingRun` がJSONL/Postgresに保存されること。
- 大きな本文やsecretらしき値がmetadataに保存されないこと。

### Phase 3: 排他lock
1. `infra/jobs` または `features/indexing` に `IndexingLock` を追加する。
2. Postgres、Redis、File fallbackの順でlockを取得できるようにする。
3. lock取得不可時はrunを `skipped` とし、理由を保存する。
4. lock TTLと最大実行時間を設定化する。

検証:
- 二重起動で片方だけが実行されること。
- lock取得不可のpayloadが `metadata.reason` に入ること。
- 異常終了後にTTLで復旧できること。

### Phase 4: source差分検出
1. `SourceChangeDetector` を追加し、`source_items`、`sync_cursors`、connector結果を比較する。
2. cursorを持つsourceでは `poll_changes(cursor)` を優先する。
3. cursor未対応sourceはbackfillとchecksum比較で差分を出す。
4. `checksum`、`metadata.revision`、`access_scope.source_acl_hash` を比較する。
5. `SourceDeleteItem` と権限喪失を `deleted` / `permission_lost` として扱う。
6. 差分結果を `IndexingRun.metadata.source_results` に保存する。

検証:
- checksum一致でskipされること。
- revision変更でchangedになること。
- ACL hash変更でchangedになること。
- 削除通知でdeletedになること。

### Phase 5: Ingestion統合
1. `IngestionService.backfill_many()` の結果を自動更新runへ接続する。
2. `RawSnapshotStore` へ保存されたraw object keyを再index対象へ紐づける。
3. `mark_deleted()` 後に該当chunkが検索除外されることを保証する。
4. `source_kind` 名を本文RAG側と統一する。
5. Minecraft Wiki connectorを自動更新対象に含める。

検証:
- source別に `seen/changed/skipped/deleted` が集計されること。
- `index_status=deleted` のchunkが検索されないこと。
- Minecraft Wikiのrevision/checksum差分を検出できること。

### Phase 6: index構築pipeline整理
1. `IndexingService.update()` を `build()` の単純委譲から自動更新向けの入口へ拡張する。
2. `stage_selection`、`full_rebuild`、`allow_cancel` は維持する。
3. 差分対象sourceがない場合はindex再構築をskipできるようにする。
4. 初期実装では差分がある場合に全体Dense/Sparse indexをstagingへ再構築する。
5. 資料名index、keyword index、BM25、FaissLikeIndexを同じsnapshotに含める。

検証:
- 差分なしrunで不要なindex上書きをしないこと。
- 差分ありrunでDense/Sparse/keyword indexが更新されること。
- `--stage` 指定が引き続き効くこと。

### Phase 7: 画像index連携
1. `ImageAssetBuildService.build_from_raw_sources()` を自動更新stageに位置付ける。
2. 画像ごとの `content_hash`、`source_item_id`、`image_index` を差分判定に使う。
3. 差分がある画像だけcaption、OCR、feature vectorを再作成する方針を固定する。
4. 削除済み画像は `metadata.index_status=deleted` として検索除外する。
5. 画像stage結果を `IndexingRun.metadata.stage_results.image` に保存する。

検証:
- 画像差分なしでcaption/OCR/vectorを不要に再生成しないこと。
- 削除済み画像が画像検索に出ないこと。
- feature vector失敗時にDense検索fallbackで継続できること。

### Phase 8: member_profiles連携
1. `member_profiles_rebuild` worker jobを自動更新stageへ接続する。
2. Guild member追加、退会、role変更、display name変更を差分として扱う。
3. 関連RAG差分からprofile再生成候補を推定するhookを追加する。
4. 退会・検索対象外profileは検索indexから除外する。
5. 件数と対象profile IDを `IndexingRun.metadata` に保存する。

検証:
- role/display name変更が再生成対象になること。
- 退会ユーザーが検索結果から除外されること。
- 差分更新で全profileを不要に再生成しないこと。

### Phase 9: Task / Event 正本index
1. workflow repositoryからTask/Event正本を検索用documentへ変換するadapterを追加する。
2. 正本の作成、更新、削除、status変更を差分判定する。
3. 承認待ち候補は検索indexに含めない。
4. Task/Event検索用chunkまたは専用indexを作成する。
5. 論理削除済み正本を検索除外する。

検証:
- 承認済みTask/Eventだけが検索対象になること。
- 削除済みTask/Eventが返らないこと。
- 変更後のタイトル/日時/statusが検索結果に反映されること。

### Phase 10: atomic publishとsnapshot
1. `data/index/staging/{run_id}` に成果物を作る経路を追加する。
2. `data/index/current` と `data/index/previous` のpointerまたはsymlink方針を決める。
3. 既存 `data/index` 直接参照との互換adapterを追加する。
4. publish成功後にcurrent pointerを切り替える。
5. 古いsnapshotを設定数だけ残して整理する。

検証:
- 更新中にcurrent indexが壊れないこと。
- publish後に検索が新snapshotを読むこと。
- 古いsnapshot整理でcurrent/previousを消さないこと。

### Phase 11: 品質smoke check
1. `IndexQualitySmokeChecker` を追加する。
2. Dense/Sparse/keyword/image/member/task/event indexのロード可否を確認する。
3. 代表クエリを設定から読み、1件以上返ることを確認する。
4. chunk数の前回比下限を確認する。
5. 権限外source、deleted/quarantined/permission_lost混入を検査する。
6. 結果を `IndexingRun.metadata.quality_check` に保存する。

検証:
- indexロード失敗で重大失敗になること。
- chunk急減で重大失敗になること。
- 権限違反で重大失敗になること。
- smoke query結果がmetadataに保存されること。

### Phase 12: rollback
1. 品質重大失敗時にcurrentを切り替えない。
2. publish後失敗に備え、previousへ戻すrollback処理を追加する。
3. rollback結果を `IndexingRun.status` と `metadata.rollback` に保存する。
4. rollback失敗時も旧snapshotの場所と手動復旧手順をmetadataへ残す。

検証:
- 品質失敗時に旧indexが維持されること。
- publish後rollbackでpreviousがcurrentに戻ること。
- rollback情報が保存されること。

### Phase 13: worker / automation / CLI統合
1. `apps.worker` に `auto_index_update` job typeを追加する。
2. `features.automation._default_rules()` に `auto_index_daily` ruleを追加する。
3. `scheduler.auto_index_enabled/time/weekdays` を使う実行可否判定を追加する。
4. `kumc-agent admin --action sync/reindex` と `index update` を可能な範囲で同じusecaseへ寄せる。
5. CLI/worker payloadの診断情報を `metadata` 配下に整理する。

検証:
- worker jobで自動更新が1回実行できること。
- schedule条件外ではskipされること。
- automation dry_runで副作用なしに予定run内容を確認できること。
- payload schema方針に反しないこと。

### Phase 14: 通知
1. admin通知用の `IndexingNotification` payloadを作る。
2. 失敗、degraded、rollback時に通知対象を作成する。
3. Discord admin DMまたは管理チャンネルへの送信adapterを追加する。
4. 送信失敗はindex更新の成否に影響させず、metadataに保存する。

検証:
- 失敗時に通知payloadが作られること。
- 通知本文にsecretや大きな本文断片が含まれないこと。
- 通知失敗でもrollback処理が継続すること。

### Phase 15: docs / runbook
1. `docs/explanation/cli.md` に `worker --job-type auto_index_update` と自動更新運用を追記する。
2. rollback手順を `docs/runbooks/` に追加する。
3. `docs/design/circle-info-rag.md`、画像、メンバー、Task/Event関連docsと参照関係を整える。
4. 設定追加がある場合は `configs/` の説明を更新する。

検証:
- 手動実行、定期実行、rollbackの手順がdocsで追えること。
- `.env` / `.env.example` の片側だけに変更がないこと。

## 4. 推奨ファイル変更範囲
想定される主な変更範囲は次の通り。

| 領域 | ファイル候補 |
| --- | --- |
| usecase | `src/kumc_agent/usecases/indexing/auto_update.py` 新規、`build.py`、`update.py` |
| indexing feature | `src/kumc_agent/features/indexing/service.py`、`src/kumc_agent/features/indexing/quality.py` 新規、`snapshot.py` 新規 |
| ingestion | `src/kumc_agent/features/ingestion/service.py`、`src/kumc_agent/infra/ingestion/repository.py` |
| connectors | `src/kumc_agent/infra/connectors/`、`src/kumc_agent/infra/loaders/` |
| operations | `src/kumc_agent/domain/models/operations.py`、`src/kumc_agent/infra/operations/repository.py` |
| jobs/worker | `src/kumc_agent/apps/worker/app.py`、`src/kumc_agent/infra/jobs/lifecycle.py` |
| automation | `src/kumc_agent/features/automation/service.py` |
| runtime | `src/kumc_agent/runtime/container.py`、`src/kumc_agent/runtime/context.py` |
| config | `src/kumc_agent/config/schema.py`、`src/kumc_agent/config/load.py`、`src/kumc_agent/config/env_map.py`、`configs/main/scheduler.yaml` |
| CLI | `src/kumc_agent/cli.py` |
| retrieval | `src/kumc_agent/infra/retrieval/faiss.py`、`src/kumc_agent/infra/retrieval/sudachi_bm25.py` 必要に応じて |
| image/member/workflow | `src/kumc_agent/features/image_search/`、`src/kumc_agent/features/member_search/`、`src/kumc_agent/features/workflow/` |
| migrations | `infrastructure/migrations/011_ingestion_indexing_assets.sql`、必要なら新規migration |
| docs | `docs/explanation/cli.md`、`docs/runbooks/auto_index_update.md` 新規 |
| tests | `tests/unit/test_auto_index_*.py`、`tests/integration/test_chat_index_eval.py` |

`.env` または `.env.example` に設定項目を追加する場合は、必ず他方にも反映する。自動更新のパラメータは `.env` / `.env.example` ではなく `configs/` に置く。

## 5. リスクと対策
| リスク | 対策 |
| --- | --- |
| index更新中に検索が壊れた成果物を読む | stagingで構築し、品質確認後にatomic publishする |
| 削除済み・権限喪失データが返る | `index_status` とAccessScopeを検索前/回答前に検査し、smoke checkに含める |
| source差分検出の取りこぼし | cursorだけに依存せずchecksum/revision/ACL hashを併用する |
| 全体再構築が重い | 初期は全体再構築、後続でvector store upsert/deleteへ移行する |
| 画像caption/OCRが高コスト | `content_hash` で差分がある画像だけ再生成する |
| member_profilesが過剰再生成される | Guild member差分と関連chunk差分から対象を絞る |
| 品質checkの誤検知で更新が止まる | 閾値を設定化し、degraded許容と重大失敗を分ける |
| rollbackとsource stateが不整合になる | source stateと公開index snapshotを分離し、次回再構築可能にする |
| payloadに内部情報が出る | 診断情報は `metadata` 配下、本文/context/secretは除外・マスクする |
| legacy依存が混入する | import検査または静的テストで `infra.legacy` 参照を禁止する |

## 6. テスト計画
pytestは未導入前提のため、既存方式に合わせて `unittest` で追加する。

追加候補:

- `tests/unit/test_auto_index_scheduler.py`
- `tests/unit/test_auto_index_lock.py`
- `tests/unit/test_auto_index_change_detection.py`
- `tests/unit/test_auto_index_indexing_run.py`
- `tests/unit/test_auto_index_snapshot.py`
- `tests/unit/test_auto_index_quality.py`
- `tests/unit/test_auto_index_rollback.py`
- `tests/unit/test_auto_index_payload.py`
- `tests/unit/test_auto_index_no_legacy_import.py`
- `tests/integration/test_auto_index_update.py`

優先度の高い検証項目:

- schedulerの実行可否判定
- 二重起動skip
- checksum/revision/ACL hash差分
- deleted / permission_lostの検索除外
- `IndexingRun` の成功/失敗/skipped/rollback保存
- staging/current/previous snapshot切替
- 品質smoke check失敗時の旧index維持
- worker `auto_index_update` job
- automation `auto_index_daily` dry_run
- CLI payloadのmetadata方針
- `infra.legacy` 非依存
