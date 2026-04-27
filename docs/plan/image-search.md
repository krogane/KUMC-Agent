# 画像検索 実装計画

## 1. 方針
`docs/design/kumc-agent.md` と `docs/design/image-search.md` に従い、画像検索を実装する。

実装では `src/kumc_agent/infra/legacy` を参照・依存しない。既存の共通部品は `domain.models.operations.Asset`、`infra.operations.repository`、`features.workflow`、`features.rag`、`infra.indexing`、`domain.models.retrieval.AccessContext` を優先して使う。現行実装と設計が矛盾する場合は `kumc-agent.md` を優先する。

画像検索は候補提示のみを行う。`image_usage_request`、`AssetUsageRequest`、`asset_usage_requests`、画像利用承認UI、外部公開可否判定は実装しない。

## 2. 完了条件
- Discord添付画像、Google Drive画像、X投稿画像、はてなブログ記事画像、クラフターズコロニー投稿画像をAsset化できる。
- 画像ごとに周辺テキスト、出典URL、投稿日時、投稿媒体、アクセス範囲を保存できる。
- 画像説明文を生成し、OCR結果とは分けて保存できる。
- OCR結果を保存し、検索indexへ投入できる。
- `Asset` に画像検索用metadataを保存・再読込できる。
- 画像説明文、OCR結果、周辺テキスト、投稿媒体からDense indexを作成できる。
- 画像特徴量vectorを保存し、Asset IDと紐づけられる。
- 入力クエリからDense検索と画像特徴量検索を実行できる。
- Dense検索と画像特徴量検索をRRFで統合できる。
- 質問者の権限に応じて閲覧不可画像を検索前・回答前に除外できる。
- 画像候補一覧、画像説明、出典を返せる。
- CLIや外部連携payloadの診断情報が `metadata` 配下に入る。
- 画像検索結果が再利用可否を断定しない。
- `image_usage_request` 経路がCLI、Discord、HTTP、workflow、repository、テストに残らない。
- 主要動作を既存テスト方式で検証できる。

## 3. 実装ステップ
### Phase 1: 利用申請フロー削除
1. `AssetUsageRequest` dataclassを削除する。
2. `WorkResponse.asset_usage_requests` を削除する。
3. `OperationsRepository` から `save_asset_usage_request()`、`get_asset_usage_request()`、`list_asset_usage_requests()` を削除する。
4. File/Postgres repositoryから `asset_usage_requests` 保存処理を削除する。
5. workflow dispatchから `image_usage_request` を削除する。
6. CLI `work --type` とDiscord `/work` の選択肢から `image_usage_request` を削除する。
7. approval typeから `asset_usage` を削除する。
8. HTTP/CLI payloadから `asset_usage_requests` を削除する。
9. migrationとテストから `asset_usage_requests` を削除する。

検証:
- `rg "image_usage_request|AssetUsageRequest|asset_usage_requests"` で実装配下に不要な参照が残らないこと。
- `work_type=image_usage_request` がサポートされないこと。
- 既存の `image_search` が画像候補だけを返すこと。

### Phase 2: AssetモデルとRepository拡張
1. `Asset.metadata` に画像検索用keyを保存する方針をテストで固定する。
2. `metadata.ocr_text`、`metadata.surrounding_text`、`metadata.source_url`、`metadata.source_label`、`metadata.image_index`、`metadata.content_hash`、`metadata.feature_vector_ref` を扱えるようにする。
3. `rights_status` と `contains_people` は互換のため残すが、検索結果の利用可否判定には使わない。
4. `list_assets()` は暫定互換として残し、専用検索service導入後はfallbackにする。
5. JSONL/Postgres payloadで既存Assetを読み込める後方互換を維持する。

検証:
- 画像検索metadata付きAssetを保存・再読込できること。
- 既存Assetを読み込めること。
- 診断情報がトップレベルへ出ないこと。

### Phase 3: 画像取得・Asset化
1. サークル情報RAGのデータ取得pipelineから画像ファイルと周辺テキストを受け取るadapterを追加する。
2. Discord添付画像をAsset化する。
3. Google Drive画像・スクリーンショットをAsset化する。
4. X投稿画像をAsset化する。
5. はてなブログ記事画像をAsset化する。
6. クラフターズコロニー投稿画像をAsset化する。
7. sourceごとに `source_kind`、`source_item_id`、`title`、`uri`、`captured_at`、`access_scope` を正規化する。
8. `content_hash` と `source_item_id` で重複排除する。

検証:
- sourceごとにAssetが作成されること。
- 同一source item内の複数画像を `image_index` で区別できること。
- 削除済み・権限変更済み画像が検索対象から外れること。

### Phase 4: 画像説明文生成
1. 画像説明文生成componentを追加する。
2. caption用プロンプトを `assets/prompts/` に追加する。
3. 画像の主対象、資料・画面の概要、検索に有用な視覚特徴を説明文に含める。
4. OCR結果を説明文に混ぜず、別フィールドとして扱う。
5. 生成失敗時は周辺テキストだけのfallback Assetを作成する。

検証:
- caption成功時に `description` または `metadata.caption` が保存されること。
- caption失敗時に `metadata.caption_status=fallback` が入ること。
- 生成結果にsecretや過剰な個人情報が保存されないこと。

### Phase 5: OCR
1. 画像OCR componentを追加する。
2. Google Drive PDF OCR周辺の既存実装を参考にするが、legacyには依存しない。
3. OCR結果を `metadata.ocr_text` に保存する。
4. OCR結果は検索indexに投入する。
5. 外部出力時にOCR結果を長さ制限し、secretをマスクする。

検証:
- OCR文字列で画像を検索できること。
- OCR失敗時もcaptionと周辺テキストで検索できること。
- 外部payloadに長大なOCR全文が出ないこと。

### Phase 6: 画像検索index作成
1. Assetから埋め込み用テキストを構築する関数を追加する。
2. 画像説明文、OCR結果、周辺テキスト、投稿媒体をFaissLikeIndexに投入する。
3. 画像特徴量vectorを作成するcomponentを追加する。
4. 画像特徴量vectorをAsset IDと紐づけて保存する。
5. `metadata.feature_vector_ref` にvector保存先参照を入れる。
6. 自動インデックス更新の対象に画像caption、OCR、feature vectorを追加する。

検証:
- caption、OCR、周辺テキスト、source_kindがDense検索対象に含まれること。
- feature vector作成失敗時にDense検索だけで継続できること。
- vector本体が外部payloadに出ないこと。

### Phase 7: 権限確認
1. 画像検索用AccessPolicyを追加する。
2. サークル情報RAGと同じsource_kind別権限設定を使う。
3. Google DriveとDiscordは指定Guild内チャットまたは指定admin user idのDMに限定する。
4. Hatena、X、Crafters Colonyは全ユーザーに許可する。
5. 検索前と回答前の両方でAccessScopeを確認する。
6. 権限外sourceの候補数や存在有無を返さない。

検証:
- 対象Guild内ではDrive/Discord画像が返ること。
- admin DMではDrive/Discord画像が返ること。
- 対象外Guild、非admin DMではDrive/Discord画像が返らないこと。
- public source画像は権限外ユーザーにも返ること。

### Phase 8: ImageSearchService追加
1. `src/kumc_agent/features/image_search/` を新設する。
2. `ImageSearchRequest` と `ImageSearchResult` を追加する。
3. `ImageSearchService.search()` を追加する。
4. 検索前にAccessPolicyとsource_filterを適用する。
5. Dense検索を実行する。
6. 画像特徴量検索を実行する。
7. Dense検索と画像特徴量検索をRRFで統合する。
8. 重複Assetを統合または制限する。
9. 回答前にAccessScopeとmetadataマスクを適用する。

検証:
- Dense検索だけで候補を返せること。
- 画像特徴量検索が利用可能な場合に類似画像を補えること。
- RRFで統合rankが安定すること。
- degraded時に `metadata.degraded=true` が入ること。

### Phase 9: workflow・統合入力受付連携
1. `features.workflow.service.image_search()` を専用 `ImageSearchService` 呼び出しへ置き換える。
2. 専用service未設定時は現行の `operations.list_assets(query=...)` fallbackを維持する。
3. `WorkResponse.assets` を主結果として維持する。
4. `detail_markdown` に候補説明と出典を含める。
5. 統合入力受付で画像検索intentを `image_search` へルーティングする。
6. route、degraded理由、検索スコア、trace idは `metadata` 配下に入れる。

検証:
- 既存workflow APIの `assets` が壊れないこと。
- `image_search` が候補数と候補一覧を返すこと。
- `image_usage_request` が復活しないこと。

### Phase 10: CLI・HTTP・Discord出力
1. CLIで `image_search` routeのpayloadを整える。
2. HTTP `/ask` または該当endpointで `assets` を返せるようにする。
3. Discordでは権限付きで画像候補を返す。
4. 長い結果はthreadまたはattachmentに分離する。
5. 検索スコア、内部rank、trace id、検索条件は `metadata` 配下に入れる。
6. 大きなOCR全文、周辺テキスト、secretを含む可能性があるmetadataを出力前に除外・マスクする。

検証:
- payloadトップレベルが安定フィールドだけであること。
- Discord応答で権限外sourceの存在有無が漏れないこと。
- `asset_usage_requests` がpayloadに含まれないこと。

### Phase 11: 運用・自動更新
1. 自動インデックス更新に画像caption、OCR、feature vectorを追加する。
2. sourceごとのcursor、checksum、revision、content_hashを使って差分検出する。
3. 差分がある画像だけcaption/OCR/vectorを再作成する。
4. 削除済み画像と権限変更画像を検索結果から除外する。
5. `indexing_runs` に処理件数、失敗件数、差分情報を保存する。
6. 重大失敗時のrollback手順をrunbookに追記する。

検証:
- 差分更新で全画像を不要に再生成しないこと。
- 削除済み画像が検索結果から除外されること。
- 失敗時に前回indexを維持できること。

### Phase 12: 評価
1. 画像検索専用のeval setを追加する。
2. 画像候補、OCR、類似画像、権限違反、出典表示を評価する。
3. 再利用可否を断定しないことを安全性評価に含める。
4. PRごとの小規模評価と定期full evalに組み込む。

検証:
- OCR文字列だけで該当画像に到達できること。
- caption由来の意味検索で該当画像に到達できること。
- 類似画像検索が重複・関連画像を補えること。
- 権限違反が0件であること。

## 4. 推奨ファイル変更範囲
想定される主な変更範囲は次の通り。

| 領域 | ファイル候補 |
| --- | --- |
| domain model | `src/kumc_agent/domain/models/operations.py` |
| workflow response | `src/kumc_agent/domain/models/workflow.py` |
| repository | `src/kumc_agent/infra/operations/repository.py` |
| image search feature | `src/kumc_agent/features/image_search/` 新規 |
| source adapters | `src/kumc_agent/infra/loaders/discord.py`、`src/kumc_agent/infra/loaders/google_drive.py`、`src/kumc_agent/infra/loaders/x.py`、`src/kumc_agent/infra/loaders/hatenablog.py`、`src/kumc_agent/infra/loaders/crafters_colony.py` |
| indexing | `src/kumc_agent/features/indexing/service.py`、`src/kumc_agent/infra/indexing/`、`src/kumc_agent/infra/retrieval/` |
| workflow | `src/kumc_agent/features/workflow/service.py` |
| entry routing | `src/kumc_agent/features/rag/components/entry_routing.py` または統合入力受付側 |
| CLI | `src/kumc_agent/cli.py` |
| HTTP | `src/kumc_agent/frontends/http/app.py` |
| Discord | `src/kumc_agent/frontends/discord/app.py` |
| config | `src/kumc_agent/config/schema.py`、`src/kumc_agent/config/load.py`、`configs/` |
| prompts | `assets/prompts/image_caption.md` 新規候補 |
| migrations | `infrastructure/migrations/011_ingestion_indexing_assets.sql` |
| tests | `tests/unit/test_image_search_*.py` |

`.env` または `.env.example` に設定項目を追加する場合は、必ず他方にも反映する。画像検索のパラメータやプロンプトは `.env` / `.env.example` ではなく `configs/` と `assets/prompts/` に置く。

## 5. リスクと対策
| リスク | 対策 |
| --- | --- |
| 権限外画像の存在有無が漏れる | 検索前と回答前にAccessScopeを確認し、件数にも含めない |
| OCRにsecretや個人情報が混入する | index投入前、保存前、外部出力前にマスクする |
| 画像説明文が誤認識を含む | 説明文は検索補助として扱い、出力では断定しすぎない |
| 画像特徴量index未構築で検索不能になる | Dense検索fallbackと `metadata.degraded` を用意する |
| 大きな画像・OCR本文でpayloadが肥大化する | metadata保存・外部出力の文字数制限を設ける |
| `image_usage_request` が再導入される | route、型、repository、payload、テストで削除状態を固定する |
| legacy依存が混入する | import検査または静的テストで `infra.legacy` 参照を禁止する |

## 6. テスト計画
pytestは未導入前提のため、既存方式に合わせて `unittest` で追加する。

追加候補:

- `tests/unit/test_image_asset_repository.py`
- `tests/unit/test_image_search_access.py`
- `tests/unit/test_image_search_ranking.py`
- `tests/unit/test_image_search_workflow.py`
- `tests/unit/test_image_search_payload.py`

主なテスト観点:

- Asset metadataの保存・再読込
- image_search workflowの正常系
- repository未設定時の応答
- Dense unavailable時のfallback
- feature vector unavailable時のfallback
- RRFランキング
- source_kind別権限フィルタ
- OCR/周辺テキストの外部出力マスク
- `image_usage_request`、`AssetUsageRequest`、`asset_usage_requests` が実装に残っていないこと

## 7. 実装順序
推奨順序は次の通り。

1. 利用申請フロー削除
2. Asset metadata保存方針の固定
3. 画像取得adapterとAsset化
4. caption生成
5. OCR
6. Dense index
7. 画像特徴量vector index
8. AccessPolicy
9. ImageSearchService
10. workflow/CLI/HTTP/Discord連携
11. 自動インデックス更新
12. 評価セット

この順序にすると、不要な利用申請経路を先に消した上で、保存、取得、index、検索、出力の各段階を小さく検証できる。
