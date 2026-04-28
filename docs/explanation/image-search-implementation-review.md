# 画像検索 実装調査

調査日: 2026-04-28

参照仕様:

- `docs/design/image-search.md`
- `docs/plan/image-search.md`
- 補助参照: `docs/design/kumc-agent.md` の「4. 画像検索」

## 結論

現状の画像検索は、初期実装だけではなく、専用 service、Asset 化、caption/OCR fallback、Dense 検索、画像特徴量風の類似検索、RRF、workflow 連携、payload sanitization まで実装されている。

ただし、仕様でいう「完全実装」には未達である。特に、Postgres repository の読込経路、`source_filter`/`limit` の統合入力連携、Google Drive 埋め込み画像の抽出、権限変更の差分検知、画像特徴量 vector の品質と設定反映、評価・テスト網羅性が不足している。

判定: **部分実装。運用可能な骨格はあるが、完全実装とは言えない。**

## 調査した主な実装

| 領域 | 実装 |
| --- | --- |
| domain model | `src/kumc_agent/domain/models/operations.py` の `Asset` |
| repository | `src/kumc_agent/infra/operations/repository.py` |
| image search feature | `src/kumc_agent/features/image_search/service.py` |
| workflow | `src/kumc_agent/features/workflow/service.py` |
| integrated input | `src/kumc_agent/usecases/integrated_input/entry.py` |
| indexing | `src/kumc_agent/features/indexing/service.py` |
| runtime wiring | `src/kumc_agent/runtime/container.py`, `src/kumc_agent/apps/workflow.py` |
| CLI/HTTP payload | `src/kumc_agent/cli.py`, `src/kumc_agent/frontends/http/app.py` |
| source loaders | `src/kumc_agent/infra/loaders/*_impl.py` |
| tests | `tests/unit/test_image_search.py` |

## 実装済みと判断できる点

| 仕様項目 | 状態 | 根拠 |
| --- | --- | --- |
| `Asset` に画像検索用 metadata を保存する | 実装済み | `Asset.metadata` があり、builder が `caption`, `ocr_text`, `surrounding_text`, `source_url`, `source_label`, `image_index`, `content_hash`, `feature_vector_ref`, `index_version` を保存している |
| Discord 添付画像の Asset 化 | 実装済み | raw `messages/**/*.jsonl` の `attachments` から画像候補を作る |
| Google Drive 画像ファイルの Asset 化 | 部分実装 | `raw/images/google_drive` の画像ファイルと sidecar metadata を読む |
| X 投稿画像の Asset 化 | 実装済み | `raw/x/posts.jsonl` の `x_media_urls` を読む |
| はてな/クラフターズコロニー記事画像の Asset 化 | 実装済み | markdown/html の画像参照を抽出する |
| caption 生成 | 部分実装 | Gemini captioner がある。未設定・失敗時は fallback |
| OCR | 部分実装 | local OCR extractor がある。未設定時は skip |
| Dense index | 実装済み | `image_text_vectors.npy` と `image_assets.jsonl` を作成し検索する |
| 画像特徴量検索 | 部分実装 | ローカル画像の色統計または hash vector を使う。専用モデル連携ではない |
| RRF | 実装済み | Dense と feature の rank list を RRF で統合する |
| 検索前・回答前の権限 filter | 部分実装 | `ImageSearchService` では pre/post filter あり。ただし fallback 経路に問題あり |
| workflow 連携 | 実装済み | `WorkflowService.image_search()` が専用 service を呼ぶ |
| CLI/HTTP payload sanitization | 部分実装 | work payload では OCR/周辺テキストを短縮し、内部パスを除外する |
| 再利用可否を断定しない | 実装済み | `text` と `detail_markdown` に候補提示のみである旨を含める |
| `image_usage_request` 削除 | 実装配下は実装済み | `src`, `tests`, `infrastructure` では該当 route/model/repository は見つからない |

## 仕様との差分

### 1. Postgres repository では画像検索が成立しない可能性が高い

`PostgresOperationsRepository.save_asset()` は Postgres へ保存するが、`list_assets()` と `get_asset()` は override されていない。そのため Postgres 構成では、保存後の読込が fallback JSONL 側を見に行く。

影響:

- `ImageAssetBuildService` が asset を Postgres に保存しても、同じ build 内の `get_asset()` / `list_assets()` がその asset を読めない。
- `target_index.build(searchable_assets)` が空になりうる。
- `ImageSearchService.search()` も repository から候補を読めず、0 件になりうる。

仕様では JSONL/Postgres の保存・再読込が対象なので、これは完全実装の阻害要因である。

### 2. `source_filter` と `limit` が workflow / integrated input / CLI / HTTP に通っていない

`ImageSearchRequest` には `limit` と `source_filter` があるが、`WorkRequest` には対応フィールドがない。`WorkflowService.image_search()` も query と access だけを渡している。

影響:

- 仕様上の検索入力である `source_filter` が、統合入力受付や `/work` 経由では使えない。
- HTTP `/work` payload や CLI `work` から source ごとの絞り込みができない。
- integrated input の routing decision に `source_filters` があっても画像検索 service へ渡らない。

### 3. `features.image_search.enabled` が実行制御に使われていない

config schema と `configs/main/features.yaml` には `features.image_search.enabled` があるが、runtime wiring や workflow routing で無効化判定に使われていない。

影響:

- 設定で無効化しても builder/service/route が動く可能性がある。
- 仕様上の設定項目としては未完了。

### 4. 権限変更の差分検知が不足している

`source_fingerprint` は `image_ref`, `content_hash`, `surrounding_text`, `caption`, `ocr_text` から作られているが、`access_scope`, `source_url`, `source_label`, source metadata は含まれていない。

影響:

- 画像本体や周辺テキストが変わらず、権限だけ変わった場合に既存 Asset が skip され、古い `access_scope` が残る可能性がある。
- 仕様の「削除済み・権限変更済み画像を検索対象から外す」に未達。

### 5. Google Drive の Docs/Slides 内埋め込み画像は完全には Asset 化されていない

実装は次を扱っている。

- Drive 上の単体画像ファイル: `raw/images/google_drive`
- Drive markdown 内の markdown/html 画像参照: `raw/docs/*.md`

一方で、Google Docs export の image placeholder は除去され、Slides/PPTX 抽出は主に text 抽出で、埋め込み画像を画像ファイルとして取り出して周辺テキスト付き Asset にする処理は確認できない。

影響:

- 仕様の「Google Drive 上の画像・スクリーンショット」「Slide/Docs 内の画像前後テキスト」には部分対応に留まる。

### 6. 画像特徴量 vector は仕様上の完全実装としては弱い

実装は、ローカル画像がある場合は 32x32 の色統計・ヒストグラム、ない場合は `content_hash` などから hash vector を作る。

不足:

- `configs/main/features.yaml` の `feature_model` が `ImageSearchConfig` や runtime wiring に反映されていない。
- `metadata.feature_vector_ref` は保存されるが、asset ごとの vector store 参照としては実体化されていない。
- vector 作成失敗時の `metadata.feature_status=failed` がない。
- hash vector fallback は「類似画像検索」としての意味的品質が限定的。

### 7. Dense 検索結果に `matched_fields` がない

仕様では Dense 検索結果に `asset_id`, `rank`, `score`, `matched_fields` を持つとしているが、実装の search metadata は `asset_id`, `rank`, `score`, `sources` であり、どの field が効いたかは出ない。

影響:

- OCR 由来で当たったのか、caption 由来で当たったのか、source label 由来で当たったのかを診断できない。
- 評価・デバッグ・説明可能性が不足する。

### 8. fallback 経路の権限 filter が不十分

`ImageSearchService` がある場合は権限 filter がある。しかし `WorkflowService.image_search()` は service 未設定時に `operations.list_assets(query=query)` を直接返す。

影響:

- service 未設定時に protected source の Asset が access check なしで返りうる。
- `tests/unit/test_design_gap_foundation.py` には、この fallback で asset が返ることを期待する古いテストが残っている。
- 仕様の「検索前・回答前 filter の両方で除外」に未達。

### 9. 重複画像の扱いが仕様より弱い

Asset ID には `content_hash` が入り、候補 dedupe もあるが、仕様にある `duplicate_group_id` の付与や RRF 後の duplicate group 単位の制限は未実装である。

影響:

- 同じ画像が複数媒体にある場合の統合表示・過剰表示抑制ができない。

### 10. 評価とテストが完全実装の範囲を覆っていない

確認できる画像検索テストは主に次の 3 件である。

- Discord attachment の Asset 化と検索
- Discord primary URL 失敗時の proxy URL fallback
- protected/public source の access filter

不足しているテスト:

- Google Drive 単体画像と Docs/Slides 埋め込み画像
- X/はてな/クラフターズコロニー画像の Asset 化
- caption 成功/失敗
- OCR 成功/失敗、OCR 文字列検索
- feature vector 作成失敗時の degraded
- RRF ranking と duplicate 制限
- `source_filter`, `limit`
- CLI/HTTP/Discord payload の image search 専用検証
- Postgres repository での保存・再読込
- `image_usage_request` が復活しないことを固定するテスト
- 画像検索 eval set

## Phase 別の完了度

| Phase | 判定 | コメント |
| --- | --- | --- |
| Phase 1: 利用申請フロー削除 | ほぼ完了 | 実装配下に route/model/repository は見当たらない。ただし古い上位設計文書には `asset_usage` 記述が残る |
| Phase 2: Asset/Repository 拡張 | 部分完了 | File repository は保存・検索できるが Postgres 読込が未実装 |
| Phase 3: 画像取得・Asset 化 | 部分完了 | 5 source の scanner はあるが、Drive 埋め込み画像と source 別テストが不足 |
| Phase 4: caption 生成 | 部分完了 | Gemini 実装と prompt はあるが、実運用品質・テストが不足 |
| Phase 5: OCR | 部分完了 | extractor はあるが、設定未導入時 skip。OCR 検索テストが不足 |
| Phase 6: 画像検索 index 作成 | 部分完了 | Dense/feature npy はあるが、`feature_model` 未反映、`matched_fields` なし |
| Phase 7: 権限確認 | 部分完了 | service では実装。fallback と権限変更差分検知に問題 |
| Phase 8: ImageSearchService | おおむね実装 | Search/RRF/output はある。duplicate/matched_fields/feature 品質が不足 |
| Phase 9: workflow・統合入力連携 | 部分完了 | route 連携はあるが `source_filter`/`limit` が落ちる |
| Phase 10: CLI・HTTP・Discord 出力 | 部分完了 | work payload sanitization はある。source/limit と Discord structured payload は不足 |
| Phase 11: 運用・自動更新 | 部分完了 | indexing build から builder は呼ばれる。ACL 変更・Postgres・rollback 検証が不足 |
| Phase 12: 評価 | 未完了 | 画像検索専用 eval set は確認できない |

## 仕様改善点

### 1. 「完全実装」の合格条件を source 別に明文化する

現行仕様は範囲が広い一方で、source ごとの最低合格条件が曖昧である。次を表にするべきである。

- raw artifact の入力形式
- 必須 metadata
- 対応する画像種別
- 権限 scope の決定方法
- 削除・権限変更の検知方法
- 必須テスト

### 2. Google Drive の対象範囲を分けて定義する

「Google Drive 画像・スクリーンショット」と「Docs/Slides 内埋め込み画像」は実装難度が違う。仕様上も次を分けるとよい。

- Drive 上の単体画像ファイル
- Google Docs export 内の画像
- Google Slides / PPTX 内の画像
- PDF ページ画像またはスクリーンショット

それぞれ、画像ファイル抽出方法と周辺テキストの取り方を定義する。

### 3. repository の backend contract を明確にする

JSONL と Postgres の両方を対象にするなら、`save_asset`, `get_asset`, `list_assets`, `save_indexing_run` の読み書き contract を backend 共通で必須にするべきである。

追加すべき受け入れテスト:

- File repository で Asset metadata を保存・再読込できる
- Postgres repository で Asset metadata を保存・再読込できる
- Postgres 構成で image index が空にならない

### 4. 権限 policy をサークル情報 RAG と共通 module に寄せる

仕様では「サークル情報 RAG と同じ権限設定」とあるが、実装は画像検索独自の `ImageAccessPolicy` である。共通 policy を使うか、差分を仕様に明記するべきである。

特に定義が必要な点:

- Drive/Discord の `guild_id` が空の場合の扱い
- admin DM の条件
- role/private visibility の扱い
- `permission_lost` と redaction policy の扱い

### 5. 差分検知 fingerprint に ACL と出典 metadata を含める

仕様に「権限変更済み画像を検索対象から外す」とあるため、fingerprint には少なくとも次を含めるべきである。

- `access_scope`
- `source_url`
- `source_label`
- source の revision / modified time
- deletion / permission status

### 6. 画像特徴量 vector の品質レベルを定義する

`local_hash` や色ヒストグラムを fallback として許すのか、完全実装にも含めるのかを明確にする必要がある。

仕様に追加したい項目:

- `feature_model` の反映先
- vector dimensions
- vector storage path と `feature_vector_ref` の形式
- 作成失敗時の `feature_status`
- 類似画像検索の最低評価指標
- hash fallback を degraded とみなすかどうか

### 7. 検索結果説明用 metadata を標準化する

Dense 検索結果に `matched_fields` が必要なら、field 別 embedding または field 別 scoring の設計を追加する必要がある。

例:

- `matched_fields`: `caption`, `ocr_text`, `surrounding_text`, `source_label`, `source_kind`
- `metadata.search_results[].components`
- external payload では score/rank を `metadata` 配下に限定

### 8. fallback の安全要件を明文化する

Dense index 未構築時の repository keyword fallback と、ImageSearchService 未設定時の workflow fallback は別物である。仕様では次を分けるべきである。

- index missing fallback: service 内で access filter 後に keyword 検索する
- service missing fallback: protected asset を返さない、または 503/設定不足として返す

### 9. output channel 別の payload contract を定義する

CLI、HTTP `/work`、HTTP `/ask`、Discord `/work`、integrated input で、どの structured fields を返すかが曖昧である。

追加すべき定義:

- `assets` を返す endpoint
- Discord で attachment にする条件
- OCR/周辺テキストの最大長
- 除外する metadata key
- source URL を表示してよい条件

### 10. eval set と最低スコアを仕様に入れる

完全実装の判断には eval が必要である。仕様には観点だけでなく、最低限の dataset と acceptance threshold を入れるべきである。

例:

- OCR-only query top-k hit rate
- caption semantic query top-k hit rate
- duplicate/similar image recall
- protected source leakage 0 件
- source attribution coverage
- 再利用可否を断定しない出力 100%

## 確認コマンド

実行済み:

```bash
rg -n "image_usage_request|AssetUsageRequest|asset_usage_requests|asset_usage" src tests infrastructure docs -g '!src/kumc_agent/infra/legacy/**'
python3 -m unittest tests.unit.test_image_search
```

結果:

- `image_usage_request` 系は実装配下では検出されなかった。古い設計文書と今回の plan/design には記述が残る。
- `tests.unit.test_image_search` は 3 tests OK。

