# Docs取得データ品質改善 実装計画

## 1. 目的

`data/raw/docs` に保存された Google Drive 由来のDocs系取得データを、RAG・画像検索・資料検索のraw sourceとして安定して使える品質にする。

現状は `.md` 本文と `.meta.json` の対応は揃っているが、「取得できたMarkdown」をそのままindexへ流すには、画像主体資料、PDF/PPTXの抽出粒度、metadata不足、保存先契約のずれが残っている。以下では、2026-04-29時点のローカル `data/raw/docs` を調査した結果と、それに基づく改善実装計画をまとめる。

## 2. 調査対象

- `data/raw/docs/*.md`
- `data/raw/docs/*.md.meta.json`
- 関連画像artifact: `data/raw/images/google_drive/*`
- 取得実装: `src/kumc_agent/infra/loaders/google_drive.py`
- Drive取得詳細: `src/kumc_agent/infra/loaders/google_drive_impl.py`
- chunk化実装: `src/kumc_agent/infra/indexing/chunking.py`
- index読み込み実装: `src/kumc_agent/features/indexing/service.py`
- 画像検索読み込み実装: `src/kumc_agent/features/image_search/service.py`
- 日付推定実装: `src/kumc_agent/infra/indexing/date_metadata.py`

調査コマンド:

```bash
rg --files data/raw/docs
rg -n "data/raw/docs|raw/docs" .
rg -n "drive_file|GoogleDrive|google drive|docs" src/kumc_agent configs docs/design docs/plans docs/explanations
PYTHONPATH=src /usr/bin/python3 - <<'PY'
# data/raw/docs の本文サイズ、metadata、MIME、source_date推定、短文資料を集計
PY
```

## 3. 実測結果

### 3.1 保存状況

| 項目 | 結果 |
| --- | ---: |
| Markdown本文 | 242 |
| `.meta.json` | 242 |
| metadata欠落 | 0 |
| orphan metadata | 0 |
| JSON parse失敗 | 0 |
| 総サイズ | 1,234,969 bytes |
| 中央値 | 2,269 bytes |
| 100 bytes未満 | 25 |
| 500 bytes未満 | 58 |
| 1,000 bytes未満 | 86 |
| 非空白文字200字未満 | 60 |
| 文字化け置換文字を含むファイル | 0 |
| Markdown/HTML画像参照を含む本文 | 0 |
| 完全重複本文グループ | 1グループ / 2ファイル |

MIME type の内訳:

| MIME type | 件数 |
| --- | ---: |
| `application/pdf` | 74 |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | 67 |
| `application/vnd.google-apps.document` | 63 |
| `application/vnd.openxmlformats-officedocument.presentationml.presentation` | 37 |
| `application/vnd.google-apps.presentation` | 1 |

短文率はPDF/PPTXで特に高い。

| MIME type | 件数 | 500 bytes未満 | 1,000 bytes未満 |
| --- | ---: | ---: | ---: |
| `application/pdf` | 74 | 29 | 40 |
| `application/vnd.openxmlformats-officedocument.presentationml.presentation` | 37 | 16 | 21 |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | 67 | 8 | 10 |
| `application/vnd.google-apps.document` | 63 | 5 | 14 |
| `application/vnd.google-apps.presentation` | 1 | 0 | 1 |

### 3.2 metadataの現状

全242件の `.meta.json` は次の7キーのみを持つ。

- `drive_file_id`
- `drive_file_name`
- `drive_name`
- `drive_path`
- `drive_mime_type`
- `drive_modified_time`
- `drive_url`

一方、raw sidecarには次のような品質管理・検索制御用metadataがない。

- `source_date`
- `updated_at`
- `checksum`
- `content_sha256`
- `access_scope`
- `visibility`
- `index_status`
- `extraction_method`
- `extraction_status`
- `text_bytes`
- `nonempty_characters`
- `page_count`
- `slide_count`
- `ocr_page_count`
- `embedded_image_count`
- `quality_flags`

`source_date` はchunk化・index側でDriveファイル名またはDriveパスから推定される。今回のデータでは242件中234件が何らかの日付に推定できたが、130件が `2025/11/01`、40件が `2024/11/01` のように月初へ丸められており、イベント当日や資料作成日としては粗い。8件は `不明` になった。

### 3.3 画像artifactの現状

`data/raw/images/google_drive` には496件の画像ファイルと496件のmetadataがある。このうち296件はPPTX内の `ppt/media/*` から抽出された画像である。

ただし `data/raw/docs/*.md` 本文にはMarkdown/HTML画像参照が0件で、Docs本文と画像artifactの対応はsidecar metadataとファイル名から推測する形になっている。RAG本文chunkから「このスライドの画像」「このページの図」を直接辿るmetadataはない。

## 4. 現行実装の挙動

### 4.1 保存先

`GoogleDriveLoader` は `config.app.ingestion_dir / "docs"` と `config.app.ingestion_dir / "sheets"` に保存する。現行設定では `configs/main/app.yaml` の `app.ingestion_dir` が `data/ingestion` のため、現在の実装上の正規保存先は `data/ingestion/docs` である。

今回調査した `data/raw/docs` は、実データとして存在しているが、現行のindex buildが直接参照する標準パスではない。改善実装は `config.app.ingestion_dir` 配下へ適用しつつ、監査コマンドでは `--raw-dir data/raw/docs` のような任意パスも検査できる必要がある。

### 4.2 ファイル形式ごとの抽出

`src/kumc_agent/infra/loaders/google_drive_impl.py` の現在の抽出は次の通り。

- Google Docs: Drive export APIで `text/markdown` として取得し、Drive画像placeholderを削除する。
- Google Slides: PPTX exportを試し、失敗時は `text/plain` にfallbackする。
- DOCX: `word/document.xml`、header、footerから段落テキストを抽出する。
- PPTX: `ppt/slides/slide*.xml` の `a:t` だけを `## Slide N` に並べる。画像は別途 `data/.../images/google_drive` に抽出する。
- PDF: PyMuPDFの `page.get_text("text")` を使い、空ページだけOCRにfallbackする。

抽出後に `text.strip()` が空なら保存されないが、1文字だけのDOCXやページ番号だけのPDF/PPTXは保存される。

### 4.3 chunk化

Docsは通常のMarkdownテキストとして `recursive_chunk_dir()` に渡される。chunk metadataはDrive sidecarから `drive_file_id`、`drive_file_name`、`drive_mime_type`、`drive_file_path` などを構築するが、ページ、スライド、表、画像、OCR、品質状態の粒度は持たない。

## 5. データから見えた問題点

### 5.1 保存先契約が `data/raw/docs` と `data/ingestion/docs` で分裂している

調査対象の実データは `data/raw/docs` にあるが、現行 `GoogleDriveLoader` と `IndexingService` は `config.app.ingestion_dir`、つまり既定では `data/ingestion/docs` を正規入力として扱う。

影響:

- `data/raw/docs` を手元で調査しても、現行indexが同じデータを読んでいるとは限らない。
- raw監査、ingestion repository、index更新、rollbackのどれが正本か判断しづらい。
- 過去取得データを再監査・移行する導線がない。

### 5.2 画像主体のPDF/PPTXが短文Markdownとして保存される

PDFは74件中40件、PPTXは37件中21件が1,000 bytes未満だった。例として、次のような本文が存在した。

- `Pop.docx`: `a` のみ
- `40-41 レッドストーン回路講座.pdf`: `## Page 1` / `40` / `## Page 2` / `41` のみ
- `回路講座.pptx`: `## Slide 1` / `40` / `## Slide 2` / `41` のみ

影響:

- ページ番号や単語だけのchunkがDense/Sparse indexに入り、検索ノイズになる。
- 見た目では情報があるポスター・掲示・スライドでも、本文indexでは内容をほぼ検索できない。
- 画像検索側に画像artifactがあっても、RAG本文側のcitationと結びつかない。

### 5.3 抽出が構造を保持していない

DOCXは段落テキストのみ、PPTXはテキストノードのみ、PDFはページ本文のみを保存する。表、箇条書き階層、図表、脚注、ページ内位置、スライド内の画像と周辺文脈はraw本文に残らない。

影響:

- 表やマニュアルの「行・列・見出し」の関係を検索結果で説明しづらい。
- スライド資料ではページ単位の視覚情報とテキスト情報が分断される。
- PDFのOCR対象が「テキスト抽出が完全に空のページ」に限られるため、ページ番号だけ抽出できた画像主体ページはOCRされない。

### 5.4 metadataがファイル単位に偏っている

現在のmetadataはDriveファイル単位の追跡には十分だが、index投入可否や品質監査に必要な情報を持たない。

影響:

- 「短すぎる」「ページ番号だけ」「画像主体」「OCR未実施」「抽出失敗に近い」などを機械的に判定できない。
- `redaction_policy=deny`、`index_status=quarantined`、`permission_lost` のような既存アクセスフィルタ用metadataをraw段階で付けられない。
- Drive ACL、共有範囲、削除・権限変更の検知結果をchunk側へ渡せない。

### 5.5 重複・派生資料をまとめる情報がない

今回のデータには `KU匠vol2.pdf` が完全同一本文で2ファイル存在した。また、同じ掲示物や資料がPDF/PPTX/DOCXの別形式で存在する例も多い。

影響:

- 同じ内容のchunkが複数hitし、検索結果の多様性が下がる。
- PDF版とPPTX版のどちらをcitationに出すべきか判断できない。
- 「編集元」「配布用PDF」「画像化された掲示」の関係を後から追跡しづらい。

### 5.6 品質ゲートがない

保存時は `text.strip()` が空かどうかだけで判定している。chunk化・index publish前にも、Docs rawに特化した品質監査はない。

影響:

- 短文・低情報量・画像主体のrawセットでもindex publishへ進める。
- 取得ライブラリ変更やDrive API fallbackにより本文品質が下がっても検知しづらい。
- 問題発生時に、取得失敗、抽出品質、chunk化設定、検索ロジックのどこが原因か切り分けづらい。

## 6. 改善方針

### 方針A: raw本文を証跡として残し、検索用正規化artifactを追加する

既存の `.md` と `.meta.json` は証跡として残す。検索用にはページ・スライド・段落・表・画像参照を持つ正規化artifactを追加する。

候補:

- `data/processed/docs/*.jsonl`
- または `config.app.ingestion_dir / "docs_normalized"` 相当のindexing中間成果物
- ingestion repository利用時は `source_items` / `documents` / `chunks` に同等の構造metadataを保存する

### 方針B: Docs raw監査を独立コマンド化する

Minecraft Wiki監査と同じ考え方で、Docs rawも「件数」ではなく「検索根拠として使える本文か」を監査する。

最低限出す指標:

- 本文件数、metadata件数、欠落・orphan・JSON parse失敗
- MIME type別件数
- bytes、非空白文字数、短文率
- page/slide見出し数
- OCR実施・未実施ページ数
- 画像artifact件数と本文への紐付け状況
- metadata必須項目の欠落
- `source_date` 推定結果と `不明` 件数
- content hash重複
- `index_status=quarantined` 候補

### 方針C: metadata schemaを拡張する

Drive由来metadataに、品質・構造・アクセス制御用のmetadataを追加する。

最低限追加するmetadata:

| metadata | 用途 |
| --- | --- |
| `source_date` | recency補正と質問時の時系列判断 |
| `updated_at` | 取得・更新時刻 |
| `checksum` / `content_sha256` | 差分検知・重複検知 |
| `extraction_method` | `google_docs_markdown`, `pdf_text`, `pdf_ocr`, `pptx_xml`, `docx_xml` など |
| `extraction_status` | `ok`, `low_text`, `ocr_needed`, `failed`, `partial` など |
| `text_bytes` | 品質監査 |
| `nonempty_characters` | 品質監査 |
| `page_count` / `slide_count` | citation粒度 |
| `ocr_page_count` | OCR実施状況 |
| `embedded_image_count` | 画像主体資料の判定 |
| `quality_flags` | `too_short`, `page_number_only`, `image_heavy`, `duplicate_content` など |
| `index_status` | `active`, `quarantined`, `deleted`, `permission_lost` |
| `access_scope` | Drive ACLまたは運用上の公開範囲 |

CLIや外部連携payloadに診断を出す場合、品質診断はトップレベルではなく `metadata` 配下へ置く。

### 方針D: 形式別の正規化を実装する

PDF、PPTX、DOCX、Google Docsを同じMarkdown文字列へ潰すのではなく、検索用には構造化レコードへ変換する。

推奨レコード単位:

- PDF: page単位。`page_number`、抽出本文、OCR本文、ページ画像参照、低情報量フラグ。
- PPTX/Google Slides: slide単位。`slide_number`、本文、speaker notes、embedded image refs、周辺文脈。
- DOCX/Google Docs: heading/paragraph/table単位。表は行・列見出しを保持したテキストへ変換。
- 画像主体資料: 本文chunkへ無理に入れず、画像検索artifactと相互リンクする。

### 方針E: 低情報量資料を隔離する

本文が極端に短い資料は、無条件にRAG本文indexへ入れない。用途に応じて扱いを分ける。

- ページ番号だけ、1文字だけ、タイトルだけ: `index_status=quarantined` または `quality_flags=["too_short"]`
- ポスター・掲示・写真中心: 画像検索へ優先投入し、RAG本文には資料名・Driveパス・OCR結果だけを入れる
- 同一本文の重複: canonical sourceへ集約し、別形式はvariant metadataとして保持する

### 方針F: 現行 `data/ingestion` と過去 `data/raw` を接続する

改善対象の実装は `config.app.ingestion_dir` を正本にする。一方、既存の `data/raw/docs` は過去取得データの監査・移行対象として扱えるようにする。

実装上は次を用意する。

- 監査コマンドの `--raw-dir`
- `data/raw/docs` から `data/ingestion/docs` またはingestion repositoryへ移行するdry-run
- stale raw検知
- sync_deleted時のmanifest出力

## 7. 実装計画

### Phase 1: Docs raw監査コマンドを追加する

実装候補:

- `src/kumc_agent/usecases/ingestion/google_drive_docs_audit.py`
- CLI: `PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest audit --source docs --raw-dir data/raw/docs`
- 出力: 人間向けMarkdownまたはテーブル、機械可読JSON

受け入れ条件:

- `data/raw/docs` と `config.app.ingestion_dir / "docs"` の両方を検査できる。
- 本文・metadata対応、MIME別件数、短文率、重複、画像artifact連携、source_date推定結果を出せる。
- 監査結果を `metadata` 配下に入れたCLI JSON payloadとして返せる。

### Phase 2: metadata拡張と品質フラグを実装する

実装候補:

- `src/kumc_agent/infra/loaders/google_drive_impl.py`
- `src/kumc_agent/infra/indexing/chunking.py`
- 必要に応じて `src/kumc_agent/config/schema.py`
- 閾値は `.env` ではなく `configs/main/*.yaml` に置く。

実装内容:

- `_drive_metadata_payload()` に品質・構造metadataを追加する。
- 保存本文の `content_sha256`、`text_bytes`、`nonempty_characters` を記録する。
- PDF page数、PPTX slide数、PPTX embedded image数、OCR実施ページ数を記録する。
- `too_short`、`page_number_only`、`image_heavy`、`duplicate_content` などの `quality_flags` を付ける。
- `index_status` の初期値を `active` にし、品質ゲートで `quarantined` にできるようにする。

受け入れ条件:

- 既存7キーは維持される。
- chunk metadataにも拡張metadataが伝播する。
- `redaction_policy` / `index_status` による既存アクセスフィルタと矛盾しない。

### Phase 3: 形式別正規化artifactを追加する

実装候補:

- `src/kumc_agent/infra/loaders/google_drive_normalizer.py`
- `src/kumc_agent/infra/indexing/docs_normalized.py`
- 保存先: `data/processed/docs` または `config.app.ingestion_dir / "docs_normalized"`

実装内容:

- PDFをpage単位JSONLへ変換する。
- PPTX/Google Slidesをslide単位JSONLへ変換し、embedded image refsを保持する。
- DOCX/Google Docsのheading、paragraph、tableを保持する。
- raw `.md` は証跡として残し、index投入は正規化artifactを優先する。

受け入れ条件:

- PDF/PPTXのchunkに `page_number` または `slide_number` が入る。
- 画像artifactに `source_document_id`、`page_number` または `slide_number` を付けられる。
- 正規化artifactから元のDriveファイルとrawファイルへ戻れる。

### Phase 4: OCRと画像主体資料の扱いを改善する

実装内容:

- PDFでテキストが空のページだけでなく、抽出本文がページ番号・単語だけのページもOCR候補にする。
- OCR実施可否、失敗理由、OCRモデル名をmetadataへ残す。
- PDFページレンダリング画像を画像検索artifactとして扱うかどうかをconfigで制御する。
- PPTX embedded imageとslide textを相互リンクする。

受け入れ条件:

- ページ番号だけのPDFが本文RAGへノイズとして入らない。
- 画像主体PDF/PPTXは画像検索またはOCR結果へ誘導できる。
- OCR未実施の理由が監査結果に出る。

### Phase 5: chunk化と検索結果citationを構造metadata対応にする

実装候補:

- `src/kumc_agent/infra/indexing/chunking.py`
- `src/kumc_agent/features/rag/components/generation.py`
- `src/kumc_agent/features/image_search/service.py`

実装内容:

- `source_type=docs` のchunkで `page_number`、`slide_number`、`heading_path`、`quality_flags` を保持する。
- `index_status=quarantined` や `redaction_policy=deny` のchunkを既存アクセスフィルタで除外する。
- citation表示でDriveファイル名だけでなく、ページ・スライド番号を出せるようにする。
- 画像検索候補とRAG chunkを `drive_file_id` + `page_number/slide_number` で結びつける。

受け入れ条件:

- 「どのファイルの何ページ/何スライドか」を回答根拠に出せる。
- 低情報量chunkが検索上位を占有しない。
- 画像主体資料は画像検索とRAGの両方から追跡できる。

### Phase 6: 重複・派生資料のcanonical化を実装する

実装候補:

- `src/kumc_agent/infra/indexing/material_catalog.py`
- `src/kumc_agent/features/indexing/service.py`

実装内容:

- `content_sha256` とDrive metadataから重複候補を検出する。
- 同一本文の別Drive fileはcanonical sourceへまとめ、variantとして保持する。
- PDF/PPTX/DOCXの同名派生資料は `variant_group_id` を付ける。
- Material catalogにcanonical名、aliases、variant refsを出す。

受け入れ条件:

- 完全同一本文が複数hitして検索枠を埋めない。
- citationではcanonicalとvariantの関係を説明できる。
- Drive file id単位の追跡は失われない。

### Phase 7: Quality Gateをindex更新へ組み込む

実装候補:

- `src/kumc_agent/features/indexing/service.py`
- `src/kumc_agent/usecases/indexing/auto_update.py`
- `configs/main/indexing.yaml` または専用config

実装内容:

- Docs raw監査の閾値をconfig化する。
- 開発ではwarning、本番publish前はfail-fastにできるようにする。
- `AutoIndexUpdateUsecase` のquality結果にDocs rawサマリを含める。
- CLI payloadでは診断を `metadata` 配下に入れる。

受け入れ条件:

- 短文率やmetadata欠落が閾値を超えるrawセットはpublish前に止められる。
- 警告・失敗の理由が具体的なファイル名付きで分かる。
- rollbackやstaging index publishの既存フローと衝突しない。

### Phase 8: テストとドキュメント更新

追加するテスト観点:

- metadata欠落・orphan・invalid JSONを監査できる。
- PDF/PPTX/DOCX/Google Docsの短文・正常本文・画像主体fixtureを分類できる。
- `index_status=quarantined` が検索対象から除外される。
- `page_number` / `slide_number` がchunk metadataとcitationに残る。
- content hash重複がvariantとして検出される。
- `--raw-dir data/raw/docs` と既定 `data/ingestion/docs` の両方で監査できる。

検証コマンド例:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_google_drive_docs_audit
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_indexing_docs_quality
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest audit --source docs --raw-dir data/raw/docs --json
```

ドキュメント更新候補:

- `docs/design/circle-info-rag.md`
- `docs/design/image-search.md`
- `docs/design/auto-index-update.md`
- `docs/runbooks/auto_index_update.md`
- 必要なら `docs/explanations/` 配下に実装後の検証メモを追加する。

## 8. 完了条件

この改善は、次の状態をもって完了とする。

- `data/raw/docs` と `config.app.ingestion_dir / "docs"` のDocs rawを同じ基準で監査できる。
- 本文・metadata欠落、短文率、重複、画像主体資料、source_date不明、ACL/visibility不足を機械的に検出できる。
- raw `.md` は証跡として残しつつ、検索用にはページ・スライド・表・画像参照を持つ正規化artifactを使える。
- PDF/PPTX/DOCX/Google Docsのchunk metadataに、少なくとも `drive_file_id`、`drive_file_name`、`drive_file_path`、`source_date`、`page_number` または `slide_number`、`quality_flags`、`index_status` が入る。
- 低情報量・画像主体・重複資料がRAG本文indexを汚染しない。
- 画像検索artifactとDocs本文chunkが同じDrive file、ページ、スライド単位で相互参照できる。
- 品質ゲートに失敗したDocs rawセットはindex publish前にwarningまたは停止できる。
- CLIや外部連携payloadの診断情報はトップレベルではなく `metadata` 配下に格納される。
