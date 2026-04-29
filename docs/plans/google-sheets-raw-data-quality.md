# Sheets取得データ品質改善 実装計画

## 1. 目的

`data/raw/sheets` に保存された Google Sheets / Excel 由来の取得データを、検索 index に入れる前の raw source として扱いやすくする。

現状は「CSVとして保存できている」段階ではあるが、RAGで質問に答えるための表構造、タブ単位の文脈、秘匿列の扱い、品質検査が弱い。以下では、2026-04-29 時点のローカル `data/raw/sheets` を調査した結果と、それに基づく改善実装計画をまとめる。

## 2. 調査対象

- `data/raw/sheets/*.csv`
- `data/raw/sheets/*.csv.meta.json`
- 取得実装: `src/kumc_agent/infra/loaders/google_drive_impl.py`
- chunk化実装: `src/kumc_agent/infra/indexing/chunking.py`
- Sheets用separator: `src/kumc_agent/infra/indexing/constants.py`

調査コマンド:

```bash
find data/raw/sheets -maxdepth 1 -type f -name '*.csv' -print | wc -l
find data/raw/sheets -maxdepth 1 -type f -name '*.meta.json' -print | wc -l
PYTHONPATH=src /usr/bin/python3 - <<'PY'
# csv / meta を csv モジュールと json で集計
PY
```

## 3. 現状

### 3.1 保存状況

| 項目 | 結果 |
| --- | ---: |
| CSVファイル数 | 33 |
| `.meta.json` 数 | 33 |
| CSV総サイズ | 228,292 bytes |
| CSV解析行数 | 3,024 rows |
| 非空行数 | 907 rows |
| 空行数 | 2,117 rows |
| 非空セル数 | 5,085 cells |
| 文字化け置換文字を含むファイル | 0 |
| CSVに対応するmetadata欠落 | 0 |

MIME type の内訳:

| MIME type | 件数 |
| --- | ---: |
| `application/vnd.google-apps.spreadsheet` | 24 |
| `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` | 9 |

### 3.2 取得実装の挙動

- Google Sheets は Drive export API で `mime_type="text/csv"` として取得している。
  - `src/kumc_agent/infra/loaders/google_drive_impl.py:1494`
- Excel `.xlsx` は zip内XMLを直接読んで、各worksheetを `# sheet: {sheet_name}` 付きのCSV断片として連結している。
  - `src/kumc_agent/infra/loaders/google_drive_impl.py:430`
- 保存時はCSV本文と `.meta.json` だけを書き、worksheetごとのmetadataや正規化済みの表構造は保存していない。
  - `src/kumc_agent/infra/loaders/google_drive_impl.py:1522`

### 3.3 chunk化実装の挙動

- Sheetsは通常のテキストとして読み、`recursive_chunk_dir()` に渡される。
  - `src/kumc_agent/infra/indexing/chunking.py:605`
- Sheets用separatorは `"\n|"`, `"\n\n"`, `"\n"`, `" "`, `""` である。
  - `src/kumc_agent/infra/indexing/constants.py:13`
- 実際のraw CSVはカンマ区切りなので、`"\n|"` はほぼ効かない。結果として、行・列・ヘッダの意味を理解せず、改行と空白ベースでchunk化される。

## 4. データから見えた問題点

### 4.1 Google Sheetsのタブ情報が消える

Excel由来の9ファイルには `# sheet:` マーカーがある。一方、Google Sheets由来の24ファイルには同等のタブ名マーカーがない。

Drive export APIの `text/csv` は、取得できる表を単一CSVに平坦化するため、複数タブを持つGoogle Sheetsでは「どのタブの内容か」「取得されなかったタブがあるか」をrawデータだけでは検証できない。

影響:

- タブ単位の質問に答えにくい。
- 重要なタブが未取得でも検知できない。
- ExcelとGoogle Sheetsで保存形式が不揃いになる。

### 4.2 表構造がRAG向けに正規化されていない

調査対象には、横に広いスケジュール表や、空白セルを前提に見た目で意味が伝わる表が多い。最大列数は68列で、先頭行の大半が空白セルのファイルも複数あった。

現状のchunk化はCSV本文を文字列として分割するだけなので、次の情報が落ちやすい。

- 列見出しと各セル値の対応
- 行見出しと各セル値の対応
- 結合セル相当の見出しの継承
- タブ名、表名、Driveパスとの結びつき
- 空白セルが「未入力」なのか「上位見出しの継承」なのかの区別

影響:

- 検索時に断片だけがhitし、何の表の何行か説明できない。
- スケジュール・シフト・会計・フォーム回答のような表で、列の意味を取り違えやすい。
- summary chunk がCSV断片の羅列になりやすい。

### 4.3 空行・空列ノイズが大きい

全体で3,024行中2,117行が空行だった。特にExcel由来の一部ファイルは、2,014解析行のうち非空行が39行のみで、ほぼ空グリッドがrawに残っている。

影響:

- chunk化前処理と差分判定に不要なノイズが入る。
- 同じ意味の表でも空行数・空列数の違いでchecksumが変わりやすい。
- LLM summary / embedding の入力に無意味な改行が混ざる。

### 4.4 metadataがファイル単位に偏っている

現在の `.meta.json` は、全ファイルで次の7キーだけだった。

- `drive_file_id`
- `drive_name`
- `drive_file_name`
- `drive_path`
- `drive_mime_type`
- `drive_modified_time`
- `drive_url`

chunk化側で `source_date` は推定されるが、raw sidecar自体には `source_date`、`sheet_name`、`sheet_index`、`row_range`、`column_range`、`table_profile` がない。

影響:

- raw取得品質を後から監査しづらい。
- 検索結果のcitationで「どのシートのどの行範囲か」を出せない。
- source freshness と表構造品質を区別して監視できない。

### 4.5 Excel抽出で型・表示形式・数式情報が失われる

`.xlsx` 抽出はXMLからセル値を読み、CSV writerで文字列化している。数値、日付、数式、表示形式、結合セル、非表示行・列、セルコメントなどは保持されない。

影響:

- 日付がシリアル値や不自然な数値として扱われる可能性がある。
- 数式セルでキャッシュ値がない場合、実データが空になる可能性がある。
- 見た目上の表意味とCSV化後のテキストが一致しない可能性がある。

## 5. 改善方針

### 方針A: raw CSVを残しつつ、検索用の正規化成果物を追加する

`data/raw/sheets/*.csv` は証跡として残す。検索用には別途、表構造を保持したJSONLまたはMarkdown風テキストを生成する。

推奨追加先:

- `data/processed/sheets/*.jsonl`
- または indexing stage 内の `first_recursive/sheets` 直前に `normalized_sheets` を挟む

rawを破壊しないことで、既存運用・差分取得・rollbackと衝突しにくい。

### 方針B: Google SheetsはSheets APIでタブ単位に取得する

Google Sheets MIMEの場合は、Drive export `text/csv` ではなく Sheets API `spreadsheets.get` / `values.batchGet` を使う。

保存単位:

- 1 Drive file
- 1 worksheet
- 1 detected table range

最低限保存するmetadata:

| metadata | 用途 |
| --- | --- |
| `drive_file_id` | source追跡 |
| `drive_file_name` | 表示名 |
| `drive_path` | citation |
| `drive_modified_time` | freshness |
| `sheet_id` | worksheet識別 |
| `sheet_name` | citation / RAG文脈 |
| `sheet_index` | 並び順 |
| `row_range` | 行範囲 |
| `column_range` | 列範囲 |
| `table_profile` | 空行率、空列率、推定header行など |
| `sensitivity` | index可否・mask方針 |

### 方針C: 表を「行レコード」に変換してchunk化する

CSV文字列をそのまま分割するのではなく、行ごとに次のようなテキストへ変換する。

```text
Drive: {drive_path}
Sheet: {sheet_name}
Table: rows {row_start}-{row_end}, columns {col_start}-{col_end}
Row {row_number}:
- {column_header_1}: {value_1}
- {column_header_2}: {value_2}
```

表形式がスケジュールグリッドの場合は、行見出し・列見出しを推定して `time_slot`、`date_or_column_label`、`value` のような構造に寄せる。

## 6. 実装計画

### Phase 1: Sheets raw profilerを追加

目的:

ローカルの `data/raw/sheets` に対して、品質問題を定量化できるようにする。

追加候補:

- `src/kumc_agent/infra/loaders/sheets_profile.py`
- `tests/unit/test_google_drive_sheets_profile.py`

実装内容:

- CSVファイルごとの行数、非空行数、空行率、最大列数、非空セル率を計算する。
- `.meta.json` の必須キーを検査する。
- `# sheet:` マーカーの有無を検出する。
- 高リスク列名候補を `metadata.sensitivity_findings` として返す。
- 結果は `data/raw/sheets_profile.json` または `data/processed/sheets/profile.json` に保存する。

完了条件:

- 33 CSV / 33 meta の現在データでprofileを生成できる。
- 空行率・タブ情報欠落・高リスク列候補が機械的に確認できる。
- raw本文やsecret候補をそのままログに出さない。

### Phase 2: Google Sheetsをタブ単位で取得する

目的:

Google Sheets由来データのタブ欠落・タブ名欠落を解消する。

変更候補:

- `src/kumc_agent/infra/loaders/google_drive_impl.py`
- `src/kumc_agent/infra/loaders/common.py`
- `tests/unit/test_google_drive_sheets_loading.py`

実装内容:

- Google Sheets MIMEの場合、Sheets APIでworksheet一覧を取得する。
- 各worksheetの `title`、`sheetId`、`index`、grid propertiesをmetadata化する。
- 値は `values.batchGet` で取得し、タブ単位ファイルとして保存する。
- 既存の `*.csv` 保存は互換のため維持しつつ、新形式を追加する。

保存案:

```text
data/raw/sheets/
  {drive_file_id}__{safe_file_name}.csv
  {drive_file_id}__{safe_file_name}.csv.meta.json
data/raw/sheets_structured/
  {drive_file_id}__{sheet_index}__{safe_sheet_name}.jsonl
  {drive_file_id}__{sheet_index}__{safe_sheet_name}.jsonl.meta.json
```

完了条件:

- 複数タブを持つGoogle Sheetsの全タブを取得できる。
- `sheet_name` をcitation metadataへ伝搬できる。
- 既存の `download_drive_markdown()` 呼び出し元を壊さない。

### Phase 3: CSV / worksheet正規化器を追加

目的:

表をRAG向けの行レコードへ変換する。

追加候補:

- `src/kumc_agent/infra/indexing/sheets_normalizer.py`
- `tests/unit/test_sheets_normalizer.py`

実装内容:

- 空行・空列をtrimする。
- ヘッダ行を推定する。
- 結合セル由来の空白を、近傍見出しとして安全に補完する。
- 横長スケジュール表は、列見出しと行見出しを保持したレコードへ変換する。
- 通常のフォーム回答表は、1回答1レコードに変換する。
- 正規化できない表は、fallbackとして現行CSV textを使う。

完了条件:

- 既存33 CSVのうち、少なくとも通常表・フォーム回答・横長スケジュール表・Excel複数シートの代表例を正規化できる。
- 正規化失敗時もindex build全体は失敗しない。
- 正規化結果に `row_range` / `column_range` / `sheet_name` が入る。

### Phase 4: Sheets専用chunk化に差し替える

目的:

CSV文字列の汎用recursive splitではなく、表単位・行単位の意味を保ったchunkを作る。

変更候補:

- `src/kumc_agent/infra/indexing/chunking.py`
- `src/kumc_agent/features/indexing/service.py`
- `src/kumc_agent/infra/indexing/constants.py`

実装内容:

- `recursive_chunk_dir(... source_type="sheets")` の前にSheets専用処理を挟む。
- 正規化済みJSONLがある場合は、それを優先してchunk化する。
- chunk metadataに `sheet_name`、`row_range`、`column_range`、`table_kind` を入れる。
- `SHEETS_SEPARATORS` はlegacy CSV fallback用に限定する。

完了条件:

- 検索結果から「Driveファイル名 + sheet名 + 行範囲」を返せる。
- 既存docs/messages/x_posts等のchunk化には影響しない。
- 現行CSV fallbackも残る。

### Phase 6: 品質ゲートと運用レポートを追加する

目的:

取得後に問題を早期発見できるようにする。

追加候補:

- `src/kumc_agent/usecases/indexing/sheets_quality.py`
- `docs/runbooks/auto_index_update.md`
- `tests/unit/test_google_drive_sheets_quality.py`

実装内容:

- `index update` のrefresh後にSheets quality summaryを作る。
- 警告条件:
  - metadata欠落
  - 空行率が高すぎる
  - Google Sheetsでタブ数不明
  - 非空セルが極端に少ない
  - 高リスク列がmaskされていない
- 結果は `metadata.sheets_quality` と運用ログに入れる。

完了条件:

- `index update` 実行後にSheets品質の概要を確認できる。
- warningはindex buildを即失敗させず、設定でfail-fastに切り替えられる。
- 既存のauto-index update metadata方針に従う。

## 7. テスト計画

既存環境ではpytest未導入前提のため、`unittest` で追加する。

推奨コマンド:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest \
  tests.unit.test_google_drive_sheets_profile \
  tests.unit.test_google_drive_sheets_loading \
  tests.unit.test_sheets_normalizer \
```

既存回帰:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest \
  tests.unit.test_google_drive_batching \
  tests.unit.test_google_drive_slides_fallback \
  tests.unit.test_raw_loaders_update_policy
```

確認観点:

- Google Sheets複数タブがタブ単位metadata付きで保存される。
- Excel複数worksheetの `# sheet:` 情報が構造化metadataへ移る。
- 空行・空列が検索用成果物から除外される。
- 既存CSV fallbackが維持される。

## 8. 優先順位

1. Phase 1: profiler追加
3. Phase 2: Google Sheetsタブ単位取得
4. Phase 3: 正規化器追加
5. Phase 4: Sheets専用chunk化
6. Phase 6: 品質ゲート

理由:

- まずprofileで現状を継続監視できるようにする。
- その後、取得粒度とchunk化品質を改善する。

## 9. 完了定義

- `data/raw/sheets` の取得結果を、ファイル単位だけでなくタブ・表・行範囲単位で追跡できる。
- Sheets由来chunkのmetadataに、少なくとも `drive_file_id`、`drive_file_path`、`sheet_name`、`row_range`、`column_range` が入る。
- 空行・空列ノイズが検索用textから除去される。
- `index update` のmetadataにSheets品質サマリが入り、運用者が問題を確認できる。
- 既存のCSV保存形式と既存Drive取得フローは、移行期間中も壊さない。
