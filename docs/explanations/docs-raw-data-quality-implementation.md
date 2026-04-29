# Docs取得データ品質改善 実装メモ

作成日: 2026-04-29

## 概要

`docs/plans/docs-raw-data-quality.md` に基づき、Google Drive由来のDocs rawデータをRAG・画像検索・資料検索で安全に使うための取得、正規化、監査、品質ゲートを追加した。

既存の `data/ingestion/docs/*.md` は証跡として残し、検索用には `data/ingestion/docs_normalized/*.jsonl` を優先してchunk化する。`docs_normalized` がない既存データは従来どおり `.md` からfallback chunk化できる。

## 実装内容

- Drive取得時に `.md.meta.json` へ品質metadataを追加する。
  - `source_date`
  - `updated_at`
  - `checksum` / `content_sha256`
  - `extraction_method`
  - `extraction_status`
  - `text_bytes`
  - `nonempty_characters`
  - `page_count` / `slide_count`
  - `ocr_page_count` / `ocr_candidate_count`
  - `embedded_image_count`
  - `quality_flags`
  - `index_status`
  - `access_scope`
- PDFはページ単位、PPTX/Google Slidesはスライド単位、DOCX/Google Docsはブロック単位で `docs_normalized` に保存する。
- PDFのOCR候補を「空ページ」だけでなく「ページ番号だけ」「極端に短いページ」に拡張した。OCRモデルがない場合は失敗させず、metadataに `ocr_status=skipped_model_unavailable` を残す。
- PPTX embedded image metadataに `source_document_id`、`slide_number`、`slide_ref` を追加し、画像検索artifactから元スライドへ戻れるようにした。
- `docs_normalized` をchunk化の優先入力にし、`index_status=quarantined/deleted/permission_lost` または `redaction_policy=deny` のrecordはRAG本文indexへ入れない。
- ingestion repository を使う自動更新経路でも `docs_normalized` のrecordを優先して `SourceRawItem` 化し、ページ・スライド単位のmetadataを保持する。
- content hashが同じDocsは `variant_group_id` とcanonical metadataを付与し、非canonical側を `quarantined` にする。
- `ingest audit --source docs` を追加し、Docs raw監査結果をCLI JSON payloadの `metadata` 配下に返す。
- index buildのstage resultに `docs_quality` を追加し、`configs/main/indexing.yaml` の `indexing.docs_quality.fail_fast=true` でpublish前に停止できるようにした。
- citation/context生成でDocsの `page_number` / `slide_number` / `heading_path` を表示できるようにした。

## 設定

`configs/main/indexing.yaml`:

```yaml
indexing:
  docs_quality:
    enabled: true
    fail_fast: false
    min_text_bytes: 100
    min_nonempty_characters: 200
    max_short_document_ratio: 0.4
    max_source_date_unknown_ratio: 0.2
    quarantine_low_information: true
```

`.env` / `.env.example` には追加していない。閾値や挙動は設定値なので `configs/main` に置く方針とした。

## 運用コマンド

Docs raw監査:

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest audit --source docs --raw-dir data/raw/docs --format json
```

Markdown出力:

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest audit --source docs --raw-dir data/raw/docs --format markdown
```

index更新時は `data/ingestion/docs_normalized` があればそれを優先し、なければ `data/ingestion/docs/*.md` にfallbackする。

## 検証

実行済み:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest \
  tests.unit.test_docs_connector_records \
  tests.unit.test_docs_normalizer \
  tests.unit.test_docs_chunking \
  tests.unit.test_google_drive_docs_audit \
  tests.unit.test_google_drive_docs_loading \
  tests.unit.test_google_drive_sheets_loading \
  tests.unit.test_google_drive_slides_fallback \
  tests.unit.test_google_drive_batching

PYTHONPATH=src app/.venv/bin/python -m compileall -q \
  src/kumc_agent \
  tests/unit/test_docs_normalizer.py \
  tests/unit/test_docs_chunking.py \
  tests/unit/test_google_drive_docs_audit.py \
  tests/unit/test_google_drive_docs_loading.py

PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest audit --source docs --raw-dir data/raw/docs --format json
```

ローカル `data/raw/docs` 監査では、242件の本文と242件のmetadataが揃っていること、短文資料60件、重複本文1グループ、画像artifact 496件を検出できることを確認した。既存 `data/raw/docs` にはまだ `docs_normalized` がないため、監査結果には `normalized_docs_missing` warning が出る。
