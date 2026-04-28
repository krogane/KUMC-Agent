# 画像検索 実装後再調査

調査日: 2026-04-28

対象仕様:

- `docs/design/image-search.md`
- `docs/plan/image-search.md`
- `docs/explanation/image-search-implementation-review.md` で指摘した差分

## 結論

今回の実装後、更新後の画像検索仕様に対する主要差分は解消済みと判断する。

外部画像特徴量モデルについては、仕様どおり **ローカルに配置済みの CLIP 互換モデルだけを `local_files_only` で使用する**。モデルが未配置の場合はダウンロードせず、`metadata.feature_status=fallback` と `metadata.degraded=true` で明示し、ローカル特徴量またはmetadata hash vectorへfallbackする。

## 解消した差分

| 旧差分 | 対応 |
| --- | --- |
| Postgres repository で Asset を読めない | `PostgresOperationsRepository.list_assets()` / `get_asset()` を実装 |
| `source_filter` / `limit` が workflow 等へ通らない | `WorkRequest`、workflow、統合入力受付、CLI、HTTP、Discord、comprehensive agent に配線 |
| `features.image_search.enabled` が未使用 | runtime/indexing と workflow service で無効化を反映 |
| 権限変更差分検知不足 | `source_fingerprint` に `access_scope`、出典、source metadata を追加 |
| Drive 埋め込み画像不足 | Google Slides / PowerPoint の `ppt/media/*` を `raw/images/google_drive` に展開 |
| 画像特徴量 vector が弱い | ローカル配置済み外部 CLIP モデル対応、fallback/degraded metadata、設定反映を追加 |
| `matched_fields` 不足 | `metadata.search_results[].matched_fields` と asset `metadata.search.matched_fields` を追加 |
| fallback 権限 filter 不足 | service 未設定 fallback でも `ImageAccessPolicy` を通し、protected asset を無条件に返さない |
| duplicate group 制限不足 | `duplicate_group_id` を保存し、既定で同一group 1件に制限 |
| 評価・テスト不足 | `docs/evals/image-search.jsonl` と source/filter/duplicate/Postgres/PPTX media tests を追加 |

## 仕様改善点の反映

| 改善点 | 対応 |
| --- | --- |
| source別合格条件の明文化 | `docs/design/image-search.md` に source別 artifact / metadata / 削除検知表を追加 |
| Google Drive 対象範囲の分離 | 単体画像、Markdown/HTML画像参照、PPTX `ppt/media/*` を仕様化 |
| repository backend contract | JSONL/Postgres共通 contract を仕様化し、Postgres Asset readを実装 |
| 権限 policy 共通化/明確化 | role/private/admin/guild/public の判定と protected source の guild id 欠落時の扱いを明記・実装 |
| ACLを含む fingerprint | 実装済み |
| 特徴量 vector 品質レベル | `feature_model`, `feature_dimensions`, `duplicate_group_limit`, fallback/degradedを仕様・実装に反映 |
| 検索結果診断metadata | `matched_fields` を追加 |
| fallback安全要件 | service内fallbackとservice未設定fallbackの安全条件を仕様化・実装 |
| channel別payload contract | 除外metadataと短縮対象を仕様化し、sanitizer/outputで除外 |
| eval set / threshold | 仕様に threshold を追加し、初期 eval set JSONL を追加 |

## 再調査結果

- `image_usage_request`、`AssetUsageRequest`、`asset_usage_requests` は `src`, `tests`, `infrastructure` に残っていない。
- `feature_model`、`feature_dimensions`、`duplicate_group_limit` は config schema/load/YAML から service へ反映される。
- `features.image_search.enabled=false` は image asset builder と workflow image search route に反映される。
- `source_filter` と `limit` は workflow/統合入力/CLI/HTTP/Discord 経由で渡せる。
- Drive PPTX media 抽出はテストで確認した。
- Postgres Asset 保存後の `get_asset()` / `list_assets()` はテストで確認した。

## 確認コマンド

```bash
python3 -m py_compile src/kumc_agent/features/image_search/service.py src/kumc_agent/features/workflow/service.py src/kumc_agent/infra/loaders/google_drive_impl.py src/kumc_agent/usecases/integrated_input/entry.py src/kumc_agent/features/agentic/comprehensive.py src/kumc_agent/frontends/http/app.py src/kumc_agent/frontends/discord/app.py src/kumc_agent/cli.py src/kumc_agent/config/load.py
python3 -m unittest tests.unit.test_image_search
python3 -m unittest tests.unit.test_google_drive_batching tests.unit.test_google_drive_slides_fallback tests.unit.test_integrated_input tests.unit.test_design_gap_foundation
rg -n "image_usage_request|AssetUsageRequest|asset_usage_requests" src tests infrastructure -g '!src/kumc_agent/infra/legacy/**'
```

結果:

- compile: OK
- `tests.unit.test_image_search`: 6 tests OK
- Google Drive / integrated input / design gap関連: 20 tests OK
- 利用申請フロー禁止語句: 実装配下では検出なし

追加で `python3 -m unittest discover tests/unit` も実行したが、現環境の未導入依存・未設定環境変数により失敗した。主な原因は `discord`, `torch`, `langchain_core`, `fastapi`, `sudachipy` の import 不可と、`KUMC_DISCORD_BOT_TOKEN`, `KUMC_GEMINI_API_KEY`, `KUMC_DRIVE_FOLDER_ID` 未設定である。画像検索関連の追加・変更テストは上記の通り成功している。
