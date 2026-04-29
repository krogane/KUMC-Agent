# Minecraft Wiki RAG 運用メモ

## 取得対象

取得元は日本語版 Minecraft Wiki のみです。

- `integrations.minecraft_wiki.api_url`: `https://ja.minecraft.wiki/api.php`
- `integrations.minecraft_wiki.page_url_base`: `https://ja.minecraft.wiki/w/`
- `integrations.minecraft_wiki.namespaces`: 通常は `[0]`

開発時は `integrations.minecraft_wiki.page_titles` または `KUMC_MINECRAFT_WIKI_PAGES` で対象ページを限定します。
代表的な少数取得を行う場合は `integrations.minecraft_wiki.acquisition_mode` を `category_sample` にし、`integrations.minecraft_wiki.category_sample.categories` と `per_category_limit` を調整します。
全記事取得は `integrations.minecraft_wiki.full_backfill_enabled` を `true` にした場合だけ実行されます。

## 速度制限

Wiki 側への負荷を避けるため、次の設定で取得間隔を制御します。

- `integrations.minecraft_wiki.rate_limit_per_minute`
- `integrations.minecraft_wiki.request_interval_seconds`
- `integrations.minecraft_wiki.max_pages`

`max_pages` は全記事取得時も安全弁として効きます。

## 再取得

Raw cache は `data/ingestion/minecraft_wiki` に保存されます。
redirect alias は実体記事へ解決され、alias元・解決先・page id の対応は各 `.meta.json` と `manifest.json` に保存されます。
同じ `minecraft_wiki_page_id` に解決されるaliasと実体記事はRaw cache上で1件に統合され、重複する旧cacheファイルは該当ページの再取得時に削除されます。
通常の再取得は次のコマンドを使います。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest backfill --source minecraft_wiki --limit 20
```

取得後はRaw品質を確認します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest audit --source minecraft_wiki
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest audit --source minecraft_wiki --format markdown
```

主な確認項目は `metadata.redirect_ratio`、`metadata.indexable_page_count`、`metadata.missing_revision_count`、`metadata.canonical_hosts`、`metadata.chunk_count` です。JSON payloadの診断情報はすべて `metadata` 配下に入ります。

index 更新は Minecraft Wiki 専用 chunk 設定を使います。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index update --stage first_recursive --stage second_recursive --stage sparse_second_recursive --stage summary
```

`integrations.minecraft_wiki.quality_gate.policy` が `warn` の場合、品質問題は `stage_results.minecraft_wiki_quality` に警告として残ります。`fail` の場合、転送本文のみページ率、index可能ページ数、chunk数、metadata必須項目、canonical host のいずれかが閾値を満たさないとpublish前に停止します。

## Rollback

問題がある場合は feature flag を無効化します。

```env
KUMC_FEATURE_SOURCE_MINECRAFT_WIKI=false
```

既存のサークル情報 RAG とは chunk/index 設定を分けていますが、同じ dense index に投入されるため、必要に応じて直近の `data/index` と `data/chunks/*/minecraft_wiki` をバックアップから戻してください。
