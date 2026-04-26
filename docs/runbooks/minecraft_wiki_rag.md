# Minecraft Wiki RAG 運用メモ

## 取得対象

取得元は日本語版 Minecraft Wiki のみです。

- `integrations.minecraft_wiki.api_url`: `https://ja.minecraft.wiki/api.php`
- `integrations.minecraft_wiki.page_url_base`: `https://ja.minecraft.wiki/w/`
- `integrations.minecraft_wiki.namespaces`: 通常は `[0]`

開発時は `integrations.minecraft_wiki.page_titles` または `KUMC_MINECRAFT_WIKI_PAGES` で対象ページを限定します。
全記事取得は `integrations.minecraft_wiki.full_backfill_enabled` を `true` にした場合だけ実行されます。

## 速度制限

Wiki 側への負荷を避けるため、次の設定で取得間隔を制御します。

- `integrations.minecraft_wiki.rate_limit_per_minute`
- `integrations.minecraft_wiki.request_interval_seconds`
- `integrations.minecraft_wiki.max_pages`

`max_pages` は全記事取得時も安全弁として効きます。

## 再取得

Raw cache は `data/raw/minecraft_wiki` に保存されます。
通常の再取得は次のコマンドを使います。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest backfill --source minecraft_wiki --limit 20
```

index 更新は Minecraft Wiki 専用 chunk 設定を使います。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index update --stage first_recursive --stage second_recursive --stage sparse_second_recursive --stage summary
```

## Rollback

問題がある場合は feature flag を無効化します。

```env
KUMC_FEATURE_SOURCE_MINECRAFT_WIKI=false
```

既存のサークル情報 RAG とは chunk/index 設定を分けていますが、同じ dense index に投入されるため、必要に応じて直近の `data/index` と `data/chunks/*/minecraft_wiki` をバックアップから戻してください。
