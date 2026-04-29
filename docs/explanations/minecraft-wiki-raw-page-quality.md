# Minecraft Wiki Raw取得品質改善 実装メモ

## 実装内容
`docs/plans/minecraft-wiki-raw-page-quality.md` に基づき、Minecraft Wiki取得を「保存件数」ではなく「RAGで根拠として使える実体記事本文」を基準に改善した。

- `MinecraftWikiConnector` は `action=parse` の `redirects=1` と `#転送` / `#REDIRECT` 検出でaliasを実体記事へ解決する。
- Raw本文は `data/ingestion/minecraft_wiki` に保存し、alias元・解決先・実体page id・revision id・保存ファイル対応は `.meta.json` と `manifest.json` に保持する。
- `integrations.minecraft_wiki.acquisition_mode`、`category_sample.*`、`quality_gate.*` を `configs/main/integrations.yaml` で管理する。
- `ingest audit --source minecraft_wiki` はRaw品質をJSON/Markdownで出力し、診断情報はpayloadの `metadata` 配下に格納する。
- redirect本文だけのRawは ingestion chunker と legacy chunk pipeline の両方でchunk化しない。
- index buildは `stage_results.minecraft_wiki_quality` に品質サマリを残し、`quality_gate.policy=fail` の場合はpublish前に停止できる。

## 主要コマンド
```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest backfill --source minecraft_wiki --limit 20
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest audit --source minecraft_wiki
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest audit --source minecraft_wiki --format markdown
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index update --stage first_recursive --stage second_recursive --stage sparse_second_recursive --stage summary
```

## payload方針
Raw監査JSONのトップレベルは `source`、`status`、`can_continue` のみを安定フィールドとして扱う。本文長分布、転送率、metadata欠落、canonical host、短文ページ、閾値、critical failuresなどの診断情報は `metadata` 配下に置く。
