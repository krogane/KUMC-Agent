# Minecraft Wiki Raw取得ページ品質改善 実装計画

## 1. 調査対象
2026-04-29時点で `data/raw/minecraft_wiki` に保存されているMinecraft Wiki取得ページを対象に、RAGのindex正本として使える品質かを確認した。

現行設計上のRaw保存先は `docs/design/minecraft-wiki-rag.md` と `docs/runbooks/minecraft_wiki_rag.md` で `data/ingestion/minecraft_wiki` と定義されている。一方、今回調査した保存先は `data/raw/minecraft_wiki` であり、現行の `MinecraftWikiConnector` は `config.app.ingestion_dir / "minecraft_wiki"` を読み書きする。

## 2. 実測結果
`data/raw/minecraft_wiki` の実測値は次の通り。

| 観点 | 結果 |
| --- | --- |
| Markdown本文 | 1,421件 |
| sidecar metadata | 1,421件 |
| 100 bytes未満の本文 | 1,288件 |
| 500 bytes未満の本文 | 1,370件 |
| 1 KB超の本文 | 36件 |
| 3 KB超の本文 | 12件 |
| 10 KB超の本文 | 1件 |
| `#転送` / `#REDIRECT` で始まる転送ページ | 1,281件 |
| metadata欠落 | 0件 |
| `minecraft_wiki_revision_id` 欠落 | 0件 |
| `https://ja.minecraft.wiki/` 以外のcanonical URL | 0件 |
| `data/ingestion/minecraft_wiki` | 未作成 |

metadataの基本項目は揃っているが、本文の大半は転送ページまたは極短い曖昧さ回避ページである。例として `Amethyst_Cluster.md` は `#転送 アメジストの塊` のみで、実体ページ本文は保存されていない。

## 3. 問題点
### 3.1 取得ページの大半が転送ページ
1,421件中1,281件が転送ページで、RAGの検索・回答に必要な説明本文を持たない。これをそのままchunk化すると、Dense/Sparse indexが別名・転送先名だけで埋まり、実体説明の検索率が低下する。

### 3.2 限定取得がページ一覧の先頭に偏っている
ファイル名は `0.0.1`、`1.0.0`、`Amethyst Cluster` などアルファベット順の前方に偏っている。`allpages` の列挙結果を `max_pages` で切ると、全体を代表するページ集合にならず、ブロック・Mob・レシピ・コマンドなどの主要トピックを十分に取得できない。

### 3.3 転送解決・alias管理のmetadataがない
転送ページであること、転送先タイトル、転送解決後のpage id、alias元タイトルをmetadataに保持していない。検索結果や引用URLでalias元と実体記事のどちらを出すべきか判断できない。

### 3.4 正規化後本文にWiki/HTML由来のノイズが残る
`<div>`, `<code>`, `<gallery>`、MediaWikiの表記法、画像ファイル名、履歴表の横並びテキストが残っている。特に表やテンプレート由来の情報は1行へ潰れやすく、チャンク境界と検索語の対応が悪くなる。

### 3.5 取得品質のゲートがない
転送率、本文長、主要カテゴリ網羅、revision差分、chunk生成数を検証する仕組みがない。結果として、metadataが正常でもRAGに不適切なRawセットがindexへ進む可能性がある。

### 3.6 ファイル名がpage idを含まない
保存名は `_safe_name(title)` ベースであり、タイトル変更・表記差・安全化後の衝突に弱い。metadataにはpage idがあるため、保存名にもpage idまたはmanifestを併用した方が追跡しやすい。

## 4. 改善方針
Raw取得は「保存できた件数」ではなく「RAGで根拠として使える本文を持つ実体記事数」を品質指標にする。

実装では次の方針を採る。

- 転送ページは本文としてindexしない。転送先の実体記事へ解決し、alias metadataとして保持する。
- `max_pages` によるアルファベット順の打ち切りを、開発用の明示ページリストまたはカテゴリ別サンプリングへ置き換える。
- 品質チェックをingestion完了後、chunk生成前、index publish前に実行する。
- パラメータは `.env` ではなく `configs` 配下に置く。

## 5. 実装計画
### Phase 1: Raw品質監査コマンド
`data/ingestion/minecraft_wiki` の両方を検査できる監査処理を追加する。

実装候補:
- `src/kumc_agent/usecases/ingestion/minecraft_wiki_audit.py`
- CLI: `PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest audit --source minecraft_wiki --raw-dir ...`
- 出力: 件数、転送率、本文長分布、metadata欠落、canonical URL host、更新日時分布、top Nの短文ページ

受け入れ条件:
- 転送ページ率と本文長分布を機械可読JSONと人間向けMarkdownで出せる。
- 監査結果に基づき、index更新を続行できるか判定できる。

### Phase 2: 転送解決の実装
`MinecraftWikiConnector` の取得時にMediaWiki redirectを解決する。

実装方針:
- `action=query` または `action=parse` にredirect解決を組み込み、実体ページ本文を保存する。
- wikitextが `#転送` / `#REDIRECT` で始まる場合は、転送先タイトルを抽出して再取得する。
- metadataに `minecraft_wiki_is_redirect`, `minecraft_wiki_redirect_from`, `minecraft_wiki_redirect_to`, `minecraft_wiki_resolved_title`, `minecraft_wiki_resolved_page_id` を追加する。
- alias元は検索補助metadataとして保持し、本文chunkとしては実体記事を使う。

受け入れ条件:
- `Amethyst Cluster` のようなalias取得でも「アメジストの塊」の実体本文を保存できる。
- 転送ページのみのRaw本文をchunk化しない。
- alias元URLと実体canonical URLの扱いがテストで固定される。

### Phase 3: 取得対象選定の改善
全ページ一覧を単純に先頭から切るのではなく、取得モードを分ける。

実装方針:
- `configured`: `integrations.minecraft_wiki.page_titles` または `KUMC_MINECRAFT_WIKI_PAGES` の明示タイトルだけ取得する。
- `category_sample`: ブロック、アイテム、Mob、バイオーム、エンチャント、コマンド、レシピなど主要カテゴリから均等に取得する。
- `full_backfill`: `full_backfill_enabled=true` のときだけ全件取得する。
- `max_pages` は安全弁として残すが、カテゴリ別上限・ページ種別上限を `configs/main/integrations.yaml` または専用configに置く。

受け入れ条件:
- 開発用の少数取得でも転送ページだけに偏らない。
- 主要カテゴリごとの取得数を監査結果に出せる。
- `max_pages=0` は従来通り無制限として扱う。

### Phase 4: 正規化の改善
検索に不要なHTML/Wiki記法を除去し、表・テンプレート由来の情報を検索可能な文章に変換する。

実装方針:
- `<code>` は中身を保持し、タグは除去する。
- `<div>`, `<gallery>`, 画像ファイル参照は検索価値がある説明だけ残す。
- 表は見出しとセル値の対応が残るテキストへ変換する。
- 長い履歴表は「版 / snapshot / 変更内容」の単位で分割できるよう整形する。
- 追加ライブラリを導入する場合は `requirements.txt` に追加し、既存の軽量正規化で足りる範囲は標準ライブラリで対応する。

受け入れ条件:
- `<div class=...>` や `<gallery>` がchunk本文に残らない。
- `minecraft:...` ID、アイテム名、見出しは失われない。
- 表由来の変更履歴が1行の巨大なノイズにならない。

### Phase 5: Quality Gateの組み込み
ingestionとindex更新に品質ゲートを入れる。

実装方針:
- 転送のみページ率、本文長下限、metadata必須項目、主要カテゴリ網羅率、chunk生成数の閾値をconfig化する。
- 開発モードではwarning、本番publish前はfail-fastにできるようにする。
- `IndexBuildResult` またはmetadataにMinecraft Wiki品質サマリを残す。

受け入れ条件:
- 転送率が高すぎるRawセットは自動で警告または停止できる。
- 本文を持つ実体記事数が一定未満の場合、index publishへ進まない設定にできる。
- CLI payloadでは品質診断をトップレベルではなく `metadata` 配下に置く。

### Phase 6: テストと回帰防止
転送・短文・HTML混在・正常記事のfixtureを追加する。

検証観点:
- `MinecraftWikiConnector` がredirect aliasを実体記事へ解決する。
- 転送ページだけのRawがchunk化対象から除外される。
- 監査コマンドが `data/ingestion/minecraft_wiki` を同じ基準で評価する。
- `data/ingestion/minecraft_wiki` が存在しない場合も明確な診断を返す。
- 正規化で重要語を消さずにHTML/Wikiノイズを落とせる。
- `PYTHONPATH=src app/.venv/bin/python -m unittest ...` で対象テストを実行できる。

## 6. 完了条件
この改善は、次の状態をもって完了とする。

- `data/ingestion/minecraft_wiki` にRAG正本として使える実体記事Rawが保存される。
- 転送ページはaliasとして扱われ、転送本文だけがindexされない。
- 開発用少数取得でも、転送ページに偏らない代表的なページ集合を取得できる。
- Raw監査で本文長・転送率・metadata・カテゴリ分布を確認できる。
- 品質ゲートに失敗したRawセットはpublish前に止められる。
- CLIや外部連携payloadの診断情報は `metadata` 配下に格納される。
- 関連設計・運用メモに保存先、転送解決、品質ゲートの仕様が反映される。