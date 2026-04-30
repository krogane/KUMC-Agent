# X Indexing データ調査と改善計画

調査日: 2026-04-30

## 対象

- Ingestion: `data/ingestion/x/`, `data/ingestion/source_items.jsonl`, `data/ingestion/documents.jsonl`, `data/ingestion/chunks.jsonl`
- Chunks: `data/chunks/first_rec_chunk/x.jsonl`, `data/chunks/second_rec_chunk/x.jsonl`, `data/chunks/sparse_second_rec_chunk/x.jsonl`, `data/chunks/summary_chunk/x.jsonl`, `data/chunks/chunks.jsonl`
- 公開 index staging: `data/index/staging/auto-index-20260429T161458453928Z-manual/*_chunks.jsonl`

## 実測サマリ

### Ingestion

- `data/ingestion/x` は約 325MB。
- `data/ingestion/x/posts.jsonl` は 445 投稿、重複 post id は 0。
- 投稿日時の範囲は `2023-11-26T11:03:07+00:00` から `2026-01-23T01:20:59+00:00`。
- `x_media_urls` を持つ投稿は 202 件、media URL は合計 322 件。
- `data/ingestion/x/data/tweets_media` には 323 ファイルがあり、内訳は画像 315 件、動画 8 件。
- `https://t.co/` を本文に含む投稿は 329 件、本文内の t.co URL は合計 358 件。
- `x_author_handle` が空の投稿は 254 件、`KUMC_X` が入っている投稿は 191 件。
- `data/ingestion/x/data` には `direct-messages*.js`、DM media、account/profile 系ファイルなど、投稿 index には不要な X archive 全体が同居している。

### Chunks / Index

- X の active chunk は `data/ingestion/chunks.jsonl` 上で 72 件。
- `data/chunks/chunks.jsonl` と staging の `dense_chunks.jsonl` / `bm25_chunks.jsonl` でも X は 72 件。
- 内訳は `second_recursive` 36 件、`summary` 36 件。
- `first_rec_chunk/x.jsonl`、`second_rec_chunk/x.jsonl`、`sparse_second_rec_chunk/x.jsonl`、`summary_chunk/x.jsonl` はそれぞれ 36 行。
- すべての X chunk の `external_id` は `x:posts.jsonl` で、投稿単位の `x_post_id`、`x_post_url`、`message_timestamp`、`x_media_urls` は chunk metadata に残っていない。
- chunk metadata の `source_type` は `x`。一方、`posts.jsonl` の投稿 metadata は `source_type: x_posts`。

## 問題点

### 1. 投稿単位の同一性が失われている

現在の X connector は `posts.jsonl` を 1 ファイルとして扱い、`iter_raw_files()` が JSONL の各行から本文だけを連結している。そのため 445 投稿が 1 つの raw item / document になり、IngestionChunker が文字数で 36 chunk に分割している。

影響:

- 1 chunk に複数投稿が混在し、検索結果が質問と無関係な投稿を一緒に返しやすい。
- chunk boundary が投稿境界ではないため、1 投稿の途中で切れる可能性がある。
- citation が `x:posts.jsonl` になり、個別投稿 URL に戻れない。
- 1 投稿の変更でも `posts.jsonl` 全体の checksum が変わり、X 全体の再処理になりやすい。

関連箇所:

- `src/kumc_agent/infra/connectors/registry.py`: X connector が `iter_raw_files(... extensions={".jsonl"})` を使用。
- `src/kumc_agent/infra/connectors/file_scanner.py`: `.jsonl` は `text` だけを連結し、行単位 metadata を落としている。

### 2. X 用 metadata が chunk / citation まで届いていない

`posts.jsonl` の各行には `x_post_id`、`x_post_url`、`message_timestamp`、`x_author_handle`、`x_media_urls` があるが、現在の repository chunk には引き継がれていない。

影響:

- `src/kumc_agent/features/rag/components/generation.py` の `_x_url_from_metadata()` は `source_type == "x_posts"` を期待しているが、chunk 側は `source_type == "x"` になっているため X URL citation が生成されない。
- RAG delta や時系列処理が `message_timestamp` を使えず、X 投稿の公開日時ではなくファイル単位の更新扱いになる。
- 画像検索や回答説明で投稿画像・動画との対応を失う。

### 3. author handle の欠損が多い

445 投稿中 254 件で `x_author_handle` が空。現在の変換は URL entity から status URL を見つけた場合だけ handle を補完しているため、URL entity がない投稿で欠損する。

改善余地:

- `data/ingestion/x/data/account.js` には archive owner の `username` があるため、少なくとも自アカウント投稿の default handle として利用できる。
- `profile.js` / `account.js` 由来の account id、screen name、display name を normalized metadata として保持する。

### 4. t.co URL が本文ノイズとして残っている

329 投稿に `https://t.co/` が残っている。短縮 URL は検索語として弱く、media だけを指す t.co は本文検索ではノイズになる。

改善余地:

- `entities.urls[].expanded_url` / `display_url` を使って本文中 URL を展開する。
- media URL 用の t.co は本文から除去し、`x_media` metadata に移す。
- 展開後 URL は `x_expanded_urls` として保持する。

### 5. media の扱いが remote URL 偏重で、動画が弱い

`x_media_urls` は remote URL / thumbnail URL であり、同じ archive 内の `tweets_media` local file を参照していない。動画 8 件は local `.mp4` があるが、投稿 metadata では動画として扱えていない。

影響:

- image search が archive 内 media を再利用できず、remote URL 再取得に依存しやすい。
- 動画投稿は thumbnail だけが画像として残り、動画ファイル・種別・代表 thumbnail の関係が失われる。

### 6. 投稿 index に不要な X archive 全体が ingestion root に同居している

`data/ingestion/x/data` には DM、DM media、account/email/profile、広告・端末・IP 系の archive ファイルがある。現時点では connector が `.jsonl` の `posts.jsonl` だけを見るため index には入っていないが、X connector の scope を広げる変更や一括 raw scan の導入時に漏洩リスクがある。

改善余地:

- index 対象は `posts.jsonl` に明示 allowlist する。
- X archive raw は `data/raw/x_archive` など indexing root 外へ分離するか、`data/ingestion/x/raw_archive` を scanner denylist にする。
- DM 系ファイルは public visibility の X source と同じ tree に置かない。

## 改善方針

### Phase 1: X 投稿を行単位の SourceRawItem として扱う

- X 専用 scanner `iter_x_posts()` を追加する。
- `posts.jsonl` を 1 行 1 投稿として読み、以下を設定する。
  - `source_kind`: `x`
  - `external_id`: `x_post_id` または `message_id`
  - `title`: `X post @<handle> <YYYY-MM-DD>`
  - `canonical_url`: `x_post_url`
  - `created_at` / `updated_at`: `message_timestamp`
  - `access_scope`: public
  - `metadata.source_type`: `x_posts`
  - `metadata.x_post_id`, `x_post_url`, `x_author_handle`, `x_media_urls`, `message_timestamp`, `source_date`
- `registry.py` の X connector は `iter_raw_files()` ではなく X 専用 scanner を使う。
- `posts.jsonl` 変換時に top-level `id` も出すか、scanner 側で `metadata.x_post_id` を必須扱いする。

期待値:

- source item は `x:posts.jsonl` 1 件ではなく投稿単位で 445 件程度になる。
- chunk metadata に個別投稿 URL と投稿日時が残る。

### Phase 2: X chunk を投稿境界で作る

- X の通常投稿は 1 投稿 1 chunk を基本にする。
- 長文投稿のみ同一 `x_post_id` の parent/child chunk に分割する。
- `chunk_kind` は `x_post` に寄せ、現状の `tweet` は互換 metadata として必要なら残す。
- `source_type` は回答生成側に合わせて `x_posts` に統一する。
- summary chunk は短文投稿ごとに作ると重複が増えるため、次のどちらかにする。
  - 初期対応では X の summary chunk を無効化し、post chunk だけを index する。
  - 月次・イベント単位の timeline summary を別 `chunk_kind` として作る。

期待値:

- `second_recursive` は最低でも投稿数に近い 445 件程度になる。
- citation は投稿 URL を返せる。
- 親チャンク上限や RRF が「1 ファイル」ではなく「1 投稿」を単位に効く。

### Phase 3: URL / media 正規化を強化する

- `x_impl.py` で URL entity を使い、本文中の t.co を expanded/display URL へ置換する。
- media の t.co は本文から除去し、metadata に移す。
- `x_media` を配列で持ち、少なくとも以下を保持する。
  - `type`: `photo` / `video` / `animated_gif`
  - `remote_url`
  - `local_relative_path`
  - `content_hash`
  - `thumbnail_remote_url`
- `tweets_media` の local file と raw tweet media entity を post id で対応付ける。
- image search は remote URL だけでなく local file を優先して asset 化する。

### Phase 4: 安全境界を明確化する

- X connector の raw scan は `posts.jsonl` allowlist に固定する。
- `data/ingestion/x/data/direct-messages*.js`、`direct_messages*_media/`、`account*.js`、`ip-audit.js`、`phone-number.js` などは index 対象 denylist に入れる。
- 可能なら X archive 原本は `data/ingestion` 外へ移し、`posts.jsonl` と必要 media manifest だけを ingestion root に置く。
- public source として出る metadata から email、phone、DM、IP、device token、secret 相当値を確実に除外する。

### Phase 5: 品質ゲートと回帰テストを追加する

最低限の検証:

- `posts.jsonl` の投稿数と X source item 数が一致する。
- X chunk の全件に `x_post_id`、`x_post_url`、`message_timestamp`、`source_type=x_posts` がある。
- `external_id=x:posts.jsonl` の X chunk が残っていない。
- `x_author_handle` 欠損率が許容値以下になる。
- X chunk 数が投稿数を大きく下回らない。
- X chunk text に media 用 t.co が残り続けない。
- DM / account / IP / phone 系 raw ファイルが public index 候補に入らない。
- `_x_url_from_metadata()` が X chunk から URL を復元できる。

テスト方針:

- pytest 未導入前提のため、既存方針に合わせて `unittest` ベースの unit test を追加する。
- `x_impl.py` の変換テスト、X scanner の行単位 SourceRawItem 化テスト、generation の citation URL 復元テストを分ける。

## 移行手順案

1. X 専用 scanner とテストを実装する。
2. `x_impl.py` で account default handle、URL 展開、media manifest を追加する。
3. 既存 `data/ingestion/x/posts.jsonl` を再生成する。
4. 旧 source item `x:posts.jsonl` を `deleted` 扱いにするか、repository を再構築する。
5. index を full rebuild する。
6. `data/chunks/chunks.jsonl` と staging index で X chunk が投稿単位になっていることを確認する。
7. X 投稿 URL が citation として返る smoke query を追加する。

## 優先度

1. 最優先: `posts.jsonl` を 1 ファイルではなく 1 投稿 1 source item として ingest する。
2. 高: chunk metadata の `source_type=x_posts`、`x_post_id`、`x_post_url`、`message_timestamp` を保持する。
3. 高: X archive raw の allowlist / denylist を明文化し、DM・account 系の誤 index を防ぐ。
4. 中: t.co 展開と media local file 対応を入れる。
5. 中: X summary chunk の扱いを見直し、投稿単位 retrieval と矛盾しない summary にする。
