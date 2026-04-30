# Notion Indexingデータ改善計画

## 目的

`data/` 配下に保存されている Notion の取り込みデータと chunk データを確認し、RAG / 自動Index更新で問題になり得る点と改善策を整理する。

本調査では、ユーザー指定の "Indestion" は `Ingestion` の意図として扱い、主に以下を対象にした。

- `data/ingestion/notion/**/*.md`
- `data/ingestion/notion/**/*.md.meta.json`
- `data/ingestion/source_items.jsonl`
- `data/ingestion/documents.jsonl`
- `data/ingestion/chunks.jsonl`
- `data/ingestion/sync_cursors.jsonl`
- `data/ingestion/source_deletes.jsonl`
- `data/chunks/chunks.jsonl`
- `data/chunks/first_rec_chunk/notion.jsonl`
- `data/chunks/second_rec_chunk/notion.jsonl`
- `data/chunks/sparse_second_rec_chunk/notion.jsonl`
- `data/chunks/summary_chunk/notion.jsonl`
- `data/object_storage/kumc-agent/raw/notion/**`
- `data/index/staging/**/{dense_chunks,bm25_chunks}.jsonl`

## 現状サマリ

ローカルの Notion raw ingestion には Markdown 45件が存在するが、ingestion repository と最終chunkには standalone page 1件分しか反映されていない。

| 層 | 件数・状態 |
| --- | --- |
| `data/ingestion/notion` | Markdown 45件、metadata sidecar 45件。sidecar欠落は0件 |
| `data/object_storage/kumc-agent/raw/notion` | raw snapshot 1件 |
| `data/ingestion/source_items.jsonl` | Notion行 1件 |
| `data/ingestion/documents.jsonl` | Notion行 1件 |
| `data/ingestion/chunks.jsonl` | Notion chunk 5件。すべて同一 standalone page |
| `data/chunks/chunks.jsonl` | Notion chunk 10件。すべて同一 standalone page |
| `data/chunks/first_rec_chunk/notion.jsonl` | 5 chunk。旧形式の単一JSONL |
| `data/chunks/second_rec_chunk/notion.jsonl` | 5 chunk。旧形式の単一JSONL |
| `data/chunks/sparse_second_rec_chunk/notion.jsonl` | 5 chunk。旧形式の単一JSONL |
| `data/chunks/summary_chunk/notion.jsonl` | 5 chunk。旧形式の単一JSONL |
| `data/chunks/*/notion/` | 現行コードが使うsource別ディレクトリは存在しない |
| `data/index/staging/.../dense_chunks.jsonl` | Notion chunk 10件。すべて同一 standalone page |

Notion Markdown 45件の内訳は、database配下44件、standalone page配下1件である。metadata上の `notion_page_id` は45件すべて一意で、`notion_title` / `notion_url` / `notion_created_time` / `notion_last_edited_time` / `source_type` はすべて存在している。

一方で、repository側に取り込まれている Notion page id は `0d6b66811ed783178b2f01a71c04695f` の1件のみである。raw ingestionに存在する残り44件は、現在の `source_items` / `documents` / `chunks` / staging index に入っていない。

## 実行経路

現行コード上の主経路は以下である。

1. `configs/main/features.yaml` で `features.sources.notion: true`。
2. `configs/main/integrations.yaml` の `integrations.notion.database_ids` / `page_ids` が Notion 取得対象。
3. `NotionLoader` は `data/ingestion/notion` に Markdown と sidecar を保存する。
4. `build_source_connectors()` は `data/ingestion/notion/**/*.md` を `iter_raw_files(source_kind="notion", default_visibility="admin")` で読み、ingestion repository に保存する。
5. `AutoIndexUpdateUsecase` 経由では `prefer_ingestion_repository=true` になり、`IndexingService` は `data/ingestion/chunks.jsonl` の active chunk を主入力にして index artifacts を作る。
6. repository chunk が存在する場合、`data/chunks/first_rec_chunk` / `second_rec_chunk` / `sparse_second_rec_chunk` / `summary_chunk` は repository chunk から再生成される。旧来の `data/chunks/*/notion.jsonl` は現行の主入力ではない。

このため、raw Markdown が45件あっても、ingestion repository に1件しか入っていなければ、最終検索対象も1件分に縮退する。

## 問題点

### 1. raw ingestion と repository / index のNotion件数が一致していない

`data/ingestion/notion` には45ページ分の Markdown があるが、`data/ingestion/source_items.jsonl` と `data/ingestion/documents.jsonl` の Notion行は1件だけである。`data/ingestion/chunks.jsonl` も5 chunkで、すべて standalone page 1件に由来する。

結果として、database配下44ページはローカルrawとして存在しているにもかかわらず、現行RAGの検索対象に入っていない。`data/chunks/chunks.jsonl` と staging index でも Notion chunk は10件だけで、すべて同じ page id である。

`data/ingestion/sync_cursors.jsonl` でも Notion は過去2回 `seen=0 changed=0`、直近で `seen=1 changed=1` になっている。database配下44件を処理した痕跡が repository 側に残っていない。

### 2. Notion IDのpage/database誤分類が運用データに残っている

`logs/execution.log` には、`0d6b66811ed783178b2f01a71c04695f` を database として `/databases/{id}/query` に渡し、Notion APIから「pageでありdatabaseではない」と返されたエラーが残っている。

その後 standalone page としては1件取り込めているが、database配下44件は repository に反映されていない。設定上の `database_ids` / `page_ids` の分類ミス、または分類修正後の再同期不足が、現在のカバレッジ欠落につながっている可能性が高い。

### 3. 旧形式のNotion chunkが残り、現行形式のsource別ディレクトリが存在しない

`data/chunks/first_rec_chunk/notion.jsonl` などの旧形式ファイルは各5行存在する。しかし、現行の chunk pipeline は `data/chunks/first_rec_chunk/notion/` のようなsource別ディレクトリ配下にファイルを作る。

現在、以下のディレクトリはいずれも存在しない。

- `data/chunks/first_rec_chunk/notion/`
- `data/chunks/second_rec_chunk/notion/`
- `data/chunks/sparse_second_rec_chunk/notion/`
- `data/chunks/summary_chunk/notion/`

現行の repository-backed build では旧形式ファイルは主入力にならないため、運用者が `data/chunks/*/notion.jsonl` を見て「Notion chunkがある」と判断すると、実際のindex投入状態を誤読する。

### 4. 低情報量ページが多い

45 Markdownの合計サイズは38,386 bytesで、中央値は478 bytesだった。18件は200 bytes未満、17件は50 bytes未満である。さらに、17件は見出しのみ、または見出しとURLだけの低情報量ページだった。

低情報量ページの多くは `5d6b66811ed783e18a8c8134bee9a3f0` 配下に集中している。この種のページをそのまま chunk 化すると、検索語には反応するが回答根拠として使える本文がほとんどない chunk が増える。

現在は Notion 専用の品質ゲートがなく、Docs向けの `docs_quality` や Sheets向けの `sheets_quality` と同等の、短文率・本文抽出失敗率・重複率を止める仕組みが存在しない。

### 5. exact duplicate本文が存在する

Markdown本文の完全一致が2グループ、計4ファイルで見つかった。いずれも異なる `notion_page_id` だが本文が完全一致している。

Notionではテンプレートページ、リンク集、埋め込み先の重複などで同一本文が発生し得る。現状は duplicate group をmetadataに持たないため、検索・material catalog・要約chunkで同じ内容が別資料として重複表示される可能性がある。

### 6. raw metadataにpage pathとaccess_scopeがない

`data/ingestion/notion/**/*.md.meta.json` には、`notion_database_id` / `notion_page_id` / `notion_title` / `notion_url` / 作成日時 / 最終編集日時は入っている。しかし、設計で求めている「ページパス」に相当する階層情報はない。

また raw sidecar には `access_scope` がなく、connector側の `default_visibility="admin"` により repository の Notion chunk は `visibility=admin` になっている。`docs/design/kumc-agent.md` と `docs/design/circle-info-rag.md` では Notion は「全ユーザー」を既定としているため、現行データのaccess_scopeは設計上の公開範囲とずれている。

### 7. Notion画像・ファイル添付の抽出状態を検証できない

Notion loader の Markdown renderer は本文ブロック、見出し、リスト、table row、bookmark、link preview などをテキスト化する。一方、画像・添付ファイル・PDFなどの取得結果を示す metadata は現在の sidecar にない。

`docs/design/circle-info-rag.md` は「ページ画像はOCRと画像認識説明文を本文に含める」としているが、現行データからは画像ブロックが取得対象外だったのか、存在したが落ちたのか、取得したが本文に入らなかったのかを判定できない。

### 8. indexing runがrunningのまま残っている

`data/operations/indexing_runs.jsonl` には `status=running` の auto-index run が4件残っている。少なくとも直近の staging index は作成されているが、run状態が完了へ遷移していないため、Notionの取り込み不足が「成功した最新index」なのか「途中で止まったstaging」なのかを運用データだけで判断しにくい。

Notion単体の問題ではないが、今回のような source coverage 調査では、run状態の未完了が原因切り分けを難しくしている。

## 改善方針

### Phase 1: Notion設定とrepository同期を再検証する

優先度: 高

対応案:

- `integrations.notion.database_ids` と `integrations.notion.page_ids` を再確認し、page IDを `database_ids` に入れない。
- Notion loader の同期結果に `databases`, `pages_seen`, `pages_updated`, `pages_skipped`, `pages_deleted` を戻り値またはmetadataとして保持し、`IngestionService` の audit / cursor に保存する。
- `data/ingestion/notion` の raw Markdown件数と、repository の `source_items` 件数を sourceごとに突き合わせる検証CLIを追加する。
- `notion` backfill後、raw 45件に対して repository active source item も45件になることを確認する。

検証観点:

- `data/ingestion/notion/**/*.md` の page id 集合と `data/ingestion/source_items.jsonl` の `source_kind=notion` active external id 集合が一致する。
- `sync_cursors.jsonl` の Notion metadata に `seen=45` 相当が残る。
- `data/object_storage/kumc-agent/raw/notion/**` も45件分作られる。

### Phase 2: Notion chunk成果物を現行形式に統一する

優先度: 高

対応案:

- repository-backed build後の `data/chunks/first_rec_chunk/notion/` などを source別ディレクトリとして生成する状態を正とする。
- 旧形式の `data/chunks/*/notion.jsonl` は、現行buildで参照されないことを docs に明記するか、移行後に削除する。
- `data/chunks/chunks.jsonl`、`dense_chunks.jsonl`、`bm25_chunks.jsonl` の Notion page id 数が raw / repository と一致するかを自動検査する。

検証観点:

- `data/chunks/first_rec_chunk/notion/` 配下に45ページ分のJSONLが作られる。
- `data/chunks/chunks.jsonl` の `source_type=notion` が standalone page 1件に偏らない。
- staging index の Notion `notion_page_id` unique count が raw件数と一致する。

### Phase 3: Notion品質ゲートを追加する

優先度: 高

対応案:

- `configs/main/indexing.yaml` に `notion_quality` を追加する。
- 最低限、以下を検査する。
  - `min_text_bytes`
  - `min_nonempty_characters`
  - `max_short_document_ratio`
  - `max_heading_only_ratio`
  - `max_duplicate_text_ratio`
  - `min_repository_coverage_ratio`
- policyは初期値 `warn` とし、実データの改善後に `fail` へ切り替えられるようにする。
- `docs_quality` と同様に、quality payload は indexing stage result の `metadata` に入れる。

検証観点:

- 現状データでは「18/45が200 bytes未満」「17/45がheading-only」と警告される。
- repository coverage が `1/45` の状態では警告またはfailになる。
- quality payloadに本文サンプルやsecretを出さず、page id / path / 統計値だけを出す。

### Phase 4: access_scopeを設計に合わせる

優先度: 中

対応案:

- Notionの既定公開範囲を `public` または「全ユーザー相当」のaccess scopeに変更する。
- ただし private Notionを扱う可能性を残すため、sourceごとのaccess policyを config 化する。
- raw sidecarにも `access_scope` または `visibility` を書けるようにし、connectorの暗黙defaultだけに依存しない。

検証観点:

- `data/ingestion/source_items.jsonl` / `chunks.jsonl` の Notion `access_scope.visibility` が設計どおりになる。
- `docs/design/kumc-agent.md` / `docs/design/circle-info-rag.md` の公開範囲と実装・データが一致する。
- admin限定にする必要があるNotionページがある場合は、設計側に例外ルールを明記する。

### Phase 5: page path / 階層metadataを保存する

優先度: 中

対応案:

- Notion再帰取得時に、親database / 親page / child_page / child_database の関係を辿った `notion_page_path` を保存する。
- chunk metadataにも `notion_page_path` を伝播し、citationとmaterial catalogのaliasに使う。
- 同一titleが複数ある場合でも、ページ階層で区別できるようにする。

検証観点:

- `Hypixel` など重複titleを持つページが、pathで区別できる。
- citationにURLだけでなく、必要に応じてNotion内のページパスを表示できる。

### Phase 6: 低情報量ページと重複本文を扱う

優先度: 中

対応案:

- heading-only / URL-only page は `index_status=quarantined` または `quality_flags=["low_information"]` を付与する。
- 完全一致本文は `duplicate_group_id` / `duplicate_group_size` をmetadataに保存する。
- 低情報量ページでもリンク集として意味がある場合は、本文ではなくリンク・子ページ参照metadataとして扱う。
- summary chunk生成では、低情報量pageをLLM要約対象から外す。

検証観点:

- 17件のheading-only pageが通常本文chunkとして検索上位に出にくくなる。
- duplicate group 2件がmaterial catalogや検索結果で重複表示されにくくなる。

### Phase 7: Notion画像・添付ファイルの取得可否を明示する

優先度: 低から中

対応案:

- Notion block renderer に `image`, `file`, `pdf`, `video`, `embed` などの検出metadataを追加する。
- 画像を取得しない場合でも、`notion_asset_count` / `notion_unsupported_block_types` を sidecar に保存する。
- 画像検索対象にするかは別設計として扱い、まずサークル情報RAGの本文補助として OCR / caption を入れるか決める。

検証観点:

- 画像ブロックがあるページで、取得対象外なのか、取得失敗なのか、OCR未対応なのかを audit できる。
- Notion画像を本文RAGに含める場合、secretやprivate asset URLを外部payloadに出さない。

### Phase 8: auto-index run状態の完了記録を確認する

優先度: 中

対応案:

- auto-index usecase の成功・失敗・キャンセル時に `data/operations/indexing_runs.jsonl` のrun状態が `succeeded` / `failed` / `cancelled` に終わることを再確認する。
- staging index 作成後にpublish前で止まった場合、rollback / failure metadata が残るようにする。
- source別の coverage failure が起きた場合、run metadataに `notion_repository_coverage_ratio` などを残す。

検証観点:

- `status=running` のまま残るrunが新規に増えない。
- Notionのraw/repository/index件数不一致が、auto-index結果だけで発見できる。

## 推奨実装順

1. Phase 1で Notion page/database設定と repository coverage を直す。
2. Phase 2で chunk成果物を現行形式に揃え、旧形式との誤読をなくす。
3. Phase 3で Notion品質ゲートを入れ、今回の欠落が再発しても検出できるようにする。
4. Phase 4とPhase 5で access_scope と page path を設計・検索表示と一致させる。
5. Phase 6以降で低情報量・重複・画像添付・run状態を順に強化する。

最優先は、raw 45件に対して repository / index が1件しかないカバレッジ欠落の解消である。ここが直らない限り、chunk品質やsummary品質を改善しても、Notion database配下の大半は検索に使われない。

## 調査時点の未検証事項

- 実際の Notion API への再接続は行っていない。
- `.env` の実値は読んでいないため、現在設定されている `KUMC_NOTION_DATABASE_IDS` / `KUMC_NOTION_PAGE_IDS` の実運用値は未確認。
- Notion本文の中身は統計中心に確認し、secretを含む可能性がある本文断片は計画書に引用していない。
- 既存の未コミット変更があるため、今回はコード修正・データ削除・chunk再生成は行っていない。
