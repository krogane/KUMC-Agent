# Minecraft Wiki RAG 実装計画

## 1. 方針
`docs/design/kumc-agent.md` と `docs/design/minecraft-wiki-rag.md` に従い、Minecraft Wiki RAGを段階実装する。

実装では `src/kumc_agent/infra/legacy` を参照・依存しない。既存の共通部品は `features/ingestion`、`features/rag`、`infra/connectors`、`infra/retrieval`、`domain/models` を優先して使う。現行実装と設計が矛盾する場合は `kumc-agent.md` を優先するが、Minecraft Version / Minecraft Edition判定、属性フィルタリング、additional_queriesは実装対象から除外する。

## 2. 完了条件
- 日本語版Minecraft Wiki記事のみを取得し、Raw cacheとmetadataを保存できる。
- 取得済み記事から第1 Recursive Chunk、第2 Recursive Chunk、Summary Chunkを作成できる。
- チャンク・検索設定をサークル情報RAGとは別に設定できる。
- 第2 Recursive ChunkとSummary ChunkをDense indexとSparse転置インデックスへ投入できる。
- 入力クエリからMinecraft Wiki RAG経路を選択できる。
- `additional_queries`、合成クエリ生成、Minecraft Version / Minecraft Edition判定、属性フィルタリングを行わない。
- Dense、通常Sparse、ステミングSparse、RRF、ReRank、Doc Cap、MMRで検索できる。
- ReRankの後にDoc Capを行う。
- 追加履歴が有効な場合は同一チャンネル履歴だけを回答生成に含める。
- 回答生成時はJava版前提で回答を作成する。
- 回答フィルタリングを行わずに回答を返す。
- CLIや外部連携payloadの診断情報が `metadata` 配下に入る。
- 主要動作を既存テスト方式で検証できる。

## 3. 実装ステップ
### Phase 1: 取得・正規化
1. `MinecraftWikiConnector` に取得速度制限を追加する。
2. `api_url` と `page_url_base` が日本語版Minecraft Wikiを指す設定にする。
3. MediaWiki APIから日本語版記事タイトルを列挙する処理を追加する。
4. `page_titles` が指定された場合は限定取得、未指定かつ全取得が有効な場合は日本語版全記事取得にする。
5. revision id、revision timestamp、page id、canonical URLをmetadataに保存する。
6. Wiki記法から検索向けMarkdownへ正規化する軽量パーサを追加する。
7. Minecraft Version / Minecraft Edition判定用の抽出処理を追加しない。

検証:
- 日本語版API URLと日本語版canonical URLが使われること。
- Raw cacheが再利用されること。
- `max_pages` と `BackfillScope.limit` が効くこと。
- edition/version抽出metadataが生成されないこと。

### Phase 2: Minecraft Wiki専用設定
1. `minecraft_wiki_rag.chunking.*` 設定を追加する。
2. `minecraft_wiki_rag.retrieval.*` 設定を追加する。
3. 専用設定が未指定の場合だけ既存RAG設定へfallbackする。
4. 設定ロード、env map、schema、デフォルト値を追加する。
5. `.env` または `.env.example` に項目を追加する場合は、必ず他方にも反映する。

検証:
- サークル情報RAGと異なるチャンクサイズを設定できること。
- サークル情報RAGと異なる検索件数を設定できること。
- 専用設定未指定時のfallbackが動作すること。

### Phase 3: Minecraft Wiki向けチャンク生成
1. `features/ingestion` または新規のMinecraft Wiki indexing serviceで、第1 Recursive Chunkを生成する。
2. 第1 Recursive Chunkから第2 Recursive Chunkを生成する。
3. Summary Chunkを生成する。LLM不可時はfallback要約にする。
4. `data/chunks/*/minecraft_wiki` にJSONL成果物を保存する。
5. metadataに `source_type=minecraft_wiki`、`minecraft_wiki_page_id`、`parent_chunk_id`、`heading_path` を保持する。
6. metadataに `minecraft_version`、`minecraft_versions`、`minecraft_edition` を追加しない。

検証:
- Minecraft Wiki専用チャンク設定が使われること。
- 見出し境界を尊重して分割されること。
- 子チャンクから親チャンクを引けること。
- `skip_parent_context` が必要な場合に付与されること。

### Phase 4: Indexing
1. Minecraft WikiチャンクをDense indexへ投入する経路を追加する。
2. 埋め込みテキストに記事名と見出しを前置する。
3. Sparse用チャンクと転置インデックスを生成する。
4. 通常SparseとステミングSparseの混合比率をMinecraft Wiki専用設定で制御する。
5. index更新時に `minecraft_wiki` だけを再構築できるようにする。

検証:
- Dense indexの対象にMinecraft Wikiチャンクが含まれること。
- Keyword indexでMinecraft固有語、日本語表記、英語表記、数値が検索できること。
- 既存サークル情報RAGのindex成果物を壊さないこと。

### Phase 5: Routing
1. Entry routingでMinecraft Wiki向けクエリを `minecraft_wiki` に振り分ける。
2. Minecraft Wiki専用routing resultを追加し、`use_additional_memory` と `fast_mode` を保持する。
3. `additional_queries` をrouting resultに含めない。
4. 合成クエリ生成を呼ばず、検索には入力クエリをそのまま使う。
5. Minecraft Version / Minecraft Editionの属性抽出を行わない。
6. CLI payloadではrouting詳細を `metadata` 配下に置く。

検証:
- Minecraft Wiki向けクエリが `minecraft_wiki` に振り分けられること。
- `additional_queries` が生成・参照されないこと。
- 検索クエリが入力クエリそのものになること。
- ルーティング失敗時に安全なデフォルトへfallbackすること。

### Phase 6: Retrieval
1. Minecraft Wiki検索経路を追加し、検索対象を `source_type=minecraft_wiki` に限定する。
2. Dense、通常Sparse、ステミングSparseを実行する。
3. 属性フィルタリングを実装しない。
4. RRFで候補を統合する。
5. 通常モードではRRF後にReRankを適用する。
6. ReRank後にDoc Capを `source_type + page_id + parent_chunk_id` 単位で適用する。
7. Doc Cap後にMMRを適用する。
8. ファストモードではReRankとMMRをスキップし、RRF後にDoc Capを適用する。

検証:
- source filterが常に適用されること。
- 属性フィルタリングが呼ばれないこと。
- 通常モードで処理順序がRRF -> ReRank -> Doc Cap -> MMRになること。
- ファストモードで処理順序がRRF -> Doc Capになること。
- Minecraft Wiki専用検索件数が使われること。

### Phase 7: Generation
1. Minecraft Wiki専用の回答生成プロンプトを追加する。
2. プロンプトに「日本語版Minecraft Wikiを根拠にする」「Java版前提で回答する」を明記する。
3. Java版/統合版やMinecraft Versionの判定・比較を回答生成機能として扱わない。
4. 親/子チャンク展開ではSummary Chunkを優先する。
5. `use_additional_memory=true` の場合は同一チャンネル履歴だけを含める。
6. 回答フィルタリングを無効化する。
7. 出典表示を記事タイトル、見出し、canonical URL中心に整形する。

検証:
- 根拠不足時に推測回答しないこと。
- 回答生成プロンプトにJava版前提が含まれること。
- Edition差分判定やVersion判定を行わないこと。
- `AnswerFilterComponent` が呼ばれないこと。
- 出典の重複が抑制されること。

### Phase 8: CLI・外部連携
1. CLIのscopeに `minecraft_wiki` を指定した検索・回答を整える。
2. Minecraft Wiki RAGのrouteを `minecraft_wiki_rag` として返す。
3. payloadトップレベルは `answer`、`route`、`sources` などの安定フィールドに限定する。
4. `routing_decision`、`fast_mode`、`trace_id`、検索スコアは `metadata` 配下に入れる。
5. `additional_queries`、`attribute_filter`、edition/version判定結果をpayloadに含めない。
6. 大きなcontext本文やsecretを含む可能性のあるmetadataを出力前に除外・マスクする。

検証:
- CLI出力schemaの単体テスト。
- `metadata` に診断情報が集約されること。
- 削除対象のフィールドが出力されないこと。

### Phase 9: 運用・ドキュメント
1. `docs/explanation/cli.md` にMinecraft Wiki RAGの呼び出し例を追記する。
2. 必要な設定を `configs/ops/app.yaml`、`configs/ops/features.yaml` に追加する。
3. `.env` または `.env.example` に項目を追加する場合は、必ず他方にも反映する。
4. 取得速度制限、日本語版全記事取得、再取得、rollbackの運用手順をrunbook化する。

検証:
- 設定ロードの単体テスト。
- feature flag無効時にconnectorが作られないこと。

## 4. 推奨ファイル変更範囲
想定される主な変更範囲は次の通り。

| 領域 | ファイル候補 |
| --- | --- |
| connector | `src/kumc_agent/infra/connectors/minecraft_wiki.py` |
| connector registry | `src/kumc_agent/infra/connectors/registry.py` |
| source models | `src/kumc_agent/domain/models/source.py` 必要に応じて |
| Minecraft RAG models | `src/kumc_agent/domain/models/minecraft_wiki.py` 新規候補 |
| chunking | `src/kumc_agent/features/ingestion/chunking.py` または新規 |
| indexing | `src/kumc_agent/features/indexing/service.py` からlegacy非依存部分を分離、または新規 |
| retrieval | `src/kumc_agent/features/rag/components/retrieval.py` |
| rag service | `src/kumc_agent/features/rag/service.py` またはMinecraft Wiki専用service |
| routing | `src/kumc_agent/features/rag/components/entry_routing.py`、`routing.py` |
| prompts | `assets/prompts/answer_minecraft_wiki.md` 新規候補 |
| config | `src/kumc_agent/config/schema.py`、`src/kumc_agent/config/load.py`、`src/kumc_agent/config/env_map.py` |
| CLI | `src/kumc_agent/cli.py` |
| tests | `tests/unit/test_minecraft_wiki_*.py` |

`src/kumc_agent/infra/legacy` は変更・参照対象にしない。

## 5. リスクと対策
| リスク | 対策 |
| --- | --- |
| 日本語版以外の記事が混入する | `api_url`、`page_url_base`、canonical URLの検証を追加する |
| 全記事取得でWiki側に負荷をかける | 速度制限、max_pages、安全弁、キャッシュ再利用を必須にする |
| Wiki記法が検索ノイズになる | 軽量Markdown正規化とテンプレート除去を導入する |
| 共通RAG設定変更がMinecraft Wiki RAGに影響する | `minecraft_wiki_rag.*` 専用設定を追加し、専用設定を優先する |
| 共通RAGとindexが混在する | `source_type=minecraft_wiki` filterを検索経路に必須化する |
| 既存RAGの回答フィルタが適用される | Minecraft Wiki RAG経路ではAnswerFilterを呼ばない |
| additional_queriesの既存処理が混入する | Minecraft Wiki RAG経路でQuerySynthesizerを呼ばないテストを追加する |
| legacy依存が残る | 新規実装ではlegacy moduleをimportしないテストを追加する |

## 6. テスト計画
既存の `unittest` 形式に合わせ、次を追加する。

- `tests/unit/test_minecraft_wiki_connector.py`
- `tests/unit/test_minecraft_wiki_chunking.py`
- `tests/unit/test_minecraft_wiki_config.py`
- `tests/unit/test_minecraft_wiki_indexing.py`
- `tests/unit/test_minecraft_wiki_routing.py`
- `tests/unit/test_minecraft_wiki_retrieval_order.py`
- `tests/unit/test_minecraft_wiki_generation.py`
- `tests/unit/test_cli_minecraft_wiki_payload.py`

優先度の高い検証項目:

- 日本語版API URLと日本語版canonical URL
- Raw cache再利用
- 取得範囲制限
- 速度制限設定の反映
- Minecraft Wiki専用チャンク設定
- Minecraft Wiki専用検索設定
- edition/version判定metadataが生成されないこと
- additional_queriesが使われないこと
- source filter
- 属性フィルタリングが呼ばれないこと
- 通常モードの順序: RRF -> ReRank -> Doc Cap -> MMR
- ファストモードの順序: RRF -> Doc Cap
- 親/子チャンク展開
- Java版前提の回答生成
- 回答フィルタ無効化
- payload metadata方針

## 7. 推奨実装順
1. Connectorの日本語版固定化とmetadata保存
2. Minecraft Wiki専用設定
3. Minecraft Wiki専用チャンク生成
4. Dense/Sparse index投入
5. additional_queriesなしのMinecraft Wiki routing
6. ReRank後Doc Capの検索service
7. Java版前提の回答生成と引用整形
8. CLI payload整備
9. 運用ドキュメントと追加設定

この順序にすると、取得データ、設定、チャンク、index、検索、回答の各段階を小さく検証できる。
