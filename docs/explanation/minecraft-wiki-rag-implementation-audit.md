# Minecraft Wiki RAG 実装・再調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/minecraft-wiki-rag.md`
- `docs/plan/minecraft-wiki-rag.md`

## 結論

Minecraft Wiki RAGは、初期実装ではなく完全実装として仕様に合わせた。

実装後の再調査では、設計書・計画書に対する既知の実装差分は残っていない。正式入口はサークル情報RAGと同じ `ChatAnswerUsecase` / `RagService` に統一され、Minecraft Wiki RAG固有の取得、chunk、index、retrieval、generation、payload整形が同じ経路で適用される。

外部ネットワークを使う日本語版Minecraft Wikiへのlive backfillは、現在の実行環境では未実行である。代わりにconnectorのrevision更新、URL検証、metadata、service経路、検索設定、payload sanitizeはunit testで検証した。

## 実装内容

| 領域 | 実装内容 | 主な変更箇所 |
| --- | --- | --- |
| 入口統一 | 統合入力の `minecraft_wiki_rag` を `AskService` ではなく `ChatAnswerUsecase` へ接続し、`route_override=minecraft_wiki` を指定 | `src/kumc_agent/usecases/integrated_input/entry.py` |
| 手動index build | `index build` のsource refresh時にMinecraft Wiki connector ingestionを呼び、Raw取得からindex作成まで完結 | `src/kumc_agent/usecases/indexing/build.py`, `src/kumc_agent/runtime/container.py` |
| index正本 | auto-indexでingestion repositoryを使う場合も、Minecraft Wikiは `data/chunks/*/minecraft_wiki` のraw chunk pipeline成果物をDense/keyword indexへ投入 | `src/kumc_agent/features/indexing/service.py` |
| Summary Chunk | Minecraft Wiki専用LLM要約を実行し、失敗時のみfallback要約へ戻す | `src/kumc_agent/features/indexing/service.py` |
| Sparse検索 | Dense+Sparse同時有効時にSparse候補を保持し、Minecraft Wiki専用Sudachi設定を検索時に渡す | `src/kumc_agent/features/rag/components/retrieval.py`, `src/kumc_agent/features/rag/service.py` |
| keyword index | Minecraft Wiki専用BM25/Sudachi設定で専用corpusを構築し、検索時に専用corpusを使う | `src/kumc_agent/features/indexing/service.py`, `src/kumc_agent/infra/indexing/keyword_inverted_index.py` |
| MediaWiki取得 | 日本語版URL検証、429/5xx retry/backoff、User-Agent、revision id比較によるcache invalidationを追加 | `src/kumc_agent/infra/connectors/minecraft_wiki.py` |
| Wiki正規化 | テンプレート・表を単純削除せず、検索可能な本文情報へ軽量変換 | `src/kumc_agent/infra/connectors/minecraft_wiki.py` |
| routing | Minecraft Wiki routeでは `additional_queries=[]`、`recency_mode=off`、`use_additional_memory=false` に正規化 | `src/kumc_agent/features/rag/components/routing.py`, `src/kumc_agent/features/rag/service.py` |
| ranking | ReRank pool外候補を保持し、Doc Cap後にMMRと記事/見出し重複抑制を適用 | `src/kumc_agent/features/rag/service.py` |
| 出典 | `Source.label` を記事名・見出し、`Source.uri` を記事URLに分離。安定anchorがないため記事URLのみを使用 | `src/kumc_agent/features/rag/components/generation.py` |
| payload | RAG tool payloadのmetadataを共通sanitizerで再帰的に除外・マスク | `src/kumc_agent/cli.py` |
| access metadata | Raw sidecarの `access_scope` をchunk metadataへ伝播 | `src/kumc_agent/infra/indexing/chunking.py` |

## 仕様との差分

実装後の再調査結果は次の通り。

| 仕様項目 | 判定 | 確認内容 |
| --- | --- | --- |
| 日本語版Minecraft Wikiのみ取得 | 達成 | `api_url` / `page_url_base` は `https://ja.minecraft.wiki` のみ許可 |
| 取得速度制限、backoff、cache更新 | 達成 | rate limit、429/5xx retry、revision id比較を実装 |
| Raw cacheとmetadata sidecar | 達成 | page id、revision id、canonical URL、public access metadataを保持 |
| 第1/第2/sparse/Summary Chunk | 達成 | 専用 `data/chunks/*/minecraft_wiki` を生成 |
| Summary Chunk専用LLM | 達成 | provider有効時は専用LLM、失敗時fallback |
| Dense index投入 | 達成 | 第2 Recursive ChunkとSummary Chunkを投入し、記事名・見出しをembedding textに前置 |
| Sparse / keyword index | 達成 | Minecraft Wiki専用corpusを専用BM25/Sudachi設定で構築 |
| 検索時専用設定 | 達成 | Sudachi mode、normalized form、remove_symbols、RRF、混合比率をrouteから渡す |
| source filter | 達成 | `source_type=minecraft_wiki` を必ず適用 |
| QuerySynthesizer不使用 | 達成 | Minecraft Wiki経路では入力クエリをそのまま検索 |
| additional_queries不使用 | 達成 | routing/service/payloadで空に正規化し、metadataにも出さない |
| ReRank -> Doc Cap -> MMR | 達成 | pool外候補保持後にDoc Cap、MMR、記事/見出し重複抑制 |
| fast mode | 達成 | ReRank/MMRをskipし、RRF後にDoc Cap |
| 親/子チャンク展開 | 達成 | Summary Chunk優先、なければ第1 Recursive Chunk |
| 履歴・サークル情報混入防止 | 達成 | `use_additional_memory=false`、Minecraft Wiki promptではcircle infoを除外 |
| Java版前提prompt | 達成 | `assets/prompts/answer_minecraft_wiki.md` |
| 回答filter無効化 | 達成 | Minecraft Wiki経路では `AnswerFilterComponent` を呼ばない |
| 出典schema | 達成 | labelとURLを分離し、`sources[].uri` は純粋な記事URL |
| section anchor | 達成 | 安定生成できないため記事URLのみ、と仕様・promptに明記 |
| CLI/外部payload | 達成 | 主結果はtop-level、診断情報はmetadata、危険metadataは再帰sanitize |
| `.env` / `.env.example` | 達成 | パラメータはconfigs管理。secret以外は追加しない方針を仕様化 |

残差分: なし。

## 仕様改善点の実装状況

| No. | 改善点 | 対応 |
| --- | --- | --- |
| 1 | 完全実装の対象入口を明文化 | サークル情報RAGと同じ `ChatAnswerUsecase` / `RagService` を正式入口にし、統合入力も接続 |
| 2 | index正本を1つに決める | Minecraft Wikiはraw chunk pipeline成果物を正本、ingestion repositoryは取得・変更検知に限定 |
| 3 | 専用retrieval設定の適用範囲 | index構築時は専用BM25/Sudachi corpus、検索時は専用tokenizer/RRF/混合比率を適用 |
| 4 | Sparse検索の受け入れテスト | Dense+Sparse保持、専用corpus、専用検索時token settingsのunit testを追加 |
| 5 | MediaWiki取得の運用仕様 | URL検証、User-Agent、rate limit、429/5xx backoff、revision比較を実装し設計書に反映 |
| 6 | Raw cache更新判定 | cached revision idとremote revision idを比較し、変更時だけ本文再取得 |
| 7 | Wiki記法正規化 | テンプレート引数と表セルの検索可能な情報を保持する軽量変換を追加 |
| 8 | 出典schema分離 | `Source.label` と `Source.uri` を分離 |
| 9 | セクションアンカー仕様 | 安定生成できない場合は記事URLのみ、と設計・promptへ明記。現行は記事URLのみ |
| 10 | Edition / Version対象外時の応答方針 | Java版前提、差分判定しない、根拠不足時は回答不可をpromptに明記 |
| 11 | payload sanitizer共通化 | `sanitize_payload_metadata()` をRAG tool payloadにも適用 |
| 12 | 完全実装の受け入れテスト | connector、retrieval、generation、integrated input、CLI、routing、indexing周辺のunit testを追加・更新 |

## 入口ごとの再調査

| 入口 | 実装後の状態 |
| --- | --- |
| `kumc-agent tool rag --scope minecraft_wiki` | `ChatAnswerUsecase` / `RagService` を通り、専用routeを強制 |
| `ChatAnswerUsecase` 直接利用 | `RagService` のMinecraft Wiki経路を使用 |
| 評価 `EvaluateRagasUsecase` | `ChatAnswerUsecase` を使用 |
| Discord `/ask source=minecraft_wiki` | 統合入力から `ChatAnswerUsecase` へ委譲 |
| HTTP / 統合入力の `minecraft_wiki_rag` | 統合入力から `ChatAnswerUsecase` へ委譲 |
| `kumc-agent index build` | source refresh時にMinecraft Wiki connector ingestionを実行 |
| auto-index update | ingestion repository使用時もMinecraft Wikiはraw chunk pipeline成果物を正本化 |

## 検証

実行した検証:

```bash
app/.venv/bin/python -m compileall -q src/kumc_agent tests/unit
KUMC_OPENAI_API_KEY= OPENAI_API_KEY= app/.venv/bin/python -m unittest tests.unit.test_minecraft_wiki_rag tests.unit.test_rag_retrieval_rrf tests.unit.test_integrated_input tests.unit.test_generation_component tests.unit.test_cli_tool_rag tests.unit.test_query_router tests.unit.test_config_loading
app/.venv/bin/python -m unittest tests.unit.test_indexing_repository_artifacts tests.unit.test_indexing_summary_chunking_llm tests.unit.test_auto_index_update tests.unit.test_material_name_keyword_index tests.unit.test_material_search_matching tests.unit.test_rag_access_filter tests.unit.test_answer_filter
app/.venv/bin/python -m unittest discover tests/unit
```

結果:

- compileall: OK
- Minecraft Wiki / retrieval / integrated input / generation / CLI / routing / config: 49 tests / OK
- indexing repository / summary chunking / auto index / material search / access / filter: 23 tests / OK
- unit test discovery: 234 tests / OK

補足:

- `QueryRouter` のretry failure testは意図的に例外ログを出すが、テスト結果はOK。
- full unit discovery中にHugging Face名前解決失敗のretryログが出るが、該当テストはfallbackを検証しており結果はOK。
- ローカルの `.env` にAPI keyがある環境でもconfig loading testを再現可能にするため、対象テスト再実行ではOpenAI key環境変数を空に固定した。
- live MediaWiki API取得はネットワーク制限により未実行。
