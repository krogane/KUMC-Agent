# Minecraft Wiki RAG 実装調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/minecraft-wiki-rag.md`
- `docs/plan/minecraft-wiki-rag.md`

調査対象:

- `src/kumc_agent/infra/connectors/minecraft_wiki.py`
- `src/kumc_agent/features/indexing`
- `src/kumc_agent/infra/indexing`
- `src/kumc_agent/features/rag`
- `src/kumc_agent/features/retrieval`
- `src/kumc_agent/usecases`
- `src/kumc_agent/runtime/container.py`
- `src/kumc_agent/cli.py`
- `configs/main/*`
- 関連unit test

`src/kumc_agent/infra/legacy` は調査対象外方針に従い、現行経路から直接参照していないことだけ確認した。現行の `features/indexing/service.py` には `legacy_cfg` という互換設定名が残るが、参照先は `src/kumc_agent/infra/indexing` であり、`src/kumc_agent/infra/legacy` ではない。

## 結論

Minecraft Wiki RAGは、主要部品はかなり実装されているが、現時点では「仕様通りの完全実装」とは判断できない。

`ChatAnswerUsecase` / `RagService` を直接通る `tool rag --scope minecraft_wiki` などの経路では、Minecraft Wiki専用route、source filter、専用prompt、回答filter無効化、ReRank -> Doc Cap -> MMR順序、CLI payload整理の一部が実装されている。

一方で、完全実装としては次がブロッカーである。

- DenseとSparseを同時に有効化した通常検索で、Sparse結果が常に破棄される。
- 手動index buildはMinecraft Wiki connectorを呼ばず、auto-indexはingestion repository chunkを優先するため、仕様の第1/第2/Summary Chunk成果物を正本としてDense/BM25へ投入する経路が安定していない。
- Minecraft Wiki専用のSparse/BM25/Sudachi設定が、オンライン検索runtimeやkeyword index構築の一部に反映されていない。
- Summary ChunkはMinecraft Wiki専用pipelineで常にfallback要約であり、仕様の専用LLM要約になっていない。
- Discord `/ask` などの統合入力経路は `RagService` ではなく `AskService` を使うため、Java版前提prompt、回答filter無効化、親/子chunk展開などのMinecraft Wiki RAG仕様を通らない。
- MediaWiki取得は速度制限と全記事列挙を持つが、429/5xx指数バックオフ、revision差分再取得、言語URL検証は未実装。

## 実装済みの主な要素

| 仕様項目 | 状態 | 主な実装箇所 |
| --- | --- | --- |
| Minecraft Wiki connector | 実装済み | `src/kumc_agent/infra/connectors/minecraft_wiki.py` |
| 日本語版URLの既定値 | 実装済み | `src/kumc_agent/config/load.py`, `configs/main/integrations.yaml` |
| page_titles / full_backfill / namespaces / max_pages | 実装済み | `MinecraftWikiConnector._resolve_backfill_titles()`, `_list_all_page_titles()` |
| Raw cacheとmetadata sidecar | 実装済み | `MinecraftWikiConnector._fetch_page()` |
| page id / revision id / timestamp / canonical URL metadata | 実装済み | `MinecraftWikiConnector._download_page()`, `_revision_metadata()` |
| Wiki記法の軽量正規化 | 部分実装済み | `_normalize_wikitext()` |
| Minecraft Wiki専用設定schema | 実装済み | `MinecraftWikiRagSection`, `configs/main/indexing.yaml` |
| 専用第1/第2/sparse/Summary出力ディレクトリ | 実装済み | `IndexingService._run_minecraft_wiki_chunk_pipeline()` |
| heading_path付与 | 部分実装済み | `infra/indexing/chunking.py` |
| Dense embedding時の記事名・見出し前置 | 実装済み | `IndexingService._chunk_embedding_text_for_dense()` |
| Minecraft Wiki route | 実装済み | `QueryRouter`, `RagService._answer_minecraft_wiki()` |
| QuerySynthesizer不使用 | 実装済み | `RagService._answer_minecraft_wiki()` |
| source_type=minecraft_wiki filter | 実装済み | `RagService._retrieve_minecraft_wiki_chunks()` |
| ReRank -> Doc Cap -> MMR順序 | 実装済み | `RagService._rank_and_select_minecraft_wiki_chunks()` |
| fast modeでReRank/MMR skip | 実装済み | `RagService._rank_and_select_minecraft_wiki_chunks()` |
| 専用回答prompt | 実装済み | `assets/prompts/answer_minecraft_wiki.md` |
| 回答filter無効化 | 実装済み | `RagService._answer_minecraft_wiki()` |
| Minecraft Wiki向け出典文字列 | 部分実装済み | `GenerationComponent._minecraft_wiki_ref_from_metadata()` |
| CLI `tool rag --scope minecraft_wiki` | 実装済み | `src/kumc_agent/cli.py` |

## 仕様との差分

| 優先度 | 差分 | 影響 | 根拠 |
| --- | --- | --- | --- |
| Critical | Dense + Sparse同時検索時、Sparse結果が無条件で空になる | Minecraft Wiki RAGのDense、通常Sparse、ステミングSparse、RRF統合が実質Dense寄りになる | `src/kumc_agent/features/rag/components/retrieval.py:117-121` |
| Critical | 手動 `index build` はMinecraft Wiki connectorを呼ばない | rawが事前に存在しない環境では、仕様の「日本語版Minecraft Wiki記事の取得」からindex作成までが手動buildで完結しない | `src/kumc_agent/usecases/indexing/build.py:49-60` はDrive/Discord等のloaderだけを呼ぶ |
| Critical | auto-index時はingestion repository chunkがDense/BM25の正本になり、専用第1/第2/Summary成果物を使わない場合がある | 仕様で要求される第2 Recursive Chunk + Summary Chunk投入、親/子chunk展開、専用chunk設定の効果が自動更新の正本indexに反映されない可能性が高い | `src/kumc_agent/usecases/indexing/auto_update.py:223-230`, `src/kumc_agent/features/indexing/service.py:176-185` |
| High | Minecraft Wiki専用Sparse/BM25/Sudachi設定がオンライン検索runtimeに渡っていない | `minecraft_wiki_rag.retrieval.sudachi_mode` や `sparse_remove_symbols` 等を変えても検索時tokenizeに反映されない | `runtime/container.py:194-200` は `features.retrieval.*` で `RetrievalComponent` を構築 |
| High | keyword inverted index構築がMinecraft Wiki専用retrieval設定ではなく共通設定を使う | 仕様の専用BM25パラメータ、Sudachi mode、symbol処理がkeyword indexに反映されない | `features/indexing/service.py:193-195`, `1130-1235` |
| High | Minecraft Wiki専用Summary ChunkはLLMを呼ばず常にfallback要約 | 仕様の「専用LLMで要約し、失敗時fallback」ではなく、重要な仕様・数値・条件をLLMで抽出する品質要件を満たせない | `features/indexing/service.py:885-936` |
| High | 統合入力経路の `minecraft_wiki_rag` は `RagService` ではなく `AskService` | Discord `/ask` では専用prompt、回答filter無効化、親/子chunk展開、Minecraft Wiki専用検索設定を通らない | `usecases/integrated_input/entry.py:133-147`, `features/retrieval/ask.py` |
| High | MediaWiki APIエラー、HTTP 429、5xxの指数バックオフがない | 全記事取得時に失敗しやすく、Wiki側への負荷制御も仕様より弱い | `MinecraftWikiConnector._request_json()` は単発 `urlopen()` |
| High | Raw cacheはrevision id / checksumを見て再取得しない | 既存rawがあるとWiki側の更新を検知せず、差分更新仕様を満たさない | `MinecraftWikiConnector._fetch_page()` は `path.exists()` で即cacheを返す |
| Medium | 日本語版Minecraft Wikiのみを対象にするURL検証がない | 設定誤りで英語版や別サイトを向いても実行される | `api_url`, `page_url_base` は設定値をそのまま使用 |
| Medium | `_normalize_wikitext()` は表・テンプレート情報を大きく落とす | レシピ、耐久値、爆発耐性など表に入る重要情報が検索不能になる可能性がある | `minecraft_wiki.py:282-300` |
| Medium | routing schemaはMinecraft Wiki routeでも `additional_queries` を要求し得る | `RagService` では無視されるが、仕様の「Minecraft Wiki RAGでは additional_queries を使わない」とrouting設計が完全には一致しない | `features/rag/components/routing.py:52-70`, `198-238` |
| Medium | ReRank pool外の候補がDoc Cap前に捨てられる | 同一親chunkでpool内候補が削られた後、pool外から補充されず、`top_k` を満たしにくい | `RagService._rank_and_select_minecraft_wiki_chunks()` |
| Medium | MMRは記事単位・見出し単位の重複抑制を明示的には行わない | 仕様の「同一記事内の類似セクション偏り抑制」はembedding類似度頼みになる | `RetrievalComponent._apply_mmr()` |
| Medium | 出典の `Source.uri` に「タイトル: URL」文字列を入れる | 外部連携payloadでURL欄が純粋なURLにならず、仕様のcanonical URL出力とずれる | `generation.py:630-635`, `1050-1071` |
| Medium | セクションアンカー生成がない | 見出しへの直接リンク仕様が未達 | `GenerationComponent._minecraft_wiki_ref_from_metadata()` |
| Medium | CLI RAG payload metadataのsanitizeが浅い | `contexts`, `llm_prompt`, `raw` は除外するが、任意metadata内のsecretや大きい本文の再帰sanitizeは行わない | `cli.py:29-45` |
| Low | `.env` / `.env.example` に新規Minecraft Wiki設定の一部しか例示がない | envで運用する場合、max_pagesやrate limit、専用RAG設定の存在に気づきにくい | `src/kumc_agent/config/env_map.py` と `.env.example` の差 |

## 主要入口ごとの状態

| 入口 | 現状 | 完全仕様利用 |
| --- | --- | --- |
| `kumc-agent tool rag --scope minecraft_wiki` | `ChatAnswerUsecase` / `RagService` を通る | 部分可。専用prompt等は使うが、Sparse破棄やindex正本の問題あり |
| `ChatAnswerUsecase` 直接利用 | `RagService` を通る | 部分可 |
| 評価 `EvaluateRagasUsecase` | `ChatAnswerUsecase` を通る | 部分可 |
| Discord `/ask source=minecraft_wiki` | `IntegratedInputUsecase` / `AskService` を通る | 不可 |
| HTTPや統合入力からの `minecraft_wiki_rag` | `AskService` を通る | 不可 |
| `kumc-agent index build` | Minecraft Wiki connectorを呼ばない | raw準備済みなら部分可、初回取得からは不可 |
| auto-index update | connector ingestionは通るがrepository chunk優先 | 取得は可、専用chunk成果物を正本にした完全indexとは言いにくい |

## 完了条件ごとの判定

| 完了条件 | 判定 | コメント |
| --- | --- | --- |
| 日本語版Minecraft Wiki記事のみ取得 | 部分達成 | 既定値は日本語版だが設定検証なし |
| Raw cacheとmetadata保存 | 部分達成 | 保存はあるがrevision差分再取得なし |
| 第1/第2/Summary Chunk作成 | 部分達成 | raw pipelineでは作成。auto-index正本とは分離 |
| 専用チャンク・検索設定 | 部分達成 | schema/configはあるがruntime検索とkeyword indexに未反映の設定あり |
| Dense/Sparse index投入 | 部分達成 | raw chunk pipeline由来では可能。auto-indexではrepository chunk優先 |
| Minecraft Wiki RAG経路選択 | 部分達成 | `RagService` 経路は可。統合入力経路は別service |
| additional_queries/合成クエリなし | 部分達成 | 回答時は使わないがrouting schemaには残る |
| Dense/通常Sparse/ステミングSparse/RRF/ReRank/Doc Cap/MMR | 未達 | Sparse破棄バグにより通常構成で満たせない |
| ReRank後Doc Cap | 達成 | `RagService` 経路では実装済み |
| 同一チャンネル履歴だけ回答生成に含める | 部分達成 | `RagService` のhistory_scope単位では可。統合入力経路は対象外 |
| Java版前提回答 | 部分達成 | 専用promptにはある。統合入力経路にはない |
| 回答filterなし | 部分達成 | `RagService` 経路では無効。統合入力経路はそもそも別回答器 |
| CLI/外部payload診断情報metadata化 | 部分達成 | top-level方針は達成。sanitizeとsource uriに差分 |
| 主要動作の既存テスト | 部分達成 | connector/service/CLI/configの一部のみ。indexing/retrieval E2Eが不足 |

## 仕様改善点

1. 完全実装の対象入口を明文化する  
   `RagService`、`AskService`、`IntegratedInputUsecase` が併存しているため、Discord `/ask`、HTTP、CLI `chat`、CLI `tool rag` のどれをMinecraft Wiki RAGの正式入口にするかを仕様で固定する。正式入口が複数あるなら、すべて同じMinecraft Wiki RAG serviceを呼ぶことを受け入れ条件にする。

2. index正本を1つに決める  
   raw chunk pipeline成果物を正本にするのか、ingestion repository chunkを正本にするのかを仕様で決める。ingestion repositoryを正本にするなら、第1/第2/Summary/sparse chunk、parent_chunk_id、skip_parent_context、heading_pathをrepository schemaにも保存する必要がある。

3. Minecraft Wiki専用retrieval設定の適用範囲を定義する  
   Dense top_kだけでなく、Sudachi mode、normalized form、remove_symbols、BM25 k1/b、RRF k、sparse混合比率を「index構築時」と「検索時」のどちらへ適用するかを明記する。今の仕様は設定名はあるが、runtime component境界が曖昧。

4. Sparse検索の受け入れテストを仕様に追加する  
   Denseだけでは拾いにくい日本語表記、英語ID、数値を含む小さなfixtureで、通常SparseとステミングSparseの両方がRRFに入ることを必須テストにする。

5. MediaWiki取得の運用仕様を具体化する  
   User-Agent、429/5xx backoff、最大並列数、継続token、revision比較、cache invalidation、resume、失敗ページの扱いをrunbookだけでなく設計にも入れる。

6. Raw cacheの更新判定をrevision id中心にする  
   「rawがあれば再取得しない」ではなく、軽量metadata照会でrevision idを確認し、変更時だけ本文取得する仕様にする。`poll_changes()` のcursor意味も定義する。

7. Wiki記法正規化の最低保持情報を決める  
   infobox、crafting recipe、表、注釈、テンプレート、画像alt、カテゴリをどうMarkdown化するかを例で定義する。Minecraft Wikiでは表が重要なので、単純削除は品質劣化につながる。

8. 出典schemaをlabelとurlに分離する  
   `Source.label` は記事名・見出し、`Source.uri` はcanonical URLまたはsection URLだけにする。CLI payloadでも `sources[].uri` はURLとして機械処理できる必要がある。

9. セクションアンカー仕様を決める  
   日本語MediaWiki見出しのanchor生成は実装差が出やすい。安定生成できない場合は記事URLのみ、と明記するか、MediaWiki APIからsection anchorを取る仕様にする。

10. Edition / Version対象外時の応答方針を具体化する  
    「判定しない」だけだと、ユーザーが統合版差分や特定versionを聞いたときの応答が曖昧になる。Java版前提で答える、または根拠不足として明示する条件を定義する。

11. payload sanitizeを共通関数必須にする  
    RAG tool payloadも `sanitize_payload_metadata()` を通すことを仕様に入れる。大きいcontext、raw LLM出力、secret、内部traceは外部連携前に必ず除外する。

12. 「完全実装」の受け入れテスト一覧をE2E中心にする  
    connector単体だけでなく、fixture raw -> first/second/summary/sparse -> Dense/BM25/keyword -> `tool rag --scope minecraft_wiki` -> payloadまでを1本で検証する。

## 推奨修正順

1. `RetrievalComponent.retrieve()` の無条件 `sparse_hits = []` を修正し、Dense+Sparse同時有効時のRRFテストを追加する。
2. Minecraft Wiki専用retrieval設定を `RetrievalComponent` とkeyword index構築へ渡す。少なくともMinecraft Wiki routeでは専用tokenizer設定を使う。
3. index正本を統一する。短期的にはMinecraft Wiki auto-indexで専用raw chunk pipeline成果物をDense/BM25へ投入する。中期的にはrepository側にstage chunkを保存する。
4. `BuildIndexUsecase` またはCLIにMinecraft Wiki connector ingestionを接続し、手動buildでも初回取得から完結できるようにする。
5. Minecraft Wiki専用Summary ChunkでLLM要約を実行し、失敗時だけfallbackにする。
6. 統合入力の `minecraft_wiki_rag` を `RagService` / `ChatAnswerUsecase` 経路へ接続する。
7. MediaWiki取得にbackoff、revision確認、URL検証を追加する。
8. 出典schemaとCLI sanitizeを整える。

## 検証

実行した検証:

```bash
python3 -m unittest tests.unit.test_minecraft_wiki_rag tests.unit.test_config_loading
app/.venv/bin/python -m unittest tests.unit.test_cli_tool_rag
app/.venv/bin/python -m unittest tests.unit.test_minecraft_wiki_rag tests.unit.test_config_loading tests.unit.test_cli_tool_rag
```

結果:

- `python3` で `tests.unit.test_minecraft_wiki_rag` と `tests.unit.test_config_loading`: 8 tests / OK
- `python3` でCLI payload testを含めた実行: `discord` module未導入でimport error
- `app/.venv/bin/python` で `tests.unit.test_cli_tool_rag`: 3 tests / OK
- `app/.venv/bin/python` で上記3 test moduleまとめ実行: 11 tests / OK

今回の調査で追加した検証は既存テストの実行のみで、実装コードは変更していない。
