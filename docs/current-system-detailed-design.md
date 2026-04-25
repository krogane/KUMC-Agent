# KUMC-Agent 現行機能 詳細設計

作成日: 2026-04-24

> 注意: この文書は Wave 1-7 実装前の現行機能調査をもとにした詳細設計です。
> 2026-04-25 時点の実装済み entrypoint / Wave 機能 / stub の扱いは `README.md` と
> `docs/implementation-consistency-audit.md` を正としてください。

## 1. 目的とスコープ

この文書は、現在のワークスペースに存在する仮実装を本番実装へ置き換えるための詳細設計である。現行コードと同じ外部挙動を再実装できることを目的に、要件、アーキテクチャ、主要データ構造、処理フロー、永続化形式、外部サービス連携、未実装スタブまでを記述する。

調査対象の正本は `src/kumc_agent` 配下の現行実装である。`src/kumc_agent/infra/legacy` は移行前コードとして残っているが、本設計では現行実装から直接呼ばれる非 legacy モジュールを基準とする。なお `features/indexing/service.py` 内には `legacy_cfg` という変数名が残るが、呼び出し先は `src/kumc_agent/infra/indexing/*` の現行モジュールである。

## 2. 現行機能の全体像

KUMC-Agent は、京大マインクラフト同好会 KUMC 向けの Discord/CLI チャットボットである。主機能は次の通り。

- Chat: OpenClaw 優先の会話入口と、ローカル RAG フォールバック。
- Local RAG: 資料検索、追加クエリ検索、資料名指定検索、回答生成、ソース表示。
- Indexing: Discord、Google Drive、はてなブログ、Crafters Colony、X アーカイブ、Notion を raw データ化し、チャンク、dense index、BM25、keyword index、material catalog を生成。
- Eval: `data/eval/ragas.jsonl` を使う answer generation、簡易 exact/token overlap、RAGAS メトリクス評価。
- Discord frontend: `/ai` テキストコマンド、Slash command 解析、index/eval/stop、VC 操作、特殊チャンネル履歴、定期 warmup、自動 index。
- VC: Discord voice meeting の録音/文字起こし/要約/議事録連携機能。既定では無効。
- Warmup: ローカル embedding、reranker、routing、LLM の事前ロード。
- Console REPL と CLI tool bridge。
- HTTP と DocGen は現在 NotImplemented の stub。
- Summarization は単純な文字数切り詰めのみ。

## 3. レイヤ構成

実装は Clean Architecture 風に分割されている。

- `frontends`: Discord、Console、HTTP の入出力層。usecase を呼ぶだけにする。
- `usecases`: ユースケース単位の orchestration。Chat、Index、Eval、VC、Warmup など。
- `features`: 機能別サービス層。RAG、Indexing、VC、Summarization、DocGen。
- `domain`: 外部 SDK 非依存のデータモデル、ports、policies。
- `infra`: Google/Discord/OpenClaw/Gemini/llama.cpp/FAISS/BM25/storage などの外部依存実装。
- `runtime`: DI と起動時 wiring。
- `config`: YAML/env/experiment の設定ロード。

アプリ起動時は `kumc_agent.runtime.container.build_runtime_context()` が全依存を構築し、`RuntimeContext` に usecase を詰める。全フロントエンドはこの `RuntimeContext` を通じて機能へアクセスする。

## 4. 主要エントリポイント

CLI は `src/kumc_agent/cli.py` の argparse で提供される。

- `kumc-agent repl`: Console REPL を起動する。
- `kumc-agent chat --query "..."`
  - `ChatEntryUsecase` を実行し、回答本文だけを stdout に出す。
- `kumc-agent tool rag --query "..."`
  - OpenClaw から呼ばれるローカル RAG bridge。
  - 単一 query では JSON object、複数 `--query` では `{"query_count": n, "results": [...]}` を出す。
  - 出力には `answer`, `route`, `sources`, `routing_decision`, `fast_mode`, `metadata` を含める。
  - `metadata.contexts` は bridge 出力から除外する。
- `kumc-agent index build`
  - raw source refresh と index build。
  - `--no-refresh-sources`, `--full-rebuild`, `--stage <name>` を受け付ける。
- `kumc-agent index update`
  - 現状は build と同じ処理を再利用し、index artifacts を上書きする。
- `kumc-agent eval ragas`
  - RAGAS 評価を実行する。
  - `--eval-file`, `--limit`, `--result-path`, `--ragas-*`, `--answer-cache-path`, `--disable-answer-cache`, `--refresh-answer-cache`, `--disable-history-for-eval` を受け付ける。
- `kumc-agent discord`
  - Discord frontend を起動する。
- `kumc-agent http`
  - 現状は `NotImplementedError`。

Python module 実行も想定される。

- `PYTHONPATH=src python -m kumc_agent.cli ...`
- `PYTHONPATH=src python -m kumc_agent.frontends.discord.app`
- `PYTHONPATH=src python -m kumc_agent.frontends.http.app`

## 5. 設定設計

設定は `load_runtime_config()` が次の順で構築する。

1. `configs/ops/*.yaml` を固定順で deep merge。
2. `.env` を読み、`config/env_map.py` の `ENV_BINDINGS` で上書き。
3. `KUMC_EXPERIMENT_PROFILE` に対応する `configs/experiments/**/*.yaml` を未知キー禁止で deep merge。

ops ファイルのロード順は固定である。

- `app.yaml`
- `providers.yaml`
- `security.yaml`
- `scheduler.yaml`
- `features.yaml`
- `model.yaml`
- `vc.yaml`

マージ仕様:

- dict は deep merge。
- scalar は後勝ち。
- list は完全置換。
- experiment 側の未知キーは起動エラー。
- env が未知 path を指定した場合も起動エラー。
- YAML パーサとして PyYAML があれば使用し、なければ限定的な fallback parser を使う。

必須に近い環境変数:

- `KUMC_DISCORD_BOT_TOKEN`: Discord bot 起動時に必須。
- `KUMC_GEMINI_API_KEY`: Gemini LLM/embedding/routing を使う場合に必要。
- `KUMC_DRIVE_FOLDER_ID`: Drive loader を動かす場合に必要。
- `KUMC_GOOGLE_APPLICATION_CREDENTIALS`: Service Account 明示時に使用。未指定なら Google ADC。
- `KUMC_OPENCLAW_ENABLED`, `KUMC_OPENCLAW_AGENT`, `KUMC_OPENCLAW_MODEL`, `KUMC_OPENCLAW_LITE_AGENT`, `KUMC_OPENCLAW_LITE_MODEL`: OpenClaw 入口。
- `KUMC_OPENAI_API_KEY`: OpenClaw が OpenAI provider を使う場合に環境へ bridge される。
- `KUMC_EXPERIMENT_PROFILE`: 既定は `rag/baseline`。

`.env` と `.env.example` は項目の追加/削除を必ず同期する必要がある。

## 6. Domain モデル

再実装時は次の構造を維持する。

```python
Answer(
    text: str,
    route: str,
    sources: list[Source],
    metadata: dict[str, object],
)

Source(id: str, label: str, uri: str = "")

Chunk(
    id: str,
    document_id: str,
    text: str,
    index: int,
    metadata: dict[str, object],
)

Document(
    id: str,
    text: str,
    source_type: str,
    source_name: str,
    source_uri: str = "",
    updated_at: datetime | None = None,
    metadata: dict[str, object] = {},
)

RoutingDecision(
    recency_mode: "off" | "soft" | "hard",
    material_names: list[str],
    include_capabilities_info: bool,
    use_additional_memory: bool,
    additional_queries: list[str],
)

EntryRoutingDecision(
    route: "direct_rag" | "openclaw",
    reason: str,
    payload: dict[str, object],
)
```

`format_sources()` の出力仕様:

- sources が空、または uri が空なら空文字。
- uri は重複排除。
- 既定では免責文 `※回答は必ずしも正しいとは限りません。重要な情報は確認するようにしてください。` を付ける。
- 最終形は `主な情報源:` の bullet list。

## 7. Chat 入口の詳細

### 7.1 ChatEntryUsecase

`ChatEntryUsecase.execute()` はユーザー query の最上位入口である。

処理:

1. query を trim。空なら `Answer(text="", route="none", metadata={"reason": "empty_query"})`。
2. OpenClaw が無効なら `ChatAnswerUsecase` へ直接渡す。
3. OpenClaw が有効なら `EntryQueryRouter.decide()` で `direct_rag` か `openclaw` を分類する。
4. 分類器が例外なら `route="openclaw"`, `reason="fallback:classifier_error"` とする。
5. `direct_rag` ならローカル RAG を実行し、entry routing metadata を付けて返す。
6. `openclaw` なら `OpenClawClient.run_turn()` を呼ぶ。
7. OpenClaw が成功したら payload から `Answer` を復元する。
8. OpenClaw が失敗したらローカル RAG へ fallback する。この場合 `disable_history=True` にし、metadata に `openclaw_fallback=True` を入れる。

OpenClaw 成功時の Answer 化:

- `route="openclaw"`。
- payload の `sources` は `{id,label,uri}` list として Source に変換。
- payload の `metadata` を基礎 metadata にする。
- `fastmode` がある場合は `fast_mode` へ正規化。
- `rag_query`, `rag_iterations` が top-level にあれば metadata に補完。
- `openclaw_payload` に route/routing_decision を除いた payload を保存。
- `openclaw_session_id` に session id を保存。
- `append_sources_to_response=True` かつ本文に `主な情報源:` がない場合だけ source list を追記。

entry routing metadata:

- `entry_route`
- `entry_route_reason`
- `entry_route_model`
- `entry_route_fallback`
- `entry_route_payload`

### 7.2 EntryQueryRouter

OpenClaw 有効時の入口分類器である。prompt は既定 `assets/prompts/routing_openclaw_gate.md`、なければ fallback prompt を使う。

出力 JSON schema:

```json
{"route":"direct_rag|openclaw","reason":"判定理由"}
```

分類方針:

- `direct_rag`: サークル関連の事実照会、資料名指定系の質問。
- `openclaw`: 複雑な質問、サークル関連以外、ツール実行依頼、文章生成依頼。

Gemini provider は `response_mime_type="application/json"` を指定する。llama.cpp provider は JSON schema grammar を使う。retry 対象は 429/5xx、rate limit、timeout、overloaded 等の一時エラー。最終的に parse 不能なら `route="openclaw"`, `reason="fallback:classification_failed"`。

## 8. Local RAG の詳細

### 8.1 ChatAnswerUsecase と RagService

`ChatAnswerUsecase` は `RagService.answer()` の薄い wrapper である。

`RagService.answer()` の主フロー:

1. query trim。空なら none answer。
2. routing 用履歴を決定。
   - override があればそれを使用。
   - `disable_history=True` なら空。
   - 通常は scope ごとの in-memory history から `prompt_default_turns` 件。
3. `QueryRouter.route()` を実行。
4. `force_disable_additional_memory=True` なら `use_additional_memory=False` に上書き。
5. `force_fast_mode=True` なら `material_names=[]`, `additional_queries=[]` に上書き。
6. generation 用履歴を決定。
   - `use_additional_memory=True` なら `prompt_additional_turns` 件。
   - それ以外は `prompt_default_turns` 件。
7. reranker runtime を必要なら事前ロード。
8. recency mode を `decision.recency_mode` または config から解決。
9. `material_names` があれば資料名指定検索、なければ通常検索。
10. rerank、recency score、parent cap、MMR、parent chunk 追加を実行。
11. chunks が空なら no-RAG 生成。
12. chunks があれば RAG 生成。資料名指定検索なら route を `material_search` にする。
13. `_finalize_answer()` で routing metadata 付与、fast mode notice 追記、会話履歴保存を行う。

履歴は process-local な `dict[str, deque]` で保持し、永続化されない。scope は Discord では `guild:<guild_id>`、CLI/OpenClaw では指定値または default。

`force_fast_mode=True` の場合、最終回答本文の先頭に `rag.fast_model_notice` を空行区切りで付ける。

### 8.2 QueryRouter

QueryRouter は RAG 内の検索方針を JSON task として並列判定する。現在有効な task は次の 4 つ。

- `use_additional_memory`: 追加履歴を使うか。
- `additional_queries`: RAG 用の追加検索 query list。
- `material_names`: 資料名指定検索で使う資料名 list。
- `recency_mode`: `off` / `soft` / `hard`。

`include_capabilities_info` と `target_model` の config は残っているが、現行の `_ROUTING_TASK_NAMES` には含まれず、通常の task 実行対象ではない。`RoutingDecision.include_capabilities_info` は既定 `False` のままになる。

task は `ThreadPoolExecutor` で並列実行する。各 task は config の provider/model/prompt を個別に持てる。prompt は `assets/prompts/*.md` から読み、該当 task の field/rule だけを抽出して system prompt を構成する。`{today_label}` は Asia/Tokyo の現在日付に展開し、`{material_search_max_names}` は設定値に展開する。

routing が disabled、または routing 全体が失敗した場合の safe default:

```python
RoutingDecision(
    recency_mode="off",
    material_names=[],
    include_capabilities_info=False,
    use_additional_memory=False,
    additional_queries=[],
)
```

### 8.3 RetrievalComponent

検索は dense と sparse を組み合わせる。

Dense:

- query を `EmbedderPort.embed_query()`。
- `FaissLikeIndex.search()`。
- FAISS が使えない場合は NumPy cosine similarity fallback。

Sparse:

- `SudachiBM25Retriever` による BM25。
- さらに keyword inverted index がある場合、`sparse_second_rec` と `second_rec_sparse` の混合検索を優先する。
- sparse query tokenization は `SparseNormalizer` を使い、失敗時は BM25 retriever の tokenizer に fallback。

Score merge:

- dense score と sparse score をそれぞれ min-max normalize。
- 同一 chunk id は統合。
- 合計 score 降順。

MMR:

- rerank 後の chunk list に対して embedding を再計算し、多様性を加味して並べ替える。
- 最初の 3 件は固定で残し、以降を MMR で選ぶ。
- `mmr_lambda >= 0.999` の場合は並べ替えなし。

### 8.4 Rerank と recency

CrossEncoder が有効かつ `force_fast_mode=False` の場合だけ rerank する。rerank score に recency score を線形合成する。

recency 対象 metadata key:

- `updated_at`
- `message_timestamp`
- `drive_modified_time`
- `hatenablog_updated_at`
- `hatenablog_created_at`
- `crafters_colony_published_at`
- `notion_last_edited_time`
- `notion_created_time`
- `first_message_date`
- `source_date`
- `created_at`

recency score は半減期 `recency_half_life_days` で `0.5 ** (age_days / half_life_days)`。日付が取れない場合は `0.5`。`soft` は `recency_weight_soft`、`hard` は `recency_weight_hard`、`off` は 0。

### 8.5 Parent context

`parent_doc_enabled=True` の場合、選択された chunk の親文脈を追加する。

- proposition chunk は second chunk の parent をたどって first chunk または summary chunk を追加。
- second_recursive chunk は first chunk または summary chunk を追加。
- `skip_parent_context` metadata が truthy なら親追加しない。
- 親単位の重複制御として `parent_chunk_cap` を適用する。

### 8.6 資料名指定検索

`decision.material_names` が空でない場合、`material_search` 経路になる。

対象外 source type:

- `messages`
- `discord_message`
- `x_posts`

検索手順:

1. `data/index/material_catalog.json` を読む。
2. 資料名を NFKC、casefold、区切り文字除去、日付正規化で variant 化する。
3. canonical name / aliases と strict match。
4. strict match がなければ partial match。
5. partial match が複数なら semantic retrieval で最上位資料を選ぶ。
6. match がなければ資料名に対する dense fallback。
7. match した資料に限定して retrieval。
8. sparse retrieval が空なら raw file full text から context chunk を構成。
9. それも失敗したら通常 RAG に fallback。

資料名指定検索では `force_all_sources=True` で source selection を扱うが、最終的な source 数は `source_max_count` を超えない。

### 8.7 GenerationComponent

RAG/no-RAG/refusal の 3 種類の生成器を持つ。provider ごとの prompt header は config の `rag.prompt_texts` で変えられる。

RAG prompt の構成:

1. 質問
2. チャット履歴
3. サークル基本情報 `assets/prompts/circle_basic_info.md`
4. コンテキスト
5. 必要な場合のみ capability info
6. extra mode instruction
7. 出力形式 prompt `assets/prompts/answer_rag.md`

no-RAG prompt は circle basic info と context を含めない。refusal prompt は固定 prefix `安全上の理由により、この質問には回答できません。` を必ず付ける。ただし現行 `RagService.answer()` から `generate_refusal()` を呼ぶ分岐は存在せず、`security.refusal_keywords` も `QueryRouter` に渡されるだけで現在の routing task では使用されない。refusal generator と refusal profile は構築・warmup 対象だが、通常チャット経路からは到達しない。

LLM 出力は JSON object を期待する。

```json
{
  "answer": "回答本文",
  "sources": [1, "2", "3-1"]
}
```

parse 仕様:

- code fence を除去。
- JSON object または JSON string 内の object を読む。
- 前後にテキストがある場合は最初の `{` から最後の `}` を読む。
- malformed/truncated JSON でも `"answer": "..."` の断片を復旧する。
- retry は `answer_json_max_retries` 回。
- 最後まで JSON として読めない場合は best effort answer または raw text。

Discord context の subsource:

- Discord chunk は context 内で `[source-sub]` を各メッセージ行へ付与する。
- source selection `"1-2"` は chunk 1 の 2 番目メッセージを指す。
- raw messages から message id を復元できれば `https://discord.com/channels/{guild}/{channel}/{message}` を source にする。

source URL 優先順位:

1. Discord message URL
2. X post URL
3. はてなブログ URL
4. Crafters Colony article URL
5. Notion URL
6. Google Drive URL
7. VC meeting label

生成後、Discord mention は `（メンション非表示）` へ mask する。

## 9. OpenClaw 連携

`OpenClawClient` は `openclaw` CLI を subprocess 実行する wrapper である。

主な仕様:

- `enabled=False` なら disabled failure。
- query 空なら empty_query failure。
- `force_fast_mode` が user context にある場合、lite agent/model を使う。
- model 名は `gemini-*` を `google/gemini-*`、`gpt-*` を `openai/gpt-*` に正規化。
- `configs/openclaw` 配下の `AGENTS.md`, `SOUL.md`, `USER.md`, `skills/**` を OpenClaw 側に同期する。
- skills が存在する場合は skills revision を session id に付与し、skill 更新時に別 session として扱う。
- agent が存在しないエラーなら default agent で retry。
- gateway unavailable など特定条件では local mode retry がある。
- stdout から JSON line、nested response、assistant messages、embedded JSON payload を復元して `OpenClawTurnResult(text, payload)` にする。
- `DEBUG` truthy の場合、`logs/openclaw_trace.jsonl` または `KUMC_OPENCLAW_TRACE_LOG_PATH` に trace を書く。
- trace は email、OpenAI key、Google key、Discord token、Bearer、token/password/api_key 系 key を redact する。
- OpenClaw 実行環境には project root、`PYTHONPATH=src`、KUMC/Gemini/OpenAI key を bridge する。

## 10. Indexing 詳細

### 10.1 BuildIndexUsecase

`BuildIndexUsecase.execute()` は `refresh_sources=True` のとき次の loader を順に実行し、戻り件数を合算する。

1. Discord
2. Google Drive
3. HatenaBlog
4. Crafters Colony
5. X
6. Notion

その後 `IndexingService.build()` を呼ぶ。

`UpdateIndexUsecase` は現状 `BuildIndexUsecase` の再利用で、差分 index 専用ロジックはない。

### 10.2 IndexingService.build()

処理順:

1. `full_rebuild` または refresh clear flags に従って raw/chunk dirs を削除。
2. raw source dir を作成。
3. raw から `Document` を parse し、`data/raw/documents.jsonl` に保存。
4. chunk pipeline を実行。
5. index 対象 chunk を stage dir から読み、domain `Chunk` へ変換。
6. `data/chunks/chunks.jsonl` に保存。
7. dense embedding text を生成して `embed_documents()`。
8. dense FAISS/NumPy artifacts を保存。
9. BM25 artifacts を保存。
10. keyword inverted indexes を保存。
11. material catalog を保存。

戻り値:

```python
IndexBuildResult(
    loaded_sources: int,
    documents: int,
    chunks: int,
    index_dir: Path,
)
```

### 10.3 raw データ配置

既定 base:

- `data/raw/docs`: Google Docs/Slides/PDF/Word/PowerPoint などを markdown/text 化。
- `data/raw/sheets`: Google Sheets/Excel を CSV 化。
- `data/raw/messages`: Discord message JSONL。
- `data/raw/x`: X archive `tweets*.js` と変換済み `posts.jsonl`。
- `data/raw/vc`: VC transcript txt。
- `data/raw/hatenablog`: はてなブログ記事 markdown。
- `data/raw/crafters_colony`: Crafters Colony 記事 markdown。
- `data/raw/notion`: Notion page markdown。

Drive/Blog/Notion などは `.meta.json` sidecar を持つ。sidecar には file id、title、URL、modified time、source date などを保存する。

### 10.4 chunk pipeline

stage 名と出力先:

- `first_recursive`: `data/chunks/first_rec_chunk/{source_type}/*.jsonl`
- `second_recursive`: `data/chunks/second_rec_chunk/{source_type}/*.jsonl`
- `sparse_second_recursive`: `data/chunks/sparse_second_rec_chunk/{source_type}/*.jsonl`
- `summary`: `data/chunks/summary_chunk/{source_type}/*.jsonl`
- `proposition`: `data/chunks/prop_chunk/{source_type}/*.jsonl`
- `raptor`: `data/chunks/raptor_chunk/*.jsonl`

既定設定:

- first recursive: size 1024, overlap 128。
- second recursive: size 384, overlap 64。
- summary: target 200 chars, batch size 2000。
- proposition: 既定 disabled。
- raptor: 既定 disabled。
- sparse second recursive: 既定 enabled。

separator:

- docs/blog/notion: `\n## `, `\n### `, blank line, newline, space, empty。
- sheets: `\n|`, blank line, newline, space, empty。
- messages/x/vc: newline, space, empty。

skip/update/delete sync:

- 各 output `.jsonl` には `.jsonl.mtime.json` sidecar があり、source path、source mtime、output mtime、updated_at を保存。
- `skip_existing=True` かつ `update_existing=True` の場合、source mtime が更新されていない output は再生成しない。
- `sync_deleted=True` の場合、source がなくなった output と sidecar を削除する。

chunk metadata の主要 key:

- `source_file_name`
- `source_type`
- `source_date`
- `updated_at`
- `meeting_date`
- `meeting_label`
- `guild_id`, `guild_name`, `category_id`, `category_name`, `channel_id`, `channel_name`
- `first_message_id`, `first_message_date`
- `drive_file_name`, `drive_mime_type`, `drive_file_path`, `drive_file_id`
- `hatenablog_title`, `hatenablog_created_at`, `hatenablog_url`
- `crafters_colony_title`, `crafters_colony_published_at`, `crafters_colony_article_url`
- `notion_database_id`, `notion_page_id`, `notion_title`, `notion_url`, `notion_created_time`, `notion_last_edited_time`
- `x_author_handle`
- `chunk_stage`
- `chunk_id`
- `parent_chunk_id`
- `skip_parent_context`
- `chunk_uid`

### 10.5 index artifacts

`data/index` に保存する。

Dense:

- `dense_vectors.npy`
- `dense_vectors.faiss`。FAISS が使えない場合は存在しないか無視される。
- `dense_chunks.jsonl`

BM25:

- `bm25_tokens.json`
- `bm25_chunks.jsonl`

Keyword index:

- `keyword/sparse.json`
- `keyword/sparse_second_rec.json`
- `keyword/second_rec_sparse.json`

Material catalog:

- `material_catalog.json`
- schema version 1。
- root: `schema_version`, `created_at`, `materials`。
- material: `material_id`, `source_type`, `source_key`, `canonical_name`, `aliases`, `raw_path`。

### 10.6 Source loader

Google Drive:

- Folder を再帰的に走査する。
- 対象 MIME:
  - Google Docs
  - Google Sheets
  - Google Slides
  - PDF
  - Word
  - Excel
  - PowerPoint
- Drive API は shared drives 対応。
- `max_files > 0` の場合は取得件数上限。
- batch size は list を分割するための設定。
- export/download は 429/5xx/timeout/connection 系を指数 backoff retry。
- Google Docs/Slides は export を使う。Slides は export size limit の場合 text/plain fallback。
- Office OpenXML は zip/xml を直接 parse して text/CSV 化する fallback を持つ。
- PDF は PyMuPDF/Pillow/OCR runner を使う。PP-OCRv5 mobile は det/rec model dir を優先し、なければ direct vision OCR runner を試す。
- docs は `data/raw/docs`、sheets は `data/raw/sheets` へ保存する。

Discord:

- Bot token が空なら何もしない。
- guild allow list が空でなければ対象 guild を制限。
- text channel と thread、archived thread を対象にする。
- bot/webhook/system message は除外。
- default/reply message のみ許可。
- URL は本文から削除。
- user mention は可能な限り display name に置換。
- channel ごとに `data/raw/messages/{guild_id}/{channel_id}.jsonl` へ append。
- channel ごとに `{channel_id}.state.json` を持ち、last message id/timestamp 以降だけ追加取得。

HatenaBlog:

- 既定 URL は `https://kumc.hatenablog.com/`。
- Atom feed を最大 200 page まで辿る。
- article は markdown/text 化し、metadata sidecar に entry id/title/url/created/updated を保存。
- 更新日時が変わらなければ skip。
- sync_deleted で削除済み記事の出力を削除。

Crafters Colony:

- author URL が空なら何もしない。
- author page から article URL を BFS 的に収集。
- `max_pages`, `max_articles` を適用。
- HTML から article body を抽出し markdown/text 化。
- title/published_at/url を sidecar に保存。

X:

- `data/raw/x/tweets*.js` を読む。
- `window.YTD.tweets.partN = [...]` 形式を JSONDecoder で抜き出す。
- tweet id、full_text/text、created_at、status URL から handle を抽出。
- `data/raw/x/posts.jsonl` へ message 互換 JSONL として保存。
- output が input より新しければ skip。

Notion:

- `features.sources.notion=True` の場合のみ loader を構築。
- token または database id が空なら何もしない。
- Notion API version は `2022-06-28`。
- database query で page を列挙し、archived/in_trash は除外。
- block children を再帰取得して markdown 化。
- page id/title/url/created/last edited を sidecar に保存。
- last edited time が同じなら skip。

## 11. Discord frontend

### 11.1 起動

`frontends/discord/app.py` は `discord.Client` を直接使う。intents:

- default
- `message_content=True`
- `voice_states=True`

起動時:

1. RuntimeContext を構築。
2. logging を設定。
3. VC usecase に discord client と indexing active callback を bind。
4. `on_ready` で VC start。
5. startup warmup を強制実行。
6. auto index loop と periodic warmup loop を開始。

### 11.2 メッセージ起動条件

query として扱う条件:

- message が bot 自身または bot author なら無視。
- bot mention prefix の後ろを query とする。
- prefix `/ai` で始まる場合は prefix 後ろを query。
- DM では message 全体を query。
- guild channel で channel name が `rag.history.special_channel_names` に含まれる場合は message 全体を query。
- それ以外は無視。

`fast` prefix:

- `/ai fast 質問` または query 先頭 `fast` で `force_fast_mode=True`。
- `fast` だけなら query 空として無視。

入力長:

- `app.max_input_characters > 0` かつ query 長超過ならエラーメッセージを返す。

同一 channel 並列:

- channel ごとに回答生成 task は 1 つだけ。
- 既に実行中なら `/ai stop` を案内。

### 11.3 コマンド

通常 message:

- `/ai <query>`: chat。
- `/ai fast <query>`: fast mode chat。
- `/ai build-index`: index build。maintenance author のみ。
- `/ai eval`: eval。maintenance author のみ。
- `/ai stop`: 現 channel の回答生成、index、eval を cancel。
- `/ai join`: VC chat channel で join。
- `/ai quit`: VC chat channel で quit。

Slash interaction:

- `/ai build-index`
- `/ai eval`
- `/ai stop`
- `/ai join`, `/ai quit` は通常 message を案内するだけ。
- chat は通常 message を案内するだけ。

OpenClaw enabled 時:

- Discord frontend は build/eval/stop を処理しない。
- chat は OpenClaw 経由に縮退する。
- VC sidecar と auto index loop は残る。

### 11.4 特殊チャンネル履歴

設定 `rag.history.special_channel_names` に含まれる channel では、message 全体を query とする。

現行実装では special channel ではない場合、逆に `_collect_special_channel_history()` を実行して reply chain を含む履歴を集め、routing/generation history override として渡す。さらに `force_disable_additional_memory=True`, `append_sources_to_response=False`, `extra_mode_instruction=special_channel_custom_instruction` を指定する。この挙動も再実装時に維持する必要がある。

### 11.5 index/eval/warmup の排他

- indexing 中は新規 query を受け付けず、開始時刻と終了目安を返す。
- auto index は設定曜日/時刻に 1 日 1 回だけ実行。
- VC active session 中は auto index と手動 index を開始しない。
- warmup は indexing/eval/回答生成/VC model activity 中なら skip。
- index 完了後は warmup を force 実行。

### 11.6 ログ

回答ログ:

- `ops.answer_record_log_enabled=True` の場合、`logs/answer_records.jsonl` に追記。
- fields: timestamp, questioner_user_id, questioner_username, question, routing_result, answer。

prompt ログ:

- 同条件で `logs/answer_prompts.jsonl` に追記。
- LLM prompt metadata がある場合のみ。
- fields: timestamp, questioner, route, routing_result, system_prompt, user_prompt。

## 12. Eval

`EvaluateRagasUsecase` は JSONL eval file を読む。1 行 1 object。

入力 key:

- question は `question` 優先、なければ `query`。
- ground truth は `ground_truths` list 優先、なければ `ground_truth`。

処理:

1. `limit` があれば切り詰め。
2. answer cache を読む。key は question hash。
3. 未cache query は `ChatAnswerUsecase` で回答生成。
4. 生成時は既定で履歴を無効化し、`history_scope="__eval__:<index>"`, routing/generation history 空、additional memory 無効。
5. `append_sources_to_response=False`。
6. answer, contexts, ground_truths を records にする。
7. exact match は truth が answer に substring として含まれるか。
8. token overlap は lower split token の truth overlap 最大値。
9. RAGAS を実行可能なら実行。
10. result path があれば JSON 保存。

RAGAS:

- metrics: answer_relevancy, faithfulness, context_precision, context_recall。
- 個別 toggle は config/env で制御。
- Gemini evaluator LLM と embedding を構築する。
- 依存がなければ RAGAS metrics は skip し、metadata に `skipped_reason="dependency_missing"`。
- batch size/max workers/timeout/retry を設定できる。
- single pass 評価が失敗または数値 metrics なしなら chunked fallback。
- cancel event が立った場合は canceled metadata を返す。

answer cache:

- 既定 path: `data/eval/cache/ragas_answers.jsonl`。
- record: `question_hash`, `question`, `answer`, `contexts`。
- `refresh_answer_cache=True` なら既存 cache を読まず再生成。

## 13. VC

VC は `features.vc` と `vc.feature_enabled` が両方 true の場合に有効。既定では無効。

`VCUsecase` は `VCService` の wrapper で、Discord client bind 後に `infra/vc/manager.VoiceMeetingManager` を遅延生成する。

公開 API:

- `start()`, `stop()`
- `on_voice_state_update(member, before, after)`
- `capture_voice_chat_message(message)`
- `maybe_join_from_command(message)`
- `maybe_quit_from_command(message)`
- `has_active_session()`
- `has_model_activity()`
- `is_voice_chat_channel(channel)`
- `should_use_fast_model_for_query()`
- `notify_rag_started()`
- `notify_rag_finished()`

設定:

- auto join: weekday/time/duration/target voice channel/min participants。
- participant check interval。
- transcription interval/model/device/dtype/language。
- auto quit。
- final summary。
- previous summary max と target characters。
- summary LLM provider/model/path/temperature/tokens。
- minutes: Google Drive dir、fetch/apply/LLM retry、image batch size、edit LLM。

VC 中は:

- 新規 index 開始をブロック。
- RAG query では fast model を使う判断が入る。
- warmup skip 条件に model activity が入る。

## 14. Warmup

Warmup はローカル runtime の事前ロード用である。Gemini provider は外部 API のため skip する。

step:

- `embedding`: provider が `local` で model がある場合。
- `cross_encoder_reranker`: reranker enabled かつ model がある場合。
- `routing_function_calling`: routing task に llama/llama_cpp provider があり model path がある場合。
- `answer_llm`: RAG generation provider が llama/llama_cpp の場合。
- `no_rag_llm`: no-RAG generation provider が llama/llama_cpp の場合。
- `refusal_llm`: refusal generation provider が llama/llama_cpp の場合。

各 step は個別に completed/skipped/failed を記録し、失敗しても後続 step を止めない。

## 15. Summarization, DocGen, HTTP

Summarization:

- `SummarizationService.summarize(text)` は trim 後、`target_characters` 以下ならそのまま返す。
- 超過なら先頭 `target_characters` 文字に切り、末尾空白を rstrip して `...` を付ける。
- LLM 要約ではない。

DocGen:

- `DocGenService.run()` と `DocGenUsecase.execute()` は `NotImplementedError`。

HTTP:

- `frontends/http/app.py` は `NotImplementedError("HTTP frontend is not implemented in this release.")`。

これらは「未実装であること」自体が現行挙動である。

## 16. 外部依存

主要依存:

- Discord: `discord.py`, `discord-ext-voice-recv`, `PyNaCl`
- Google: `google-api-python-client`, `google-auth`, `google-genai`
- RAG/index: `langchain-*`, `sentence-transformers`, `torch`, `faiss-cpu`, `numpy`, `sudachipy`, `sudachidict_core`, `rank-bm25`
- LLM local: `llama-cpp-python`
- Eval: `ragas`, `datasets`
- OCR/PDF: `PyMuPDF`, `Pillow`, `paddleocr`, `paddlepaddle`, `transformers`, `einops`
- Config: `python-dotenv`, `PyYAML`
- Test: `pytest` は requirements にあるが、README では `python -m unittest discover tests` が案内されている。

OpenClaw CLI は pip 管理ではなく npm/global で別管理される。

## 17. 本番再実装時の受け入れ条件

最低限、次の挙動互換を満たすこと。

- CLI の subcommand/options/stdout JSON 形状が現行と一致する。
- Discord の `/ai`, `/ai fast`, `/ai build-index`, `/ai eval`, `/ai stop`, `/ai join`, `/ai quit` の挙動と権限制御が一致する。
- OpenClaw enabled/disabled、entry route、OpenClaw failure fallback が一致する。
- Local RAG の route 名 `rag`, `no_rag`, `material_search`, `none`, `openclaw` が一致する。`refusal` route は生成器としては存在するが、現行通常チャット経路からは到達しない点も維持する。
- Answer metadata に routing/openclaw/contexts/llm_prompt/source selections が必要な場所で残る。
- source list 追記、免責文、`主な情報源:` の形式が一致する。
- index build の raw/chunk/index/material catalog artifact 形式が一致する。
- Drive/Discord/Blog/Crafters/X/Notion の skip/update/sync_deleted 方針が一致する。
- RAGAS eval の cache、metrics toggle、fallback、cancel metadata が一致する。
- HTTP/DocGen は現状どおり NotImplemented。
- VC disabled 既定で no-op、enabled 時の manager API 境界が一致する。

## 18. 記述漏れしやすい箇所の確認結果

次の観点は漏れやすいため、調査後に個別確認した。

- OpenClaw 優先入口とローカル RAG fallback の両方を記述済み。
- OpenClaw 成功 payload から Answer へ戻す metadata/source 正規化を記述済み。
- `tool rag` の単一/複数 query JSON 形状を記述済み。
- QueryRouter の現行 task が 4 つのみで、config に残る `include_capabilities_info` が通常実行されない点を記述済み。
- `security.refusal_keywords` と refusal generator は存在するが、通常チャットで refusal 分岐が未接続である点を記述済み。
- `force_fast_mode` が material/additional queries を消し、reranker/MMR 系の一部を軽くする点を記述済み。
- 資料名指定検索で Discord/X を除外する点を記述済み。
- parent chunk 追加、summary 優先、`skip_parent_context` を記述済み。
- Discord source selection `"1-2"` と message URL 復元を記述済み。
- source 表示の免責文と重複抑制を記述済み。
- index の stage selection、mtime sidecar、sync_deleted を記述済み。
- raw/chunk/index/material catalog のファイル配置と schema を記述済み。
- Google Drive の Slides export fallback、PDF OCR、Office XML fallback を記述済み。
- Discord loader の state file と append 増分取得を記述済み。
- X archive の `tweets*.js` 変換を記述済み。
- Notion が既定 disabled である点を記述済み。
- Discord frontend の特殊チャンネル履歴まわりの現行挙動を記述済み。
- indexing/eval/answer generation の cancel と排他を記述済み。
- answer record と prompt log を記述済み。
- RAGAS の answer cache、履歴無効化、dependency missing skip を記述済み。
- VC が既定 disabled で、Discord 起動時に bind される点を記述済み。
- Warmup が local provider のみ対象で、Gemini は skip する点を記述済み。
- HTTP と DocGen は未実装 stub が仕様である点を記述済み。
