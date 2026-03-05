# KUMC-Agent
KUMCが保有する情報をGoogle Drive・Discord・はてなブログから収集し、RAGで回答するDiscordボットです。
ローカルの`llama.cpp`またはGeminiを使った回答生成に対応し、Index構築・評価（ragas）まで一通りの流れが揃っています。

## 主な機能
- Google DriveのDocs/Sheets/Slides/Word/Excel/PowerPoint/PDFを取得し、テキスト化して保存（Office Open XML: `.docx/.xlsx/.pptx`、PDFはPyMuPDF＋OCR対応、更新差分を判定）
- Discordメッセージログの取得（Botトークンがある場合、ギルド絞り込み可）
- はてなブログ（`https://kumc.hatenablog.com/`）の記事取得（更新差分を判定）
- Chunking: First/Second Recursive、Summery（要約）、Proposition、RAPTOR
- 検索: FAISSのDense + BM25（Sudachi）のSparse、Cross-Encoder再ランク、MMR多様化
- ルーティング: Function Callingで `RAG / No-RAG / Refusal` を自動切替
- Function Callingでメモリ参照・モデル情報挿入・検索クエリ変換・機密判定を実施
- Drive/DiscordのソースURLを回答末尾に付与
- 自動インデックス更新（スケジュール）

## データフロー
### Indexing
1. (任意) 既存データのクリア
2. Discordメッセージ取得（`DISCORD_BOT_TOKEN` がある場合）
3. Google DriveのDocs/Sheets/Word/Excel/Slides/PowerPoint/PDF取得（`app/data/raw`、Officeは`.docx/.xlsx/.pptx`）
4. はてなブログ記事取得（`app/data/raw/hatenablog`）
5. First Recursive Chunking（`app/data/first_rec_chunk`）
6. (任意) Second Recursive Chunking（`app/data/second_rec_chunk`）
7. (任意) Summery Chunking（`app/data/summery_chunk`）
8. (任意) Proposition Chunking（`app/data/prop_chunk`、Second Rec必須）
9. (任意) RAPTOR要約（`app/data/raptor_chunk`）
10. すべてのChunkをEmbeddingしてFAISS索引を作成（`app/data/index`）

### Query
1. Function Callingで回答経路（`rag` / `no_rag` / `refusal`）と各フラグを判定（無効化可）
2. `rag`: originalクエリ + 変換クエリ（最大3）で Dense + Sparse 検索
3. Cross-Encoderで再ランク + recency補正、Parent doc補完、MMRで最終選択
4. 回答LLMで回答生成（JSON出力 → パース）
5. `refusal`: 固定拒否文 + 拒否LLM補足（任意）
6. 回答にソースURLを付与（Drive/Discord、refusalは除く）

## ディレクトリ構成
- `app/src/main.py`: Discordボットのエントリポイント
- `app/src/config.py`: 設定/環境変数の集約
- `app/src/pipeline/`: RAG推論パイプライン
- `app/src/indexing/`: Drive/Discord取得・Chunking・RAPTOR・FAISS
- `app/src/eval/`: Ragas評価スクリプト
- `app/data/raw`: 取得データ（docs/sheets/messages/hatenablog）
- `app/data/first_rec_chunk`, `second_rec_chunk`, `summery_chunk`, `prop_chunk`, `raptor_chunk`
- `app/data/index`: FAISS索引
- `app/data/eval`: 評価データと結果
- `app/model/`: ローカルモデル配置（embedding/llm/cross-encoder）
- `docker/`, `docker-compose.yml`: Docker実行用

## セットアップ
### Python環境
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### `.env`
`.env`はプロジェクトルートに配置します。主要なものだけ載せています。全設定は`app/src/config.py`を参照してください。

#### 必須 / ほぼ必須
- `DISCORD_BOT_TOKEN`: Discord botトークン（Bot起動・Discordログ取得に使用）
- `FOLDER_ID`: Google DriveのフォルダID
- `GOOGLE_APPLICATION_CREDENTIALS`: サービスアカウントJSONのパス（空ならデフォルト認証を利用）
- `EMBEDDING_MODEL`: Embedding用モデル名 or パス
- `LLM_PROVIDER`: `gemini` or `llama`
- `GEMINI_API_KEY`: Gemini利用時必須
- `LLAMA_MODEL` または `LLAMA_MODEL_PATH`: llama.cpp利用時必須

#### モデルディレクトリ
- `LLM_MODEL_DIR`（既定: `app/model/llm`）
- `EMBEDDING_MODEL_DIR`（既定: `app/model/embedding`）
- `CROSS_ENCODER_MODEL_DIR`（既定: `app/model/cross-encoder`）
- `PDF_OCR_MODEL`（既定: `app/model/ocr/tencent/HunyuanOCR`。画像PDFのOCRに使用）

#### 回答LLM / Embedding
- `GEMINI_MODEL`（Gemini利用時のモデル名）
- `EMBEDDING_MODEL` / `RAPTOR_EMBEDDING_MODEL`: `gemini:<model>` または `gemini/<model>` で Gemini Embeddings を利用可能（例: `gemini:text-embedding-004`）
- `LLAMA_CTX_SIZE`, `LLAMA_THREADS`, `LLAMA_GPU_LAYERS`
- `TEMPERATURE`, `MAX_OUTPUT_TOKENS`, `THINKING_LEVEL`
- `CROSS_ENCODER_MODEL`（再ランク用モデル。未設定なら再ランクなし）

#### Function Calling (RAGルーティング)
- `FUNCTION_CALL_ENABLED` (default: true)
- `FUNCTION_CALL_PROVIDER`: `gemini` or `llama_cpp`
- `FUNCTION_CALL_GEMINI_MODEL`: Gemini利用時のモデル名
- `FUNCTION_CALL_LLAMA_MODEL`: llama_cpp用のggufパス
- `FUNCTION_CALL_TEMPERATURE`, `FUNCTION_CALL_MAX_NEW_TOKENS`, `FUNCTION_CALL_MAX_RETRIES`
- `RAG_IDEA_TEMPERATURE`: RAGアイデア生成モード用
- `recency_mode`（`off` / `soft` / `hard`）をルーティング結果として返し、RAG検索時の新しさ重視度を制御

※ Function Calling用モデルが無い場合は `FUNCTION_CALL_ENABLED=false` にしてください。

#### No-RAG回答（RAGを使わない場合のLLM設定）
- `NO_RAG_LLM_PROVIDER`, `NO_RAG_GEMINI_MODEL`, `NO_RAG_LLAMA_MODEL`
- `NO_RAG_LLAMA_CTX_SIZE`, `NO_RAG_TEMPERATURE`, `NO_RAG_MAX_OUTPUT_TOKENS`, `NO_RAG_THINKING_LEVEL`

#### Refusal回答（回答拒否用LLM設定）
- `REFUSAL_LLM_PROVIDER`, `REFUSAL_GEMINI_MODEL`, `REFUSAL_LLAMA_MODEL`
- `REFUSAL_LLAMA_CTX_SIZE`, `REFUSAL_TEMPERATURE`, `REFUSAL_MAX_OUTPUT_TOKENS`, `REFUSAL_THINKING_LEVEL`

#### Ragas評価
- `EVAL_ANSWER_RELEVANCY_ENABLED`
- `EVAL_FAITHFULNESS_ENABLED`
- `EVAL_CONTEXT_PRECISION_ENABLED`
- `EVAL_CONTEXT_RECALL_ENABLED`

#### Chunking
- `FIRST_REC_CHUNK_SIZE`, `FIRST_REC_CHUNK_OVERLAP`
- `SECOND_REC_ENABLED`, `SECOND_REC_CHUNK_SIZE`, `SECOND_REC_CHUNK_OVERLAP`
- `SUMMERY_ENABLED`, `SUMMERY_CHARACTERS`
- `SUMMERY_PROVIDER`, `SUMMERY_GEMINI_MODEL`, `SUMMERY_LLAMA_MODEL`
- `SUMMERY_LLAMA_CTX_SIZE`, `SUMMERY_TEMPERATURE`
- `SUMMERY_MAX_OUTPUT_TOKENS`, `SUMMERY_MAX_RETRIES`
- `PROP_ENABLED`, `PROP_PROVIDER`
- `PROP_GEMINI_MODEL`, `PROP_LLAMA_MODEL`
- `PROP_LLAMA_CTX_SIZE`, `PROP_TEMPERATURE`
- `PROP_MAX_OUTPUT_TOKENS`, `PROP_MAX_RETRIES`

#### RAPTOR
- `RAPTOR_ENABLED`, `RAPTOR_EMBEDDING_MODEL`
- `RAPTOR_CLUSTER_MAX_TOKENS`, `RAPTOR_SUMMERY_MAX_TOKENS`
- `RAPTOR_STOP_CHUNK_COUNT`, `RAPTOR_K_MAX`, `RAPTOR_K_SELECTION`
- `RAPTOR_SUMMERY_PROVIDER`, `RAPTOR_SUMMERY_GEMINI_MODEL`
- `RAPTOR_SUMMERY_LLAMA_MODEL`, `RAPTOR_SUMMERY_LLAMA_CTX_SIZE`
- `RAPTOR_SUMMERY_TEMPERATURE`, `RAPTOR_SUMMERY_MAX_RETRIES`

#### Retrieval
- `TOP_K`, `DENSE_SEARCH_TOP_K`
- `SPARSE_SEARCH_TOP_K`（初回のSparse検索）
- `SPARSE_SEARCH_ORIGINAL_TOP_K`, `SPARSE_SEARCH_TRANSFORM_TOP_K`（再検索時）
- `PARENT_DOC_ENABLED`, `PARENT_CHUNK_CAP`
- `RERANK_ENABLED`, `RERANK_POOL_SIZE`, `MMR_LAMBDA`
- `RECENCY_WEIGHT_SOFT`, `RECENCY_WEIGHT_HARD`, `RECENCY_HALF_LIFE_DAYS`
- `SUDACHI_MODE` (`A`/`B`/`C`)
- `SPARSE_BM25_K1`, `SPARSE_BM25_B`
- `SPARSE_USE_NORMALIZED_FORM`, `SPARSE_REMOVE_SYMBOLS`
- `SOURCE_MAX_COUNT`
- `ANSWER_JSON_MAX_RETRIES`, `ANSWER_RESEARCH_MAX_RETRIES`

#### Discord / Drive / Scheduler
- `DISCORD_GUILD_ALLOW_LIST`（カンマ区切りID。空なら全ギルド）
- `DRIVE_MAX_FILES`（0で無制限）
- `AUTO_INDEX_ENABLED`, `AUTO_INDEX_TIME`（例: `03:00`）, `AUTO_INDEX_WEEKDAYS`（例: `mon,tue,...` or `0-6`）
- `WARMUP_INTERVAL_MINUTES`（定期Warm-Up間隔[分]。`0`で無効、既定: `60`）
- `INDEX_UPDATE_ESTIMATE_MIN_MINUTES`, `INDEX_UPDATE_ESTIMATE_MAX_MINUTES`（更新中に案内する終了目安レンジ。既定: `30`〜`60`）

#### その他
- `COMMAND_PREFIX`（質問用プレフィックス）
- `CHAT_HISTORY_ENABLED`, `CHAT_HISTORY_MAX_TURNS`
- `MAX_INPUT_CHARACTERS`

#### Index更新とクリア
- `CLEAR_RAW_DATA`
- `CLEAR_FIRST_REC_CHUNK_DATA`, `CLEAR_SECOND_REC_CHUNK_DATA`
- `CLEAR_SUMMERY_CHUNK_DATA`, `CLEAR_PROP_CHUNK_DATA`, `CLEAR_RAPTOR_CHUNK_DATA`
- `UPDATE_RAW_DATA`
- `UPDATE_FIRST_REC_CHUNK_DATA`, `UPDATE_SECOND_REC_CHUNK_DATA`
- `UPDATE_SPARSE_SECOND_REC_CHUNK_DATA`
- `UPDATE_SUMMERY_CHUNK_DATA`, `UPDATE_PROP_CHUNK_DATA`, `UPDATE_RAPTOR_CHUNK_DATA`

`CLEAR_*`が`0`のときは、`UPDATE_*`が`1`なら差分更新（入力更新時のみ再生成）を行います。  
入力元が削除された場合、対応する出力Chunkと下流Chunkも同期削除されます。

##### モデルパスの解決
`EMBEDDING_MODEL`/`LLAMA_MODEL`/`CROSS_ENCODER_MODEL`/`*_LLAMA_MODEL` などは、ファイル名のみ指定した場合は各`*_MODEL_DIR`を基準に解決されます。相対パスや絶対パスでも指定可能です。  
`EMBEDDING_MODEL` と `RAPTOR_EMBEDDING_MODEL` は `gemini:` または `gemini/` で始めると、ローカルパス解決を行わず Gemini Embeddings として扱います。

## 実行方法
### Index構築
```bash
python app/src/indexing/build_index.py
```
Google Drive/Discordから取得し、ChunkingとFAISS索引の生成を行います。

### Discordボット起動
```bash
python app/src/main.py
```
Botが起動し、`COMMAND_PREFIX`で始まるメッセージに応答します。

### ローカル対話テスト
```bash
python app/src/test.py
```
コンソールから対話的にRAG回答を試せます。

### Ragas評価
```bash
python app/src/eval/evaluate_ragas.py
```
`app/data/eval/ragas.jsonl`を読み込み、評価結果を`app/data/eval/result/`に出力します。
オプション: `--eval-file`, `--limit`, `--save-dataset`, `--result-path`, `--judge-model`。

## Discordコマンド
- `COMMAND_PREFIX`（デフォルト: `/ai `）: 質問用のプレフィックス
- `/ai build-index`: Index再構築コマンド（実行中は質問受付停止）
- `/ai eval`: Ragas評価コマンド
- `/ai stop`: 回答生成 / Index更新の中止

※ `/ai eval` と `/ai stop` は固定です。`/ai build-index` は `config.py` の定数で変更できます。

## 主要モジュールの挙動
### `RagPipeline` (`app/src/pipeline/rag_pipeline.py`)
- Dense検索（FAISS）+ Sparse検索（BM25+Sudachi）のハイブリッド
- Cross-Encoder再ランク（`CROSS_ENCODER_MODEL`が設定されている場合）
- MMRによる多様化、Parent chunk追加（`PARENT_DOC_ENABLED`）
- Function Callingで `RAG / No-RAG / Refusal` と回答フラグを選択（`FUNCTION_CALL_ENABLED`）
- LLMはJSON出力（`answer`, `sources`）を想定
- クエリ変換・追加メモリ判定はFunction Calling側で実施

### Indexing (`app/src/indexing/`)
- `drive_loader.py`: Google DriveのDocs/Sheets/Word/Excel/Slides/PowerPoint/PDFを取得・テキスト化（Officeは`.docx/.xlsx/.pptx`、PDFはPyMuPDF＋OCR）
- `hatenablog_loader.py`: はてなブログ記事を取得（本文テキスト）
- `discord_loader.py`: Discordログ取得（URL除去・メンション置換）
- `chunking.py`: Recursive / Summery / Proposition Chunking
- `raptor.py`: Embedding → k-meansクラスタリング → 要約を反復
- `faiss_index.py`: ChunkをEmbeddingしてFAISS索引を生成

## Docker
`docker-compose.yml`から起動できます。
```bash
docker compose up --build
```
注意: `docker/DockerFile`はルートの`requirements.txt`を参照します。

## 参考: 評価データ形式
`app/data/eval/ragas.jsonl`は以下の形式です:
```json
{"question": "質問文", "ground_truth": "正解文"}
{"question": "質問文", "ground_truths": ["正解1", "正解2"]}
```

## Crafters Colony source
- Raw output directory: `app/data/raw/crafters_colony`
- Author URL is configurable via `.env`:
  - `CRAFTERS_COLONY_AUTHOR_URL`
  - `CRAFTERS_COLONY_MAX_PAGES`
  - `CRAFTERS_COLONY_MAX_ARTICLES`
