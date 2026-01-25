# KUMC-Agent
KUMCが保有する情報をGoogle Driveから収集し、RAGで回答するDiscordボットです。
ローカルの`llama.cpp`またはGeminiを使った回答生成に対応し、Index構築・評価(Ragas)まで一通りの流れが揃っています。

## 全体構成
- Google DriveのDocs/SheetsをMarkdown/CSVで取得し、段階的にChunkingしてFAISS索引を構築
- Discordで`/ai `から始まる質問を受け取り、検索＋再ランク＋回答生成
- Ragas評価スクリプトでRAG品質を可視化

## データフロー
Indexing:
1. (任意) 既存データをクリア
2. Google DriveからDocs/Sheetsを取得 (`app/data/raw`)
3. Recursive Chunking (`app/data/rec_chunk`)
4. (任意) Proposition Chunking (`app/data/prop_chunk`)
5. (任意) RAPTOR要約 (`app/data/raptor_chunk`)
6. すべてのChunkをEmbeddingしてFAISS索引を作成 (`app/data/index`)

Query:
1. `RagPipeline`がSemantic検索＋Keyword検索を実行
2. 必要に応じてCross-Encoderで再ランク
3. LLM (Gemini or llama.cpp) で回答生成

## ディレクトリ構成
- `app/src/main.py`: Discordボットのエントリポイント
- `app/src/config.py`: すべての設定/環境変数の集約
- `app/src/pipeline/`: RAG推論パイプライン
- `app/src/indexing/`: Index構築ロジック (Drive取得/Chunking/RAPTOR/FAISS)
- `app/src/eval/`: Ragas評価スクリプト
- `app/data/`: 取得データ・Chunk・Index・評価データ
- `app/model/`: ローカルLLM/Embedding/Cross-Encoderの配置場所
- `docker/`, `docker-compose.yml`: Docker実行用

## セットアップ
### Python環境
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### `.env`
`.env`をプロジェクトルートに配置し、必要な環境変数を設定します。
`app/src/config.py`が唯一の設定ソースです。

## 実行方法
### Index構築
```bash
python app/src/indexing/build_index.py
```
Google Driveから取得し、ChunkingとFAISS索引の生成を行います。

### Discordボット起動
```bash
python app/src/main.py
```
Botが起動し、`/ai `で始まるメッセージに応答します。

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

## Discordコマンド
- `COMMAND_PREFIX` (デフォルト: `/ai `): 質問用のプレフィックス
- `index_command_prefix` (デフォルト: `/build_index`): Index再構築コマンド
  - Index更新中は質問受付を停止し、完了後に再開します

## 主要モジュールの挙動
### `RagPipeline` (`app/src/pipeline/rag_pipeline.py`)
- `retrieve`:
  - Semantic検索 (FAISS) + Keyword検索を実施
  - 結果をマージ後、Cross-Encoderで再ランク (設定がある場合)
  - Proposition Chunk使用時は親Recursive Chunkを追加 (`PARENT_DOC_ENABLED`)
- `generate`:
  - `LLM_PROVIDER`でGemini or llama.cppを選択
  - `system_rules`に沿って回答生成

### Indexing (`app/src/indexing/`)
- `drive_loader.py`: Google DriveからDocs/Sheetsを取得し、Markdown/CSVとして保存
- `chunking.py`: Recursive Chunking + (任意) Proposition Chunking
  - Proposition ChunkingはLLMでJSON配列を生成し、失敗時はリトライ
- `raptor.py`: Embedding → k-meansクラスタリング → 要約を繰り返す
- `faiss_index.py`: ChunkをEmbeddingしてFAISS索引を生成

## 設定一覧 (.env)
必須 / よく使う:
- `DISCORD_BOT_TOKEN`: Discord botトークン
- `GEMINI_API_KEY`: Gemini APIキー (Gemini利用時)
- `FOLDER_ID`: Google DriveのフォルダID
- `GOOGLE_APPLICATION_CREDENTIALS`: サービスアカウントJSONのパス

Embedding/LLM:
- `EMBEDDING_MODEL`: Embedding用モデル(ggufパス or HF名)
- `LLM_PROVIDER`: `gemini` or `llama`
- `GEMINI_MODEL`: Geminiモデル名 (例: `gemini-3-flash-preview`)
- `LLAMA_MODEL_PATH`: llama.cpp用モデルパス
- `LLAMA_CTX_SIZE`, `LLAMA_THREADS`, `LLAMA_GPU_LAYERS`
- `TEMPERATURE`, `MAX_OUTPUT_TOKENS`, `THINKING_LEVEL`

Chunking:
- `REC_CHUNK_SIZE`, `REC_CHUNK_OVERLAP`, `REC_MIN_CHUNK_TOKENS`
- `PROP_CHUNK_ENABLED`, `PROP_CHUNK_PROVIDER`
- `PROP_CHUNK_MODEL`, `PROP_CHUNK_LLAMA_MODEL_PATH`
- `PROP_CHUNK_LLAMA_CTX_SIZE`, `PROP_CHUNK_TEMPERATURE`
- `PROP_CHUNK_MAX_OUTPUT_TOKENS`, `PROP_CHUNK_SIZE`, `PROP_CHUNK_MAX_RETRIES`

RAPTOR:
- `RAPTOR_ENABLED`, `RAPTOR_EMBEDDING_MODEL`
- `RAPTOR_CLUSTER_MAX_TOKENS`, `RAPTOR_SUMMARY_MAX_TOKENS`
- `RAPTOR_STOP_CHUNK_COUNT`, `RAPTOR_K_MAX`, `RAPTOR_K_SELECTION`
- `RAPTOR_SUMMARY_PROVIDER`, `RAPTOR_SUMMARY_MODEL`
- `RAPTOR_SUMMARY_LLAMA_MODEL_PATH`, `RAPTOR_SUMMARY_LLAMA_CTX_SIZE`
- `RAPTOR_SUMMARY_TEMPERATURE`, `RAPTOR_SUMMARY_MAX_RETRIES`

Retrieval:
- `TOP_K`, `RAPTOR_SEARCH_TOP_K`, `KEYWORD_SEARCH_TOP_K`
- `PARENT_DOC_ENABLED`

Index更新とクリア:
- `CLEAR_RAW_DATA`, `CLEAR_REC_CHUNK_DATA`
- `CLEAR_PROP_CHUNK_DATA`, `CLEAR_RAPTOR_CHUNK_DATA`

その他:
- `COMMAND_PREFIX`: Discord質問のプレフィックス
- `DRIVE_MAX_FILES`: 取得するDriveファイル数の上限
- `CROSS_ENCODER_MODEL`: 再ランク用のllama.cppモデルパス

## モデル配置の例
- Embedding: `app/model/embedding/*.gguf`
- LLM: `app/model/llm/*.gguf`
- Cross-Encoder: `app/model/cross-encoder/*.gguf`

`EMBEDDING_MODEL`や`LLAMA_MODEL_PATH`は絶対パスでも相対パスでも指定可能です。
相対パスの場合はリポジトリルート基準で解決されます。

## Docker
`docker-compose.yml`から起動できます。
```bash
docker compose up --build
```
注意: `docker/DockerFile`は`app/requirements.txt`を参照します。
現在はルートに`requirements.txt`があるため、必要に応じてDockerfileを修正してください。

## 参考: 評価データ形式
`app/data/eval/ragas.jsonl`は以下の形式です:
```json
{"question": "質問文", "ground_truth": "正解文"}
{"question": "質問文", "ground_truths": ["正解1", "正解2"]}
```

