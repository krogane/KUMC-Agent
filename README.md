# KUMC-Agent

KUMC-Agent は `Discord Bot` / `CLI` / 将来の `HTTP API` を前提に再設計された Python プロジェクトです。
現在は **RAG / Index / Eval / VC** を本移行済み、`DocGen` と `HTTP` は stub 実装です。

## 1. Architecture

実装正本は `src/kumc_agent` のみです。`app/src` 互換は提供しません。

- `frontends`
  - Discord / Console / HTTP の入出力層
  - usecase 呼び出しのみを担当
- `usecases`
  - workflow / orchestration 層
  - Chat, Index build/update, Ragas eval, VC 制御など
- `domain`
  - 不変モデル、ports(protocol)、policies
  - 外部 SDK 依存なし
- `features`
  - 機能単位サービス層（`rag`, `indexing`, `vc`, `summarization`, `docgen`）
- `infra`
  - 外部依存実装（Google API, Discord, FAISS/BM25, Gemini/llama.cpp, storage 等）
- `runtime`
  - DI と起動時 wiring (`container.py`, `context.py`)
- `config`
  - 設定読み込み・マージ・ENV マッピング

## 2. Directory Layout

```text
KUMC-Agent/
  configs/
    ops/
    experiments/
  assets/
    prompts/
    templates/
  data/
    raw/
    processed/
    chunks/
      first_rec_chunk/
      second_rec_chunk/
      sparse_second_rec_chunk/
      summery_chunk/
      prop_chunk/
      raptor_chunk/
    index/
    cache/
    eval/
  model/
    llm/
    embedding/
    cross-encoder/
    whisper/
    ocr/
  src/kumc_agent/
  tests/
```

## 3. Config System

設定は 2 層で管理します。

- 運用固定設定: `configs/ops/*.yaml`
- 実験設定: `configs/experiments/**/*.yaml`

優先順位は固定です。

1. `ops defaults`
2. `environment variables`
3. `experiment config`

マージ仕様:

- deep-merge
- scalar: 後勝ち
- list: 完全置換
- 未知キー: 起動エラー

## 4. Environment Variables

`.env` と `.env.example` は新キー体系に統一しています。

最小必須:

- `KUMC_DISCORD_BOT_TOKEN`
- `KUMC_OPENCLAW_ENABLED` (`1` で OpenClaw 優先経路を有効化)
- `KUMC_OPENCLAW_AGENT` (OpenClaw の対象 agent ID/名前。既定: `main`)
- `KUMC_OPENCLAW_MODEL` (OpenClaw で使うモデル ID。例: `google/gemini-3-flash-preview`)
- `OLLAMA_API_KEY` (OpenClaw がローカル Ollama provider を使う場合。通常は `ollama-local`)
- `KUMC_GEMINI_API_KEY`
- `KUMC_GEMINI_REQUESTS_PER_MINUTE` (Gemini API の1分あたり呼び出し上限)
- `KUMC_GOOGLE_APPLICATION_CREDENTIALS`
- `KUMC_DRIVE_FOLDER_ID`
- `KUMC_EXPERIMENT_PROFILE`
- `KUMC_LOG_LEVEL`

VC を有効化する場合:

- `KUMC_FEATURE_VC=1`
- `KUMC_VC_FEATURE_ENABLED=1`
- 必要に応じて `KUMC_VC_*` を調整

## 5. Prompt Management

Prompt の正本は `assets/prompts/*.md` です。
`PROMPT_*` 環境変数は廃止済みです。

## 6. Entrypoints

`src` パッケージを読むため、以下のいずれかを使ってください。

- `pip install -e .`
- `PYTHONPATH=src` を付与

### Discord Bot

```bash
PYTHONPATH=src python -m kumc_agent.frontends.discord.app
```

維持しているコマンド:

- `/ai <query>`
- `/ai build-index`
- `/ai eval`
- `/ai stop`
- `/ai join`
- `/ai quit`

### CLI

```bash
PYTHONPATH=src python -m kumc_agent.cli repl
PYTHONPATH=src python -m kumc_agent.cli chat --query "KUMCの活動内容は？"
PYTHONPATH=src python -m kumc_agent.cli tool rag --query "KUMCの活動内容は？"
PYTHONPATH=src python -m kumc_agent.cli index build
PYTHONPATH=src python -m kumc_agent.cli index update
PYTHONPATH=src python -m kumc_agent.cli eval ragas --eval-file data/eval/ragas.jsonl --ragas-batch-size 10
PYTHONPATH=src python -m kumc_agent.cli eval ragas --eval-file data/eval/ragas.jsonl --ragas-batch-size 200 --ragas-max-workers 4 --answer-cache-path data/eval/cache/ragas_answers.jsonl
```

### HTTP (stub)

```bash
PYTHONPATH=src python -m kumc_agent.cli http
# または
PYTHONPATH=src python -m kumc_agent.frontends.http.app
```

## 7. VC Migration Notes

VC 機能は `infra/legacy` 由来実装を `infra/vc` + `features/vc` + `usecases/vc` へ本移行しました。

- `frontends/discord` は VC の実処理を持たず usecase 呼び出しのみ
- VC の ASR / 議事録更新 / 最終要約生成は `infra/vc/manager.py` が担当
- LLM 呼び出しは `infra/vc/llm_client.py` に分離

## 8. Data / Model Re-allocation

旧配置の内容は新配置へ再配置済みです。

- `app/model/*` -> `model/*`
- `app/data/raw/*` -> `data/raw/*`
- `app/data/index/*` -> `data/index/*`
- `app/data/eval/*` -> `data/eval/*`
- `app/data/first_rec_chunk/*` -> `data/chunks/first_rec_chunk/*`
- `app/data/second_rec_chunk/*` -> `data/chunks/second_rec_chunk/*`
- `app/data/sparse_second_rec_chunk/*` -> `data/chunks/sparse_second_rec_chunk/*`
- `app/data/summery_chunk/*` -> `data/chunks/summery_chunk/*`
- `app/data/prop_chunk/*` -> `data/chunks/prop_chunk/*`
- `app/data/raptor_chunk/*` -> `data/chunks/raptor_chunk/*`

旧 `app/data` と `app/model` は削除済みです。

## 9. Tests

```bash
PYTHONPATH=src python -m unittest discover tests
```

依存ライブラリが未導入の場合は、まず以下を実行してください。

```bash
pip install -r requirements.txt
```

## 10. Operational Caution

- `.env` の実値（token/api key）はコミットしない
- `data/` と `model/` は大容量のため Git 管理対象外を前提

## 11. OpenClaw Migration Notes

- OpenClaw 連携は `openclaw` CLI を叩く薄いラッパー実装です (`src/kumc_agent/infra/openclaw`)。
- OpenClaw 用の `AGENTS.md` / `SOUL.md` / `USER.md` などは `configs/openclaw/` 配下に配置してください（`integrations.openclaw.config_dir` で変更可）。
- `chat` / `repl` は OpenClaw 優先で実行し、OpenClaw 側が失敗した場合のみ現行 RAG 経路へ自動フォールバックします。
- OpenClaw から現行 RAG を呼ぶ場合は `kumc-agent tool rag` を使います。
- OpenClaw モード (`KUMC_OPENCLAW_ENABLED=1`) では Discord frontend は VC サイドカー用途に縮退し、テキスト応答を実行しません。内部自動 index ループは OpenClaw 設定に関わらず実行されます。

## 12. Docker で OpenClaw を引き継ぐ

- `docker-compose.yml` ではプロジェクト直下の `./.openclaw` を `/root/.openclaw` にマウントしているため、OpenClaw の agent/workspace/session を引き継げます。
- `app/.venv` は匿名 volume (`/app/app/.venv`) でマスクしているため、巨大な `.venv` は引き継ぎません。
- コンテナ内の `openclaw` CLI は `docker/DockerFile` で `npm install -g openclaw@latest` しており、`.venv` 非依存で実行されます。

```bash
docker compose build --no-cache bot
docker compose up -d bot
```
