# KUMC-Agent

KUMC-Agent は KUMC の Discord Bot / CLI / HTTP API / Worker を 1 つの
`src/kumc_agent` package に集約した Python プロジェクトです。

現在のコードは `docs/kumc-agent-redesign-v4.md` の Wave 1-7 をもとに、
基盤、Connector / Ingestion、Retrieval、Workflow、Agentic Search、DocGen、
Announcement、Minecraft 支援、Automation / Production hardening を実装しています。
一部の外部サービス連携は dry-run または local repository 実装で、運用導入に必要な
設定作業は `docs/external-integration-and-operations-backlog.md` に分離しています。

## Architecture

実装正本は `src/kumc_agent` です。`src/kumc_agent/infra/legacy` は移行前コードの保持領域で、
新規実装から直接依存しない方針です。

- `apps`: 本番プロセス入口と app context 構築。`bot`、`api`、`worker`、foundation、retrieval、workflow、automation など。
- `domain`: 外部 SDK 非依存の dataclass / policy / port。
- `features`: 機能単位の service。RAG、indexing、retrieval、ingestion、workflow、docgen、announcement、minecraft、automation など。
- `infra`: 外部依存・永続化・repository・connector・migration 実装。
- `frontends`: Console / Discord / HTTP の protocol/UI adapter。context 構築は `apps` から注入します。
- `usecases`: CLI / frontend から呼ばれる orchestration。
- `runtime`: DI と runtime context。
- `config`: YAML、`.env`、環境変数の読み込みと schema 化。

## Runtime Layout

```text
KUMC-Agent/
  configs/
    ops/
    experiments/
  docs/
  infrastructure/
  src/kumc_agent/
    apps/
      api/
      bot/
      worker/
    domain/
    features/
    infra/
    runtime/
  tests/
```

## Setup

`src` package を読むため、開発時は `PYTHONPATH=src` を付けるか editable install を使います。

```bash
python -m venv app/.venv
app/.venv/bin/pip install -r requirements.txt
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli --help
```

この環境では pytest は前提にしていません。検証は `unittest` を使います。

## Configuration

設定は `configs/ops/*.yaml`、環境変数、`configs/experiments/**/*.yaml` の順に merge されます。

優先順位:

1. ops defaults
2. environment variables
3. experiment config

マージ仕様:

- dict は deep-merge
- scalar は後勝ち
- list は完全置換
- 未知キーは起動エラー

`.env` と `.env.example` は同じキー集合を保つ必要があります。片方にキーを追加・削除した場合は、必ずもう片方にも反映してください。

主な環境変数:

- `KUMC_DISCORD_BOT_TOKEN`
- `KUMC_GEMINI_API_KEY`
- `KUMC_GOOGLE_APPLICATION_CREDENTIALS`
- `KUMC_DRIVE_FOLDER_ID`
- `KUMC_OPENCLAW_ENABLED`
- `KUMC_OPENCLAW_AGENT`
- `KUMC_OPENCLAW_MODEL`
- `KUMC_OPENAI_API_KEY`
- `KUMC_EXPERIMENT_PROFILE`
- `KUMC_LOG_LEVEL`
- `KUMC_FEATURE_AUTOMATION_AUTO_RUN_MODE`

運用環境では secret manager / CI secret / cloud secret から注入し、`.env` の実値はコミットしないでください。

## Entrypoints

### App Entrypoints

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli bot
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli api --host 127.0.0.1 --port 8000
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli worker
```

- `bot`: Discord slash-command bot を起動します。
- `api`: API app を起動します。
- `worker`: worker skeleton を 1 回実行します。

### Admin / DB

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action health
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action readiness
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action sync --scope all --limit 20
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action reindex --scope all --force
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action eval --limit 10
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action feature_flags
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action permissions
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action cost_report
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli db migrate
```

`db migrate` は `infrastructure/migrations` / `infra/migrations` の migration を PostgreSQL に適用します。
PostgreSQL 未設定時は local file repository で動く機能のみ利用できます。

### Ingestion / Retrieval

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest backfill --source file --limit 20
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --question "次回の活動予定は？"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --question "Minecraftサーバーの参加方法は？" --depth deep
```

`ingest` は connector registry 経由で raw item / chunk を保存し、SecretFinding と terms review metadata を付与します。
`ask` は hybrid retrieval、citation、access filtering、prompt-injection guard を通した統合質問応答です。

### Workflow / Approval / Automation

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type task_extract --instruction "TODO: 新歓資料を作成 担当: @alice"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type task_add --instruction "新歓資料を作成"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type task_list
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type event_add --instruction "2026-05-01 18:00 新歓"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type doc_draft --instruction "新歓告知文を作成"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type x_draft --instruction "次回活動の告知"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type announcement_draft --instruction "Discord向け告知"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type mc_status
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type mc_request --instruction "whitelist add player Steve"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli approval --action list
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli approval --action approve --target-id task_...
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action list
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action dry_run --rule-id auto_index_daily
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action set_mode --rule-id auto_index_daily --mode approval_required
```

`approval` は現在 `task` type のみです。外部投稿・Minecraft 実操作・完全自動実行は、誤操作防止のため dry-run / approval-first を基本にしています。

### Local / Tool Entrypoints

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli repl
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli chat --query "KUMCの活動内容は？"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli tool rag --query "KUMCの活動内容は？"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index build
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index update
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli eval ragas --eval-file data/eval/ragas.jsonl
```

`discord` / `http` 互換入口は削除済みです。Discord は `bot`、HTTP API は `api` を使います。

## Implemented Waves

- Wave 1: apps/bot、apps/api、apps/worker、foundation、health、audit、jobs、migration runner。
- Wave 2: connector registry、ingestion、SecretFinding、raw snapshot、terms metadata。
- Wave 3: hybrid retrieval、citation、answering、access filtering、integrated `/ask`。
- Wave 4: workflow、task、event、schedule、meeting、approval。
- Wave 5: agentic search、DocGen、announcement draft。
- Wave 6: Minecraft status / request / dry-run operation support。
- Wave 7: automation、readiness、cost cap、runbook、admin hardening。

## Tests

代表的な Wave 実装の検証:

```bash
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src app/.venv/bin/python -m unittest \
  tests.unit.test_wave1_foundation \
  tests.unit.test_wave2_ingestion \
  tests.unit.test_wave3_retrieval \
  tests.unit.test_wave4_workflow \
  tests.unit.test_wave5_agentic_docgen_announcement \
  tests.unit.test_wave6_minecraft_support \
  tests.unit.test_wave7_automation_hardening \
  tests.unit.test_config_loading \
  tests.unit.test_cli_tool_rag \
  tests.unit.test_stubs \
  tests.architecture.test_layer_rules
```

全件 discovery は、一部の既存テストが外部 API / モデル / DNS / ローカルデータ前提を持つため、環境に応じて失敗することがあります。

## Related Docs

- `docs/kumc-agent-redesign-v4.md`: redesign spec。
- `docs/implementation-gap-report.md`: Wave 1-7 後の未実装・外部依存整理。
- `docs/external-integration-and-operations-backlog.md`: 外部サービス連携と運用作業。
- `docs/implementation-consistency-audit.md`: 現行コードの用途不明・一貫性・効率・dead code 調査。
- `docs/runbooks/*.md`: production hardening runbook。

## Operational Caution

- `.env` の token / api key / credential 実値はコミットしない。
- `data/` と `model/` は大容量・環境依存のため Git 管理対象外を前提にする。
- 外部投稿、Minecraft 操作、automation auto-run は approval / dry-run で検証してから有効化する。
- SecretFinding は検出補助であり、運用では secret rotation と監査ログ確認を併用する。
