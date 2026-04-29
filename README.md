# KUMC-Agent

KUMC-Agent は、KUMC の活動情報を探したり、告知文・議事録・タスク・イベント候補を作ったり、Discord Bot / CLI / HTTP API / Worker から操作できる AI エージェントです。

この README は、プロジェクトをまだ詳しく知らない人が「まず何を実行すればよいか」を分かるように、操作方法とコマンドを中心にまとめています。

## できること

- Discord、Google Drive、Notion、X、はてなブログ、クラフターズコロニー、Minecraft Wiki などの情報を取り込み、根拠付きで質問に答える
- 活動予定、タスク、イベント、会議準備、議事録、告知文、X 投稿文などの下書きや候補を作る
- 画像検索、メンバー検索、Minecraft サーバー支援など、用途別の操作を実行する
- タスクやイベントなどの候補を、人間の承認後に正本へ反映する
- インデックス更新、定期処理、readiness 確認、コスト確認などの運用コマンドを実行する

副作用のある操作は、原則として dry-run や approval を挟みます。外部投稿、Minecraft 操作、自動実行などは、いきなり本番反映するのではなく、内容を確認してから進める運用です。

## 最初に使うコマンド

まずヘルプを表示します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli --help
```

質問を 1 回だけ実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --question "次回の活動予定は？"
```

対話モードで試します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli repl
```

アプリの状態を確認します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action health
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action readiness
```

## セットアップ

Python 仮想環境を作り、依存ライブラリを入れます。

```bash
python -m venv app/.venv
app/.venv/bin/pip install -r requirements.txt
```

CLI は `src` 配下の package を読むため、実行時に `PYTHONPATH=src` を付けます。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli --help
```

`.env.example` を参考に `.env` を作ります。

```bash
cp .env.example .env
```

`.env` には API キーやトークンなどの秘密情報だけを置きます。プロンプトや通常のパラメータは `.env` ではなく、`assets/prompts/` や `configs/` 配下を使います。

よく設定する項目です。

- `KUMC_DISCORD_BOT_TOKEN`: Discord Bot を起動するための token
- `KUMC_OPENAI_API_KEY`: OpenAI API を使う場合の key
- `KUMC_GEMINI_API_KEY`: Gemini API を使う場合の key
- `KUMC_GOOGLE_APPLICATION_CREDENTIALS`: Google Drive 連携用 credential
- `KUMC_DRIVE_FOLDER_ID`: 取り込み対象の Drive folder
- `KUMC_DATABASE_URL`: PostgreSQL を使う場合の接続先
- `KUMC_DISCORD_GUILD_ALLOW_LIST`: 許可する Discord guild
- `KUMC_DISCORD_MEMBER_PROFILE_GUILD_IDS`: `member_profiles` のメンバー情報取得先 Discord guild。未設定時は `KUMC_DISCORD_GUILD_ALLOW_LIST` を使う
- `KUMC_MAINTENANCE_COMMAND_AUTHOR_IDS`: 管理操作を許可する Discord user

`.env` と `.env.example` は同じキー集合を保ってください。片方に項目を追加・削除した場合は、もう片方にも反映します。

## 起動する

Discord Bot を起動します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli bot
```

HTTP API を起動します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli api --host 127.0.0.1 --port 8000
```

Worker を 1 回実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli worker
```

特定の worker job を実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli worker --job-type auto_index_update
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli worker --job-type ingest_backfill --payload-json '{"source":"drive","limit":20}'
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli worker --job-type autonomous_agent_run --payload-json '{"dry_run":true,"scope":["tasks","events"]}'
```

## 質問する

通常の質問をします。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --question "KUMCの活動内容を教えて"
```

検索対象を絞ります。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --source minecraft_wiki --question "丸石の入手方法は？"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --source task --question "未完了タスクを教えて"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --source event --question "今月のイベントは？"
```

深く調べたい場合は `--depth deep` を使います。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --depth deep --question "次回イベントの準備に必要な作業を根拠付きで整理して"
```

速さを優先する場合は `--mode fast`、慎重に確認したい場合は `--mode careful` を使います。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --mode fast --question "次の活動日は？"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --mode careful --question "新歓準備で未確認の論点は？"
```

権限付きの情報を扱う場合は、呼び出し元のユーザー情報を渡します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask \
  --question "Drive内の新歓資料を探して" \
  --user-id 123456789 \
  --guild-id 987654321 \
  --role-id 111 \
  --admin
```

## 取り込みとインデックス更新

connector からデータを取り込みます。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest backfill --source drive --limit 20
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest backfill --source discord --limit 50
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest backfill --source minecraft_wiki --limit 20
```

有効な connector をまとめて取り込む場合は `--source` を省略します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest backfill --limit 20
```

検索インデックスを更新します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index update
```

全体を作り直す場合は `--full-rebuild` を付けます。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index update --full-rebuild
```

管理コマンドから同期・再インデックスを実行することもできます。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action sync --scope all --limit 20
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action reindex --scope all --force
```

## タスク・イベント・下書きを作る

`work` コマンドは、タスク管理、イベント管理、会議準備、文書下書き、画像検索、メンバー検索、Minecraft 支援などをまとめて扱います。

タスク候補を作ります。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type task_add --instruction "新歓資料を作成"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type task_extract --instruction "TODO: 新歓資料を作成 担当: @alice 期限: 5/1"
```

タスクを一覧します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type task_list
```

タスクの変更・完了・削除候補を作ります。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type task_update --target "<task-id>" --instruction "期限を5/3に変更"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type task_done --target "<task-id>" --instruction "完了"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type task_delete --target "<task-id>" --instruction "不要になった"
```

イベント候補を作ります。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work \
  --type event_add \
  --instruction "イベント: 新歓会 日時: 2026-05-05 14:00 場所: 部室"
```

文章からイベント候補を抽出します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work \
  --type event_extract \
  --instruction "5/5 14:00から部室で新歓会を開催します。"
```

イベントを確認します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type event_list
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type event_brief --target "<event-id>"
```

予定を作成・確認します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type schedule_add --instruction "毎週金曜 18:00 定例会"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type schedule_list
```

会議準備や議事録下書きを作ります。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type meeting_prepare --instruction "次回定例会の議題案を作って"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type meeting_minutes_draft --instruction "決定事項: 新歓日程を5/5にする TODO: 告知文作成"
```

文書・告知・X 投稿の下書きを作ります。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type doc_draft --instruction "新歓案内文を作って"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type announcement_draft --instruction "Discord向けに次回活動を告知して"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type x_draft --instruction "次回活動の告知ポストを作って"
```

画像やメンバーを探します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type image_search --instruction "新歓で使えそうな集合写真"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type member_search --instruction "動画編集を担当できそうな人" --admin
```

Minecraft サーバー支援を使います。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type mc_status --admin
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type mc_request --instruction "whitelist add player Steve" --admin
```

## 承認する

タスク、イベント、予定、告知、自動化ルール、サーバー操作などは、候補を作ってから承認できます。

承認待ちを一覧します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli approval --type task --action list
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli approval --type event --action list
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli approval --type server_operation --action list --admin
```

詳細を表示します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli approval --type task --action show --target-id "<candidate-id>"
```

承認・却下します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli approval --type task --action approve --target-id "<candidate-id>"
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli approval --type event --action reject --target-id "<candidate-id>" --comment "日時が未確定"
```

承認済みのサーバー操作を実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type server_operation_execute --target "<operation-id>" --admin
```

## 自動化する

自動化ルールを一覧します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action list
```

ルールの詳細を確認します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action show --rule-id auto_index_daily
```

実行せずに試します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action dry_run --rule-id auto_index_daily
```

承認必須モードにします。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action set_mode --rule-id auto_index_daily --mode approval_required --admin
```

有効化・無効化します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action enable --rule-id auto_index_daily --admin
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action disable --rule-id auto_index_daily --admin
```

自律エージェントを手動で 1 回動かします。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli autonomous --dry-run --slot manual --scope tasks --scope events
```

## 管理・運用コマンド

ヘルスチェックを実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action health
```

readiness を確認します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action readiness
```

feature flag を確認します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action feature_flags
```

管理者や guild の設定状況を確認します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action permissions
```

コスト関連の状態を確認します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action cost_report
```

メンバープロフィールを再構築します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action member_profiles
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli admin --action sync --scope member_profiles
```

取得先 Guild は `KUMC_DISCORD_MEMBER_PROFILE_GUILD_IDS` または `configs/main/security.yaml` の `security.discord_member_profile_guild_ids` で指定します。未設定の場合は従来通り `discord_guild_allow_list` を使います。単発で明示する場合は `--action member_profiles --scope <guild_id>` を使います。

PostgreSQL migration を適用します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli db migrate
```

## HTTP API を使う

API を起動します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli api --host 127.0.0.1 --port 8000
```

ヘルスチェックします。

```bash
curl http://127.0.0.1:8000/health
```

質問します。

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"次回の活動予定は？","source":"all","mode":"answer","depth":"normal"}'
```

workflow を実行します。

```bash
curl -X POST http://127.0.0.1:8000/work \
  -H 'Content-Type: application/json' \
  -d '{"type":"task_add","instruction":"新歓資料を作成","user_id":"admin","admin":true}'
```

承認します。

```bash
curl -X POST http://127.0.0.1:8000/approval \
  -H 'Content-Type: application/json' \
  -d '{"type":"task","action":"approve","target_id":"<candidate-id>","user_id":"admin","admin":true}'
```

自動化を確認します。

```bash
curl -X POST http://127.0.0.1:8000/automation \
  -H 'Content-Type: application/json' \
  -d '{"action":"list"}'
```

## Discord で使う

Bot を起動します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli bot
```

Discord では主に slash command を使います。

- `/ask`: 質問・検索・統合入力
- `/work`: タスク、イベント、告知、画像検索、メンバー検索、Minecraft 支援
- `/approval`: 承認待ちの確認、承認、却下、編集
- `/automation`: 自動化ルールの確認、dry-run、実行、モード変更
- `/admin action`: health、readiness、sync、reindex、feature flags、permissions、cost report など

管理系の操作は、`.env` の `KUMC_MAINTENANCE_COMMAND_AUTHOR_IDS` と `KUMC_DISCORD_GUILD_ALLOW_LIST` の設定に従って制限されます。

## 評価・検証

RAG 評価を実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli eval ragas --eval-file data/eval/ragas.jsonl
```

代表的な unit test を実行します。

```bash
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src app/.venv/bin/python -m unittest \
  tests.unit.test_foundation_services \
  tests.unit.test_ingestion_service \
  tests.unit.test_retrieval_ask_service \
  tests.unit.test_workflow_service \
  tests.unit.test_automation_hardening \
  tests.unit.test_autonomous_agent \
  tests.architecture.test_layer_rules
```

全件 discovery は、外部 API、モデル、DNS、ローカルデータを前提にしたテストが混ざることがあります。失敗した場合は、対象機能の設定や fixture を確認してください。

## よく見るファイル

- `configs/main/`: アプリ設定、RAG、indexing、scheduler、security、provider などの設定
- `assets/prompts/`: プロンプト
- `.env.example`: ローカル環境変数の見本
- `docs/design/`: 機能別の設計メモ
- `docs/explanation/cli.md`: CLI の詳しい説明
- `docs/runbooks/`: 運用手順
- `docs/kumc-agent-redesign-v4.md`: 全体設計・移行計画
- `src/kumc_agent/cli.py`: CLI 入口
- `src/kumc_agent/apps/`: bot / api / worker などのアプリ入口

## 運用時の注意

- `.env` の実値、API key、token、credential はコミットしないでください。
- 外部投稿、Minecraft 操作、自動実行は、dry-run や approval で内容を確認してから有効化してください。
- CLI や HTTP の JSON 出力では、主結果をトップレベルに置き、診断情報や内部判断は `metadata` 配下に置く方針です。
- `data/`、`model/`、ローカル index、credential は環境依存です。共有するときは含めるべきものか確認してください。
- `src/kumc_agent/infra/legacy` は移行前コードの保持領域です。通常の実装や運用では依存しない方針です。
