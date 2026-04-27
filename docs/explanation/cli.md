# `src/kumc_agent/cli.py` の解説

## このファイルの役割

`cli.py` は、KUMC-Agent をコマンドラインから起動するための入口です。

利用者は次のように `python -m kumc_agent.cli ...` という形でコマンドを実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --question "次回の活動予定は？"
```

このとき Python が最初に読むのが `src/kumc_agent/cli.py` です。
`cli.py` は、入力されたコマンドやオプションを読み取り、「どの機能を動かすべきか」を判断して、実際の処理を担当する別のモジュールへ渡します。

つまり、このファイル自体が検索、回答生成、DB 操作、Discord Bot 処理などをすべて実装しているわけではありません。
主な仕事は、コマンドライン操作とアプリケーション内部の処理をつなぐことです。

## 全体像

`cli.py` の処理は、大きく見ると次の順番で進みます。

1. `argparse` で使えるコマンドとオプションを定義する
2. ユーザーが入力したコマンドを解析する
3. コマンドに応じて必要な app context や runtime context を作る
4. 対応する service / usecase / frontend を呼び出す
5. 結果を標準出力へ表示する

`argparse` は Python 標準ライブラリのコマンドライン引数解析ツールです。
たとえば `ask --question "..."` のような入力から、`command` は `ask`、`question` は質問文、という形でプログラム内から扱える値に変換します。

## ファイル冒頭の import

冒頭では、この CLI が呼び出す可能性のある処理を読み込んでいます。

代表的なものは次の通りです。

- `argparse`: コマンドライン引数を解析するために使う
- `asyncio`: ingestion など非同期処理を CLI から実行するために使う
- `json`: 結果を JSON 形式で表示するために使う
- `logging`: 実行ログを出すために使う
- `Path`: ファイルパスを扱うために使う
- `run_repl`: コンソール対話モードを起動する
- `build_runtime_context`: 旧来互換の chat / index / eval 用の依存関係をまとめて作る
- `ChatRequest` などの request class: usecase に渡す入力データを表す
- `configure_logging`: ログ設定を行う

この import の構成からも、`cli.py` が「各機能の本体」ではなく「各機能を呼び出す入口」であることがわかります。

## 補助関数

### `_build_tool_rag_payload`

`tool rag` コマンドの回答結果を、外部ツールや別プロセスが扱いやすい JSON 用の辞書に変換します。

主に次の情報を出力します。

- `answer`: 回答本文
- `route`: どの回答ルートが使われたか
- `sources`: 回答の根拠になった情報源
- `metadata`: ルーティング判断、高速モード、その他の付加情報

ここでは `metadata` から `contexts`、`llm_prompt`、`raw` を削除しています。
`contexts` は検索で集めた本文断片、`llm_prompt` は内部プロンプト、`raw` はモデルの生出力であり、サイズが大きくなったり内部情報を含んだりする可能性があります。
CLI の出力を軽くし、ツール連携で扱いやすくするために除外しています。

この payload では、`routing_decision` や `fast_mode` のような診断情報はトップレベルには置きません。
CLI や外部連携向け payload では、主結果として扱う安定フィールドだけをトップレベルに置き、内部判断や実行モードなどの診断情報は `metadata` にまとめる方針です。

### `_workflow_response_payload`

`work` と `approval` コマンドの結果を JSON 化しやすい形に変換します。

workflow 系の処理では、タスク候補、イベント候補、予定候補、タスク、イベント、予定、会議、承認、Minecraft サーバー操作など、複数種類の結果が返る可能性があります。
この関数はそれらをまとめて辞書にします。

日時のように `isoformat()` を持つ値は文字列へ変換します。
これは、`datetime` などがそのままでは JSON にしづらいためです。

### `_automation_response_payload`

`automation` コマンドの結果を JSON 化しやすい形に変換します。

automation では自動実行ルールや実行履歴を扱うため、主に次の情報を出力します。

- `text`: 人間向けの概要
- `detail_markdown`: 詳細説明
- `rules`: 自動化ルール一覧
- `runs`: 実行履歴
- `warnings`: 警告
- `metadata`: 付加情報

## `_build_parser`

`_build_parser()` は、この CLI で使えるコマンド一覧を定義する関数です。

ここで作られた定義により、たとえば次のようなコマンドが使えるようになります。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli --help
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --help
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --help
```

この関数では `subparsers` を使っています。
`subparsers` は `kumc-agent ask ...`、`kumc-agent admin ...` のように、最初の単語で処理を分ける仕組みです。

## 定義されている主なコマンド

### `repl`

コンソール上で対話的に質問できるモードを起動します。

1 回だけ質問するのではなく、ターミナル上で会話を続けたいときの入口です。

### `chat`

1 回だけ通常のチャット質問を実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli chat --query "KUMCの活動内容は？"
```

内部では `ChatEntryRequest` を作り、`context.chat_entry.execute(...)` に渡します。

### `tool rag`

ローカルの RAG をツール連携向けに実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli tool rag --query "KUMCの活動内容は？"
```

`chat` と違い、回答文だけではなく、情報源やメタデータを含む JSON を出力します。
`--query` は複数回指定できます。
複数指定した場合は、それぞれの質問に対する結果を配列として返します。
Drive / Discord など権限付きソースを検索対象に含める場合は、呼び出し元の権限情報として `--user-id`、`--guild-id`、`--role-id`、`--admin` を渡せます。

Minecraft Wiki RAG を明示的に使う場合は、`--scope minecraft_wiki` を指定します。
この経路では日本語版 Minecraft Wiki の index だけを検索対象にし、payload の `route` は `minecraft_wiki_rag` になります。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli tool rag --scope minecraft_wiki --query "丸石の入手方法は？"
```

また、このコマンドでは履歴や追加メモリを無効化する指定が入っています。
そのため、外部ツールから安定した RAG 結果を得る用途に向いています。
ルーティング判断、高速モード、合成クエリ、回答フィルタリング結果のような診断情報が必要な場合は、トップレベルではなく `metadata` の中を確認します。

### `index build` / `index update`

検索用インデックスを作成または更新します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index build
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index update
```

`build` は従来互換のインデックス作成、`update` は自動インデックス更新 usecase の手動入口です。
どちらも次のオプションを持ちます。

- `--no-refresh-sources`: 元データの再取得をしない
- `--full-rebuild`: 全体を作り直す
- `--stage`: 実行する stage を絞る。複数回指定可能

`index update` は差分取り込み、lock、staging build、quality smoke check、publish を同じ run として扱い、`indexing_runs` に保存します。
CLI payload の安定フィールドは `status`、`run_id`、`seen`、`changed`、`skipped`、`deleted` で、差分内訳、品質結果、snapshot 情報、skip 理由などの診断情報は `metadata` 配下に入ります。
`index build` は読み込んだ source 数、document 数、chunk 数、index directory を JSON で表示します。

### `eval ragas`

RAG の評価を実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli eval ragas --eval-file data/eval/ragas.jsonl
```

評価ファイル、件数上限、結果出力先、RAGAS のバッチサイズやタイムアウト、回答キャッシュの扱いなどを指定できます。
結果として、評価件数、完全一致率、トークン重複率、RAGAS 指標などを JSON で表示します。

### `bot` / `api` / `worker`

本番プロセス向けの app entrypoint です。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli bot
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli api --host 127.0.0.1 --port 8000
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli worker
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli worker --job-type auto_index_update
```

- `bot`: Discord slash-command bot を起動する
- `api`: API app を指定 host / port で起動する
- `worker`: worker job を 1 回実行する。`--job-type auto_index_update` は自動インデックス更新 usecase を呼びます。

これらは `main()` の中で必要になったタイミングで import されています。
常にすべての app を読み込むのではなく、実行するコマンドに必要なものだけを読み込む構成です。
旧 `discord` / `http` 入口は削除され、Discord は `bot`、HTTP API は `api` に統一されています。

### `admin`

運用・管理用のコマンドです。

`--action` で実行内容を選びます。

- `health`: アプリのヘルスチェック
- `readiness`: 自動化などを含む readiness report
- `sync`: connector からデータを取り込む
- `reindex`: 強制的に再取り込み・再インデックス寄りの処理を行う
- `eval`: ローカル評価ハーネスの概要を出す
- `feature_flags`: feature flag の状態を出す
- `permissions`: 管理者 ID や guild allow list の設定状況を出す
- `cost_report`: automation のコスト関連レポートを出す

`sync` と `reindex` では自動インデックス更新 usecase を呼びます。
`reindex` の場合は `force` と `full_rebuild` が有効になるため、差分にかかわらず全体再構築寄りの処理として扱われます。

### `db migrate`

PostgreSQL migration を適用します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli db migrate
```

### `work` の event 系操作

イベント管理はadmin権限付きの workflow 操作として扱います。手動登録でもまず `EventCandidate` を作り、`approval --type event approve` で承認されるまで `Event` 正本には入りません。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work \
  --type event_add \
  --instruction "イベント: 新歓会 日時: 2026-05-05 14:00 場所: 部室" \
  --user-id admin --admin

PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli approval \
  --type event --action approve --target-id "<candidate-id>" \
  --user-id admin --admin
```

RAG差分や長文からの自動抽出は `event_extract` を使います。LLMが使えない、schemaが不正、根拠不足の場合は候補を作らず、理由は `metadata.extraction` 配下に入ります。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work \
  --type event_extract \
  --instruction "5/5 14:00から部室で新歓会を開催します。" \
  --user-id admin --admin
```

表示は日時、場所、状態、関連未完了タスクで絞り込めます。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work \
  --type event_list \
  --instruction "状態: planning 場所: 部室 2026-05-01から2026-05-31まで 未完了タスクあり" \
  --user-id admin --admin
```

正本Eventの変更・削除も直接反映せず、変更候補を作ってから承認します。削除は `status=canceled` への論理削除です。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work \
  --type event_update --target "<event-id>" --instruction "場所: 第2会議室" \
  --user-id admin --admin

PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work \
  --type event_delete --target "<event-id>" --instruction "中止になった" \
  --user-id admin --admin
```

通知対象抽出、まとめ承認、完了確認も event 系 work type として利用できます。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type event_notify --instruction "days: 1" --user-id admin --admin
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type event_batch_approval --instruction "channel: events" --user-id admin --admin
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type event_complete --target "<event-id>" --instruction "完了確認済み" --user-id admin --admin
```

workerから定期実行する場合は、`event_reminder` と `event_approval_batch` のjob typeを使います。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli worker --job-type event_reminder --payload-json '{"days":1,"kind":"before"}'
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli worker --job-type event_reminder --payload-json '{"kind":"day_of"}'
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli worker --job-type event_approval_batch --payload-json '{"instruction":"channel: events"}'
```

workflow payloadでは、主結果は `event_candidates`、`event_change_candidates`、`event_approval_batches`、`events`、`tasks`、`approvals` に出ます。抽出条件、degraded理由、重複情報、batch id、通知件数などの診断情報はトップレベルではなく `metadata` 配下に入ります。

内部では foundation app context を作り、`foundation.migrations.apply()` を呼びます。
適用された migration と、スキップされた migration を JSON で表示します。

### `ingest backfill`

connector から raw item や chunk を取り込む入口です。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ingest backfill --source file --limit 20
```

主なオプションは次の通りです。

- `--source`: 対象 connector。複数回指定可能。省略時は有効な connector 全体が対象
- `--limit`: 取り込み件数の上限
- `--force`: 強制的に処理する

非同期処理である `ingestion.service.backfill_many(...)` を、CLI から `asyncio.run(...)` で実行しています。

### `ask`

統合質問応答の入口です。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli ask --question "次回の活動予定は？"
```

通常は retrieval app context を作り、`retrieval.ask.ask(...)` を呼びます。
質問文、source filter、mode、depth、アクセス情報を `RetrievalQuery` にまとめて渡します。

`--depth deep` の場合だけ、通常の retrieval ではなく agentic search を使います。
この場合は `build_agentic_app_context()` で agentic context を作り、`agentic.agentic_search.search(...)` を呼びます。

アクセス制御に関係する値として、次のオプションがあります。

- `--user-id`
- `--guild-id`
- `--role-id`
- `--admin`

これらは `AccessContext` にまとめられ、回答時の権限判定や表示制御に使われます。
`--source` には `image`、`member`、`task`、`event` も指定できます。専用データが未登録の場合は、存在や利用可否を断定せず安全な応答になります。

### `work`

タスク管理、予定管理、文書下書き、告知文作成、Minecraft 支援などの workflow 機能を実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli work --type task_add --instruction "新歓資料を作成"
```

`--type` で作業の種類を選びます。

代表例は次の通りです。

- `meeting_prepare`: 会議準備
- `meeting_minutes_draft`: 議事録下書き
- `task_extract`: 指示文からタスク候補を抽出
- `task_add`: タスク追加
- `task_list`: タスク一覧
- `task_done`: タスク完了
- `task_update`: 正本 Task の変更候補を作成
- `task_delete`: 正本 Task の論理削除候補を作成
- `task_notify_due`: 期限前・期限超過通知対象を抽出し通知済み情報を記録
- `task_batch_approval`: 自動抽出候補と変更候補をまとめ承認 batch に集約
- `event_add`: イベント候補を作成
- `event_list`: イベント一覧
- `event_brief`: イベント概要
- `schedule_add`: スケジュール候補を作成
- `schedule_list`: スケジュール一覧
- `doc_draft`: 文書下書き
- `x_draft`: X 投稿文下書き
- `announcement_draft`: 告知文下書き
- `mc_status`: Minecraft サーバー状態確認
- `mc_request`: Minecraft 関連操作リクエスト
- `image_search`: 登録済み Asset と画像検索indexから画像候補を検索。再利用可否は判断しません。
- `member_search`: 権限付きでメンバー候補を検索

入力は `WorkRequest` にまとめられ、`workflow.workflow.run(...)` に渡されます。
結果は `_workflow_response_payload()` で JSON 向けに整形されます。
`task_add`、`task_extract`、`event_add`、`schedule_add` は正本を直接登録せず、候補として返します。Task 正本の変更・削除も `task_change_candidates` に保存され、承認されるまで `tasks` には反映されません。正本に入るのは、`approval --type task|event|schedule --action approve` で承認された後です。
タスク系の診断情報、抽出器、重複検出結果、通知条件、batch id はトップレベルではなく `metadata` に入ります。
`member_search` は organizer / admin 権限がない場合、対象情報の有無を示唆しない拒否応答を返します。

### `approval`

承認操作の入口です。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli approval --type task --action list
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli approval --type event --action approve --target-id <candidate-id>
```

`--type` には `task`、`event`、`schedule`、`announcement`、`automation_rule`、`server_operation`、`finance_record`、`member_assignment`、`other` を指定できます。
`--action` では次の操作を指定できます。

- `list`: 承認待ち一覧
- `show`: 詳細表示
- `approve`: 承認
- `reject`: 却下
- `edit`: 編集

`work` と同じ workflow app context を使い、`workflow.workflow.approval(...)` を呼びます。
`task`、`event`、`schedule` は候補を正本へ昇格できます。`task` では `TaskCandidate` の承認に加え、`TaskChangeCandidate` の承認も扱います。変更候補は承認後に正本 Task を更新し、削除候補は物理削除ではなく `status=deleted` の論理削除として扱います。それ以外の type は現時点では承認記録のみを保存し、外部投稿、サーバー操作、会計確定などの副作用は実行しません。

### `automation`

自動化ルールの確認・実行・有効化などを行う入口です。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action list
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli automation --action dry_run --rule-id auto_index_daily
```

`--action` で次の操作を選びます。

- `list`: ルール一覧
- `show`: ルール詳細
- `dry_run`: 実行せずに試す
- `run`: 実行する
- `enable`: 有効化
- `disable`: 無効化
- `set_mode`: 実行モードを変える

`--mode` には次の値を指定できます。

- `dry_run`: 試行のみ
- `approval_required`: 承認必須
- `auto_run`: 自動実行

`trigger-key` や `idempotency-key` は、自動化の起動理由や重複実行防止に使う値です。

## `main()` の処理

`main()` は、このファイルの中心です。

まず `_build_parser()` で parser を作り、`parser.parse_args()` でユーザー入力を解析します。
その後、`args.command` を見て、どの処理を実行するかを `if` 文で分岐します。

### 早い段階で処理されるコマンド

`bot`、`api`、`worker`、`admin`、`db`、`ingest`、`ask`、`work`、`approval`、`automation` は、`build_runtime_context()` を作る前に処理されます。

これは、それぞれが専用の app context を持っているためです。
たとえば `ask` は retrieval app context、`work` は workflow app context、`automation` は automation app context を使います。

### 後半で処理される互換系コマンド

前半の分岐に当てはまらなかった場合、`build_runtime_context()` を呼びます。
この context は、従来の console / chat / RAG / index / eval などの usecase をまとめた実行環境です。

その後、次のコマンドを処理します。

- `repl`
- `chat`
- `tool rag`
- `index build`
- `index update`
- `eval ragas`

この構成から、`cli.py` には「新しい app context を使うコマンド」と「従来互換の runtime context を使うコマンド」が同居していることがわかります。

## 出力形式

この CLI の出力は、コマンドによって少し違います。

`chat` は回答本文だけを表示します。

```text
KUMCは...
```

一方で、`ask`、`work`、`automation`、`index`、`eval` などは JSON を表示します。

JSON にしている理由は、人間だけでなく、スクリプト、CI、外部ツールからも結果を読み取りやすくするためです。
`ensure_ascii=False` を指定しているため、日本語は `\uXXXX` のようなエスケープではなく、そのまま表示されます。

payload のトップレベルには、回答本文、処理結果、作成されたタスクやイベントなど、利用者や外部連携先が主結果として扱う安定フィールドを置きます。
一方で、ルーティング判断、選択された内部 handler、実行モード、ポリシー判定、trace 情報のような診断情報は、payload 種別にかかわらず `metadata` に入れます。
この方針により、コマンドごとに診断情報の置き場所がばらつくことを避けています。

## ログ設定

一部のコマンドでは、実行前に `configure_logging(...)` を呼んでいます。

ログレベルは設定ファイルや環境変数から読み込まれた config に従います。
ログファイルの保存先は `default_execution_log_path(base_dir=...)` で決まります。

ログは、CLI で何を実行したか、インデックス作成で何件処理したか、評価がどう終わったかなどを追跡するために使われます。

## このファイルを読むときのポイント

`cli.py` を読むときは、次の順番で見ると理解しやすいです。

1. `_build_parser()` で「どんなコマンドがあるか」を見る
2. `main()` で「そのコマンドがどの context や service に渡されるか」を見る
3. 実際の処理内容を知りたい場合は、呼び出し先の `apps`、`features`、`usecases` を読む

たとえば `ask` の中身を詳しく知りたい場合、`cli.py` だけを読んでも検索や回答生成の詳細はわかりません。
`cli.py` では `build_retrieval_app_context()` と `retrieval.ask.ask(...)` を呼んでいるので、次は `src/kumc_agent/apps/retrieval.py` や `src/kumc_agent/features/retrieval/` を追うのが自然です。

同じように、`work` の詳細は workflow、`automation` の詳細は automation、`index` の詳細は indexing の usecase や feature を見る必要があります。

## まとめ

`src/kumc_agent/cli.py` は、KUMC-Agent のコマンドライン入口をまとめたファイルです。

このファイルは、ユーザーが入力したコマンドを解析し、適切な app context、service、usecase、frontend へ処理を渡します。
実際の業務ロジックは別ファイルに分かれており、`cli.py` はそれらを起動するための交通整理役です。

プロジェクト未理解の状態で読む場合は、まず `cli.py` で「どんな入口があるか」を把握し、その後に興味のあるコマンドの呼び出し先を追うと、全体像をつかみやすくなります。
