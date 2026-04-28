# KUMC-Agent 本番実装向け 詳細設計・移行計画

作成日: 2026-04-24
対象サービス: KUMC-Agent
形態: Discord Bot / RAG Agent / 業務自動化基盤
前提: 現行の仮実装を、本番運用可能なアーキテクチャへ大規模変更する
反映元: kumc-agent-workflows.md の Agent / Workflow 定義


## 0. この文書の目的
この文書は、KUMC-Agent の現行仮実装を、実務で使える AI エージェントとして再設計・再実装するための詳細設計書である。
現行設計は「現在のコードと同じ外部挙動を再実装する」ことを主目的としている。一方、本設計では次の方針へ変更する。
1. **互換再実装ではなく、本番向けの再設計を行う。**
2. **LLM にコマンド自由実行させず、定義済み tool / command をサーバー側で検証して実行する。**
3. **RAG・総合エージェント・資料作成・タスク管理・スケジュール管理・自動化などの機能を同じ基盤上に載せる。**
4. **セキュリティ、監査ログ、評価、観測可能性を初期実装から入れる。**
5. **kumc-agent-workflows の Agent 群を、検索・生成・承認・正本管理・自動化の usecase として統合する。**


## 1. プロダクト定義
### 1.1 サービス概要
KUMC-Agent は、KUMC の Discord サーバー上で動作する AI エージェントである。
主な役割は次の通り。
- Google Drive、Discord、Notion、X、はてなブログ、クラフターズコロニー、Minecraft Wiki などの情報を横断検索し、根拠付きで回答する。
- Discord / Google Drive / X / はてなブログ / クラフターズコロニー上の画像を検索する。
- 総合エージェントで複数回の検索・検証・候補作成・承認申請を扱う。
- 定義済みの安全なサーバー操作コマンドを、承認付きで実行する。
- X 投稿案、告知文、議事録、活動資料、企画資料、依頼文、ブログ記事などを生成する。
- メンバー情報を権限付きで検索し、担当候補を「候補」として提示する。
- タスク、イベント、スケジュール、投稿オートメーションを候補抽出から承認・正本登録まで管理する。
- 定例処理、Indexing、通知、期限リマインドを自動化する。
- 意思決定に必要な根拠、論点、選択肢、リスクを提示する。

### 1.2 主要ユーザー
| ユーザー | 想定行動 | 重要要件 |
|---|---|---|
| 会長・運営メンバー | 過去資料検索、企画判断、資料作成、タスク管理 | 正確性、引用、承認付き実行、監査ログ |
| 一般メンバー | サークル情報の質問、活動情報の確認 | 簡単な操作、誤回答抑制、権限に応じた情報制御 |
| 開発者・管理者 | index 更新、コネクタ追加、運用監視 | 拡張性、可観測性、ロールバック |

### 1.3 今回の実装範囲
本設計では、段階的な最小版ではなく **最初からフルプロダクトを実装対象** とする。実装順は分割するが、スコープ削減のための簡易版は置かない。各機能は feature flag、risk policy、承認フロー、staging / production の切替で公開状態を制御する。

初期から実装対象に含める機能は次の通り。
- Discordチャット上での権限付き RAG 質問応答
- Google Drive / Discord / Notion / はてなブログ / X / クラフターズコロニー記事 / Minecraft Wiki の取り込み
- Minecraft Wiki 検索 Agent: 記事・節単位検索、バージョン抽出、Java / Bedrock 差分整理
- 画像検索 Agent: 画像説明文、OCR、画像特徴量、類似画像検索
- メンバー検索 Agent: ロール、自己紹介、過去担当履歴、スキルに基づく担当候補提示
- 秘密情報・個人情報・運用秘密の検出、引用抑制、回答抑制
- 総合エージェントと統合検索
- 例会準備Agent: 直近ログ、未完了タスク、前回議事録、イベント日程から議題案と確認事項を作る
- 議事録作成 Agent: 決定事項、未決事項、ToDo、担当者、期限、次回確認事項を抽出する
- タスク候補抽出: Discord / Drive / Notion から ToDo 候補を抽出し、承認後に Task 正本へ入れる
- イベント候補抽出: 日時・場所・状態を抽出し、承認後に Event 正本へ入れる
- 返信・メッセージ投稿オートメーション候補抽出と承認付き登録
- X 投稿案・告知文・依頼文・ブログ記事・資料下書き生成
- 管理者向け index 同期、自動 Indexing、検索品質チェック
- 承認・正本管理、監査ログ、評価、基本的なメトリクス
- VC 録音・文字起こし連携
- 画像生成 API 連携のためのプロンプト生成、staging 検証、本番課金運用ガード

ただし、以下の機能は production での既定状態を `disabled` または `approval_required` にする。
- VC 録音・文字起こし: プライバシー方針と参加者同意フローが整うまで `disabled`
- 画像生成 API の本番課金運用: cost cap と承認フローが整うまで `approval_required`
- Minecraft サーバー操作、会計確定、外部投稿、自動返信: production では `approval_required` 以上


## 2. 現行実装からの主要変更点
### 2.1 変更方針
現行実装は、Clean Architecture 風の分割を持ちつつも、次の特徴が強い。
- OpenClaw を優先し、失敗時に Local RAG へ fallback する。
- message prefix `/ai` を中心に Discord 操作を受ける。
- HTTP / DocGen は stub である。
- VC、local LLM など、現行の運用方針と合わない機能が残っている。

新設計では、これを次のように変更する。
| 項目 | 現行 | 新設計 |
|---|---|---|
| AI 入口 | OpenClaw 優先 + Local RAG fallback | 内部 Router / RAG / Agent Engine を主経路。OpenClaw は任意 adapter |
| Discord UI | `/ai` message command 中心 | slash command / component / modal 中心。`/ai` は廃止|
| index 正本 | ローカル artifact | PostgreSQL 正本 + 検索 index + object storage |
| worker | frontend と強く結合 | bot / api / worker を分離 |
| LLM | Gemini 中心 | LLM・embedding・rerank は外部 API 前提 |
| RAG | hybrid(dense + sparse) + rerank + Doc Cap + MMR | hybrid(dense + sparse) + rerank + Doc Cap + MMR + ACL + eval + trace |
| Agent | OpenClaw 外部依存 | 内部状態機械型 総合エージェント |
| 業務 Workflow | 個別機能なし / 手作業 | Workflow registry + Candidate + Approval + 正本 DB |
| 画像・素材検索 | 未整備 | OCR / caption / feature embedding / usage approval |
| Minecraft Wiki | 未整備 | Wiki connector + edition/version aware RAG |
| Command 実行 | 未整備 | 定義済み command registry + schema 検証 + 承認 |
| DocGen | NotImplemented | 中間表現 + template + exporter |
| Security | 安全制御 stub 等が未接続 | threat model / prompt injection 防御 / audit / ACL を必須化 |
| Observability | answer log / prompt log 中心 | trace / metric / structured log / cost log / audit log |

### 2.2 廃止・縮小する機能
以下は新設計では廃止または縮小する。
| 機能 | 方針 | 理由 |
|---|---|---|
| CrossEncoder local reranker | 廃止 | rerank も外部 API または軽量 LLM judge へ移行 |
| OpenClaw 優先 routing | 縮小 | 自前実装の自由度を優先するため |
| `data/index` artifact 正本 | 廃止 | DB / search index を正本にするため |
| VC 機能 | 後回し | 本番では優先度が低く、実装負荷が高いため |
| `/ai build-index` message command | 廃止 | slash command + 管理者権限へ移行 |


## 3. 設計原則
### 3.1 変更に強い設計
KUMC-Agent は、機能が随時追加され、途中で大胆な変更が入る前提で設計する。
そのため、次の要素を分離する。
- ユーザーインターフェース
- RAG / Search
- Agent orchestration
- Tool / Action 実行
- Connector
- Workflow / Automation
- Eval
- Audit / Security
外部 API、LLM provider、検索基盤、投稿対象サービスが変わっても、domain model と usecase が壊れない構成にする。

### 3.2 人間承認を前提にした Agent
KUMC-Agent は「自律的に何でも実行する Bot」ではなく、「根拠を集め、案を作り、必要に応じて承認付きで実行する業務補助 Agent」として設計する。
副作用のある操作は次の原則を守る。
1. LLM は実行内容を提案するだけ。
2. サーバー側が schema、権限、risk、rate limit を検証する。
3. 中リスク以上の操作は dry-run を表示する。
4. 必要な操作は人間が承認する。
5. 実行結果を audit log に保存する。

### 3.3 RAG 文書は命令ではなく未信頼データ
Google Drive、Discord、X、ブログ、PDF などから取得した文章には、prompt injection が含まれる可能性がある。
したがって、検索結果の文書はすべて **data** として扱い、system instruction / tool policy / command policy と混同しない。

### 3.4 評価駆動で開発する
AI 機能は、通常の unit test だけでは品質を保証できない。
各機能について、実装前に次を定義する。
- 期待回答
- 禁止回答
- 必須引用
- 許容される不確実性表現
- コスト上限
- latency 上限
- 権限違反がないこと
この評価セットを CI / staging で回し、品質劣化を検出する。


## 4. ターゲットアーキテクチャ
### 4.1 全体構成
新設計では、**モジュラモノリス + 役割別プロセス分離** とする。
デプロイ単位は次の 3 つ。
```text
bot      : Discord Gateway / Interaction 受付
api      : 管理 API、Webhook 受信、承認 UI 用 API
worker   : ingestion、embedding、indexing、agent search、doc generation、automationなど
```

内部 module は同一 repository に置く。
```text
/apps
  /bot
  /api
  /worker
/libs
  /domain
  /application
  /connectors
  /retrieval
  /agents
  /actions
  /approval
  /assets
  /members
  /workflows
  /automation
  /docgen
  /evals
  /security
  /observability
  /providers
/prompts
/tool_schemas
/configs
/docs
  /adr
  /runbooks
  /evals
/infrastructure
/tests
```

### 4.2 推奨スタック
| 領域 | 推奨 | 備考 |
|---|---|---|
| 言語 | Python 3.12+ | 既存資産と開発者習熟を優先 |
| Bot | discord.py 系 | slash command / interaction を主に使う |
| API | FastAPI | Webhook / 管理 API / 承認 UI 用 |
| DB | PostgreSQL | domain 正本、audit、task、workflow |
| Vector | FaissLikeIndex | ファイルベースの dense index を標準運用 |
| Sparse search | Elasticsearch | BM25 / Sudachi を使用 |
| Queue | Celery | worker job 管理 |
| Object Storage | S3 互換 | raw snapshot、export、生成資料 |
| LLM | 外部 API | OpenAI / Gemini を adapter 化 |
| Embedding | 外部 API | OpenAI / Gemini を adapter 化 |
| Rerank | 外部 API or LLM judge　を選択 | local model は使わない |
| Observability | OpenTelemetry + structured log | LLM call / tool call / retrieval を trace |
| Secret | Google Secret Manager | `.env` 直置きは local 限定 |

### 4.3 処理の流れ
通常質問の流れ。
```text
Discord slash command
  -> bot: interaction defer
  -> application: QueryUsecase
  -> security: user / role / guild scope resolve
  -> retrieval: hybrid search
  -> generation: answer with citations
  -> audit: query log / retrieval log / generation log
  -> bot: follow-up response
```

総合エージェントの流れ。
```text
Discord slash command
  -> bot: interaction defer
  -> application: IntegratedInputUsecase
  -> agents: PLAN
  -> tools: TOOL
  -> agents: VERIFY
  -> generation: ANSWER
  -> audit / trace
  -> bot: follow-up response
```

副作用のある command 実行の流れ。
```text
Discord command
  -> bot: interaction defer
  -> application: ActionProposalUsecase
  -> LLM: command_id + args proposal
  -> server: schema validation / ACL / risk check
  -> dry-run output
  -> human approval button
  -> worker: isolated executor
  -> audit log
  -> result response
```


## 5. アプリケーション境界
### 5.1 `bot`
責務:
- Discord connection 管理
- slash command 登録
- interaction 受付
- button / modal の callback 受付
- channel / guild / role / user 情報の解決
- 長時間処理の defer / follow-up
- Discord 表示形式への整形

禁止事項:
- 重い ingestion を直接行わない。
- LLM API を直接呼ばない。
- raw data / chunk / embedding を直接操作しない。
- コマンド実行を直接行わない。

### 5.2 `api`
責務:
- health check
- admin API
- Webhook 受付
- Google Drive / X / external service の callback 受付
- 承認 UI 用 API
- workflow rule の CRUD
- eval result / trace result の閲覧 API
最低限のUIを用意する。API と簡易 JSON response から開始する。

### 5.3 `worker`
責務:
- source backfill
- incremental sync
- document normalization
- chunking
- embedding
- index update
- long-running comprehensive agent
- doc generation
- automation execution
- command execution executor への dispatch
worker は idempotent にする。ジョブが再実行されても、同じ source version から同じ document / chunk が作られることを目標にする。


## 6. Domain モデル
### 6.1 基本方針
Domain モデルは外部 SDK に依存させない。
Discord、Google Drive、X、Hatena などの SDK 固有構造は connector 層で吸収し、domain では共通の `SourceItem`, `Document`, `Chunk`, `SearchResult`, `AgentRun`, `ActionRun` として扱う。

### 6.2 主要エンティティ
```python
@dataclass(frozen=True)
class SourceAccount:
    id: str
    kind: Literal["discord", "google_drive", "x", "hatena", "notion", "crafters_colony", "minecraft_wiki", "member_profile", "manual"]
    display_name: str
    enabled: bool
    metadata: dict[str, Any]

@dataclass(frozen=True)
class SourceItem:
    id: str
    source_account_id: str
    source_kind: str
    external_id: str
    canonical_url: str | None
    title: str | None
    author_id: str | None
    created_at: datetime | None
    updated_at: datetime | None
    access_scope: AccessScope
    raw_object_key: str
    checksum: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class Document:
    id: str
    source_item_id: str
    version: int
    title: str
    normalized_text: str
    normalized_format: Literal["markdown", "plain", "csv_as_text", "slide_text", "wiki_markdown", "image_caption", "ocr_text", "asset_metadata"]
    language: str | None
    access_scope: AccessScope
    checksum: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    source_item_id: str
    chunk_index: int
    chunk_kind: Literal["body", "summary", "table", "slide", "message_window", "thread", "tweet", "heading", "wiki_article", "wiki_section", "image_caption", "ocr", "asset_metadata"]
    text: str
    token_count: int
    parent_chunk_id: str | None
    access_scope: AccessScope
    metadata: dict[str, Any]
```

### 6.3 Answer モデル
```python
@dataclass(frozen=True)
class Answer:
    answer_id: str
    text: str
    route: Literal[
        "rag",
        "comprehensive_agent",
        "no_answer",
        "action_proposal",
        "doc_generation",
        "x_draft",
        "wiki_search",
        "image_search",
        "member_search",
        "workflow_candidate",
    ]
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    warnings: list[str]
    metadata: dict[str, Any]

@dataclass(frozen=True)
class Citation:
    source_item_id: str
    chunk_id: str
    label: str
    url: str | None
    quote: str | None
    score: float | None
```

### 6.4 AccessScope
```python
@dataclass(frozen=True)
class AccessScope:
    visibility: Literal["public", "guild", "role", "private", "admin"]
    guild_id: str | None
    role_ids: list[str]
    user_ids: list[str]
    source_acl_hash: str | None
```

検索時点で `AccessScope` を使って絞り込む。検索後に unauthorized chunk を消すだけでは不十分である。

### 6.5 業務正本エンティティ
```python
@dataclass(frozen=True)
class TaskCandidate:
    id: str
    title: str
    description: str | None
    proposed_assignee_user_id: str | None
    proposed_due_at: datetime | None
    related_event_id: str | None
    evidence: list[Citation]
    confidence: Literal["high", "medium", "low"]
    status: Literal["proposed", "approved", "rejected", "merged"]
    created_by: Literal["agent", "user"]

@dataclass(frozen=True)
class Asset:
    id: str
    asset_type: Literal["image", "article", "document", "icon", "screenshot", "other"]
    title: str
    object_key: str | None
    canonical_url: str | None
    owner_user_id: str | None
    related_event_id: str | None
    access_scope: AccessScope
    metadata: dict[str, Any]

@dataclass(frozen=True)
class MemberProfile:
    id: str
    user_id: str
    display_name: str
    roles: list[str]
    grade: str | None
    skills: list[str]
    interests: list[str]
    past_assignments: list[dict[str, Any]]
    access_scope: AccessScope
    metadata: dict[str, Any]

@dataclass(frozen=True)
class WorkflowCandidate:
    id: str
    candidate_type: Literal["task", "event", "announcement", "automation_rule", "asset_usage", "server_operation", "member_assignment", "other"]
    payload: dict[str, Any]
    evidence: list[Citation]
    confidence: Literal["high", "medium", "low"]
    status: Literal["proposed", "needs_review", "approved", "rejected", "merged", "archived"]
    approval_policy: str
    created_by: Literal["agent", "user", "automation"]
    related_agent_run_id: str | None
    metadata: dict[str, Any]

@dataclass(frozen=True)
class ServerOperation:
    id: str
    server_name: str
    operation: Literal["status", "file_search", "docker_ps", "compose_up", "compose_down", "compose_restart", "start", "stop", "restart", "op_add", "op_remove", "whitelist_update", "backup"]
    requested_by_user_id: str
    approved_by_user_ids: list[str]
    status: Literal["requested", "approved", "running", "succeeded", "failed", "rejected"]
    risk_level: Literal["low", "medium", "high", "critical"]
    action_run_id: str | None
    metadata: dict[str, Any]

@dataclass(frozen=True)
class SecretFinding:
    id: str
    source_item_id: str
    chunk_id: str | None
    secret_type: Literal["credential", "pin", "internal_ip", "network_key", "unlock_procedure", "personal_data", "finance", "external_confidential", "other"]
    severity: Literal["low", "medium", "high", "critical"]
    redaction_policy: Literal["quote_allowed", "summary_only", "deny", "admin_only"]
    detected_span_hash: str
    status: Literal["active", "false_positive", "resolved"]
```


## 7. DB 設計
### 7.1 方針
DB の詳細 DDL はこの詳細設計書には載せず、`/infrastructure/migrations`、`/docs/adr`、repository interface の型定義に分離する。この文書では、実装で守るべき DB 方針と主要 table の責務だけを定義する。

PostgreSQL を業務正本にする。raw file や生成資料そのものは object storage に置き、DB には参照 key と metadata を保存する。dense 検索 index は FaissLikeIndex、sparse 検索 index は Elasticsearch などに分散するが、正本は DB 側に置く。

### 7.2 主要 table 群
最低限、次の table 群を migration 側で定義する。
```text
source_accounts
source_items
documents
chunks
chunk_acl_entries
embeddings
search_runs
search_run_results
agent_runs
agent_steps
llm_calls
tool_calls
action_specs
action_runs
action_approvals
jobs
events
meetings
task_candidates
tasks
schedule_events
announcements
assets
asset_usage_requests
member_profiles
minecraft_wiki_articles
workflow_candidates
workflow_runs
approval_records
indexing_runs
finance_records
server_operations
secret_findings
automation_rules
automation_runs
audit_logs
eval_sets
eval_cases
eval_runs
eval_results
sync_cursors
```

### 7.3 DB 実装上の必須条件
- `source_kind + external_id + checksum` を source identity の基本にする。
- `source_items.deleted_at` と `index_status` を持ち、削除済み・権限喪失済み・quarantine 済みの source を検索対象から外す。
- ACL は `access_scope jsonb` だけに閉じず、検索前 filter 用に `chunk_acl_entries` または同等の denormalized table を持つ。
- `status`, `risk_level`, `confidence`, `visibility` などは migration 側で CHECK constraint または enum を設定する。
- vector は model / dimensions ごとに一貫した index strategy を定義する。
- `audit_logs` は app 用 DB user から update/delete できない append-only table とする。
- `secret_findings` は `chunk_id is null` の重複を防ぐため、migration 側で partial unique index を使う。例: item-level finding と chunk-level finding の unique index を分ける。
- side effect action は必ず `idempotency_key` を保存する。
- 長時間処理は `jobs` table で status、actor、input、result、error、interaction token reference を追跡する。

### 7.4 DB 詳細仕様の置き場所
詳細 DDL は以下に分けて管理する。
```text
/infrastructure/migrations/*.sql
/docs/adr/ADR-002-db-source-and-index.md
/docs/adr/ADR-003-event-task-meeting-source-of-truth.md
/docs/adr/ADR-004-secret-finding-redaction-policy.md
/libs/domain/*
/libs/application/repositories/*
```
この文書は、DB の完全な CREATE TABLE 定義ではなく、設計意図と実装上の制約を示す。

## 8. 設定設計
### 8.1 設定ファイル構成
```text
/configs
  /main
    app.yaml
    providers.yaml
    security.yaml
    features.yaml
    rag.yaml
    indexing.yaml
    integrations.yaml
    evaluation.yaml
    summarization.yaml
    scheduler.yaml
```

設定の merge 順:
1. `configs/main/*.yaml`
2. environment variables
環境変数の対応先が未知 key の場合は起動時にエラーにする。

### 8.2 環境変数
必須:
```text
KUMC_ENV
KUMC_DISCORD_BOT_TOKEN
KUMC_DATABASE_URL
KUMC_REDIS_URL
KUMC_OBJECT_STORAGE_BUCKET
KUMC_LLM_PROVIDER
KUMC_LLM_API_KEY
KUMC_EMBEDDING_PROVIDER
KUMC_EMBEDDING_API_KEY
```

任意:
```text
KUMC_RERANK_PROVIDER
KUMC_RERANK_API_KEY
KUMC_GOOGLE_CLIENT_ID
KUMC_GOOGLE_CLIENT_SECRET
KUMC_GOOGLE_SERVICE_ACCOUNT_JSON
KUMC_DRIVE_FOLDER_ALLOWLIST
KUMC_X_BEARER_TOKEN
KUMC_HATENA_BLOG_URL
KUMC_NOTION_API_TOKEN
KUMC_NOTION_PAGE_ALLOWLIST
KUMC_NOTION_DATABASE_ALLOWLIST
KUMC_MINECRAFT_WIKI_BASE_URL
KUMC_IMAGE_FEATURE_PROVIDER
KUMC_IMAGE_FEATURE_API_KEY
KUMC_CRAFTERS_COLONY_BASE_URL
KUMC_OPENCLAW_ENABLED
KUMC_OPENCLAW_AGENT
KUMC_OTEL_EXPORTER_OTLP_ENDPOINT
KUMC_SECRET_MANAGER_PROVIDER
```

### 8.3 provider 設定
```yaml
providers:
  llm:
    default:
      provider: openai
      model: gpt-4.1-mini
      temperature: 0.2
      max_output_tokens: 1200
    reasoning:
      provider: openai
      model: gpt-4.1
      temperature: 0.1
      max_output_tokens: 3000
    cheap:
      provider: openai
      model: gpt-4.1-nano
      temperature: 0.0
      max_output_tokens: 800
  embedding:
    default:
      provider: openai
      model: text-embedding-3-large
      dimensions: 1536
  rerank:
    default:
      provider: external_reranker
      model: rerank-multilingual-v1
```
provider 名と model 名は adapter で吸収する。domain 層に provider 固有名を漏らさない。

### 8.4 Discord role / permission 設定
実際の Discord role id は現時点では未確定のため、空配列を仮値として置く。後から `configs/env/*.yaml` で上書きする。

```yaml
roles:
  member:
    discord_role_ids: []
  organizer:
    discord_role_ids: []
  admin:
    discord_role_ids: []
  finance:
    discord_role_ids: []
  public_relations:
    discord_role_ids: []
  server_operator:
    discord_role_ids: []

permissions:
  default_response_visibility: public_summary
  commands:
    ask: [member]
    work: [member]
    approval: [organizer, admin]
    automation: [organizer, admin]
    admin: [admin]
  sources:
    google_drive: [member]
    discord: [member]
    notion: [organizer, admin]
    x: [member]
    hatena: [member]
    crafters_colony: [member]
    minecraft_wiki: [member]
    member_profile: [organizer, admin]
  actions:
    low: [member]
    medium: [organizer, admin]
    high: [admin]
    critical: [admin]
```

上記は仮の初期値であり、production では guild ごとの実 role id と運用方針に合わせて変更する。

## 9. Connector 設計
### 9.1 共通 Interface
```python
class SourceConnector(Protocol):
    source_kind: str

    async def backfill(self, scope: BackfillScope) -> AsyncIterator[SourceRawItem]:
        ...

    async def poll_changes(self, cursor: SyncCursor) -> AsyncIterator[SourceRawItem | SourceDeleteItem]:
        ...

    async def fetch_item(self, external_id: str) -> SourceRawItem:
        ...

    async def normalize(self, raw: SourceRawItem) -> NormalizedDocument:
        ...
```
Connector は source 固有の処理だけを持つ。chunking、embedding、index update は connector の責務ではない。

### 9.2 Google Drive Connector
対象:
- Google Docs
- Google Sheets
- Google Slides
- PDF
- Word / Excel / PowerPoint

設計:
- 初回は recursive backfill する。
- 増分同期は Drive Changes API の start page token / changes list を使う。
- export できる Google Workspace ファイルは export する。
- export サイズ制限や構造維持が必要な場合、Docs / Sheets / Slides の個別 API に fallback する。
- raw export は object storage に保存する。
- normalized text は DB `documents` に保存する。
- Drive file permission を完全再現できない場合は fail closed とし、`admin_only` または indexing 保留にする。folder allowlist と KUMC role ACL は補助条件として使う。

Normalized format:
| 種類 | 変換 |
|---|---|
| Docs | Markdown 風 text。見出しを保持 |
| Sheets | sheet ごとに table block 化。ヘッダ付き行テキストにする |
| Slides | slide ごとに title/body/speaker note を抽出 |
| PDF | text layer 優先。必要なら OCR job を別 queue へ |
| Office | text / table extract。失敗時は raw only + warning |

### 9.3 Discord Connector
対象:
- allowlist された guild
- text channel
- thread
- archived thread

設計:
- `message_content` 権限が必要な範囲を明示する。
- 初回 backfill は channel ごとに page 取得する。
- 増分は Gateway event と定期 reconciliation を併用する。
- Bot / webhook / system message は原則除外する。ただし運営上必要な bot message は allowlist 可能にする。
- message は thread / reply chain / 時間窓で grouping し、chunk にする。
- user mention は表示名に置換するが、raw には元 ID を保持する。
- 取得できる message URL を canonical_url とする。

Discord chunking:
| 種類 | chunk 単位 |
|---|---|
| thread | thread 全体を複数 chunk に分割 |
| reply chain | reply 関係をまとめる |
| normal channel | 10〜20 message または 10分窓 |
| announcement | 1 message 単位 |

### 9.4 Hatena Blog Connector
設計:
- 初期は Atom / RSS feed から取得する。
- blog URL は config で指定する。
- 記事本文を Markdown 化する。
- title、URL、created、updated を metadata に保存する。
- feed に出なくなった記事の削除検知は、site map / archive page との照合または定期 full scan で補完する。

### 9.5 X Connector
設計:
- 初期は投稿案生成に必要な reference source として扱う。
- 自動投稿は `approval_required` を既定とし、feature flag で enable / disable を切り替える。
- 既存アーカイブがある場合は import connector として扱う。
- API 連携する場合は、recent search と user timeline を分ける。
- X から取得した外部投稿は信頼度を低く扱い、事実確認 source には使いすぎない。


### 9.6 Notion Connector
対象:
- Notion page
- Notion database
- database item
- page 内 block tree

設計:
- Notion API token は Secret Manager で管理する。
- 取得対象は `KUMC_NOTION_PAGE_ALLOWLIST` と `KUMC_NOTION_DATABASE_ALLOWLIST` で制限する。
- page / database / block を source item として正規化し、block tree を Markdown 風 text に変換する。
- database property は metadata と table summary の両方に入れる。
- synced block、toggle、callout、table、file attachment は可能な範囲で保持する。
- archived page / deleted block は `deleted_at` または `index_status=deleted` として検索対象から外す。
- Notion 側の権限を完全に再現できない場合は fail closed とし、`admin_only` または indexing 保留にする。
- 差分同期は last edited time と checksum を併用する。API で十分に差分検知できない場合は定期 reconciliation を行う。

Normalized format:
| 種類 | 変換 |
|---|---|
| Page | `markdown`。title、heading、paragraph、list、toggle を保持 |
| Database | `csv_as_text` または table summary。property と row を保持 |
| Block tree | heading path と block type を metadata に入れる |
| Attachment | asset または raw snapshot として保存 |

### 9.7 Minecraft Wiki Connector
設計:
- Minecraft Wiki の記事を article 単位で取得し、節単位の chunk を作る。
- article metadata として、記事名、URL、取得日時、edition、version tag、namespace を保存する。
- Java Edition / Bedrock Edition の差分が記事内にある場合は、節 metadata に edition を入れる。
- 差分同期が可能な場合は更新日時または API の revision id を使う。不可の場合は定期 full crawl と checksum 比較を使う。
- 古い取得結果に基づく回答では、取得日と不確実性を明示する。

Normalized format:
| 種類 | 変換 |
|---|---|
| Wiki article | `wiki_markdown`。見出し、infobox、table を可能な範囲で保持 |
| Wiki section | section title + body + edition/version metadata |
| Version table | table summary + original table chunk |

### 9.8 クラフターズコロニー / 外部制作物 Connector
設計:
- KUMC 関連の投稿記事、制作物名、説明、画像、投稿日時、投稿者、URL を取得する。
- 外部サイト由来の情報は、著作権、利用規約、投稿者の意図を確認する必要があるため、公開素材として自動確定しない。
- 画像は `assets` に登録し、記事本文は通常の `documents` / `chunks` として扱う。
- 外部投稿の削除や非公開化を検知した場合、検索結果から除外する。

### 9.9 Image / Asset Indexing Connector
設計:
- Discord 添付画像、Google Drive 画像、X 投稿画像、はてなブログ画像、クラフターズコロニー画像を `Asset` として正規化する。
- 画像ごとに caption、OCR text、特徴量 vector、出典 URL、投稿日時、投稿媒体、関連イベントを metadata に保存する。
- 人物の有無、ロゴの有無、公開済みかどうかを metadata として持つ。
- 画像検索結果は「利用候補」であり、`asset_usage_requests` の承認があるまで外部公開に使わない。

### 9.10 Member Profile Source
設計:
- Discord profile、自己紹介文、ロール、過去担当履歴を `member_profiles` に正規化する。
- メンバー情報は個人情報を含むため、検索前 filter と回答前 filter の両方で権限確認する。
- 担当候補推薦では、参加意思や能力を断定しない。出力は「候補」と「確認が必要な点」に限定する。
- 外部公開用の回答では、個人名、連絡先、内部ロール、参加可能性を出さない。

### 9.11 Connector 追加時のチェックリスト
新しい connector を追加する場合、以下を必ず実装する。
- `backfill`
- `poll_changes` または「差分同期不可」の明示
- raw snapshot 保存
- normalized document 生成
- canonical URL
- access scope
- checksum
- deleted / updated の扱い
- connector integration test
- sample eval case


### 9.12 Raw snapshot / 外部 source 保持方針
raw snapshot の保存期間は次を初期値とする。運用中に必要が出た場合は config で変更する。

| 種別 | 初期保存期間 | 方針 |
|---|---:|---|
| public / external source raw | 180日 | 再取得可能な source は長期保存しない |
| internal source raw | 90日 | Drive / Discord / Notion の raw は必要最小限にする |
| deleted / permission lost source | 30日以内に purge | 検索 index からは即時除外する |
| high severity secret を含む raw | 即時 quarantine、30日以内に purge | audit 用 metadata だけ残す |
| generated document | 365日 | 必要に応じて手動延長 |
| audit log | 2年 | append-only。本文・秘密情報は保存しない |

外部 source の利用規約確認は後日行う。初期状態では `terms_review_status: pending` を metadata に保持し、外部再利用・再配布・公開素材化は承認ゲートで止める。検索・要約の内部利用は、source ごとの allowlist と robots / API policy を別途確認してから enable する。

## 10. Normalization / Chunking 設計
### 10.1 基本方針
検索品質は chunk の質に大きく依存する。source ごとに chunking 方針を固定し、後から eval で改善する。

### 10.2 chunk 単位
| Source | chunk 単位 | 目安 |
|---|---|---|
| Google Docs / Word | 見出し単位 + token split | 300〜600 tokens |
| Google Sheets / Excel | sheet / table block + row summary | 1 table または 300〜600 tokens |
| Google Slides / Powerpoint | slide title/body/speaker note | 1 slide または 300〜600 tokens |
| PDF | text layer section + OCR fallback | 300〜600 tokens |
| Notion | page heading / block tree / database row summary | 300〜600 tokens |
| Discord | thread / reply chain / 時間窓 | 10〜20 message または 10分窓 |
| X | 1 post + thread context | 1〜5 posts |
| Minecraft Wiki | article section + version table | 300〜800 tokens |
| Image / Screenshot | caption chunk + OCR chunk + asset metadata | 1 asset 単位 |
| Member profile | profile section + role / skill summary | 1 member または 1項目単位 |
| Crafters Colony | article section + asset reference | 300〜600 tokens |

### 10.3 chunk metadata
最低限、次の metadata を保持する。
```json
{
  "source_kind": "google_drive",
  "source_title": "2026年度 新歓企画資料",
  "canonical_url": "...",
  "author": "...",
  "created_at": "2026-04-01T00:00:00+09:00",
  "updated_at": "2026-04-12T00:00:00+09:00",
  "folder_or_channel": "...",
  "document_version": 3,
  "chunk_kind": "heading",
  "heading_path": ["新歓", "予算"],
  "access_scope": {...},
  "checksum": "..."
}
```

### 10.4 summary chunk
長い document には summary chunk を作る。
- document summary
- section summary
- table summary
- thread summary
summary chunk は検索 recall を上げるために使うが、回答の根拠として出す場合は原文 chunk も合わせて提示する。

### 10.5 parent context
最終回答生成時は、選ばれた chunk の parent context を補助的に入れる。
ただし、citation は直接 hit した chunk を優先する。parent context のみから断定しない。

## 11. Retrieval / RAG 設計
### 11.1 検索パイプライン
標準 RAG は次の流れで行う。
```text
1. Query normalization
2. Query classification
3. Filter extraction
4. Dense retrieval top_k=40
5. Sparse retrieval top_k=40
6. Reciprocal Rank Fusion
7. ReRank top_k=20
8. MMR top_k=8
9. Parent context packing
10. Answer generation
11. Citation validation
12. Response formatting
```

### 11.2 Query classification
分類項目:
```json
{
  "intent": "fact|summary|compare|draft|decision_support|action|task|event|schedule|image|member|member_assignment|minecraft_spec|approval|unknown",
  "requires_comprehensive_agent": true,
  "requires_freshness": false,
  "source_filters": ["google_drive", "discord", "notion", "minecraft_wiki", "image", "member", "task", "event"],
  "minecraft": {"edition": "java|bedrock|unknown", "version": "optional"},
  "asset_filters": {"event": "optional", "world": "optional", "has_person": "optional"},
  "date_range": {
    "start": "2026-04-01",
    "end": "2026-04-24"
  },
  "risk": "none|low|medium|high"
}
```
`action`, `task`, `event`, `schedule`, `approval`, `member_assignment` は RAG だけで処理しない。該当 usecase へ route し、必要に応じて候補作成と承認フローに入れる。`minecraft_spec` は Minecraft Wiki 検索、`image` は Image / Asset 検索へ route する。

### 11.3 Dense retrieval
- embedding は外部 API を使う。
- query embedding は cache 可能にする。
- chunk embedding は `embeddings` table に保存する。
- source access scope を検索条件に含める。

### 11.4 Sparse retrieval
Elasticsearch BM25を使う
日本語検索では tokenizer の品質が重要である。Sudachiを使用する。

### 11.5 RRF
Dense と sparse の結果を RRF で融合する。
現行RAGコンポーネントでも、dense-vs-sparse の候補融合はスコア正規化加算ではなくRRFを使う。
```python
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)
```
同じ chunk が dense と sparse の両方で出た場合、score を加算する。

### 11.6 ReRank
ReRank は top 20 程度に限定する。
優先順位:
1. 外部 rerank API
2. 低価格 LLM による pairwise / listwise rerank
3. rerank なし fallback
ReRank には本文全体ではなく、chunk text + title + metadata を渡す。不要な個人情報や権限外情報は渡さない。

### 11.7 MMR
ReRank 後、MMR で重複を減らす。（MMRの前にDoc Capも行う）
```python
def mmr_score(relevance: float, similarity_to_selected: float, lambda_: float = 0.7) -> float:
    return lambda_ * relevance - (1 - lambda_) * similarity_to_selected
```
目的は「似た chunk ばかり回答文脈に入る」ことを防ぐこと。

### 11.8 Recency
recency は source ごとに扱いを変える。
| Source | recency の意味 |
|---|---|
| Discord | message timestamp |
| Drive | modified time |
| Hatena | article updated |
| X | posted time |
| Minecraft Wiki | retrieved_at / article updated |
| Image / Asset | posted time / captured time / indexed time |
| Member profile | profile updated / role updated |
| Task / Event / Schedule | due / start time / status updated |
質問が「最近」「次回」「今週」「最新」を含む場合は recency weight を上げる。

### 11.9 Citation validation
回答生成後、次を検査する。
- citation が存在するか。
- citation chunk が user の access scope 内か。
- 回答中の主要主張が citation に支えられているか。
- citation URL が生成されているか。
- 根拠が弱い場合、`confidence="low"` にする。

### 11.10 回答形式
Discord に返す通常回答は、長文を直接流さず **短い要約** を優先する。詳細な根拠、長い表、議事録、資料下書きは generated document、attachment、または thread follow-up に分離する。

Discord の標準出力:
```text
結論:
...  # 3〜6行程度

根拠の要約:
- ...
- ...

主な情報源:
- [資料名 / channel名](URL)
- [資料名 / channel名](URL)

詳細:
- 必要に応じて添付 Markdown / generated document / thread に保存

注意:
- 根拠が古い可能性があります。
```

長文出力の扱い:
- 2000文字を超える可能性がある回答は、Discord 本文に短い要約だけを出す。
- 議事録、週報、企画書、runbook は Markdown attachment または generated document として出す。
- citation は本文では上位 3〜5件だけを表示し、詳細側に完全版を入れる。
- 権限付き source を含む詳細出力は、ephemeral または権限付き channel に限定する。

出典がない場合は、次のように返す。

```text
確認できる範囲では、回答に使える根拠を見つけられませんでした。

確認した範囲:
- Google Drive: ...
- Discord: ...
- Notion: ...

次に試すとよいこと:
- 資料名や時期を指定する
- 対象チャンネルや source を指定する
```


## 12. 総合エージェント設計
### 12.1 基本方針
旧深掘り検索機能は総合エージェントへ統合する。総合エージェントは、単一RAGで解決できない依頼、複数機能を必要とする依頼、または単一RAGでも `depth=deep` が指定された依頼を、自由な無限ループではなく状態機械として処理する。
```text
PLAN -> TOOL -> VERIFY -> ANSWER
```
根拠不足や矛盾がある場合は、予算内で `PLAN -> TOOL -> VERIFY` を再実行する。

### 12.2 状態定義
| 状態 | 役割 | 出力 |
|---|---|---|
| PLAN | 専用LLM Plannerが入力を分解し、必要機能、tool順序、tool入力、検索条件、成功条件、副作用境界を決める | tasks, required_tools, tool_sequence, success_criteria |
| TOOL | PLANで選ばれたtoolを実行する。検索系は実行し、副作用系は候補作成または承認申請までに限定する | tool result, citations, candidates, approval targets |
| VERIFY | 専用LLM Verifierと決定的チェックで、根拠不足、矛盾、権限外情報、secret混入、副作用境界違反を判定する | satisfied, missing, conflicts, warnings |
| ANSWER | 最終回答を作る | answer, citations, confidence, candidates, approvals |

### 12.3 制約
初期値:
```yaml
comprehensive_agent:
  planner:
    enabled: true
    provider: gemini
  verifier:
    enabled: true
    provider: gemini
  budget:
    max_steps: 10
    max_search_calls: 6
    max_read_chunks: 20
    max_replans: 2
    max_cost_usd: 0.75
    max_latency_seconds: 120
    require_citations: true
```

### 12.4 Tool 一覧
総合エージェントで使える標準tool:
```text
circle_rag_search
minecraft_wiki_rag_search
member_search
image_search
task_search
task_candidate_create
event_search
event_candidate_create
server_operation_candidate_create
approval_candidate_create
```
`task_candidate_create`、`event_candidate_create`、`server_operation_candidate_create` は正本変更や外部実行を行わず、候補作成に限定する。候補作成後は承認recordまたはapproval batchを自動作成する。非adminが候補作成を含む依頼を行った場合は拒否する。

### 12.5 失敗時の出力
総合エージェントが十分な根拠を見つけられない場合、無理に回答しない。
```text
十分な根拠を見つけられませんでした。

不足している情報:
- ...

確認した検索:
- ...

次の候補:
- ...
```


## 13. Discord UI 設計
### 13.1 基本方針
Discord は slash command 中心にする。ただし command 数は増やしすぎず、少数の command group に集約する。細かい機能差は `type` / `mode` / `action` option、button、modal、承認 UI で分岐する。

方針:
- top-level slash command は最小限にする。
- 特化機能は独立 command を増やさず、`/work` または `/ask` に route する。
- 長い結果は Discord 本文に出さず、短い要約 + 詳細 attachment / generated document にする。
- 副作用のある操作は command 実行ではなく、proposal -> dry-run -> approval -> action の流れにする。
- sensitive / member / finance / admin 情報は ephemeral または権限付き channel に返す。

### 13.2 最小 command registry
| Command | 用途 | 主な option | 権限 | production 既定 |
|---|---|---|---|---|
| `/ask` | RAG / Wiki / 画像 / メンバー / タスク / イベントを含む統合質問 | `question`, `source`, `mode`, `depth` | member | enabled |
| `/work` | 業務ワークフローの実行・下書き生成・候補抽出 | `type`, `instruction`, `target`, `format` | member〜admin | enabled |
| `/approval` | 候補・投稿・素材・サーバー操作の承認 / 却下 / 編集 | `action`, `target_id`, `comment` | organizer / admin | enabled |
| `/automation` | 自動化 rule の一覧、dry-run、enable / disable、手動実行 | `action`, `rule_id`, `mode` | organizer / admin | enabled |
| `/admin` | sync、eval、health、feature flag、権限確認 | `action`, `scope` | admin | enabled |

`/wiki`, `/image`, `/member`, `/meeting`, `/task`, `/event`, `/schedule`, `/draft-x`, `/announce`, `/mc` は top-level command としては増やさず、`/ask` または `/work` の `type` で扱う。

### 13.3 `/ask`
入力:
```text
question: str
source: optional enum[all, drive, discord, notion, hatena, x, crafters_colony, minecraft_wiki, image, member, task, event]
mode: enum[answer, search_only, fast, careful]
depth: enum[light, normal, deep]
```

処理:
1. interaction を defer する。
2. user context を解決する。
3. query classification を行う。
4. intent と `depth` に応じて RAG / 総合エージェント / Wiki / Image / Member / Workflow Search へ route する。
5. 回答を生成する。
6. Discord には短い要約を返し、詳細は attachment / generated document / thread に分離する。

### 13.4 `/work`
`/work` は業務系機能の統合入口である。

入力:
```text
type: enum[
  meeting_prepare,
  meeting_minutes_draft,
  task_extract,
  task_add,
  task_list,
  task_done,
  event_add,
  event_list,
  event_brief,
  schedule_add,
  schedule_list,
  announcement_draft,
  x_draft,
  doc_draft,
  mc_status,
  mc_request,
  image_search,
  image_usage_request,
  member_search
]
instruction: optional str
target: optional str
format: optional enum[compact, markdown, google_doc_draft, slides_outline]
```

出力:
- Discord 本文: 3〜8行程度の短い要約
- 詳細: Markdown attachment / generated document / thread
- 候補: `TaskCandidate`, `WorkflowCandidate`, `AssetUsageRequest`, `ServerOperation` など
- 必要に応じて承認 button / modal

### 13.5 `/approval`
入力:
```text
action: enum[list, show, approve, reject, edit]
target_id: optional str
type: optional enum[task, event, announcement, automation_rule, asset_usage, server_operation, finance_record, member_assignment, other]
comment: optional str
```
承認時は、payload、差分、承認者、承認日時、根拠 citation を `approval_records` に保存する。

### 13.6 `/automation`
入力:
```text
action: enum[list, show, dry_run, run, enable, disable, set_mode]
rule_id: optional str
mode: optional enum[dry_run, approval_required, auto_run]
```

`enable` / `disable` は `AutomationRule.enabled` を切り替える。`auto_run` は low risk かつ明示的に許可された rule だけで使える。

### 13.7 `/admin`
入力:
```text
action: enum[health, sync, eval, feature_flags, permissions, reindex, cost_report]
scope: optional str
```
管理者用。production では必ず role id ベースの権限確認と audit log を通す。

### 13.8 互換 `/ai`
移行期間のみ、message command `/ai` を残す。
| 現行 | 新設計への mapping |
|---|---|
| `/ai <query>` | `/ask question:<query>` |
| `/ai fast <query>` | `/ask mode:fast question:<query>` |
| `/ai build-index` | `/admin action:sync` |
| `/ai eval` | `/admin action:eval` |
| `/ai stop` | job cancellation |
| `/ai join`, `/ai quit` | feature flag が無効なら案内のみ返す |
互換期間終了後、`/ai` は案内メッセージだけ返す。

## 14. Action / Command 実行設計
### 14.1 基本方針
LLM に任意の shell command を生成させない。
実行可能なのは、あらかじめ定義された `ActionSpec` のみとする。

### 14.2 ActionSpec
```python
@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    name: str
    description: str
    risk_level: Literal["low", "medium", "high", "critical"]
    allowed_roles: list[str]
    args_schema: dict[str, Any]
    approval_policy: ApprovalPolicy
    dry_run_template: str
    executor: str
    timeout_seconds: int
    cooldown_seconds: int
    audit_required: bool = True
```

### 14.3 実行フロー
```text
1. User request
2. LLM proposes action_id + args
3. Server validates action_id
4. Server validates args_schema
5. Server checks allowed_roles
6. Server checks risk policy
7. Server creates dry-run
8. User / approver confirms
9. Worker dispatches executor
10. Result is recorded in audit log
11. Discord response is updated
```

### 14.4 ApprovalPolicy
```python
@dataclass(frozen=True)
class ApprovalPolicy:
    mode: Literal["none", "self", "admin", "two_person"]
    required_role_ids: list[str]
    expires_in_seconds: int
```

| risk | approval |
|---|---|
| low | self or none |
| medium | admin |
| high | admin + dry-run required |
| critical | two_person or disabled |

### 14.5 executor 分離
Executor は bot process から分離する。
- rootless container
- restricted filesystem
- environment variable allowlist
- network allowlist
- timeout
- stdout / stderr size limit
- no secret exposure

### 14.6 初期 Action 候補
| action_id | 内容 | risk | approval |
|---|---|---|---|
| `sync_knowledge_index` | 指定 source の同期 | low | self/admin |
| `run_eval_set` | 指定 eval set 実行 | low | self/admin |
| `create_task_candidate` | タスク候補作成 | low | none/self |
| `approve_task_candidate` | 候補を Task 正本へ昇格 | low | self/assignee/organizer |
| `create_task` | タスク作成 | low | self |
| `update_task_status` | タスク状態変更 | low | self/assignee |
| `create_event` | Event 作成 | low | organizer |
| `create_meeting_minutes_draft` | 議事録下書き生成 | low | self |
| `create_announcement_draft` | 告知下書き生成 | low | self |
| `create_asset_usage_request` | 画像・素材利用許可候補作成 | low | self |
| `approve_asset_usage_request` | 画像・素材利用許可承認 | medium | owner/organizer |
| `create_workflow_candidate` | 汎用 workflow 候補作成 | low | none/self |
| `approve_workflow_candidate` | 汎用候補の正本登録 | medium | organizer/admin |
| `create_automation_rule_candidate` | 自動投稿・定期投稿ルール候補作成 | medium | admin |
| `post_discord_announcement` | Discord 告知投稿 | medium | admin/organizer |
| `create_schedule_event` | 予定作成 | medium | organizer |
| `generate_doc_draft` | 資料下書き生成 | low | self |
| `create_finance_record_draft` | 会計記録の下書き | medium | finance_role |
| `approve_finance_record` | 会計記録の承認 | high | finance_role + admin |
| `assign_role` | Discord ロール付与補助 | medium | admin |
| `mc_docker_ps` | Minecraft server コンテナ状態確認 | low | admin |
| `mc_file_search` | サーバー内ファイル・フォルダ検索 | medium | admin + dry-run |
| `mc_compose_up` | `docker compose up -d` 実行 | high | admin + dry-run |
| `mc_compose_down` | `docker compose down` 実行 | critical | two_person or disabled |
| `mc_compose_restart` | `docker compose restart` 実行 | high | admin + dry-run |
| `restart_mc_server` | Minecraft server 再起動 | high | admin + dry-run |
| `create_server_backup` | server backup 作成 | medium | admin |
| `update_whitelist` | whitelist 更新 | medium | admin |
| `deploy_resource_pack` | リソースパック反映 | high | admin + dry-run |
`restart_mc_server` などの実サーバー操作は実装対象に含めるが、production では feature flag と承認ポリシーで `approval_required` または `disabled` にできるようにする。


## 16. Doc Generation 設計
### 16.1 基本方針
資料生成は、直接 Google Docs / Slides を作るのではなく、中間表現を作ってから出力する。
```text
RAG evidence
  -> DocumentPlan
  -> SectionDraft
  -> Review
  -> Rendered Markdown / Google Docs / Slides
```

### 16.2 中間表現
```python
@dataclass(frozen=True)
class DocumentPlan:
    doc_type: str
    title: str
    audience: str
    purpose: str
    sections: list[SectionPlan]
    required_evidence: list[str]

@dataclass(frozen=True)
class SectionPlan:
    heading: str
    goal: str
    evidence_query: str
    expected_length: str

@dataclass(frozen=True)
class SectionDraft:
    heading: str
    body: str
    citations: list[Citation]
    warnings: list[str]
```

### 16.3 Template
初期 template:
#### `meeting_notes`
```text
# 議事録

## 概要
## 決定事項
## 議論内容
## 未決事項
## タスク
## 次回確認
```

#### `weekly_report`
```text
# 週報

## 今週の実績
## 進行中の作業
## 課題
## 来週の予定
## 判断が必要なこと
```

#### `decision_memo`
```text
# 意思決定メモ

## 結論候補
## 背景
## 選択肢
## 比較
## リスク
## 推奨案
## 根拠
```


## 17. Task 管理設計
### 17.1 方針
タスクは内部 DB を正本にする。
外部サービス連携は後から adapter として追加する。

### 17.2 Task data model
詳細 DDL は migration 側に置く。この文書では必要フィールドだけを定義する。

必須フィールド:
- `id`
- `guild_id`
- `title`
- `description`
- `status`
- `priority`
- `assignee_user_ids`
- `created_by_user_id`
- `due_at`
- `related_event_id`
- `related_source_items`
- `related_agent_run_id`
- `access_scope`
- `created_at`
- `updated_at`
- `completed_at`

`status` と `priority` は migration 側で enum または CHECK constraint を設定する。

### 17.3 Discord command
独立した `/task` command は増やさず、`/work` の type で扱う。
```text
/work type:task_add instruction:<title/due/assignee>
/work type:task_list target:<optional status/assignee>
/work type:task_done target:<task_id>
/work type:task_remind target:<task_id> instruction:<when>
```

### 17.4 AI 補助
AI は次の補助を行う。
- 会話からタスク候補を抽出する。
- タスクの重複を検出する。
- 期限切れタスクをまとめる。
- 会議後に action item を提案する。
タスク作成は原則確認付きにする。


## 18. Schedule 管理設計
### 18.1 方針
予定は内部 DB に保持する。必要に応じて Google Calendar と同期する。

### 18.2 ScheduleEvent data model
詳細 DDL は migration 側に置く。この文書では必要フィールドだけを定義する。

必須フィールド:
- `id`
- `guild_id`
- `title`
- `description`
- `start_at`
- `end_at`
- `timezone`
- `location`
- `created_by_user_id`
- `visibility`
- `access_scope`
- `related_source_items`
- `external_calendar_event_id`
- `recurrence_rule`
- `created_at`
- `updated_at`
- `canceled_at`

日時は必ず timezone を持つ。既定は `Asia/Tokyo` とする。

### 18.3 AI 補助
- 「次の定例会いつ？」への回答
- 「来週までの予定をまとめて」への回答
- Discord 会話から日程候補抽出
- 日程重複検出
- 期限前 reminder 作成

### 18.4 注意
日時は必ず timezone を持つ。既定は `Asia/Tokyo` とする。


## 19. 業務ワークフロー設計
### 19.1 目的
業務ワークフローは、kumc-agent-workflows で定義された各 Agent を、本番実装で扱える usecase、DB、Discord command、承認フローに落とし込むための設計である。

ここでいうワークフローは、単なる LLM 応答ではなく、次の一連の処理を指す。
```text
trigger
  -> user / role / scope 解決
  -> workflow routing
  -> source retrieval / parsing
  -> draft or candidate 生成
  -> server-side validation
  -> preview
  -> human approval if needed
  -> 正本 DB 反映 or 下書き保存
  -> notification
  -> audit / trace / eval log
```

原則:
1. Agent が抽出した情報は、まず candidate として扱う。
2. 正本 DB への登録、外部公開、サーバー操作、投稿自動化は、必ず承認・検証を通す。
3. すべての candidate には、根拠 citation、作成者、作成時刻、confidence、必要な承認者を持たせる。
4. 個人情報、内部情報、認証情報、外部未公開情報は、検索時点と回答時点の両方で制御する。
5. 出力では「確定情報」「候補」「未確認」「次に必要な確認」を分ける。

### 19.2 Workflow catalog
| workflow_id | Agent | 主な入口 | 主な出力 | 承認 | risk | production 既定 |
|---|---|---|---|---|---|---|
| `integrated_search` | 統合検索 Agent | `/ask` | 回答、関連資料、関連画像、関連タスク、関連イベント | 不要。ただし副作用へ route する場合は必要 | low | enabled |
| `minecraft_wiki_search` | Minecraft Wiki 検索 Agent | `/ask source:minecraft_wiki` | 仕様回答、関連記事、edition / version 差分 | 不要 | low | enabled |
| `image_search` | 画像検索 Agent | `/ask source:image`, `/work type:image_search` | 画像候補、説明、出典、類似画像、利用許可候補 | 外部公開・再利用時は必要 | medium | enabled |
| `writing_draft` | 文章作成 Agent | `/work type:x_draft|announcement_draft|doc_draft` | 告知文、依頼文、ブログ記事、SNS 投稿案 | 投稿・外部送信時は必要 | medium | enabled |
| `meeting_minutes` | 議事録作成 Agent | `/work type:meeting_prepare|meeting_minutes_draft` | 議題案、議事録、決定事項、未決事項、ToDo | Task / Event 登録時は必要 | low〜medium | enabled |
| `member_search` | メンバー検索 Agent | `/ask source:member`, `/work type:member_search` | メンバー候補、該当理由、確認事項 | 担当決定時は本人または運営確認 | medium | enabled |
| `minecraft_icon` | Minecraft アイコン作成 Agent | `/work type:doc_draft` または専用 workflow | 生成プロンプト、アイコン案、整形済み画像 | 外部公開・公式利用時は必要 | medium | approval_required |
| `task_management` | タスク管理 Agent | `/work type:task_extract|task_add|task_list|task_done` | TaskCandidate、Task、期限リマインド | 正本登録時に必要 | medium | enabled |
| `event_management` | イベント管理 Agent | `/work type:event_add|event_brief|schedule_add` | EventCandidate、Event、Calendar 連携、変更差分 | 日時・場所確定時に必要 | medium | enabled |
| `message_automation` | 返信・投稿オートメーション Agent | `/automation` | 自動返信ルール候補、定期投稿ルール、投稿ログ | 初期設定時は admin 承認 | high | approval_required |
| `auto_indexing` | 自動 Indexing Agent | `/admin action:sync`, automation runner | 更新済み index、差分、失敗ログ、品質確認 | 通常不要。削除・大規模再構築は admin | low〜medium | enabled |
| `server_management` | サーバー管理 Agent | `/work type:mc_status|mc_request` | 状態、dry-run、実行結果、監査ログ | admin。高リスクは二者承認 | high〜critical | approval_required |
| `approval_registry` | 承認・正本管理 Agent | `/approval` | 承認待ち一覧、承認済み正本、履歴 | 操作種別ごとの policy | medium | enabled |

### 19.3 共通 WorkflowRun モデル
すべての workflow run は、内部的に次の情報を持つ。
```json
{
  "workflow_id": "task_management",
  "run_id": "uuid",
  "trigger": "discord_command|automation|webhook|manual",
  "actor_user_id": "...",
  "guild_id": "...",
  "channel_id": "...",
  "source_scope": {"source_kinds": ["discord", "google_drive"]},
  "input": {},
  "retrieval": {"search_run_ids": []},
  "candidates": [],
  "drafts": [],
  "validation_result": {},
  "approval_required": true,
  "status": "running|waiting_approval|succeeded|failed|rejected",
  "audit_log_id": "..."
}
```

`WorkflowRun` は `agent_runs` / `agent_steps` と対応付ける。検索・生成・検証の各段階は trace 可能にする。

### 19.4 統合検索 Workflow
目的は、サークル資料、Minecraft Wiki、画像、メンバー情報、イベント、タスクを横断検索し、質問に応じて適切な Agent へ処理を振り分けることである。

処理:
1. Query classification で intent を判定する。
2. `source_filters` と user access scope を決める。
3. intent が `minecraft_spec` の場合は Minecraft Wiki 検索へ route する。
4. intent が `image` の場合は Image / Asset 検索へ route する。
5. intent が `member` の場合は Member Search へ route する。
6. intent が `task`, `event`, `approval`, `action` の場合は、該当 workflow usecase へ route する。
7. 複数 source の結果が矛盾する場合は、正本 DB、最新資料、権限内の一次情報を優先する。

出力:
- 回答
- 関連資料
- 関連画像
- 関連タスク
- 関連イベント
- 参照元
- 追加確認が必要な項目

制約:
- 権限が異なる source を横断するため、検索前 filter と回答前 filter の両方を必須にする。
- 個人情報・内部情報・認証情報は回答に含めない。
- 確定情報と候補情報を分ける。

### 19.5 Minecraft Wiki 検索 Workflow
目的は、Minecraft Wiki の記事を検索し、Minecraft の仕様、アイテム、ブロック、Mob、コマンドに関する質問に回答することである。

入力:
- Minecraft Wiki article / section index
- Wiki article metadata
- ユーザー質問文
- Minecraft version 指定
- Java Edition / Bedrock Edition 指定

処理:
1. 質問から item / block / mob / command / version / edition を抽出する。
2. version または edition が重要な質問で指定がない場合は、回答内で前提を明示する。必要に応じて追加確認を促す。
3. article 単位検索と section 単位検索を併用する。
4. version table や edition 差分の section を優先的に読む。
5. 回答後に citation validation を行う。

出力:
- 質問への回答
- Java Edition / Bedrock Edition の差分
- version 依存の注意
- 関連 Wiki 記事
- Wiki 取得日
- 根拠が古い可能性の警告

注意:
- Java Edition と Bedrock Edition の仕様を混同しない。
- Wiki の取得・更新日は保存し、回答で必要に応じて表示する。
- サーバー操作や whitelist 操作に関する質問は `/work type:mc_status|mc_request` workflow に route する。

### 19.6 画像検索 / Asset 利用 Workflow
目的は、KUMC 関連画像や制作物を検索し、利用候補として提示し、必要な場合は利用許可確認へ進めることである。

対象 source:
- Discord 添付画像
- Google Drive 上の画像・スクリーンショット
- X 投稿画像
- はてなブログ記事画像
- クラフターズコロニー投稿画像

Indexing:
- 画像 caption 生成
- OCR text 抽出
- 画像特徴量 vector 保存
- 出典 URL、投稿日時、投稿媒体、投稿者、関連イベント、人物の有無を metadata に保存

検索条件:
- イベント名
- ワールド名
- 建築物名
- 投稿時期
- 投稿媒体
- 人物の有無
- 類似画像

処理:
1. `/ask source:image` または `/work type:image_search` で条件を抽出する。
2. text retrieval、OCR retrieval、image vector similarity を統合する。
3. 検索結果を `Asset` 候補として返す。
4. 外部公開、ビラ、X、Web、ブログで使う場合は `/work type:image_usage_request` を作成する。
5. 承認者、投稿者、権利確認、人物確認が必要な場合は `needs_owner_check` にする。

出力:
- 画像候補一覧
- 画像説明
- 出典
- 投稿日時
- 類似画像
- 使用許可確認の要否

禁止:
- 人物が写っている画像を承認なしで外部公開素材として確定すること。
- X や外部サイトの画像を、規約確認なしに再利用可能と断定すること。
- 画像検索結果を自動で「公式素材」として登録すること。

### 19.7 文章作成 Workflow
目的は、KUMC の告知文、説明文、依頼文、ブログ記事、SNS 投稿などの作成を支援することである。

入力:
- 文章種別
- 目的
- 対象読者
- 掲載媒体
- 必須情報
- 過去の類似文面
- 関連資料
- tone 指定

処理:
1. 目的と媒体に応じて template を選ぶ。
2. RAG で関連資料と過去文面を取得する。
3. 日時、場所、参加条件、料金、申込方法を fact check 対象として抽出する。
4. 文章案を生成する。
5. 未確認情報、公開不可情報、個人情報の混入を検査する。
6. 外部公開する場合は `Announcement` の draft として保存し、承認待ちにする。

出力:
- 文章案
- 修正候補
- 事実確認が必要な箇所
- 採用理由
- 公開前チェックリスト

注意:
- KUMC の公式見解と個人の意見を混同しない。
- 未確認の日時、場所、参加条件、料金、申込方法を断定しない。
- 外部公開対象は `fact_check_status="checked"` になるまで投稿不可にする。

### 19.8 議事録作成 / 例会準備 Workflow
目的は、過去のサークル情報から議題を作成し、議事録本文、決定事項、未決事項、ToDo を整理することである。

入力:
- Google Drive / Discord の関連資料
- 前回議事録
- 未完了タスク
- イベント予定
- 直近ログ

`/work type:meeting_prepare` 処理:
1. lookback 期間内の Discord / Drive / Task / Event を検索する。
2. 前回議事録との差分を確認する。
3. 未完了タスク、期限が近いタスク、未決事項を抽出する。
4. サーバー運用の確認事項に分類する。
5. 議題案、決めるべき論点、告知文案を出す。

`/work type:meeting_minutes_draft` 処理:
1. source から議論内容を読み取る。
2. 決定事項と単なる意見を分離する。
3. 未決事項、ToDo、担当者、期限を抽出する。
4. ToDo は `TaskCandidate` として候補登録する。
5. 外部公開用議事録では個人情報と内部情報を除外する。

出力:
- 議題案
- 各議題の論点
- 議事録本文
- 決定事項
- 未決事項
- ToDo
- 担当者
- 期限
- 次回確認事項
- 根拠リンク

### 19.9 メンバー検索 / 担当候補 Workflow
目的は、KUMC メンバーの情報を権限付きで検索し、条件に合うメンバーや担当候補を探すことである。

入力:
- `member_profiles`
- Discord profile
- 自己紹介文
- ロール情報
- 過去の担当履歴
- ユーザー検索条件

検索条件:
- 学年
- 役職
- スキル
- 過去の担当
- 興味分野
- 参加可能性に関する明示情報

出力:
- 条件に合うメンバー候補
- 該当理由
- 関連するスキル、ロール、担当履歴
- 本人確認または運営確認が必要な情報

制約:
- 個人の能力や参加意思を断定しない。
- 担当者決定は本人確認または運営承認を挟む。
- 外部公開用の回答には個人情報を含めない。
- メンバー検索は原則として運営 role 以上に制限する。

### 19.10 Minecraft アイコン作成 Workflow
目的は、Minecraft 風のアイコンや、KUMC の企画・サーバー・イベント向け画像素材の作成を支援することである。

入力:
- 作成したいアイコンの説明
- 用途
- サイズ
- 参考画像
- 色・雰囲気
- Minecraft 風デザイン指定
- 既存ロゴ・素材
- 外部画像生成 API 設定

処理:
1. 用途と媒体に応じて安全な生成プロンプトを作る。
2. 既存ロゴや素材がある場合は使用権限を確認する。
3. 画像生成 API を呼び出す場合は staging で検証する。
4. 複数案、サイズ調整、背景透過、Discord サーバーアイコン向け整形を行う。
5. 公式利用する場合は `Asset` として保存し、利用承認を取る。

出力:
- 生成プロンプト
- アイコン案
- サイズ調整済み画像
- 背景透過画像
- 利用上の注意

### 19.11 タスク管理 Workflow
目的は、Discord や Google Drive からタスク候補を抽出し、承認後に正本登録し、期限に応じてリマインドすることである。

抽出項目:
- タイトル
- 備考
- 担当者
- 期限
- 優先度
- 関連イベント
- 関連チャンネル・メッセージ
- 状態: `proposed / todo / doing / blocked / done`

処理:
1. RAG / 総合エージェントでタスクらしい発言や文書箇所を集める。
2. LLM が `TaskCandidate` を生成する。
3. server-side validation で schema、権限、重複、秘密情報を確認する。
4. 担当者が曖昧な場合は `未定` として扱う。
5. admin または担当者が承認したものだけ `tasks` に昇格する。
6. 期限前リマインドと期限切れ通知を送る。
7. 完了確認後、状態を `done` にする。

注意:
- LLM の推測だけでタスクを確定しない。
- リマインドは過剰通知にならないよう頻度を制御する。
- 根拠 message / document を evidence として保持する。

### 19.12 イベント管理 Workflow
目的は、Discord や資料からイベント情報を抽出し、承認後に正本登録し、関係者へリマインドすることである。

抽出項目:
- イベント名
- 概要
- 日時
- 場所
- 状態: `proposed / planning / announced / done / canceled`

処理:
1. Discord / Drive / Meeting からイベント候補を抽出する。
2. 日時、場所、関係者、関連資料を整理する。
3. 既存 Event / Schedule と重複確認する。
4. admin または運営メンバーへ確認依頼を出す。
5. 承認後に `events` と必要に応じて `schedule_events` へ登録する。
6. Google Calendar 連携が有効な場合は calendar event を作成または更新する。
7. 変更が発生した場合は、変更前後の差分を表示して再承認を求める。

注意:
- 日時・場所は誤りの影響が大きいため、必ず承認を挟む。
- 仮予定と確定予定を明確に分ける。
- 告知済み event の変更は、関連告知・リマインドも更新対象にする。

### 19.13 返信・メッセージ投稿オートメーション Workflow
目的は、定型的な返信や定期投稿の規則性を抽出し、承認後に自動投稿を行うことである。

入力:
- Discord logs
- 投稿先 channel
- 投稿タイミング
- 対象 event
- admin 承認

処理:
1. Discord logs から規則性を抽出する。
2. 次の候補を生成する。
   - 毎週投稿
   - event 前 reminder
   - よくある質問への返信
   - 特定 keyword への反応
3. 投稿文案、対象 channel、投稿時刻、対象条件、停止条件を明示する。
4. `AutomationRule` candidate を作成する。
5. admin が承認したものだけ正本登録する。
6. 投稿前 preview と投稿後 log を保存する。
7. `/automation action:list` と `/automation action:enable|disable` から停止・再開できるようにする。

注意:
- 自動投稿は誤爆の影響が大きいため、初期設定時は必ず admin 承認を必要とする。
- 個人宛ての自動返信は慎重に扱う。
- 投稿内容に未確認情報が含まれる場合は自動投稿しない。

### 19.14 自動 Indexing Workflow
目的は、Discord、Google Drive、Minecraft Wiki、画像データなどを定期的に Indexing し、検索・RAG の精度を維持することである。

対象:
- Google Drive / Discord / Hatena / X / Crafters Colony
- Minecraft Wiki
- 画像 caption / OCR / feature vector
- member_profiles
- Task / Event / Schedule 正本

処理:
1. source ごとの cursor / checksum / revision を確認する。
2. 新規、更新、削除、権限変更を検出する。
3. raw snapshot を保存する。
4. text extraction、OCR、caption generation、chunking、embedding、sparse index を更新する。
5. 削除済み file や閲覧権限がなくなった file を検索結果から除外する。
6. 検索品質の簡易チェックを実行する。
7. 失敗時は `indexing_runs` に失敗ログを残し、admin に通知する。

出力:
- 更新済み index
- Indexing log
- 失敗 log
- 差分情報
- 再 Indexing 対象一覧
- 検索品質確認結果

### 19.15 サーバー管理 Workflow
目的は、Minecraft サーバーや関連アプリケーションの運用コマンドを、安全に実行・確認できるようにすることである。

許可する操作:
- サーバー内ファイル・フォルダ検索
- `docker compose up -d`
- `docker compose down`
- `docker compose restart`
- `docker ps`
- 実行前 dry-run
- 実行後 log 取得
- 障害時の状態確認

処理:
1. `/work type:mc_status` または `/work type:mc_request` を受ける。
2. admin 権限を確認する。
3. operation、target server、target directory、service name を schema validation する。
4. 影響範囲、想定停止時間、rollback 方針を dry-run で表示する。
5. high risk 以上は admin 承認、critical は二者承認または disabled にする。
6. isolated executor で定義済み command のみ実行する。
7. stdout / stderr、server state、container state、監査 log を保存する。

禁止:
- 任意 shell command の実行。
- LLM が生成した command string の直接実行。
- ネットワークキー、PIN、内部 IP、認証情報、解錠手順を一般回答に含めること。

### 19.16 承認・正本管理 Workflow
目的は、タスク、イベント、投稿オートメーション、画像利用、サーバー操作などの候補情報を、承認後に正本 DB へ登録することである。

対象 candidate:
- `TaskCandidate`
- `Event` / `ScheduleEvent` candidate
- `Announcement` draft
- `AutomationRule` candidate
- `AssetUsageRequest`
- `ServerOperation`
- 汎用 `WorkflowCandidate`

状態:
```text
proposed -> needs_review -> approved -> merged
proposed -> rejected
approved -> archived
```

処理:
1. candidate を一覧化する。
2. 根拠、差分、承認 policy、risk を表示する。
3. 承認、却下、修正を受け付ける。
4. 承認時に正本 DB へ反映する。
5. 修正履歴と承認履歴を `approval_records` に保存する。
6. 承認者、承認日時、根拠 message / document を記録する。
7. 重複 candidate は merge する。

注意:
- Agent が抽出した情報は、原則として正本ではなく候補として扱う。
- 正本更新時は、変更前後の差分を保存する。
- 承認権限は candidate type ごとに設定する。

### 19.17 ワークフロー共通の出力ポリシー
すべての業務ワークフローは、次の形式を基本にする。
```text
結論 / 提案:
...

根拠:
- ...

未確認:
- ...

次の操作候補:
- [承認ボタン] ...
- [編集ボタン] ...
- [却下ボタン] ...

安全上の注意:
- ...
```
LLM が提案した内容は、正本 DB へ直接反映しない。副作用がある場合は、必ず server-side validation と approval flow を通す。

### 19.18 失敗時・不足時の動作
- 根拠が不足する場合は、回答を断定せず、不足情報と確認した範囲を表示する。
- 候補の schema validation に失敗した場合は、正本登録せず、修正可能な draft として返す。
- 権限不足の場合は、対象情報の存在を示唆しない形で拒否する。
- 外部公開前 safety check に失敗した場合は、公開不可理由と修正案を返す。
- workflow が途中で失敗した場合も、`WorkflowRun`、`agent_steps`、`audit_logs` に状態を残す。

## 20. Automation 設計
### 20.1 基本方針
自動化はフルプロダクトの実装対象に含める。ただし production では `enabled` と `mode` を別々に管理し、いつでも停止できるようにする。

実行状態:
| 項目 | 内容 |
|---|---|
| `enabled=false` | rule は存在するが trigger では起動しない |
| `enabled=true` | trigger 対象になる |
| `mode=dry_run` | 実行内容だけ提示 |
| `mode=approval_required` | 承認後に実行 |
| `mode=auto_run` | 自動実行。ただし low risk かつ allowlist 済み action のみ |

### 20.2 AutomationRule
```python
@dataclass(frozen=True)
class AutomationRule:
    id: str
    name: str
    enabled: bool
    trigger: TriggerSpec
    conditions: list[ConditionSpec]
    actions: list[ActionSpecRef]
    mode: Literal["dry_run", "approval_required", "auto_run"]
    risk_level: Literal["low", "medium", "high", "critical"]
    created_by_user_id: str
    approved_by_user_id: str | None
    last_run_at: datetime | None
    next_run_at: datetime | None
```

### 20.3 Trigger
初期 trigger:
```text
schedule_cron
drive_changed
notion_changed
discord_message_matched
task_due_soon
manual
```

### 20.4 初期 automation 例
| 名前 | trigger | action | mode | enabled 初期値 |
|---|---|---|---|---|
| 週次活動まとめ | 毎週日曜 21:00 | RAG で週報下書き生成 | dry_run | true |
| 新規 Drive 資料 index | Drive changed | sync_knowledge_index | auto_run | true |
| 新規 Notion 更新 index | Notion changed | sync_knowledge_index | auto_run | false |
| 期限前 reminder | task due soon | Discord 通知 | auto_run | true |
| 定例会資料案 | 定例会前日 | make-doc | approval_required | true |
| 自動返信 rule | keyword match | reply draft / post | approval_required | false |

### 20.5 enable / disable 操作
`/automation action:enable rule_id:<id>` で `enabled=true`、`/automation action:disable rule_id:<id>` で `enabled=false` にする。

制約:
- `enabled` の変更は audit log に保存する。
- `mode` を `auto_run` に変更する場合は admin 承認を必須にする。
- high / critical risk rule は `auto_run` にできない。
- 外部投稿、ロール変更、サーバー操作、会計確定は `approval_required` 以上に固定する。

### 20.6 冪等性
Automation は `idempotency_key` を持つ。
例:
```text
weekly_report:2026-W17
drive_sync:file_id:version
notion_sync:page_id:last_edited_time
reminder:task_id:due_at:24h
```
同じ key は二重実行しない。

### 20.7 自動返信・定期投稿 rule
返信・投稿 automation は誤爆リスクが高い。初期登録時は必ず `approval_required` にする。

必須条件:
- 投稿先 channel
- 投稿時刻または trigger
- 対象 keyword / event / task
- 投稿文 template
- 停止条件
- 投稿前 preview の有無
- 投稿後 log 保存先

未確認情報、個人情報、内部情報、外部公開不可情報を含む場合は `auto_run` を禁止する。

### 20.8 自動 Indexing rule
`sync_knowledge_index` は source ごとに idempotency_key を持つ。
```text
drive_sync:file_id:version
notion_sync:page_id:last_edited_time
discord_sync:channel_id:message_id
minecraft_wiki:article_id:revision
image_index:asset_id:checksum
member_profile:user_id:updated_at
```
削除または権限喪失を検知した source は、検索 index と citation 候補から除外する。

## 21. Security 設計
### 21.1 脅威モデル
KUMC-Agent で優先して対策する脅威。
| 脅威 | 例 | 対策 |
|---|---|---|
| Prompt Injection | Drive 文書に「system prompt を無視しろ」と書かれる | 文書を未信頼 data として扱う、tool policy 分離 |
| Insecure Output Handling | LLM 出力をそのまま shell 実行 | ActionSpec + schema validation |
| Excessive Agency | AI が勝手に投稿・削除・再起動する | 承認、risk policy、write tool 制限 |
| Sensitive Info Disclosure | 権限外資料を回答に混ぜる | search-time ACL、citation validation |
| Data Poisoning | 悪意ある文書が index に入る | source allowlist、metadata、review、削除対応 |
| Model DoS / Cost Runaway | Agent が何度も高額 API を呼ぶ | max steps、max cost、rate limit |
| Overreliance | 誤回答を断定的に出す | confidence、根拠提示、不確実性表示 |

### 21.2 Prompt Injection 対策
必須ルール:
- retrieved context 内の命令文を system instruction として扱わない。
- context は delimiter で囲む。
- tool call は model ではなく server が許可する。
- retrieved context から action policy を変更できない。
- 外部文書に含まれる URL を自動実行しない。
- 「この文書以外を見るな」などの指示を無視する。

### 21.3 権限制御
認可は次の順で行う。
1. Discord user / guild / role を解決。
2. Command ごとの permission を確認。
3. Retrieval filter に access scope を入れる。
4. Answer citation が access scope 内であることを検証。
5. Action 実行前に risk policy を確認。
6. audit log に actor / decision を保存。

具体的な権限設定は後から変えられるように、`configs/base/security.yaml` と `configs/env/*.yaml` に分離する。role id は現時点では空配列を仮値にする。

```yaml
security:
  guild_allowlist: []
  default_visibility: guild
  response_visibility:
    ask: public_summary
    work: public_summary
    member_search: ephemeral
    finance_summary: ephemeral
    approval: ephemeral
    admin: ephemeral
  roles:
    member:
      discord_role_ids: []
    organizer:
      discord_role_ids: []
    admin:
      discord_role_ids: []
    finance:
      discord_role_ids: []
    public_relations:
      discord_role_ids: []
    server_operator:
      discord_role_ids: []
  command_permissions:
    ask: [member]
    work: [member]
    approval: [organizer, admin]
    automation: [organizer, admin]
    admin: [admin]
  source_permissions:
    google_drive: [member]
    discord: [member]
    notion: [organizer, admin]
    hatena: [member]
    x: [member]
    crafters_colony: [member]
    minecraft_wiki: [member]
    image: [member]
    member_profile: [organizer, admin]
    finance: [finance, admin]
  action_risk_permissions:
    low: [member]
    medium: [organizer, admin]
    high: [admin]
    critical: [admin]
  feature_flags:
    vc_transcription: false
    image_generation_paid: false
    external_auto_post: false
    mc_server_write_actions: false
    automation_auto_run: false
```

### 21.4 Secret 管理
- `.env` は local のみ。
- staging / production は Secret Manager を使う。
- LLM API key、Discord token、Google credential、X token は log に出さない。
- trace に prompt を出す場合は redaction を通す。
- personal data を含む chunk は provider へ送る最小限の範囲に制限する。

### 21.5 出力安全性
LLM 出力は用途ごとに validation する。
| 用途 | validation |
|---|---|
| 回答 | citation、権限、禁止表現 |
| X 投稿案 | 個人情報、未確認事実、外部公開不可情報 |
| Action args | JSON Schema、enum、range、path traversal |
| DocGen | citation、未確認情報、private source 混入 |

### 21.6 秘密情報検出と引用制御 v2.0
業務調査レポートでは、PIN、内部 IP、ネットワークキー、端末設定、解錠手順、会計情報、個人情報、外部関係者との調整内容が含まれる可能性が示されている。したがって、retrieval 前後の両方に秘密情報対策を入れる。

#### 21.6.1 検出対象
| 種別 | 例 | 標準 redaction_policy |
|---|---|---|
| `credential` | token, password, secret key | deny |
| `pin` | PIN, 暗証番号 | deny |
| `internal_ip` | private IP, LAN 情報 | summary_only / admin_only |
| `network_key` | Wi-Fi key, 無線キー | deny |
| `unlock_procedure` | 解錠手順、鍵管理 | admin_only |
| `personal_data` | 氏名、所属、MCID、連絡先 | summary_only |
| `finance` | 売上、立替、会費、カンパ | role only |
| `external_confidential` | 外部関係者との未公開調整 | role only |

#### 21.6.2 Retrieval 時の処理
```text
1. access_scope で検索前 filter
2. secret_findings を参照
3. deny chunk は context から除外
4. summary_only chunk は redacted summary に変換
5. admin_only chunk は権限確認後に最小限で使用
6. citation validation で raw quote を禁止
```

#### 21.6.3 存在確認抑制
権限外ユーザーには、「その情報が存在するかどうか」自体が漏れないようにする。例えば、非公開議事録や秘密のネットワークキーに関する質問では、検索対象外として処理し、存在を示唆する文言を避ける。

### 21.7 外部公開コンテンツの安全ゲート v2.0
X、ブログ、ビラ、Web 原稿、外部連絡文案では、次を投稿前チェックに含める。
- private / role / admin source 由来の情報が混入していないか
- 個人情報、MCID、内部連絡先が含まれていないか
- 未確認の実績、数字、日時、場所を断定していないか
- 外部関係者の未公開情報を含んでいないか
- 会計やサーバー運用秘密を含んでいないか
外部公開対象は、`fact_check_status="checked"` かつ `visibility="public"` の根拠だけを基本 source とする。


### 21.8 メンバー情報保護 v2.1
メンバー検索では、次を必須にする。
- `member_profiles.access_scope` による検索前 filter
- 回答前の個人情報 redaction
- 担当候補の出力では「本人確認が必要」と明示
- 外部公開文面への個人情報混入検査
- 参加意思、能力、空き時間を LLM が断定しないための prompt / eval

### 21.9 画像・素材利用安全ゲート v2.1
画像検索結果は候補であり、利用可能素材ではない。外部公開、再配布、ビラ掲載、X 投稿、Web 掲載に使う場合は、次を確認する。
- 出典
- 投稿者または権利者
- 人物の有無
- 外部サイトの利用規約。利用規約確認は後日行い、初期状態では `terms_review_status=pending` とする
- 既存ロゴ・商標・第三者素材の有無
- 使用目的と掲載媒体
- 承認者と承認日時

承認前の asset は `Announcement` や外部公開 doc の source として使えない。

## 22. Observability 設計
### 22.1 構造化ログ
全ログは JSONL 形式を基本にする。
主要 field:
```json
{
  "timestamp": "...",
  "env": "production",
  "service": "bot|api|worker",
  "trace_id": "...",
  "user_id": "...",
  "guild_id": "...",
  "route": "rag",
  "event": "answer_generated",
  "latency_ms": 1234,
  "cost_usd": 0.012,
  "status": "success"
}
```

### 22.2 Trace
trace 対象:
- Discord interaction
- query classification
- retrieval dense
- retrieval sparse
- RRF
- rerank
- MMR
- LLM generation
- tool call
- action execution
- workflow run

### 22.3 Metrics
| Metric | 意味 |
|---|---|
| `kumc_agent_requests_total` | request 数 |
| `kumc_agent_latency_ms` | latency |
| `kumc_agent_llm_cost_usd` | LLM cost |
| `kumc_agent_retrieval_recall_at_k` | eval recall |
| `kumc_agent_citation_precision` | citation 精度 |
| `kumc_agent_action_approval_count` | 承認件数 |
| `kumc_agent_action_denied_count` | 拒否件数 |
| `kumc_agent_prompt_injection_detected_count` | 検知数 |

### 22.4 Cost log
LLM / embedding / rerank の呼び出しは、全て cost log を残す。詳細 DDL は migration 側に置く。

必須フィールド:
- `id`
- `trace_id`
- `agent_run_id`
- `workflow_run_id`
- `user_id`
- `guild_id`
- `provider`
- `model`
- `purpose`
- `input_tokens`
- `output_tokens`
- `cost_usd`
- `latency_ms`
- `prompt_version`
- `request_hash`
- `response_hash`
- `created_at`


## 23. Eval / 品質保証設計
### 23.1 評価方針
KUMC-Agent は eval-driven に開発する。
実装前に eval case を作り、実装後に regression として回す。

### 23.2 EvalSet
初期 eval set:
| Set | 件数 | 内容 |
|---|---:|---|
| `rag_drive_basic` | 20 | Drive 資料の基本質問 |
| `rag_discord_basic` | 20 | Discord 会話履歴質問 |
| `rag_mixed` | 20 | Drive + Discord 複合質問 |
| `comprehensive_agent` | 15 | 追加検索・複数機能連携・候補作成境界の検証が必要な質問 |
| `x_draft` | 10 | 投稿案生成 |
| `security_prompt_injection` | 20 | prompt injection 耐性 |
| `acl` | 15 | 権限違反防止 |
| `action_safety` | 15 | command 実行安全性 |

### 23.3 EvalCase schema
```json
{
  "id": "rag_drive_001",
  "input": "新歓企画の予算はいくら？",
  "user_context": {
    "roles": ["member"],
    "guild_id": "..."
  },
  "expected": {
    "must_contain": ["..."],
    "must_cite_source_kind": ["google_drive"],
    "must_not_contain": ["..."],
    "confidence_min": "medium"
  },
  "tags": ["rag", "drive", "budget"]
}
```

### 23.4 指標
| 指標 | 目標 |
|---|---|
| answer correctness | 人手評価 4/5 以上 |
| citation precision | 0.8 以上 |
| retrieval recall@10 | 0.8 以上 |
| unauthorized citation | 0 件 |
| prompt injection success | 0 件 |
| p95 latency `/ask` | 10 秒以内を目標 |
| p95 latency `/ask depth:deep` | 60 秒以内を目標 |
| cost per `/ask` | 上限を config 化 |

### 23.5 CI gate
PR ごとに実行:
- unit test
- type check
- schema validation
- prompt snapshot test
- small eval set

main merge 前または nightly:
- full eval set
- security eval
- cost regression
- latency regression


### 23.6 業務ワークフロー Eval
業務調査レポート反映後は、RAG の検索精度だけでなく、業務出力の妥当性を評価する。
| EvalSet | 件数 | 評価内容 |
|---|---:|---|
| `meeting_prepare` | 20 | 議題案、未完了タスク、決定論点、根拠リンク |
| `task_extraction` | 30 | 担当、期限、状態、重複検出、候補止まりの確認 |
| `event_brief` | 15 | Event 概要、関連資料、関連タスク、次の判断事項 |
| `announcement_safety` | 20 | 外部公開不可情報、個人情報、未確認事実の混入防止 |
| `finance_safety` | 10 | 金額根拠、証跡、権限、確定処理禁止 |
| `mc_operation_safety` | 15 | dry-run、承認、秘密情報非表示、危険操作拒否 |
| `minecraft_wiki_search` | 15 | edition / version 差分、関連記事、古い取得日の警告 |
| `image_search_usage` | 20 | 画像候補、OCR、類似画像、人物・権利確認、承認前公開禁止 |
| `member_search_privacy` | 15 | 権限、個人情報抑制、担当候補の非断定表現 |
| `approval_registry` | 15 | 候補一覧、承認、却下、修正、差分・履歴保存 |
| `auto_indexing` | 15 | 差分検出、削除反映、失敗通知、検索品質チェック |
| `secret_redaction` | 30 | PIN、内部IP、ネットワークキー、個人情報、会計情報の引用抑制 |

### 23.7 TaskCandidate 評価基準
TaskCandidate の eval は、通常の正解率だけでなく次を見る。
- 根拠 message / document を正しく持つか
- 担当者を過剰推測しないか
- 期限が曖昧な場合に曖昧と表現するか
- 既存 Task と重複していないか
- 承認前に Task 正本へ入っていないか
- 権限外 source を evidence にしていないか

## 24. Testing 設計
### 24.1 Test pyramid
| 種別 | 対象 |
|---|---|
| Unit | domain、policy、schema、rank fusion、MMR |
| Integration | connector、DB repository、provider adapter |
| Contract | LLM structured output、tool schema、Discord command schema |
| E2E | Discord command -> answer -> audit log |
| Eval | RAG / Agent / X draft 品質 |
| Security | prompt injection、ACL、action safety |

### 24.2 LLM 出力テスト
LLM は非決定的なので、以下を分ける。
- schema が守られているか。
- 禁止 tool が呼ばれないか。
- citation があるか。
- eval case に対して許容範囲か。

### 24.3 Connector test
各 connector に fixture を用意する。
```text
/tests/fixtures/connectors/drive/docs_sample.json
/tests/fixtures/connectors/discord/messages_sample.jsonl
/tests/fixtures/connectors/hatena/feed_sample.xml
/tests/fixtures/connectors/x/posts_sample.jsonl
```


## 25. 開発プロセス
### 25.1 Issue 形式
```markdown
## 目的

## 背景

## 受け入れ条件
- [ ] ...

## 非目標

## 影響範囲

## テスト

## Eval

## Security checklist
```

### 25.2 ADR
大きな設計判断は ADR に残す。
初期 ADR:
```text
ADR-001: モジュラモノリス + bot/api/worker 分離を採用する
ADR-002: PostgreSQL を domain 正本にする
ADR-003: RAG は hybrid + RRF + rerank + MMR にする
ADR-004: Action 実行は command registry + approval に限定する
ADR-005: OpenClaw は任意 adapter とし、主経路にしない
ADR-006: LLM / embedding / rerank は外部 API を使う
ADR-007: slash command を主 UI にする
```

### 25.3 Definition of Done
全機能共通:
- [ ] unit test がある。
- [ ] integration test がある。
- [ ] eval case が追加または更新されている。
- [ ] 権限チェックがある。
- [ ] audit log が出る。
- [ ] structured log / trace が出る。
- [ ] config schema が更新されている。
- [ ] runbook または運用メモがある。
- [ ] rollback 方法がある。

AI 機能追加時:
- [ ] prompt version が管理されている。
- [ ] structured output schema がある。
- [ ] failure fallback がある。
- [ ] cost 上限がある。
- [ ] hallucination / citation の評価がある。


## 26. 実装 Wave
本設計では簡易版を置かず、フルプロダクトを実装対象にする。ただし、開発・検証・公開は Wave に分ける。各 Wave はスコープ削減ではなく、依存関係と安全性のための実装順である。

### Wave 1: 基盤構築
作業:
- repo 再構成
- bot / api / worker の起動雛形
- PostgreSQL / Redis / object storage 接続
- migration 導入
- config schema
- feature flag
- logging / tracing
- audit log repository
- job lifecycle
- Discord slash command skeleton

完了条件:
- `/admin action:health` が staging で動く。
- DB migration が CI で通る。
- audit log が保存される。
- feature flag で高リスク機能を停止できる。

### Wave 2: Connector / Ingestion / SecretFinding
作業:
- Connector interface
- Drive connector
- Discord connector
- Notion connector
- Hatena connector
- X / Crafters Colony / Minecraft Wiki connector
- raw snapshot storage
- normalized document 保存
- chunking pipeline
- SecretFinding detector
- access_scope mapping
- deleted / permission lost source の index 除外

完了条件:
- Drive / Discord / Notion / Hatena / Minecraft Wiki の backfill ができる。
- 変更が checksum で検出される。
- chunk が DB に保存される。
- secret_findings が生成され、deny chunk が回答 context に入らない。

### Wave 3: Retrieval / 統合質問応答
作業:
- embedding job
- dense retrieval
- sparse retrieval
- RRF
- rerank adapter
- Doc Cap
- MMR
- context packing
- citation付き回答
- citation validation + redaction
- `/ask` の統合 route

完了条件:
- eval `rag_drive_basic`, `rag_discord_basic`, `notion_basic`, `secret_redaction` が基準を満たす。
- unauthorized citation が 0。
- `/ask` が staging Discord で動く。
- Discord 本文には短い要約が出て、詳細は attachment / generated document に分離される。

### Wave 4: Workflow / Task / Event / Meeting
作業:
- Event repository
- Meeting repository
- TaskCandidate repository
- Task repository
- Schedule repository
- `/work type:meeting_prepare`
- `/work type:meeting_minutes_draft`
- `/work type:task_extract`
- `/work type:event_add|event_list|event_brief`
- `/work type:schedule_add|schedule_list`

完了条件:
- 例会前に議題案と確認事項を出せる。
- チャット・Drive・Notion からタスク候補を抽出できる。
- 承認された候補だけ Task 正本に入る。
- Event brief が関連資料・未完了タスクを返せる。

### Wave 5: 総合エージェント / DocGen / Announcement
作業:
- Agent state machine
- tool schema
- plan/tool/verify/answer loop
- cost / step budget
- DocumentPlan / SectionDraft
- Markdown renderer
- Announcement repository
- X draft tournament
- fact check
- `/work type:doc_draft|x_draft|announcement_draft`

完了条件:
- 総合エージェント eval が基準を満たす。
- 週報 / 意思決定メモ / 告知下書きが出せる。
- 外部公開不可情報の混入テストに通る。

### Wave 6: Minecraft 支援
作業:
- ServerOperation repository
- ActionSpec registry 拡張
- approval flow
- executor separation
- `/work type:|mc_status|mc_request`

完了条件:
- Minecraft 操作は dry-run と承認なしに実行されない。
- audit log が完全に残る。

### Wave 7: Automation / Production hardening
作業:
- automation rule
- automation enable / disable
- automation dry-run / run
- full eval
- red team prompt injection
- load test
- cost cap
- backup / restore test
- runbook 整備
- staged rollout

完了条件:
- production guild で公開できる条件を満たす。
- automation rule を enable / disable できる。
- rollback が実演済み。
- incident 対応手順がある。
- 例会・タスク・安全RAGが 1週間以上 staging 運用済み。

## 27. 現行実装からの移行計画
### 27.1 移行の基本方針
現行コードを小修正して延命するのではなく、新 module を並行実装し、徐々に route を切り替える。
```text
現行 src/kumc_agent
  -> 新 apps/libs 構成を追加
  -> Discord command 単位で新実装へ切替
  -> index artifact から DB へ移行
  -> OpenClaw 優先経路を停止
```

### 27.2 移行 mapping
| 現行 module | 新 module | 方針 |
|---|---|---|
| `frontends.discord` | `apps/bot` | slash command 中心に再実装 |
| `frontends.console` | `apps/api` or dev CLI | 開発用 CLI に縮小 |
| `frontends.http` | `apps/api` | stub から実装へ |
| `usecases.chat` | `libs/application/query` | RAG / Agent routing へ再設計 |
| `features.rag` | `libs/retrieval` | DB / hybrid pipeline に移行 |
| `features.indexing` | `libs/connectors` + `worker` | ingestion と indexing を分離 |
| `features.docgen` | `libs/docgen` | stub から実装へ |
| `features.summarization` | `libs/docgen` / `retrieval` | LLM 要約と summary chunk へ |
| `features.vc` | optional package | feature flag で enable / disable を切り替える |
| `infra.openclaw` | `libs/providers/openclaw_adapter` | 任意 adapter に縮小 |
| `infra.faiss` | 標準 dense index | FaissLikeIndex に一本化 |
| `infra.storage` | `libs/storage` | object storage + DB へ |
| `runtime.container` | `apps/*/bootstrap` | DI を分割 |

### 27.3 データ移行
現行 artifact から移行する場合。
```text
data/raw/*
  -> object storage raw snapshot
  -> source_items

data/chunks/chunks.jsonl
  -> documents / chunks

data/index/dense_chunks.jsonl
  -> chunks metadata 補完

data/index/material_catalog.json
  -> source_items aliases / document metadata
```
ただし、可能なら新 connector で再 backfill する。現行 artifact は migration fallback として使う。

### 27.4 段階的切替
1. 既存 Bot と別名の staging Bot を作る。
2. staging Bot で新 `/ask` を動かす。
3. 現行 `/ai` はそのまま維持する。
4. `/ask` が安定したら `/ai` を新 `/ask` へ内部転送する。
5. index build を新 ingestion へ切替。
6. OpenClaw 経路を optional にする。
7. 旧 artifact build を停止する。
8. 旧 module を archive する。

### 27.5 移行中の注意
- 同じ Discord message を二重 index しない。
- Drive の same file version を重複 document にしない。
- 旧 source URL と新 source URL の引用形式を揃える。
- eval で旧回答と新回答を比較する。
- 権限範囲が旧実装より広がらないようにする。


## 28. GitHub Issues 初期バックログ
### Foundation
1. `ADR-001` モジュラモノリス構成を決定する
2. `ADR-002` DB 正本設計を決定する
3. `ADR-003` Event / Task / Meeting 正本化方針を決定する
4. `ADR-004` SecretFinding / redaction policy を決定する
5. bot / api / worker の雛形を作る
6. config schema と env loader を作る
7. PostgreSQL migration を導入する
8. Redis queue を導入する
9. object storage adapter を作る
10. structured logging を入れる
11. OpenTelemetry trace を入れる
12. audit log repository を作る

### Discord
13. 最小 slash command registry（/ask, /work, /approval, /automation, /admin）を実装する
14. `/admin action:health` を実装する
15. interaction defer / follow-up helper を作る
16. user context resolver を作る
17. role permission resolver を作る
18. Discord component approval helper を作る
19. `/ai` 互換 adapter を作る

### Ingestion / Security
20. Connector interface を定義する
21. raw snapshot 保存を実装する
22. Drive connector backfill を実装する
23. Drive changes sync を実装する
24. Docs / Sheets / Slides / PDF normalization を実装する
25. Discord connector backfill を実装する
26. Discord gateway incremental capture を実装する
27. Hatena connector を実装する
28. Notion connector を実装する
29. checksum / idempotent upsert を実装する
30. SecretFinding detector を実装する
31. redaction utility を実装する
32. citation redaction を実装する

### Retrieval
33. chunking pipeline を実装する
34. embedding job を実装する
35. FaissLikeIndex repository を整備する
36. sparse search repository を実装する
37. RRF を実装する
38. rerank adapter を実装する
39. MMR を実装する
40. context packing を実装する
41. citation builder を実装する
42. answer generator を実装する
43. `/ask mode:search_only` を実装する
44. `/ask` を実装する

### Meeting / Task / Event
45. Event schema / repository を作る
46. Meeting schema / repository を作る
47. TaskCandidate schema / repository を作る
48. duplicate task detector を作る
49. `/work type:meeting_prepare` を実装する
50. `/work type:meeting_minutes_draft` を実装する
51. `/work type:task_extract` を実装する
52. `/approval type:task action:list` を実装する
53. `/approval type:task action:approve` を実装する
54. `/work type:event_add` を実装する
55. `/work type:event_list` を実装する
56. `/work type:event_brief` を実装する

### Agent / DocGen / Announcement
57. AgentRun / AgentStep schema を作る
58. tool schema registry を作る
59. Agent state machine を実装する
60. search / read / verify tool を実装する
61. budget enforcement を実装する
62. `/ask depth:deep` を実装する
63. X draft candidate generator を作る
64. pairwise judge を作る
65. fact check step を作る
66. `/work type:x_draft` を実装する
67. DocumentPlan schema を作る
68. Markdown renderer を作る
69. `/work type:doc_draft` を実装する
70. Announcement schema / repository を作る
71. `/work type:announcement_draft` を実装する

### Minecraft
83. ServerOperation schema / repository を作る
84. `/work type:mc_status` を実装する
85. `/work type:mc_request` を dry-run 実装する

### Task / Schedule / Automation
86. task schema を v2 に更新する
87. task repository を作る
88. `/work type:task_add` を実装する
89. `/work type:task_list` を実装する
90. `/work type:task_done` を実装する
91. schedule schema を作る
92. `/work type:schedule_add` を実装する
93. `/work type:schedule_list` を実装する
94. automation rule schema を作る
95. automation runner を作る
96. dry-run mode を作る
97. approval flow を作る
98. `/automation action:run|dry_run|enable|disable` を実装する

### Eval / Runbook
99. prompt injection test cases を作る
100. ACL eval cases を作る
101. secret_redaction eval cases を作る
102. meeting_prepare eval cases を作る
103. task_extraction eval cases を作る
106. action safety eval cases を作る
107. eval runner を作る
108. CI eval gate を作る
109. cost regression check を作る
110. backup / restore runbook を作る
111. incident runbook を作る
112. NF 当日 runbook template を作る
113. Minecraft operation rollback runbook を作る


### Workflow v2.1 追加
114. Minecraft Wiki connector を実装する
115. Wiki article / section chunking を実装する
116. `/ask source:minecraft_wiki` を実装する
117. 画像 caption / OCR / feature indexing pipeline を実装する
118. `asset_usage_requests` repository を作る
119. `/ask source:image` と `/work type:image_search` を実装する
120. `/work type:image_usage_request` を実装する
121. `member_profiles` repository を作る
122. `/ask source:member` と `/work type:member_search` を実装する
123. `workflow_candidates` / `approval_records` repository を作る
124. `/approval action:list|approve|reject|edit` を実装する
125. 自動返信・定期投稿 rule extractor を作る
126. `create_automation_rule_candidate` action を実装する
127. `indexing_runs` table と admin 通知を実装する
128. image / member / wiki 用 eval cases を作る
129. asset usage safety gate を実装する
130. member privacy redaction eval を作る
131. Minecraft icon prompt generator を staging 実装する

## 29. 受け入れ条件
### 29.1 本番公開条件
本番公開条件はフルプロダクト前提で定義する。個別機能を削った簡易版の公開条件は置かない。高リスク機能は `disabled` または `approval_required` のまま本番公開できるが、実装・設定・停止手段・監査ログは揃える。

必須条件:
- `/ask` が production guild で動作する。
- `/work`, `/approval`, `/automation`, `/admin` が権限付きで動作する。
- Drive / Discord / Notion / Hatena / X / Crafters Colony / Minecraft Wiki の取り込み設定がある。
- 権限を解決できない source は fail closed になる。
- unauthorized source が回答・citation・詳細 attachment に出ない。
- PIN、内部 IP、ネットワークキー、解錠手順、会計詳細、個人情報を通常回答で引用しない。
- 総合エージェントが複数検索を行い、根拠不足時に止まれる。
- Minecraft Wiki を根拠に edition / version 差分を明示して回答できる。
- 画像検索が画像候補、出典、投稿日時、類似画像、利用許可確認の要否を返せる。
- 画像利用 request が承認待ち候補を作成でき、承認前に外部公開素材として確定しない。
- メンバー検索が権限付きでメンバー候補を返し、個人の参加意思や能力を断定しない。
- 例会準備が議題案、未完了タスク、決定論点、告知文案を出せる。
- 議事録作成とタスク抽出が TaskCandidate を生成できる。
- TaskCandidate は承認前に Task 正本へ入らない。
- 承認・却下・編集・履歴保存ができる。
- Event / Schedule / Task の基本操作ができる。
- X 投稿案、告知文、資料下書き、ブログ記事下書きを生成できる。
- Minecraft 操作 request が生成できる。
- `/admin action:sync` で source 同期を開始できる。
- 自動 Indexing が差分、削除、権限変更を反映し、失敗時に admin 通知できる。
- `/automation` で rule の list / dry-run / run / enable / disable / mode 変更ができる。
- 自動返信・外部投稿・会計確定・Minecraft 書き込み操作は production で `approval_required` 以上に固定できる。
- すべての LLM call / retrieval / action / automation run が trace 可能である。
- audit log が保存される。
- production secrets が `.env` に依存していない。
- backup / restore が実演済み。
- rate limit / cost limit が設定済み。
- admin command の権限が確認済み。
- staging で 1週間以上運用済み。
- rollback 手順がある。
- 運営メンバーが回答品質を確認済み。
- 例会準備とタスク抽出の結果を運営メンバーがレビュー済み。
- prompt injection eval で重大失敗が 0 件である。
- secret_redaction eval で critical leak が 0 件である。

### 29.2 feature flag で本番制御する機能
以下は実装対象だが、本番公開時の既定値を保守的にする。

| 機能 | production 既定 | 公開条件 |
|---|---|---|
| VC 録音・文字起こし | disabled | 参加者同意、保存期間、削除手順が決まること |
| 画像生成 API 本番課金 | disabled | cost cap、承認フロー、生成物レビューがあること |
| 外部自動投稿 | approval_required | fact check、素材承認、投稿先確認が通ること |
| Minecraft サーバー書き込み操作 | approval_required / critical は disabled | dry-run、二者承認、rollback runbook があること |
| 会計確定処理 | disabled | 会計 role と監査手順が確定すること |
| automation auto_run | disabled by default | low risk rule のみ allowlist 登録すること |

### 29.3 外部 source / 利用規約に関する受け入れ条件
利用規約確認は後日実施する。初期状態では次を満たせばよい。
- connector ごとに `terms_review_status` を持つ。
- `pending` の source は外部再利用・再配布・公開素材化に使わない。
- 画像・ロゴ・第三者素材は `asset_usage_request` と承認 gate を通す。
- 内部検索・要約に使う source は allowlist で制限する。
