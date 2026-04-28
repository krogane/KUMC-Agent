# 総合エージェント実装調査

作成日: 2026-04-28

## 結論

`docs/design/comprehensive-agent.md` と `docs/plan/comprehensive-agent.md` が求める「完全実装」には未達である。

現行実装には `ComprehensiveAgentService`、総合エージェント用モデル、tool registry、統合入力受付経由の CLI / HTTP / Discord 入口、trace 保存、trace 読み取り API が存在する。旧 `AgenticSearchService` / `AgenticSearchRequest` / `AgenticSearchResponse` は `src` と `tests` の非legacy範囲から削除されている。

一方で、Planner / Verifier は専用LLMによる計画・検証コンポーネントとして実装されるべきところ、現行は決定的ルール中心の部分実装に留まる。加えて、承認待ち候補作成 / payload の安定契約 / Discord承認UI / 評価・監視も不足している。現状は「初期実装を超えた骨格実装」ではあるが、仕様上の完全実装とは言えない。

## 調査対象

- `docs/design/comprehensive-agent.md`
- `docs/plan/comprehensive-agent.md`
- `docs/design/kumc-agent.md` の「10. 総合エージェント」
- `src/kumc_agent/domain/models/agentic.py`
- `src/kumc_agent/features/agentic/comprehensive.py`
- `src/kumc_agent/features/agentic/tools.py`
- `src/kumc_agent/apps/agentic.py`
- `src/kumc_agent/usecases/integrated_input/entry.py`
- `src/kumc_agent/features/rag/components/integrated_input_routing.py`
- `src/kumc_agent/cli.py`
- `src/kumc_agent/frontends/http/app.py`
- `src/kumc_agent/frontends/discord/app.py`
- `src/kumc_agent/infra/agentic/repository.py`
- `tests/unit/test_agentic_docgen_announcement.py`
- `tests/unit/test_integrated_input.py`

`src/kumc_agent/infra/legacy` は設計指示通り調査対象から除外した。

## 検証コマンド

```bash
rg -n "AgenticSearch|Agentic Search|AgenticSearchRequest|AgenticSearchResponse" src tests --glob '!src/kumc_agent/infra/legacy/**'
rg -n "state=.*SEARCH|state=.*READ|\"SEARCH\"|\"READ\"" src tests --glob '!src/kumc_agent/infra/legacy/**'
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_agentic_docgen_announcement tests.unit.test_integrated_input tests.architecture.test_layer_rules
```

結果:

- `AgenticSearch*` の残存は非legacyの `src` / `tests` では検出されなかった。
- `SEARCH` / `READ` state の残存は非legacyの `src` / `tests` では検出されなかった。
- unittest は 15 tests / OK。

## 実装済みの範囲

### モデル

`domain.models.agentic` には以下が実装されている。

- `AgentBudget`
- `ToolSchema`
- `AgentRun`
- `AgentStep`
- `AgentTask`
- `ToolCallPlan`
- `AgentPlan`
- `AgentToolResult`
- `VerificationResult`
- `ComprehensiveAgentRequest`
- `ComprehensiveAgentResponse`

`AgentBudget` には `max_replans`、`allow_write_tools`、`require_citations` があり、計画で想定された主要項目は概ね揃っている。

### 状態機械

`ComprehensiveAgentService.run()` は以下を実行する。

- `AgentRun(status="running")` の保存
- `PLAN` step の保存
- `TOOL` step の保存
- `VERIFY` step の保存
- 必要に応じた再計画ループ
- `ANSWER` step の保存
- 最終 `AgentRun` の保存

状態名は `PLAN` / `TOOL` / `VERIFY` / `ANSWER` に揃っており、旧 `SEARCH` / `READ` state は使われていない。

### Tool registry

`ToolSchemaRegistry` には設計で列挙された標準toolが登録されている。

- `circle_rag_search`
- `minecraft_wiki_rag_search`
- `member_search`
- `image_search`
- `task_search`
- `task_candidate_create`
- `event_search`
- `event_candidate_create`
- `server_operation_candidate_create`
- `approval_candidate_create`

`read_only` の設定も検索系は `true`、候補作成系は `false` になっている。

### Tool adapter

`ComprehensiveToolAdapters` は以下へ委譲している。

- `circle_rag_search`: `ask_service.ask(RetrievalQuery(...))`
- `minecraft_wiki_rag_search`: `source_filter="minecraft_wiki"` の `ask_service.ask(...)`
- workflow系tool: `WorkflowService.run(WorkRequest(...))`

`AccessContext` は `RetrievalQuery` / `WorkRequest` に渡されている。

### 統合入力受付と入口

統合入力受付は `comprehensive_agent` route を持ち、`required_features` が2つ以上のとき総合エージェントへ昇格する。

CLI `ask`、HTTP `/ask`、Discord `/ask` は統合入力受付を経由するため、旧Agentic Search専用入口ではなく総合エージェントへ到達できる構成になっている。`depth=deep` はリクエストmetadataとして総合エージェントに渡され、Planner側で `circle_rag` を補うヒントとして使われている。

### Trace保存・読み取り

`FileAgentTraceRepository` と `PostgresAgentTraceRepository` は以下を持つ。

- `save_run`
- `save_step`
- `get_run`
- `list_steps`
- `latest_runs`

HTTPには以下のtrace参照APIがある。

- `GET /agent/runs`
- `GET /agent/runs/{run_id}`

### Sanitization

総合エージェント側の `sanitize_payload()` / `sanitize_text()` は、少なくとも以下を除去またはマスクする。

- `secret` / `password` / `token` / `api_key`
- `raw` / `context` / `contexts` / `llm_prompt`
- 内部IP
- PIN / network key / unlock steps
- 長文の切り詰め

統合入力受付、CLI、HTTPでも `features.foundation.payload_sanitizer` が使われている。

## 仕様と実装の差分

| 仕様項目 | 現行実装 | 判定 |
| --- | --- | --- |
| AgenticSearch関連コード削除 | 非legacyの `src` / `tests` から `AgenticSearch*` は消えている | 概ね完了 |
| 状態機械 | `PLAN` / `TOOL` / `VERIFY` / `ANSWER` を保存する | 骨格は完了 |
| PLANで入力分解、必要機能、tool順序、検索条件、成功条件、副作用境界を決定 | 専用LLM Plannerは未実装。現行はキーワードベースでfeatureからtoolを選ぶ。分解は「toolごとのtask」程度で、検索条件・成功条件は汎用文言に留まる | 部分実装 |
| 入力が曖昧な場合に質問を返す | 空または極端に短いqueryのみ質問。副作用候補作成に必要な必須項目不足はWorkflow側の結果依存 | 部分実装 |
| 単一機能は直接ルーティング | 統合入力受付では直接ルーティングされる。ただし `ComprehensiveAgentService` を直接呼ぶと単一toolでも実行する | 部分実装 |
| RAG / Minecraft Wiki / メンバー / 画像 / タスク / イベント / サーバー候補tool | registry上は揃っており、adapterも主要toolへ委譲する | 概ね実装 |
| `approval_candidate_create` | registryにはあるがPlannerが選ばず、adapterの明示実装もない。未対応toolは実質 `task_list` へ落ちる | 未実装 |
| 副作用は候補作成または承認申請まで | Workflow側の task/event/server は候補・dry-run中心。総合エージェント側は `allow_write_tools` で候補作成を止められる | 部分実装 |
| 承認待ち候補IDと承認対象をresponseへ含める | candidate IDは回答文と一部top-levelに出る。approval recordや承認targetの統一payloadはない | 部分実装 |
| VERIFYで根拠不足、矛盾、権限外情報、副作用境界違反を検出 | 専用LLM Verifierは未実装。現行はcitation有無とcandidate statusの一部のみ確認。ACL再検証、tool間矛盾、secret混入、正本変更の前後比較は未実装 | 未達 |
| 根拠不足・矛盾時の再計画 | `max_replans` まで再度PLANするが、検索queryやtool入力を実質改善しないため同じ実行を繰り返しやすい | 部分実装 |
| 最終回答に結論、根拠、使用機能、未確認事項、承認待ち候補 | `ComprehensiveAgentAnswerBuilder` が該当セクションを出す | 実装 |
| `ComprehensiveAgentResponse` のトップレベルは安定主結果のみ | dataclassに `run: AgentRun` がトップレベルで存在する。統合入力受付経由では落とされるが、ServiceのI/Fとしては仕様とずれる | 部分不一致 |
| task/event change candidate、schedule candidate、approvalsの扱い | `_work_candidates()` は task_change / event_change を内部候補化するが、response top-levelは `task_candidates` / `event_candidates` / `server_operations` 中心。schedule / approvals は総合エージェントresponseから落ちる | 不足 |
| CLI/HTTP/Discord payloadの診断情報をmetadata配下へ | 統合入力受付経由では概ねmetadata配下。ただし直接 `ComprehensiveAgentResponse` の `run` は診断情報を含む | 部分実装 |
| 大きなcontext、secret、権限外情報を外部payloadやtraceに出さない | sanitizerはあるが、VERIFY項目としての検査やtool別のACL証跡確認はない | 部分実装 |
| Discordで承認候補を扱う | 統合入力応答は最初のtask候補にだけtask承認viewを付ける。event/server候補は総合エージェント経由では専用viewが付かない | 部分実装 |
| 評価・監視連携 | run/step保存とreadinessのcost集計はある。tool別成功率、総合エージェント評価、再計画率、latency等の集計は限定的 | 未達 |
| テスト | 状態機械、citation不足、registry、統合入力昇格はある。tool全種、承認境界、replan、payload契約、HTTP/Discordは不足 | 部分実装 |

## 主要な未達点

### 1. 専用LLM Plannerが未実装である

総合エージェントのPlannerは、専用LLMを用いて入力分解、tool選択、検索条件、成功条件、副作用境界、再計画条件をJSON schemaに沿って生成するコンポーネントとして実装する必要がある。

現行Plannerは `detect_required_features()` とキーワード分岐でtool列を作る。入力を小タスクに分解しているように見えるが、実態は「各toolを実行する」というtaskを生成しているだけであり、専用LLM Plannerとしては未実装である。

不足しているもの:

- toolごとの具体的な検索条件
- toolごとの成功条件
- 副作用境界の理由
- 必須項目不足時のclarification
- 再計画時に変える検索queryやtool入力
- 単一機能時のdirect route返却
- 専用LLMのJSON schema validationを伴うplanner出力

### 2. `approval_candidate_create` が実装されていない

仕様では承認待ち候補作成toolが標準toolに含まれる。しかし現行実装では registry に登録されているだけで、Plannerが選択しない。adapterも明示的に `approval_candidate_create` を処理せず、未知toolは `_work_type_for_tool()` の末尾で `task_list` に落ちる。

このため「候補作成」と「承認申請作成」が仕様上は分かれているにもかかわらず、実装上は候補IDを返すだけで、approval recordや承認targetを統一的に作成する経路がない。

### 3. 専用LLM Verifierが未実装である

総合エージェントのVerifierは、専用LLMを用いてtool結果、引用根拠、候補、矛盾、未確認事項、副作用境界違反を検証し、構造化された `VerificationResult` を返すコンポーネントとして実装する必要がある。決定的に検査できる項目はルールで前後比較し、意味的な矛盾や根拠の妥当性は専用LLM Verifierで補完する構成が望ましい。

現行Verifierは以下を主に見る。

- 予定toolの成功結果があるか
- RAG系toolでcitationがあるか
- write系toolのcandidate statusが `done` / `completed` / `executed` になっていないか
- `execution_allowed=True` がmetadataにないか

仕様で求める以下は未実装または限定的である。

- tool間矛盾検出
- 権限外情報の混入検査
- secretや巨大contextの混入検査をVERIFY結果として扱うこと
- 正本変更が発生していないことのrepository前後比較
- 成功条件ごとの `satisfied` / `missing` / `conflicts`
- 候補IDとapproval IDの対応確認

### 4. 再計画が実質的に同じ実行を繰り返す

citation不足時などに再計画ループは動くが、現行Plannerは `previous_results` や `previous_verification` をほとんど使わない。`circle_rag_search` を追加する以外、検索queryや条件を変えないため、同じtoolを重複実行しやすい。

仕様上の「根拠不足や矛盾時の再計画」は、検索条件の変更、別tool追加、質問への切り替え、停止理由の明示まで含めて定義する必要がある。

### 5. Response契約が完全には固まっていない

`ComprehensiveAgentResponse` は `run: AgentRun` をトップレベルに持つ。統合入力受付経由の外部payloadでは落ちるが、Service I/Fとしては「トップレベルには主結果だけを置く」という方針とずれる。

また、現行responseには以下の安定フィールドが足りない。

- `task_change_candidates`
- `event_change_candidates`
- `schedule_candidates`
- `approvals`
- approval target type / target id の統一表現

仕様側も `task_candidates` / `event_candidates` / `server_operations` は列挙しているが、変更候補やschedule候補をどう扱うかが曖昧である。

### 6. Discordの総合エージェント承認導線が不足している

Discord `/ask` は統合入力受付を使うため総合エージェントには到達できる。しかし `_send_integrated_response()` は最初のtask候補IDだけを見てtask承認viewを付ける。

不足しているもの:

- event候補へのevent承認view
- server operationへの承認view
- 複数候補がある場合の選択UI
- task_change / event_change / schedule候補への導線

### 7. 評価・監視が仕様の粒度に足りない

trace保存とHTTP参照APIはある。readiness側でagentic JSONLのcostを集計する実装もある。

ただし仕様が求める以下は確認できない。

- tool別失敗率
- 平均step数
- `insufficient_evidence` 率
- `needs_approval` 率
- 再計画回数
- 総合エージェント専用eval
- routing / plan / tool / verify / answer品質の分解評価

## 仕様改善点

### 1. 「完全実装」の受け入れ基準を明文化する

現行planはPhaseが多い一方、どこまでをMVP、どこからを完全実装とするかが曖昧である。以下のように段階を分けると実装判定しやすい。

- Level 1: 状態機械、trace、read-only RAG tool
- Level 2: workflow tool、候補作成、payload sanitization
- Level 3: approval作成、Verifier前後比較、HTTP/Discord承認UI
- Level 4: tool別評価、監視、専用LLM Planner / Verifier、schema validation、実運用runbook

### 2. 承認モデルを再定義する

`task_candidate_create` / `event_candidate_create` / `server_operation_candidate_create` と `approval_candidate_create` の責務境界を明確にするべきである。

決めるべきこと:

- 候補作成時にapproval recordも作るのか
- approval recordは手動承認画面を開いた時に作るのか
- `needs_approval` の必須payloadは何か
- approval target type / target id / candidate id / batch id の正規化形式
- 複数候補を1つのapproval batchにまとめる条件

### 3. 専用LLM Planner schemaをtool別に具体化する

専用LLM Plannerが安定した構造化出力を返せるように、`AgentPlan` とtool入力schemaを具体化するべきである。現行の `AgentPlan` schemaは存在するが、tool入力の仕様が粗い。tool別に以下を定義したほうがよい。

- `circle_rag_search`: query、source_filters、recency、required_citation_count
- `minecraft_wiki_rag_search`: query、edition、version、article_filter
- `member_search`: query、role、skill、availability、guild_id
- `image_search`: query、source_filters、media_type、usage_context
- `task_candidate_create`: title、description、assignee_hint、due_at、related_event_id、evidence_required
- `event_candidate_create`: title、starts_at、place、summary、related_sources
- `server_operation_candidate_create`: operation、target、dry_run_required、risk
- `approval_candidate_create`: target_type、target_ids、reason、required_approver_role

### 4. 専用LLM Verifierを仕様として詳細化する

専用LLM Verifierは、決定的チェックの結果とtool出力の短い正規化要約を入力に取り、成功条件ごとの充足、根拠不足、矛盾、未確認事項を構造化して返す仕様にするべきである。現行仕様はさらに実装可能な形に落とす必要がある。

推奨する検証単位:

- `EvidenceCheck`: toolごとのcitation件数、citationのACL、引用長
- `CandidateBoundaryCheck`: repository前後の正本件数、candidate status、server executor未実行
- `ConflictCheck`: 同一フィールドの不一致、日付・担当者・場所の矛盾
- `SanitizationCheck`: secret、内部IP、PIN、巨大context
- `ApprovalCheck`: candidate IDとapproval targetの対応
- `BudgetCheck`: step、search calls、latency、cost、replan

### 5. Response payloadを統一する

総合エージェントresponseと統合入力responseの差を小さくした方がよい。

提案:

- `run` はトップレベルに置かず、`metadata.agent_run_id` とtrace APIで参照する。
- 候補系は `candidates` の正規化配列を追加し、互換用に `task_candidates` 等も残す。
- `approvals` をtop-level主結果として追加する。
- `task_change_candidates`、`event_change_candidates`、`schedule_candidates` を含めるか、正規化 `candidates` に一本化する。
- `metadata` に入れてよいキーと外部出力前に落とすキーを表で定義する。

### 6. `depth=deep` の扱いを整理する

設計では `depth=deep` を互換経路として残さない方針だが、現行CLI/HTTP/Discordにはdepth引数があり、Plannerのmetadata hintとして使われている。

仕様側で以下を明記した方がよい。

- `depth=deep` はrouteを強制するのか、read-only深掘りhintなのか
- 単一RAG + `depth=deep` を総合エージェントへ昇格するのか
- `depth=deep` と複数機能昇格が同時に起きた場合の優先順位
- 外部payloadに `depth` を残す場合のmetadata配置

### 7. 権限方針を統合する

`docs/design/kumc-agent.md` ではタスク・イベント管理はadmin限定とされている。一方、総合エージェント詳細設計では `candidate_only` が実行可能と読める。現行実装は `allow_write_tools = risk in {candidate_only, approval_required} and access.is_admin` で、adminでない場合は候補作成も止める。

仕様上、次を明記すべきである。

- 非adminの候補作成依頼は拒否するのか、予定だけ返すのか、clarifyにするのか
- メンバー検索や画像検索と組み合わせた場合も同じ権限にするのか
- server管理はadmin限定かつapproval_requiredでよいか
- Discord guild / role / user IDごとの権限をtool入力にどう記録するか

### 8. テスト計画を機能別に分割する

現行テストは状態機械と統合入力昇格の最低限に留まる。完全実装の受け入れには以下が必要である。

- `tests/unit/test_comprehensive_agent.py` を新設する。
- 全標準toolのadapterテストを追加する。
- `allow_write_tools=False` で候補作成が実行されないことを検証する。
- task/event/server候補作成で正本が増えないことを検証する。
- `approval_candidate_create` の作成・payloadを検証する。
- citation不足時に意味のあるreplanになることを検証する。
- secret / context / internal IPがtraceとpayloadに残らないことを検証する。
- CLI / HTTP / Discord payloadのsnapshot系テストを追加する。

## 推奨対応順

1. `approval_candidate_create` の仕様と実装を確定する。
2. `ComprehensiveAgentResponse` から `run` を外部I/F上分離し、候補・approval payloadを正規化する。
3. 専用LLM Plannerを実装し、tool別input schema、clarification条件、JSON schema validationを接続する。
4. 専用LLM Verifierを実装し、repository前後比較、secret/context検査、tool別citation検査などの決定的チェック結果を入力として渡す。
5. 再計画時に検索query・source filter・tool列を変えるロジックを入れる。
6. Discord統合入力応答でevent/server/change/schedule候補の承認導線を追加する。
7. tool別評価と監視集計を追加する。
8. `docs/kumc-agent-redesign-v4.md` に残る旧Agentic Search説明を現行仕様へ寄せるか、旧文書として扱う旨を明記する。
