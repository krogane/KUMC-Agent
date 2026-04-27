# 総合エージェント 実装計画

## 1. 方針
`docs/design/kumc-agent.md` と `docs/design/comprehensive-agent.md` に従い、総合エージェントを実装する。

実装では `src/kumc_agent/infra/legacy` を参照・依存しない。既存の共通部品は `domain.models.agentic` の汎用モデル、`infra.agentic.repository`、`features.workflow.service.WorkflowService`、`features.retrieval`、`domain.models.retrieval.AccessContext`、`domain.models.workflow.WorkResponse` を優先して使う。現行の `AgenticSearchService`、`AgenticSearchRequest`、`AgenticSearchResponse`、Agentic Search専用起動経路は削除し、ComprehensiveAgentのコードとして作り直す。現行実装と設計が矛盾する場合は `kumc-agent.md` を優先する。

初期実装では、`ComprehensiveAgentService` を新設し、AgenticSearch関連コードを置き換える。統合入力受付から2つ以上の機能が必要な入力を検出した場合に総合エージェントへルーティングする。深掘り検索が必要な単一検索依頼も、Agentic SearchではなくComprehensiveAgentのread-only計画として扱う。

## 2. 完了条件
- 統合入力受付が、必要機能を分類し、2つ以上の場合に総合エージェントへ昇格できる。
- 単一機能で解決できる入力は、既存の単一機能へ直接ルーティングできる。
- 総合エージェントが PLAN / TOOL / VERIFY の状態機械として実行される。
- PLANで入力分解、必要機能、tool順序、検索条件、成功条件、副作用境界を決定できる。
- サークル情報RAG、Minecraft Wiki RAG、メンバー検索、画像検索、タスク管理、イベント管理、サーバー管理候補作成をtoolとして呼び出せる。
- 副作用のある操作は候補作成または承認申請までに限定され、承認前に正本変更や実行が起きない。
- VERIFYで根拠不足、矛盾、権限外情報、副作用境界違反を検出できる。
- 根拠不足や矛盾時に最大n回まで再計画できる。
- 最終回答に結論、根拠、使用した機能、未確認事項、承認待ち候補を含められる。
- `agent_runs` と `agent_steps` にrun/step traceを保存できる。
- CLI、HTTP、Discord payloadの診断情報が `metadata` 配下に入る。
- 大きなcontext、secret、権限外情報を外部payloadやtraceに出さない。
- 既存の `depth=deep` Agentic Search経路が削除され、必要な深掘り検索はComprehensiveAgentへ接続される。
- AgenticSearch関連のテスト、fixture、payload期待値がComprehensiveAgent前提へ更新される。
- 主要動作を既存のunittest方式で検証できる。

## 3. 実装ステップ
### Phase 1: AgenticSearch削除範囲の確定
1. `AgenticSearchService`、`AgenticSearchRequest`、`AgenticSearchResponse` の参照箇所を洗い出す。
2. CLI、HTTP、Discord、workflow、app context、テストからAgentic Search専用起動経路を洗い出す。
3. `SEARCH`、`READ` stateを前提にしたtrace期待値を洗い出す。
4. `AgentRun`、`AgentStep`、`AgentBudget`、`ToolSchema`、`AgentTraceRepository` のうち、ComprehensiveAgentでも使う汎用部品を確定する。
5. AgenticSearch専用の名前、payload、テストは削除またはComprehensiveAgent向けに改名する。

主な削除・置換候補は次の通り。

- `src/kumc_agent/features/agentic/service.py` の `AgenticSearchService`
- `src/kumc_agent/domain/models/agentic.py` の `AgenticSearchRequest`、`AgenticSearchResponse`
- `src/kumc_agent/features/agentic/__init__.py` のAgenticSearch export
- `src/kumc_agent/apps/agentic.py` のAgenticSearch app context
- `src/kumc_agent/cli.py` のAgentic Search起動分岐
- `src/kumc_agent/frontends/http/app.py` のAgentic Search起動分岐
- `src/kumc_agent/frontends/discord/app.py` のAgentic Search起動分岐
- `src/kumc_agent/features/workflow/service.py` の `agentic_search` 依存
- Agentic Search前提の単体テストとpayload期待値

検証:
- `rg "AgenticSearch|Agentic Search|AgenticSearchRequest|AgenticSearchResponse"` で削除・移行対象が明確になっていること。
- 汎用部品として残す対象が総合エージェントの責務と矛盾しないこと。

### Phase 2: 総合エージェント用モデル追加
1. `ComprehensiveAgentRequest`、`ComprehensiveAgentResponse` を追加する。
2. `AgentPlan`、`AgentTask`、`ToolCallPlan`、`AgentToolResult` 相当のモデルを追加する。
3. 状態名は `PLAN`、`TOOL`、`VERIFY`、`ANSWER` に揃える。
4. `AgentRun.status` に `needs_approval` を扱えるようにする。
5. `AgentBudget` に `max_replans` を追加するか、`metadata` で明示する。
6. `AgenticSearchRequest/Response` は削除し、外部I/Fは `ComprehensiveAgentRequest/Response` に統一する。

検証:
- `AgenticSearchRequest/Response` のimportが残らないこと。
- 新responseのトップレベルに安定結果だけが出ること。

### Phase 3: ToolSchemaRegistry拡張
1. `ToolSchemaRegistry` に総合エージェント標準toolを追加する。
2. `circle_rag_search` を登録する。
3. `minecraft_wiki_rag_search` を登録する。
4. `member_search` を登録する。
5. `image_search` を登録する。
6. `task_search`、`task_candidate_create` を登録する。
7. `event_search`、`event_candidate_create` を登録する。
8. `server_operation_candidate_create` を登録する。
9. `approval_candidate_create` を登録する。
10. `read_only=false` のtoolが候補作成に限定されることをschemaとテストで明示する。

検証:
- registryのtool名が重複しないこと。
- read_only属性が正しいこと。
- required input schemaが空や不正入力を拒否できること。

### Phase 4: Tool Adapter実装
1. `features/agentic/adapters.py` を追加する。
2. `circle_rag_search` adapterで `RetrievalQuery` を呼ぶ。
3. `minecraft_wiki_rag_search` adapterでMinecraft Wiki向けsource filterまたは専用serviceを呼ぶ。
4. `member_search` adapterで `WorkflowService.run(work_type="member_search")` を呼ぶ。
5. `image_search` adapterで `WorkflowService.run(work_type="image_search")` を呼ぶ。
6. `task_candidate_create` adapterで `task_extract`、`task_add`、`task_update`、`task_delete` を呼び分ける。
7. `event_candidate_create` adapterで `event_extract`、`event_add`、`event_update`、`event_delete` を呼び分ける。
8. `server_operation_candidate_create` adapterで `mc_request` を呼ぶ。
9. 各adapterで `AccessContext` を必ず渡す。
10. tool固有responseを `AgentToolResult` に正規化する。

検証:
- 各toolが期待する既存serviceへ委譲されること。
- `AccessContext` が失われないこと。
- candidate作成toolが正本変更を行わないこと。
- toolエラーが総合エージェント全体を即時クラッシュさせず、VERIFYで扱えるstatusになること。

### Phase 5: Planner実装
1. `ComprehensiveAgentPlanner` を追加する。
2. 初期実装では決定的ルールで必要機能を抽出する。
3. LLM plannerを使う場合は、`assets/prompts/comprehensive_agent_plan.md` を追加し、JSON schema validationを通す。
4. PLAN出力に `tasks`、`required_tools`、`tool_sequence`、`success_criteria`、`side_effect_boundary`、`retry_policy` を含める。
5. 入力が曖昧で必要機能を決められない場合は、toolを実行せず質問を返す。
6. 単一機能だけで十分な場合は、総合エージェント内で処理せず、呼び出し元へ直接ルーティング指示を返す。

検証:
- 複合依頼が2つ以上のtool計画になること。
- 単一依頼は直接ルーティング扱いになること。
- 副作用を含む依頼に `candidate_only` または `approval_required` が付くこと。

### Phase 6: ComprehensiveAgentService実装
1. `features/agentic/comprehensive.py` を追加する。
2. `search()` ではなく `run()` を公開する。
3. run開始時に `AgentRun(status="running")` を保存する。
4. PLAN stepを保存する。
5. budget内でTOOL stepを順に実行する。
6. 各tool結果をstepとして保存する。
7. VERIFY stepを実行する。
8. 根拠不足や矛盾時に `max_replans` までPLAN/TOOLへ戻る。
9. ANSWER stepを保存する。
10. 最終 `AgentRun` にanswer、citations、confidence、metadataを保存する。

検証:
- PLAN / TOOL / VERIFY / ANSWER の順でstepが保存されること。
- 複数tool結果が最終responseへ集約されること。
- budget超過で停止し、warningを返すこと。
- `needs_approval` statusを返せること。

### Phase 7: VERIFY実装
1. `ComprehensiveAgentVerifier` を追加する。
2. citation必須の計画でcitationがない場合は不足扱いにする。
3. tool結果の矛盾を検出するため、同一fieldに異なる値がある場合はwarningにする。
4. `read_only=false` toolの結果に正本変更済みpayloadが含まれていないか検査する。
5. 権限外情報、secret、巨大contextの混入を検査する。
6. 成功条件ごとに `satisfied`、`missing`、`conflicts` を返す。

検証:
- citation不足で再計画または `insufficient_evidence` になること。
- 候補作成toolがTask/Event正本を返した場合に失敗扱いになること。
- secretらしき文字列が最終出力から除外されること。

### Phase 8: 回答生成
1. `ComprehensiveAgentAnswerBuilder` を追加する。
2. 結論、根拠、使用した機能、未確認事項、承認待ち候補を整形する。
3. citationsを重複除去する。
4. candidate ID、target type、承認が必要であることを明示する。
5. 実行済みと誤解される表現を避ける。
6. `detail_markdown` にtrace summaryを入れる。
7. Discord向けには長文をattachmentへ逃がせる構造にする。

検証:
- 使用した機能一覧が回答に含まれること。
- 未確認事項が空でない場合に明示されること。
- 承認待ち候補が候補として表示され、実行済みと書かれないこと。

### Phase 9: app context配線
1. `AgenticAppContext` は `ComprehensiveAgentAppContext` へ改名するか、`comprehensive_agent` のみを持つcontextへ変更する。
2. `build_agentic_app_context()` は `build_comprehensive_agent_app_context()` へ改名する。
3. 循環が発生する場合は、総合エージェント用app contextを `apps/workflow.py` 側で組み立てる。
4. repositoryは既存 `build_agent_trace_repository()` を共有する。
5. `AgenticSearchService` の組み立て、export、importを削除する。

検証:
- app context生成で循環importしないこと。
- Postgres有効時はPostgres repository、未設定時はJSONL repositoryになること。
- `apps/agentic.py` にAgenticSearch専用contextが残らないこと。

### Phase 10: 統合入力受付の分類拡張
1. `EntryRoutingDecision` を拡張するか、新しい統合router resultを追加する。
2. routeに `comprehensive_agent` を追加する。
3. `required_features`、`risk`、`source_filters`、`attribute_filters` を分類結果に含める。
4. 既存 `direct_rag` / `openclaw` の扱いは、統合入力受付の既存挙動を壊さない範囲で維持する。ただしAgentic Search fallbackは残さない。
5. `required_features` が2つ以上の場合に総合エージェントへ渡す。
6. 分類失敗時のfallback方針を決める。副作用を含む可能性がある場合は直接実行しない。
7. ルーティング判断は最終payloadのトップレベルではなく `metadata` に入れる。

検証:
- 複合依頼が `comprehensive_agent` になること。
- 単一RAG質問が直接RAGになること。
- 分類失敗時に副作用候補が作られないこと。

### Phase 11: CLI/HTTP/Discord配線
1. CLIに総合エージェント起動経路を追加する。
2. `ask` の `depth=deep` はAgentic Searchを呼ばず、必要に応じてComprehensiveAgentのread-only計画を起動する。
3. 複合依頼用の `mode` またはroute結果からComprehensiveAgentを呼び出す。
4. HTTP `/ask` がroute結果に応じて総合エージェントを呼べるようにする。
5. Discord `/ask` がroute結果に応じて総合エージェントを呼べるようにする。
6. 長い `detail_markdown` はDiscord attachmentにする。
7. payloadの診断情報を `metadata` 配下に整理する。
8. CLI、HTTP、Discordから `AgenticSearchRequest` をimportする分岐を削除する。

検証:
- CLIで複合依頼のJSON payloadが安定schemaで返ること。
- HTTPで `task_candidates`、`event_candidates`、`server_operations` がトップレベル主結果として返ること。
- Discordで権限外情報やsecretが表示されないこと。
- `depth=deep` がAgentic Searchに到達しないこと。

### Phase 12: 候補作成と承認境界の統合
1. タスク候補作成toolが `TaskCandidate` または `TaskChangeCandidate` だけを返すことを保証する。
2. イベント候補作成toolが `EventCandidate` または `EventChangeCandidate` だけを返すことを保証する。
3. サーバー管理toolが `ServerOperation` dry-run候補だけを返すことを保証する。
4. 承認対象を `approval_records` または各候補モデルに紐づける。
5. 総合エージェントの最終statusを、候補がある場合は `needs_approval` にする。
6. 承認UIに必要なtarget type、target id、候補概要をresponseへ含める。

検証:
- 承認前に `tasks`、`events`、server executor結果が増えないこと。
- 候補IDが最終回答とmetadataに含まれること。
- `needs_approval` statusが監査・traceに残ること。

### Phase 13: Trace read APIと評価連携
1. `AgentTraceRepository` にrun/step取得APIを追加する。
2. JSONL repositoryで最新runとstep列を読み戻せるようにする。
3. Postgres repositoryでrun id検索とstep一覧取得を実装する。
4. 評価基盤からtool単位の成否、使用機能、安全性を参照できるようにする。
5. readiness/monitoringで総合エージェントの成功率、latency、コスト、tool別失敗率を集計する。

検証:
- run idから全stepを復元できること。
- tool別statusを評価対象として抽出できること。

### Phase 14: 安全性とpayload sanitization
1. 総合エージェント共通のsanitizerを追加する。
2. tool入力、tool出力、trace、最終payloadに対してsecret検出と長さ制限を適用する。
3. RAG context全文は外部payloadへ出さず、citation idと短い要約にする。
4. 内部IP、PIN、token、招待URL、学籍番号、連絡先をマスクする。
5. `metadata` に入れる値も外部出力前にマスクする。

検証:
- secretを含むtool出力が最終回答に出ないこと。
- traceにも巨大contextが保存されないこと。
- `routing_decision`、`selected_tool`、`trace_id` がトップレベルへ出ないこと。

### Phase 15: ドキュメント更新
1. `docs/explanation/cli.md` に総合エージェントのCLI例を追加する。
2. 統合入力受付の設計文書がある場合はroute追加を反映する。
3. 評価基盤のtarget一覧に総合エージェントのtool単位評価を追記する。
4. 運用runbookに、`insufficient_evidence`、`needs_approval`、tool failureの確認手順を追加する。
5. Agentic Searchの説明、コマンド例、payload例が残っている場合はComprehensiveAgentへ置き換える。

検証:
- docs上のコマンド例が現行CLIと一致すること。
- payload例で診断情報が `metadata` 配下にあること。

### Phase 16: 回帰テスト
1. ComprehensiveAgent用に更新したテストを実行する。
2. Workflow関連テストを実行する。
3. Entry routing関連テストを実行する。
4. CLI payload関連テストを実行する。
5. Architecture testでlegacy依存がないことを確認する。
6. AgenticSearch関連のimportや起動分岐が残っていないことを確認する。

検証候補:
- `python -m unittest tests.unit.test_comprehensive_agent`
- `python -m unittest tests.unit.test_workflow_service`
- `python -m unittest tests.unit.test_chat_entry_usecase`
- `python -m unittest tests.unit.test_entry_query_router`
- `python -m unittest tests.architecture.test_layer_rules`
- `rg "AgenticSearch|Agentic Search|AgenticSearchRequest|AgenticSearchResponse" src tests docs`

## 4. 実装順の推奨
最初にAgenticSearch関連コードの削除範囲を確定し、その後 `ComprehensiveAgentService` を新規実装する。互換経路は残さず、`depth=deep` を含む深掘り系の入口もComprehensiveAgentへ統一する。

推奨順は次の通り。

1. AgenticSearch関連のservice、request/response、起動経路、テスト期待値を削除対象として確定する。
2. ComprehensiveAgent用モデルとtool schemaを追加する。
3. read-only tool adapterを実装する。
4. PLAN / TOOL / VERIFY / ANSWER の最小serviceを実装する。
5. タスク、イベント、サーバー管理の候補作成toolを追加する。
6. 統合入力受付からの昇格を接続する。
7. CLI/HTTP/Discord出力をComprehensiveAgentへ統一する。
8. AgenticSearch関連参照を削除する。
9. 評価、監視、sanitizationを強化する。

## 5. リスク
- `apps/agentic.py` と `apps/workflow.py` の依存が循環しやすい。総合エージェントの組み立て位置を早めに固定する。
- AgenticSearchを削除するため、`depth=deep` の既存利用者には挙動変更が出る。入口はComprehensiveAgentへ統一し、payload変更をdocsに明示する。
- 候補作成toolは `read_only=false` だが、正本変更は禁止である。テストで承認境界を固定する。
- tool結果に大きなRAG contextやsecretが混ざりやすい。trace保存前と外部出力前の両方でsanitizationする。
- OpenClaw経路とローカル総合エージェント経路が併存する。統合入力受付のroute方針を明示し、fallback時に副作用候補を作らない。
- `domain.models.agentic` や `infra.agentic.repository` の名称が残る場合、AgenticSearchと誤読されやすい。必要に応じて `comprehensive` 系の名称へ改名する。
