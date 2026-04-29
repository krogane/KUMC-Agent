# 統合入力受付 実装計画

## 1. 方針
`docs/design/kumc-agent.md` と `docs/design/integrated-input.md` に従い、統合入力受付を実装する。

実装では `src/kumc_agent/infra/legacy` を参照・依存しない。既存の共通部品は `domain.models.retrieval.AccessContext`、`domain.models.retrieval.RetrievalQuery`、`domain.models.workflow.WorkRequest`、`domain.models.workflow.WorkResponse`、`features.workflow.service.WorkflowService`、新規実装後の `features.agentic.comprehensive.ComprehensiveAgentService` を優先して使う。現行実装と設計が矛盾する場合は `kumc-agent.md` を優先する。

`ChatEntryUsecase`、`EntryQueryRouter`、`EntryRoutingDecision`、CLI/HTTP/Discordに分散している旧 `/ask` 分岐は削除する。互換経路や後方互換adapterは残さず、入口は `IntegratedInputUsecase` に統一する。

## 2. 完了条件
- Discord統合コマンドから質問、依頼、管理操作を受け取れる。
- 入力本文、source、mode、depth、ユーザー権限情報を正規化できる。
- ルーティング前にGuild、role、admin設定を解決し、全route先へ同じ `AccessContext` を渡せる。
- 分類結果にintent、source_filters、risk、freshness要否、属性フィルタ、required_featuresを含められる。
- 診断情報やルーティング判断が最終payloadのトップレベルではなく `metadata` 配下に入る。
- サークル情報RAG、Minecraft Wiki RAG、メンバー検索、画像検索、タスク管理、イベント管理、サーバー管理、総合エージェントへルーティングできる。
- required_featuresが2つ以上の場合に総合エージェントへ昇格できる。
- 副作用のある依頼を直接実行せず、候補作成または承認待ちに限定できる。
- 分類失敗時に副作用候補を作成せず、read-only fallbackまたは確認質問を返せる。
- HTTPでは `maintenance_command_author_ids` に含まれる `user_id` だけをadminにし、CLIの `--admin` はローカル実行として信頼できる。
- `history_scope` をRAG、workflow metadata、総合エージェントmetadataへ伝播できる。
- task/event完了と通知依頼が統合入力受付経由で正本を直接変更しないことを検証できる。
- Discordへの送信責務が統合入力受付adapterに集約される。
- CLI、HTTP、Discordで同じ出力envelopeを利用できる。
- secret、巨大context、raw prompt、画像local pathが外部payloadに出ない。
- `ChatEntryUsecase`、`EntryQueryRouter`、旧 `/ask` 分岐が削除され、通常入力が `IntegratedInputUsecase` 経由だけになる。
- 主要動作を既存のunittest方式で検証できる。

## 3. 実装ステップ
### Phase 1: 旧入口の削除範囲確定
1. `ChatEntryUsecase`、`ChatEntryRequest`、`EntryRoutingDecision`、`EntryRoute`、`EntryQueryRouter` の参照箇所を列挙する。
2. CLI `ask` の `source=member`、`depth=deep`、通常RAGの個別分岐を削除対象として特定する。
3. HTTP `/ask` の `source=member`、`depth=deep`、通常RAGの個別分岐を削除対象として特定する。
4. Discord `/ask` の `source=member`、`depth=deep`、通常RAGの個別分岐を削除対象として特定する。
5. `direct_rag` / `openclaw` とOpenClaw入口を削除対象として特定する。
6. 削除後に `IntegratedInputUsecase` が引き継ぐ入力、分類、権限、payload項目を一覧化する。

検証:
- `rg "ChatEntryUsecase|EntryQueryRouter|EntryRoutingDecision|direct_rag|openclaw"` で削除対象を把握できていること。
- 削除対象の責務が後続Phaseの新componentへ割り当てられていること。

### Phase 2: domain model追加
1. `domain/models/integrated_input.py` を追加する。
2. `IntegratedInputRequest` を定義する。
3. `IntegratedInputDecision` を定義する。
4. `IntegratedInputResponse` を定義する。
5. `IntegratedRoute` のLiteralに `circle_rag`、`minecraft_wiki_rag`、`member_search`、`image_search`、`task_management`、`event_management`、`server_management`、`comprehensive_agent`、`clarify`、`deny` を定義する。
6. `RequiredFeature`、`RiskLevel`、`InputIntent` のLiteralを定義する。
7. `IntegratedInputResponse` は `AskResponse` と `WorkResponse` の主要フィールドを包含する。
8. `metadata` は全modelで `dict[str, object]` として保持する。

検証:
- dataclass初期値が空入力や未指定sourceで安全に動くこと。
- トップレベルに診断専用フィールドを置かないこと。

### Phase 3: 共通sanitizer追加
1. CLI/HTTPに重複している `_sanitize_payload_metadata`、`_compact_payload_text`、`_mask_payload_secret` を共通moduleへ移す。
2. 候補: `features/foundation/payload_sanitizer.py` または `usecases/integrated_input/sanitizer.py`。
3. 除外keyとして `contexts`、`context`、`llm_prompt`、`raw`、`secret` を定義する。
4. 画像metadataの `downloaded_image_path`、`original_image_ref` を除外する。
5. OCR本文、周辺本文、検索contextの長さ上限を定義する。
6. API key、token、password、secretらしき文字列をマスクする。
7. CLI/HTTP/統合入力受付で同じsanitizerを使う。

検証:
- secretがtext、metadata、nested itemからマスクされること。
- CLI/HTTP/Discordの出力が同じsanitizerを通ること。

### Phase 4: IntegratedInputRouter実装
1. `features/rag/components/integrated_input_routing.py` を追加する。
2. Gemini呼び出し、retry、JSON抽出、code fence除去を `IntegratedInputRouter` 内に実装する。
3. promptは `assets/prompts/integrated_input_routing.md` に保存する。
4. 出力schemaにroute、intent、required_features、source_filters、attribute_filters、risk、freshness_required、needs_clarification、clarification_question、reasonを定義する。
5. schema外の診断情報は `metadata` に移す。
6. invalid payload時はretry後にread-only fallback decisionを返す。
7. `EntryRoutingDecision` との互換adapterは作らず、新しい `IntegratedInputDecision` のみを返す。

検証:
- 全routeの正常parse。
- invalid routeがfallbackになること。
- fallback時に `risk=read_only` または `route=clarify` になること。
- raw出力がトップレベルに出ないこと。

### Phase 5: 決定的ルーティング規則
1. LLM分類結果の後処理として `IntegratedRoutingPolicy` を追加する。
2. `source=minecraft_wiki` は `minecraft_wiki_rag` を優先する。
3. `source=member` は `member_search` を優先する。
4. `source=image` は `image_search` を優先する。
5. `source=task` は `task_management` を優先する。
6. `source=event` は `event_management` を優先する。
7. サーバー管理語彙を検出した場合は `server_management` にする。
8. 明示sourceは初期featureとして保持し、本文から追加featureが検出されて `required_features` が2つ以上になった場合は `comprehensive_agent` にする。
9. 副作用語彙があり情報不足の場合は `clarify` にする。
10. 権限不足が入口で確定する場合は `deny` にする。

検証:
- 明示sourceがLLM分類より優先されること。
- 複合依頼が総合エージェントへ昇格すること。
- 分類失敗時に副作用routeへ進まないこと。
- `source=member` などの明示source付き複合依頼でも追加featureが検出されること。

### Phase 6: IntegratedInputUsecase追加
1. `usecases/integrated_input/entry.py` を追加する。
2. 依存としてretrieval ask、workflow service、agentic/comprehensive agent、router、sanitizerを受け取る。
3. 空入力処理を実装する。
4. requestを正規化する。
5. `AccessContext` を構築または受け取る。
6. routerを実行する。
7. routing policyを適用する。
8. routeに応じてhandlerへ委譲する。
9. handler responseを `IntegratedInputResponse` に正規化する。
10. sanitizerを適用する。
11. trace用metadataを付与する。
12. `history_scope` をRAG request、workflow metadata、総合エージェントmetadataへ渡す。

検証:
- 各routeが期待するservice mockへ1回だけ委譲されること。
- `AccessContext` が失われないこと。
- `history_scope` がroute先metadataに残ること。
- 例外時に利用者向けmessageとmetadata traceが返ること。

### Phase 7: route handler実装
1. `circle_rag` handlerで `RetrievalQuery` を呼ぶ。
2. `minecraft_wiki_rag` handlerで `source_filter="minecraft_wiki"` の `RetrievalQuery` を呼ぶ。
3. `member_search` handlerで `WorkRequest(work_type="member_search")` を呼ぶ。
4. `image_search` handlerで `WorkRequest(work_type="image_search")` を呼ぶ。
5. `task_management` handlerでintentから `task_extract`、`task_add`、`task_list`、`task_done`、`task_update`、`task_delete`、`task_notify_due`、`task_batch_approval` を選ぶ。
6. `event_management` handlerでintentから `event_extract`、`event_add`、`event_list`、`event_brief`、`event_update`、`event_delete`、`event_notify`、`event_batch_approval`、`event_complete`、`schedule_add`、`schedule_list` を選ぶ。
7. `server_management` handlerで状態確認は `mc_status`、操作依頼は `mc_request` を選ぶ。
8. `comprehensive_agent` handlerで分類結果、AccessContext、source_filters、attribute_filters、riskを渡す。
9. `clarify` と `deny` は外部serviceを呼ばずresponseを返す。
10. `task_done` は `task_update(status=done)` 候補、`event_complete` は `event_update(status=done)` 候補へ変換する。
11. `task_notify_due` / `event_notify` は通知送信やmetadata更新をせず `WorkflowCandidate` として返す。
12. task/event/schedule/server操作で必要情報が不足する場合はdispatch前に `clarify` を返す。

検証:
- source、mode、depthがRAGへ渡ること。
- work_type選択がintentと一致すること。
- server操作が直接executorへ行かないこと。
- 完了・通知系work_typeが正本更新serviceへ直接到達しないこと。

### Phase 8: 副作用境界の検査
1. `RiskLevel` ごとの許可操作を定義する。
2. `read_only` では候補作成work_typeも呼ばない。
3. `candidate_only` では候補作成まで許可する。
4. `approval_required` では承認待ち候補または操作候補のみ許可する。
5. `admin_only` では `AccessContext.is_admin` がfalseなら `deny` にする。
6. `WorkResponse` に正本変更済みの `tasks`、`events`、実行済みserver結果が含まれる場合の扱いを検査する。
7. 境界違反時は出力を停止しwarningを返す。
8. `direct_mutation` として `task_done`、`task_notify_due`、`event_notify`、`event_complete`、`server_operation_execute` を一覧化し、dispatch前に候補化またはclarifyへ変換する。

検証:
- read_only routeで `task_add` が呼ばれないこと。
- server操作依頼が `mc_request` の候補に留まること。
- admin以外が管理操作を依頼した場合にdenyになること。
- 実repositoryを使い、統合入力受付経由の完了・通知依頼で正本task/eventが変わらないこと。

### Phase 9: app context配線
1. `apps/integrated_input.py` を追加する。
2. `IntegratedInputAppContext` を定義する。
3. `build_integrated_input_app_context()` でfoundation、retrieval、workflow、agenticを組み立てる。
4. import循環が起きる場合は、依存をprotocolまたは薄いadapterで切る。
5. `runtime/container.py` の `RuntimeContext` へ必要に応じて統合入力受付を追加する。
6. `RuntimeContext` やapp contextから `ChatEntryUsecase` 参照を取り除く。

検証:
- app context生成で循環importしないこと。
- Discord bot起動時にcontext構築が重複しすぎないこと。

### Phase 10: CLI配線と旧分岐削除
1. CLI `ask` を `IntegratedInputUsecase` 経由に変更する。
2. CLI `ask` 内の `source=member`、`depth=deep`、通常RAGの個別分岐を削除する。
3. `--source`、`--mode`、`--depth`、`--user-id`、`--guild-id`、`--role-id`、`--admin` を `IntegratedInputRequest` へ渡す。
4. 出力は `IntegratedInputResponse` のpayload builderでJSON化する。
5. 安定フィールド `text`、`detail_markdown`、`citations`、`confidence`、`warnings` は `IntegratedInputResponse` から出力する。
6. 診断情報は `metadata` 配下へ入れる。

検証:
- CLI `ask` が `IntegratedInputUsecase` を1回だけ呼ぶこと。
- member/image/task/event/serverの主結果がトップレベルに出ること。
- `ChatEntryUsecase`、`EntryQueryRouter`、OpenClaw分類を参照しないこと。

### Phase 11: HTTP配線と旧分岐削除
1. HTTP `/ask` を `IntegratedInputUsecase` 経由に変更する。
2. payloadの `question` / `query`、`source`、`mode`、`depth`、権限情報をrequestへ渡す。
3. `user_id` が `maintenance_command_author_ids` に含まれる場合だけadminにする。payloadの `admin` / `is_admin` は信頼しない。
4. `_workflow_payload` とRAG payloadの重複を統合payload builderへ寄せる。
5. エラー時はHTTP 400/403/500の使い分けを整理する。
6. 主結果フィールド名は `IntegratedInputResponse` の安定schemaに揃える。

検証:
- sourceごとのrouteがHTTPでもCLIと一致すること。
- metadata sanitizerが適用されること。
- HTTP `/ask` 内に個別route分岐が残っていないこと。
- HTTP payloadの `is_admin=true` だけではadminにならないこと。

### Phase 12: Discord配線と旧分岐削除
1. Discord `/ask` を `IntegratedInputUsecase` 経由に変更する。
2. `_access_context(interaction)` を統合入力受付へ渡す。
3. route先のresponseを直接送信せず、`DiscordOutputAdapter` で送信する。
4. `text` が長い場合または `detail_markdown` が本文より長い場合はattachmentを付ける。
5. task/event/schedule/server/generic候補がある場合は、`approval_target_type` と `approval_target_id` に基づいて承認viewを必要に応じて添付する。
6. `/work` と `/approval` は明示操作として残すが、通常の依頼は `/ask` へ集約する。
7. 各ルーティング先serviceにDiscord送信責務を持たせない。

検証:
- Discord `/ask` が `IntegratedInputUsecase` を1回だけ呼ぶこと。
- attachment判定が `IntegratedInputResponse` に基づくこと。
- 候補作成時に承認ボタンが必要な場合だけ付くこと。
- Discord `/ask` 内に個別route分岐が残っていないこと。

### Phase 13: 総合エージェント連携
1. `required_features` が2つ以上の場合のrequest変換を実装する。
2. `docs/design/comprehensive-agent.md` の `ComprehensiveAgentService` を呼ぶ。
3. `ComprehensiveAgentService` が未実装の場合は、総合エージェントrouteをstub化せず、実装順を前倒しする。
4. 分類結果のrisk、source_filters、attribute_filtersを総合エージェントへ渡す。
5. 総合エージェントのrun idは `metadata.trace_id` または `metadata.agent_run_id` に入れる。
6. 候補や承認待ちがある場合はトップレベル主結果に正規化する。
7. `candidates`、task/event change候補、schedule候補、approvals、assets、member_profilesを `IntegratedInputResponse` と同じschemaで返す。

検証:
- 複合依頼が総合エージェントrouteとして識別されること。
- run idがトップレベルへ昇格しないこと。

### Phase 14: 監査・trace
1. request idを発行する。
2. frontend、route、intent、required_features、risk、handler、latency、fallback有無をmetadataへ入れる。
3. audit logに必要最小限のイベントを記録する。
4. user input本文を保存する場合はsecretマスクと長さ制限をかける。
5. route先run id、workflow run id、agent run idをmetadataへ入れる。

検証:
- traceにsecretと巨大contextが保存されないこと。
- fallbackやdenyの理由が追跡できること。

### Phase 15: ドキュメント更新
1. `docs/explanation/cli.md` に統合入力受付経由の `ask` 例を追加する。
2. Discord `/ask` のsource、mode、depthの意味をrunbookまたは説明docsに追記する。
3. payload schema方針として、診断情報は `metadata` 配下に入れることを明記する。
4. 総合エージェント計画と重複する箇所は相互参照にする。

検証:
- docs内のファイル名、route名、work_type名が実装と一致すること。

## 4. 推奨実装順
1. Phase 1で旧入口の削除範囲と責務移管先を確定する。
2. Phase 2からPhase 8で新しいmodel、router、usecase、route handler、安全境界を実装する。
3. Phase 9でapp contextを統合入力受付中心に組み替える。
4. Phase 10からPhase 12でCLI/HTTP/Discordの旧 `/ask` 分岐を削除し、薄いadapterへ置き換える。
5. Phase 13以降で総合エージェント、trace、docsを仕上げる。

この順序にすると、旧入口を残したまま新旧併存させる期間を作らず、`IntegratedInputUsecase` への一本化を前提に各frontendを置き換えられる。

## 5. テスト一覧
追加または更新するテスト候補は次の通り。

| ファイル | 内容 |
| --- | --- |
| `tests/unit/test_integrated_input_models.py` | request/decision/responseの初期値とpayload化 |
| `tests/unit/test_integrated_input_router.py` | LLM JSON parse、fallback、metadata格納 |
| `tests/unit/test_integrated_routing_policy.py` | source優先、複合依頼昇格、副作用fallback |
| `tests/unit/test_integrated_input_usecase.py` | routeごとのservice委譲、AccessContext伝播 |
| `tests/unit/test_integrated_input_side_effects.py` | fallback時の副作用遮断、direct mutation変換、正本不変性 |
| `tests/unit/test_integrated_input_sanitizer.py` | secret、context、画像pathの除外 |
| `tests/unit/test_cli_integrated_ask.py` | CLI askが統合入力受付だけを呼ぶこと |
| `tests/unit/test_http_integrated_ask.py` | HTTP /askが統合入力受付だけを呼び、admin allowlistだけを信頼すること |
| `tests/unit/test_discord_integrated_ask.py` | Discord送信adapter、attachment判定、汎用候補承認view |
| `tests/unit/test_removed_legacy_entrypoints.py` | 旧入口のimport参照が残っていないこと |

pytestは未導入のため、既存と同じunittest形式で追加する。

## 6. リスクと対策
| リスク | 対策 |
| --- | --- |
| 旧入口削除時に責務が抜け落ちる | Phase 1で削除対象と移管先を対応表にする |
| 分類LLMが誤って副作用routeを選ぶ | 決定的policyとrisk検査で副作用を候補作成に限定する |
| Discord送信責務の移行漏れ | route handlerはresponse返却のみ、送信はadapterに限定するテストを置く |
| app context循環 | 統合入力受付用contextを独立させ、必要ならprotocol adapterを使う |
| metadataからsecretが漏れる | 共通sanitizerを全出口とtrace保存前に適用する |
| 総合エージェント未完成でrouteが宙に浮く | 統合入力受付の総合エージェントroute実装前に `ComprehensiveAgentService` を用意する |
