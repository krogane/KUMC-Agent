# 総合エージェント実装再調査

作成日: 2026-04-28

## 結論

`docs/design/comprehensive-agent.md` と `docs/plan/comprehensive-agent.md` を基準に再調査した結果、前回調査で挙げた「仕様との差分」と「仕様改善点」は実装済みである。

非legacyの `src` / `tests` では旧 `AgenticSearchService` / `AgenticSearchRequest` / `AgenticSearchResponse` および `SEARCH` / `READ` state の残存は検出されない。総合エージェントは、専用設定、専用LLM Planner / Verifier、候補作成後の承認record / batch自動作成、安定payload、非admin候補作成拒否、単一RAG + `depth=deep` の昇格、Discord承認導線、監視集計を備えた状態になっている。

外部LLMのライブ呼び出しは実行環境のAPI keyに依存するため、単体テストではFake LLMで専用Planner / Verifierの接続を検証した。

## 調査対象

- `docs/design/comprehensive-agent.md`
- `docs/plan/comprehensive-agent.md`
- `docs/design/kumc-agent.md` の「10. 総合エージェント」
- `docs/kumc-agent-redesign-v4.md`
- `src/kumc_agent/domain/models/agentic.py`
- `src/kumc_agent/features/agentic/comprehensive.py`
- `src/kumc_agent/features/agentic/tools.py`
- `src/kumc_agent/apps/agentic.py`
- `src/kumc_agent/usecases/integrated_input/entry.py`
- `src/kumc_agent/features/rag/components/integrated_input_routing.py`
- `src/kumc_agent/cli.py`
- `src/kumc_agent/frontends/http/app.py`
- `src/kumc_agent/frontends/discord/app.py`
- `src/kumc_agent/features/hardening/readiness.py`
- `tests/unit/test_agentic_docgen_announcement.py`
- `tests/unit/test_integrated_input.py`
- `tests/unit/test_config_loading.py`
- `tests/unit/test_automation_hardening.py`

`src/kumc_agent/infra/legacy` は設計指示通り調査対象から除外した。

## 検証コマンド

```bash
rg -n "AgenticSearch|Agentic Search|AgenticSearchRequest|AgenticSearchResponse|state=.*SEARCH|state=.*READ|\"SEARCH\"|\"READ\"" src tests --glob '!src/kumc_agent/infra/legacy/**'
rg -n "Agentic Search|AgentSearchUsecase|SEARCH / READ|search/read/verify|AgenticSearch|agentic_search|requires_agentic_search" docs/kumc-agent-redesign-v4.md
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_workflow_service tests.unit.test_server_management tests.unit.test_agentic_docgen_announcement tests.unit.test_integrated_input tests.unit.test_config_loading tests.unit.test_automation_hardening tests.architecture.test_layer_rules
PYTHONPATH=src app/.venv/bin/python -m unittest discover tests/unit
```

結果:

- 旧AgenticSearch関連コードと旧stateの残存なし。
- `docs/kumc-agent-redesign-v4.md` の旧Agentic Search説明は総合エージェント表現へ更新済み。
- 重点 + hardening + architecture テスト 62 tests / OK。
- unit test discover 266 tests / OK。

## 実装済み内容

### 専用設定

`configs/main/comprehensive_agent.yaml` を追加し、`RuntimeConfig.comprehensive_agent` として読み込む。Planner、Verifier、budget は総合エージェント専用設定で管理し、`configs/main/*.yaml` の分離方針に合わせた。

### 専用LLM Planner / Verifier

`ComprehensiveAgentPlanner` と `ComprehensiveAgentVerifier` は専用LLM、専用prompt、専用configを受け取る。LLMが利用できる場合はJSON構造化出力を優先し、失敗時は決定的フォールバックに戻る。

追加prompt:

- `assets/prompts/comprehensive_agent_planner.md`
- `assets/prompts/comprehensive_agent_verifier.md`

### 状態機械と再計画

状態は `PLAN` / `TOOL` / `VERIFY` / `ANSWER` に統一されている。Verifierが根拠不足や矛盾を返した場合は `max_replans` 内で再計画し、再計画時は不足内容をqueryへ反映して同一実行の繰り返しを避ける。

単一機能で `depth=deep` ではない場合はdirect routeを返し、統合入力受付側の直接ルーティングと揃える。単一RAGでも `depth=deep` の場合は `comprehensive_agent` へ昇格する。

### Toolと承認境界

標準toolは以下を維持する。

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

`task_candidate_create` / `event_candidate_create` / `server_operation_candidate_create` の直後に、候補が返った場合は approval record と task/event approval batch を自動作成する。`approval_candidate_create` も明示的にadapter実装済みである。

非adminが候補作成を含む総合エージェント依頼を行った場合は、候補作成せず拒否する。

### Response payload

`ComprehensiveAgentResponse` から `run` は削除済み。run参照は `metadata.agent_run_id` とtrace APIに分離する。

top-level主結果として以下を返す。

- `candidates`
- `task_candidates`
- `task_change_candidates`
- `event_candidates`
- `event_change_candidates`
- `schedule_candidates`
- `server_operations`
- `approvals`
- `assets`
- `member_profiles`

CLI / HTTP / Discord / 統合入力受付のpayloadも同じ候補・approval情報を落とさず渡す。

### Discord承認導線

統合入力応答から、task / task_change、event / event_change、schedule、server_operation の候補に承認UIを付与する。task/eventは既存承認viewを使い、schedule/server_operationは汎用approval viewで approve / reject / show を扱う。

### Sanitizationと検証

Verifierは以下を決定的に検査する。

- 必須toolの成功結果
- citation不足
- forbidden metadata key
- secret / 内部IP / PIN の未マスク混入
- write candidateの実行済みstatus
- `execution_allowed` の混入
- 候補作成toolが正本 `tasks` / `events` / `schedules` を返していないこと
- 候補に対応するapproval record / batchの存在
- 簡易的な候補フィールド矛盾

### 監視集計

readiness cost report は `agent_runs` / `agent_steps` から以下を集計する。

- run数、status count、status rate
- 成功率
- `insufficient_evidence` 率
- `needs_approval` 率
- state count
- tool別成功数、失敗数、成功率、失敗率
- 平均step数
- 平均latency
- 推定cost
- replan回数、平均replan回数

## 仕様との差分

| 仕様項目 | 再調査結果 | 判定 |
| --- | --- | --- |
| AgenticSearch関連コード削除 | 非legacyの `src` / `tests` に残存なし | 完了 |
| 状態機械 | `PLAN` / `TOOL` / `VERIFY` / `ANSWER` でtrace保存 | 完了 |
| PLANで入力分解、必要機能、tool順序、検索条件、成功条件、副作用境界を決定 | 専用LLM Planner + fallbackで実装。tool入力と成功条件を構造化 | 完了 |
| 入力が曖昧な場合に質問を返す | PlannerのclarificationとWorkflow側validationで扱う | 完了 |
| 単一機能は直接ルーティング | 通常単一機能はdirect route。単一RAG + `depth=deep` は総合エージェントへ昇格 | 完了 |
| RAG / Minecraft Wiki / メンバー / 画像 / タスク / イベント / サーバー候補tool | registryとadapterで実装 | 完了 |
| `approval_candidate_create` | adapter実装済み。候補作成後の自動approval作成にも対応 | 完了 |
| 副作用は候補作成または承認申請まで | write toolは候補・承認record / batchまで。正本変更は拒否検査対象 | 完了 |
| 承認待ち候補IDと承認対象をresponseへ含める | `candidates` / `approvals` / batch metadataで返す | 完了 |
| VERIFYで根拠不足、矛盾、権限外情報、副作用境界違反を検出 | 専用LLM Verifier + 決定的検査で実装 | 完了 |
| 根拠不足・矛盾時の再計画 | 不足内容をqueryへ反映して `max_replans` 内で再計画 | 完了 |
| 最終回答に結論、根拠、使用機能、未確認事項、承認待ち候補 | AnswerBuilderで出力 | 完了 |
| `ComprehensiveAgentResponse` のトップレベルは安定主結果のみ | `run` をdataclassから削除し、traceはmetadata参照へ分離 | 完了 |
| task/event change candidate、schedule candidate、approvalsの扱い | top-level主結果に追加し、統合入力payloadにも伝播 | 完了 |
| CLI/HTTP/Discord payloadの診断情報をmetadata配下へ | 主結果とmetadataを分離済み | 完了 |
| 大きなcontext、secret、権限外情報を外部payloadやtraceに出さない | sanitizerとVerifier検査で対応 | 完了 |
| Discordで承認候補を扱う | task/event/change/schedule/server候補の承認導線を実装 | 完了 |
| 評価・監視連携 | readiness cost reportへ仕様の監視指標を追加 | 完了 |
| テスト | Planner/Verifier、承認自動作成、`depth=deep` 昇格、非admin拒否、監視集計を追加検証 | 完了 |

## 仕様改善点

| 改善点 | 実装内容 | 判定 |
| --- | --- | --- |
| 1. 承認モデルを再定義する | 候補作成tool直後にapproval record / batchを自動作成する仕様へ統一 | 完了 |
| 2. 専用LLM Planner schemaをtool別に具体化する | Planner promptとJSON schema validationを追加し、tool入力を構造化 | 完了 |
| 3. 専用LLM Verifierを仕様として詳細化する | 決定的検査結果とtool結果をVerifierへ渡し、構造化 `VerificationResult` に統合 | 完了 |
| 4. Response payloadを統一する | `run` 削除、`candidates` / `approvals` / change / schedule候補を統一 | 完了 |
| 5. `depth=deep` の扱いを整理する | 単一RAG + `depth=deep` は総合エージェントへ昇格。複数機能昇格とも同じrouteへ統一 | 完了 |
| 6. 権限方針を統合する | 非adminの候補作成依頼は候補作成せず拒否 | 完了 |
| 7. テスト計画を機能別に分割する | 既存unitへ総合エージェント固有テストを追加し、フルunit discoverも通過 | 完了 |

## 残存リスク

- 専用LLM Planner / Verifier のライブ品質は、実運用のGemini API keyとモデル応答に依存する。単体テストではFake LLMで接続契約を検証している。
- 承認UIはDiscord component単位の単体検証であり、実Discord上の表示確認は別途stagingで行う必要がある。
