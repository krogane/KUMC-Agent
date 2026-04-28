# 統合入力受付 実装調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/integrated-input.md`
- `docs/plan/integrated-input.md`

調査対象:

- `src/kumc_agent/domain/models/integrated_input.py`
- `src/kumc_agent/features/rag/components/integrated_input_routing.py`
- `src/kumc_agent/usecases/integrated_input/entry.py`
- `src/kumc_agent/apps/integrated_input.py`
- CLI / HTTP / Discord の `/ask` 相当入口
- `tests/unit/test_integrated_input.py`

`src/kumc_agent/infra/legacy` は実装対象外方針に従い、調査対象から除外した。

## 結論

統合入力受付は、初期実装ではなく主要な構成要素まで実装済みである。`IntegratedInputRequest` / `IntegratedInputDecision` / `IntegratedInputResponse`、`IntegratedInputRouter`、`IntegratedRoutingPolicy`、`IntegratedInputUsecase`、CLI / HTTP / Discord `/ask` への配線、共通sanitizer、総合エージェント連携は存在する。

ただし、現時点では「仕様通りの完全実装」とは判断できない。特に、分類fallback時の副作用route、task/event系の直接正本更新、HTTP/CLIのadmin信頼、Discord出力adapterの候補種別対応に差分がある。安全境界に関わるため、単なる仕様の細部差分ではなく公開前に直すべき未達である。

## 実装済みの主な要素

| 仕様項目 | 状態 | 主な実装箇所 |
| --- | --- | --- |
| 統合入力domain model | 実装済み。route、intent、required_features、risk、metadata、主結果フィールドを保持 | `domain/models/integrated_input.py` |
| 統合router | 実装済み。Gemini分類、JSON parse、code fence除去、heuristic fallback、routing policyを持つ | `features/rag/components/integrated_input_routing.py` |
| 統合usecase | 実装済み。正規化、trace id、router実行、route dispatch、sanitizer、payload化を行う | `usecases/integrated_input/entry.py` |
| route先 | RAG、Minecraft Wiki RAG、member、image、task、event、server、comprehensive_agentへ委譲 | `usecases/integrated_input/entry.py` |
| app context | 実装済み。retrieval / workflow / agentic / routerを組み立てる | `apps/integrated_input.py` |
| CLI `ask` | `IntegratedInputUsecase` 経由 | `cli.py` |
| HTTP `/ask` | `IntegratedInputUsecase` 経由 | `frontends/http/app.py` |
| Discord `/ask` | `IntegratedInputUsecase` 経由。送信は `_send_integrated_response` に集約 | `frontends/discord/app.py` |
| 共通sanitizer | `contexts`、`context`、`llm_prompt`、`raw`、secret、画像local pathなどを除外またはマスク | `features/foundation/payload_sanitizer.py` |
| 旧entrypoint削除 | `ChatEntryUsecase`、`EntryQueryRouter`、`EntryRoutingDecision`、`direct_rag` 参照は現行srcから削除済み | `tests/unit/test_integrated_input.py` |

## 仕様との差分

| 優先度 | 差分 | 影響 | 根拠 |
| --- | --- | --- | --- |
| Critical | 分類器が使えない場合のfallbackが副作用routeを返し得る | 仕様では「分類失敗時はread-only fallbackまたは確認質問。副作用候補は作らない」だが、Gemini未設定や失敗時にheuristicが `task_management` / `event_management` / `server_management` と `candidate_only` / `approval_required` を返し、候補作成へ進む可能性がある | `IntegratedInputRouter.decide()` は provider不在時に `_heuristic_decision()` を返す。`_heuristic_decision()` はタスク語彙などから副作用routeを選ぶ |
| Critical | task/eventの一部work_typeが統合入力受付から直接呼ばれると正本を更新する | 仕様では副作用依頼を候補作成または承認待ちに限定する必要があるが、`task_done`、`task_notify_due`、`event_notify`、`event_complete` はworkflow内で `save_task()` / `save_event()` を実行する。統合入力受付の `_guard_workflow_response()` は呼び出し後に検出するため、正本更新を防げない | `IntegratedInputUsecase._run_workflow_route()` はworkflow実行後にguardする。`WorkflowService.task_done()`、`task_notify_due()`、`event_notify()`、`event_complete()` は正本保存を行う |
| High | `is_admin` の解決が統合入力受付で一貫していない | 仕様では入力値をそのまま信頼せずresolverで決定する必要がある。Discordは設定から判定するが、HTTPはpayloadの `admin` / `is_admin` をそのまま信頼し、CLIも `--admin` をそのまま渡す | `frontends/http/app.py` の `_access()` は `is_admin=bool(payload.get("admin") or payload.get("is_admin"))`。`cli.py ask` は `is_admin=bool(args.admin)` |
| High | Discord出力adapterがtask候補中心で、event / schedule / server候補の承認UIを統合出力できない | 仕様では候補や承認待ちをroute先から受け取り、Discord最終出力で提示する必要がある。現実装は `_first_candidate_id()` が task candidate / task change candidate のみを見て、task approval viewだけを付ける | `frontends/discord/app.py` の `_first_candidate_id()` と `_send_integrated_response()` |
| High | 副作用に必要な不足情報を統合入力受付で事前clarifyしきれていない | 仕様では不足情報がある場合、候補作成や承認申請を行わず確認質問を返す。現実装は多くをworkflowへ委譲し、`task_done` / `task_update` / `task_delete` などはtarget不足時に例外となり、統合入力受付は一般エラーへ変換する | `WorkflowService.task_done()` などはtarget不足で `ValueError` を投げる。`IntegratedInputUsecase.execute()` はroute handler例外を一般失敗messageにする |
| Medium | `history_scope` が正規化後のroute先に渡らない | 設計上の入力項目だが、RAG / workflow / comprehensive_agentへ明示的に伝播していない。Discordのguild/channel/thread相当の履歴範囲を使う実装にはなっていない | `IntegratedInputRequest.history_scope` はあるが、`RetrievalQuery` 生成や `WorkRequest` 生成に使われていない |
| Medium | comprehensive agentの候補payloadが統合出力schemaより狭い | 統合出力は task change / event change / schedule / approvals も持つが、`ComprehensiveAgentResponse` は task_candidates / event_candidates / server_operations中心で、change候補やschedule候補を型として返せない | `domain/models/agentic.py` の `ComprehensiveAgentResponse` |
| Medium | 明示source優先と複数機能昇格の優先順位が仕様上曖昧 | 設計には「source=member等は優先」と「required_featuresが2つ以上なら総合エージェントへ昇格」が併存する。現実装は一部で明示sourceがrequired_featuresを単一に上書きするため、複合依頼でも昇格しないケースがあり得る | `IntegratedRoutingPolicy.apply()` は `source == member/image/task/event` で required を単一化した後、`len(required) >= 2` を判定する |

## 仕様改善点

1. 副作用境界を `work_type` 単位で明文化する。`read_only`、`candidate_only`、`approval_required`、`direct_mutation` を一覧化し、統合入力受付がdispatch前に禁止できるようにする。特に `task_done`、`task_notify_due`、`event_notify`、`event_complete` は、候補作成用work_typeへ分離するか、dry-run/candidateモードを明示する。
2. 分類fallbackの定義を分ける。LLM分類失敗時は仕様通りread-onlyまたはclarifyに固定し、決定的heuristicを「通常分類器」として使う場合はconfidence、required fields、side-effect可否の条件を別途定義する。
3. `AccessResolver` の責務とtrust boundaryを明文化する。Discord、HTTP、CLIで `user_id`、`guild_id`、`role_ids`、`is_admin` をどう信頼・検証するかを分け、HTTP payloadの `is_admin` を直接admin化しない方針を仕様に入れる。
4. 候補出力と承認UIの契約を定義する。candidate種別ごとに `approval_target_type`、`approval_target_id`、表示本文、ボタンaction、batch扱いを統一し、Discord adapterがtask/event/schedule/serverを同じ仕組みで扱えるようにする。
5. 不足情報clarificationの事前ルールを追加する。task更新・完了・削除にはtarget id、event追加にはtitle/date、server操作には対象server/actionなど、dispatch前に確認すべき最小項目を仕様化する。
6. comprehensive agentの出力schemaを `IntegratedInputResponse` と揃える。task_change、event_change、schedule_candidates、approvals、workflow_candidatesを落とさず返せるようにする。
7. 明示sourceと複合依頼昇格の優先順位を決める。例: `source=member` でも「担当候補を探してタスク候補を作る」は総合エージェントへ昇格するのか、member_searchだけに限定するのかを仕様で固定する。
8. 完全実装判定用テストに「repositoryが変更されないこと」を入れる。副作用境界はpayloadだけでは検証できないため、候補作成前後、完了・通知・削除依頼前後の正本件数・status・metadataの不変性をテスト条件にする。
9. `history_scope` の利用方針を具体化する。入力として受けるだけなら説明を弱め、履歴検索に使うなら `RetrievalQuery` または別contextへ伝播する契約を追加する。

## 推奨修正順

1. `IntegratedInputUsecase` のdispatch前に `work_type` 副作用表を適用し、直接正本更新work_typeを統合入力受付から呼べないようにする。
2. `task_done` / `task_notify_due` / `event_notify` / `event_complete` を候補作成経路へ分離するか、統合入力受付からはclarifyまたはdenyにする。
3. LLM分類失敗時のheuristic fallbackをread-only/clarifyに制限する。副作用語彙を検出したら確認質問にする。
4. HTTP/CLI向けの `AccessResolver` を追加し、少なくともHTTPではpayloadの `is_admin` を直接信頼しない。
5. Discordの統合出力adapterをcandidate種別共通にし、event/schedule/server候補も承認導線を出せるようにする。
6. comprehensive agentのresponse schemaを統合出力schemaへ拡張する。
7. 上記を `tests/unit/test_integrated_input.py` または分割テストに追加する。

## 検証

実行した検証:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_integrated_input
rg -n "ChatEntryUsecase|ChatEntryRequest|EntryQueryRouter|EntryRoutingDecision|EntryRoute|entry_routing|direct_rag" src/kumc_agent tests -g '!src/kumc_agent/infra/legacy/**'
rg -n "OpenClawClient|infra.openclaw|openclaw" src/kumc_agent/apps src/kumc_agent/features src/kumc_agent/usecases src/kumc_agent/frontends src/kumc_agent/runtime -g '!src/kumc_agent/infra/legacy/**'
```

結果:

- `tests.unit.test_integrated_input`: 7 tests / OK
- 旧統合入口名の現行src参照: なし。テスト内の削除確認のみ検出
- OpenClawの現行app / feature / usecase / frontend / runtime参照: なし。`infra.openclaw` とconfig/testは残るが、統合入力受付の通常routeには接続されていない

既存テストは通るが、副作用境界については「payloadで止まること」ではなく「正本が変更されないこと」の検証が不足している。
