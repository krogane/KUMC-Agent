# 統合入力受付 実装後再調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/integrated-input.md`
- `docs/plan/integrated-input.md`

調査対象:

- `src/kumc_agent/domain/models/integrated_input.py`
- `src/kumc_agent/features/rag/components/integrated_input_routing.py`
- `src/kumc_agent/usecases/integrated_input/entry.py`
- `src/kumc_agent/apps/integrated_input.py`
- `src/kumc_agent/frontends/http/app.py`
- `src/kumc_agent/frontends/discord/app.py`
- `src/kumc_agent/features/agentic/comprehensive.py`
- CLI / HTTP / Discord の `/ask` 相当入口
- `tests/unit/test_integrated_input.py`

`src/kumc_agent/infra/legacy` は実装対象外方針に従い、調査対象から除外した。

## 結論

前回調査で確認した「仕様との差分」と「仕様改善点」は実装済みである。現時点の統合入力受付は、主要入口を `IntegratedInputUsecase` に集約し、分類fallback、副作用境界、AccessContext、候補出力、history_scope伝播、総合エージェント出力schema、payload sanitizationについて、参照仕様を満たしている。

ただし、明示的な `/work` や承認componentの直接操作は、設計通り「通常入力」ではなく明示操作として残している。統合入力受付経由では、正本更新系work_typeをdispatch前に候補化またはclarifyへ変換する。

## 実装確認

| 仕様項目 | 実装後の状態 | 主な実装箇所 |
| --- | --- | --- |
| 統合入力domain model | route、intent、required_features、risk、metadata、主結果、汎用candidateを保持 | `domain/models/integrated_input.py` |
| 統合router | Gemini分類、JSON parse、code fence除去、決定的policy、safe fallbackを実装 | `features/rag/components/integrated_input_routing.py` |
| 分類失敗時fallback | LLM未設定・失敗時に副作用語彙を検出した場合は `clarify`。read-onlyだけfallback routeへ進む | `IntegratedInputRouter._safe_fallback_decision()` |
| 明示sourceと複合依頼 | 明示sourceを尊重しつつ、本文が複数機能を要求する場合は `comprehensive_agent` へ昇格 | `IntegratedRoutingPolicy.apply()` |
| 副作用境界 | `read_only`、candidate系、direct mutation系work_typeをdispatch前に判定。統合入力経由で正本更新work_typeを直接呼ばない | `IntegratedInputUsecase._preflight_workflow_route()` |
| task/event完了 | `task_done` は `task_update(status=done)` 候補、`event_complete` は `event_update(status=done)` 候補へ変換 | `usecases/integrated_input/entry.py` |
| 通知系 | `task_notify_due` / `event_notify` は通知送信やmetadata更新をせず `WorkflowCandidate` を返す | `usecases/integrated_input/entry.py` |
| 不足情報clarify | task/event変更対象、event/schedule日時、server操作内容をdispatch前に確認 | `usecases/integrated_input/entry.py` |
| HTTP admin解決 | HTTP payloadの `admin` / `is_admin` を直接信頼せず、`maintenance_command_author_ids` の `user_id` 一致だけadminにする | `frontends/http/app.py` |
| CLI admin解決 | ローカルCLIの `--admin` は明示指定として信頼 | `cli.py` |
| Discord admin解決 | Discord user idとguild allow listからadmin判定 | `frontends/discord/app.py` |
| Discord出力adapter | task/event/schedule/server/generic candidateの承認viewを統合出力で扱う | `frontends/discord/app.py` |
| history_scope | RAG、workflow metadata、comprehensive agent metadata、最終metadataへ伝播 | `usecases/integrated_input/entry.py` |
| comprehensive agent schema | candidates、task/event change、schedule、approvals、assets、member_profilesを統合出力へ返せる | `domain/models/agentic.py`, `features/agentic/comprehensive.py` |
| sanitizer | secret、raw、context、画像local pathなどを外部payloadから除外またはマスク | `features/foundation/payload_sanitizer.py` |
| 旧entrypoint削除 | `ChatEntryUsecase`、`EntryQueryRouter`、`EntryRoutingDecision`、`direct_rag` の現行src参照なし | `tests/unit/test_integrated_input.py` |
| OpenClaw通常route廃止 | app / feature / usecase / frontend / runtimeの通常route参照なし | `rg` 確認 |

## 仕様との差分 再調査

| 前回差分 | 再調査結果 |
| --- | --- |
| 分類器が使えない場合のfallbackが副作用routeを返し得る | 解消。`_safe_fallback_decision()` が副作用語彙を含むfallbackを `clarify` に固定し、副作用候補を作らない |
| task/eventの一部work_typeが統合入力受付から直接呼ばれると正本を更新する | 解消。統合入力受付のdispatch前preflightで `task_done` / `event_complete` を変更候補作成work_typeへ変換し、通知系は `WorkflowCandidate` に留める |
| `is_admin` の解決が統合入力受付で一貫していない | 解消。HTTPはallowlistのみ、CLIはローカル明示指定、Discordは設定ベースの判定に整理 |
| Discord出力adapterがtask候補中心 | 解消。event / schedule / server_operation / generic candidateも承認view対象になった |
| 副作用に必要な不足情報を事前clarifyしきれていない | 解消。target id、event title/date、schedule date、server operationをdispatch前に検査 |
| `history_scope` がroute先に渡らない | 解消。RAG `ChatRequest`、workflow `WorkRequest.metadata`、comprehensive metadata、最終metadataへ伝播 |
| comprehensive agentの候補payloadが狭い | 解消。統合出力schemaと同じ候補カテゴリを返せる |
| 明示source優先と複数機能昇格の優先順位が曖昧 | 解消。明示sourceを初期featureにしつつ、本文から検出した追加featureがあれば総合エージェントへ昇格する |

## 仕様改善点 実装状況

| 改善点 | 実装状況 |
| --- | --- |
| 副作用境界をwork_type単位で明文化 | `_READ_WORK_TYPES`、`_CANDIDATE_WORK_TYPES`、`_DIRECT_MUTATION_WORK_TYPES` とpreflightで実装 |
| fallback定義をread-only / clarifyに分離 | `_safe_fallback_decision()` とテストで実装 |
| AccessResolverのtrust boundary明確化 | HTTP allowlist、CLI trusted local flag、Discord config-based判定に整理 |
| 候補出力と承認UI契約 | `candidates` / `workflow_candidates` に `approval_target_type` / `approval_target_id` を持たせ、Discord adapterが汎用candidateを扱う |
| 不足情報clarificationの事前ルール | `_missing_required_fields()` と `_clarification_question()` で実装 |
| comprehensive agent出力schema統一 | `ComprehensiveAgentResponse` とresponse builderが統合出力の候補カテゴリを返す |
| 明示sourceと複合依頼昇格の優先順位 | `IntegratedRoutingPolicy.apply()` に実装し、`source=member` 複合依頼の昇格テストを追加 |
| 正本不変性のテスト | 実際の `FileWorkflowRepository` と `WorkflowService` を使い、完了・通知依頼で正本task/eventが変わらないことを検証 |
| `history_scope` 方針具体化 | route先metadataへ伝播する方針で実装 |

## 検証

実行した検証:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_integrated_input
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_stubs tests.unit.test_discord_commands
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_agentic_docgen_announcement tests.unit.test_workflow_service
PYTHONPATH=src app/.venv/bin/python -m unittest discover tests/unit
rg -n "ChatEntryUsecase|ChatEntryRequest|EntryQueryRouter|EntryRoutingDecision|EntryRoute|entry_routing|direct_rag" src/kumc_agent tests -g '!src/kumc_agent/infra/legacy/**'
rg -n "OpenClawClient|infra.openclaw|openclaw" src/kumc_agent/apps src/kumc_agent/features src/kumc_agent/usecases src/kumc_agent/frontends src/kumc_agent/runtime -g '!src/kumc_agent/infra/legacy/**'
```

結果:

- `tests.unit.test_integrated_input`: 23 tests / OK
- `tests.unit.test_stubs` + `tests.unit.test_discord_commands`: 4 tests / OK
- `tests.unit.test_agentic_docgen_announcement` + `tests.unit.test_workflow_service`: 23 tests / OK
- full unit discovery: 278 tests / OK
- 旧統合入口名の現行src参照: なし。テスト内の削除確認のみ検出
- OpenClawのapp / feature / usecase / frontend / runtime参照: なし

full unit discovery中にHugging Face名前解決失敗のretry logが出たが、該当テストはfallback込みで完了し、最終結果はOKだった。
