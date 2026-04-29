# 自律エージェント 実装再調査結果

調査日: 2026-04-29

参照仕様:

- `docs/design/autonomous-agent.md`
- `docs/plan/autonomous-agent.md`

調査対象:

- `src/kumc_agent/domain/models/autonomous_agent.py`
- `src/kumc_agent/features/autonomous_agent/*`
- `src/kumc_agent/apps/autonomous_agent.py`
- `src/kumc_agent/apps/automation.py`
- `src/kumc_agent/apps/worker/app.py`
- `src/kumc_agent/cli.py`
- `configs/main/autonomous_agent.yaml`
- `assets/prompts/autonomous_agent_planner.md`
- `assets/prompts/autonomous_agent_verifier.md`
- `tests/unit/test_autonomous_agent.py`

`src/kumc_agent/infra/legacy` は、依存が混入していないことの検索確認だけを行い、実装対象外とした。

## 結論

自律エージェントは、今回の実装後に仕様上の主要要件を満たす状態になった。

実装済みになった範囲は、1日にn回のAutomation default rule生成、worker/CLI/manual共通service実行、RAG差分collector、統合入力受付の標準配線、Workflow/Automation/Retrieval/Server限定adapter、専用LLM Planner / Verifier、再検索/再計画ループ、承認・通知候補の履歴保存、構造化副作用検証、system actor権限設定、dry-run三値化、idempotency開始時予約、worker payload metadata方針、runbook、unit test拡張である。

現時点の仕様との差分は、外部副作用を承認後に実行するUI/通知送信コンポーネントが本機能の対象外である点だけで、これは仕様上も対象外として定義されている。自律エージェント単体としては、承認前に外部投稿、サーバー操作、Task/Event正本更新を行わず、候補・承認申請・ログに限定する仕様を満たしている。

## 実装済みの主な範囲

| 項目 | 実装状況 | 主な実装箇所 |
| --- | --- | --- |
| 定期起動 | `schedule_times` から `autonomous_agent_{HHMM}` のAutomation default ruleを生成し、`autonomous_agent_run` worker jobへ接続 | `src/kumc_agent/apps/automation.py`, `src/kumc_agent/features/automation/service.py` |
| enabled制御 | Automation/schedule trigger時に `enabled=false` なら `blocked` として履歴・監査に記録 | `src/kumc_agent/features/autonomous_agent/service.py` |
| RAG差分 | ingestion active chunksのmetadata時刻を `rag_delta_lookback_hours` で抽出し、summaryとcitationだけをsnapshot化 | `src/kumc_agent/features/autonomous_agent/snapshot.py` |
| 統合入力受付 | app contextで `IntegratedInputUsecase` を標準経路に配線し、Workflowはfallbackとして保持 | `src/kumc_agent/apps/autonomous_agent.py` |
| 限定adapter | Workflow / Automation dry-run / Retrieval ask / Server pending queryを副作用なしで正規化 | `src/kumc_agent/features/autonomous_agent/integrated_input.py` |
| 専用LLM Planner | Gemini/OpenAIを個別設定可能。JSON schema検証、risk/boundary正規化、決定的planとのmerge、fallbackを実装 | `src/kumc_agent/features/autonomous_agent/planner.py`, `src/kumc_agent/features/autonomous_agent/llm.py` |
| 専用LLM Verifier | Gemini/OpenAIを個別設定可能。LLM判断を決定的Verifierの副作用・根拠guardとmerge | `src/kumc_agent/features/autonomous_agent/verifier.py` |
| OpenAI接続 | 既存 `integrations.openai_api_key` を使う軽量LLM adapterを追加 | `src/kumc_agent/infra/llm/openai.py` |
| dry-run三値 | `AutonomousAgentRequest.dry_run` を `bool | None` にし、`None` はconfigに従う | `src/kumc_agent/domain/models/autonomous_agent.py`, `src/kumc_agent/features/autonomous_agent/service.py` |
| system actor | `system_user_id`、`guild_id`、`role_ids`、`is_admin` をconfig化し、workerのadmin defaultを廃止 | `configs/main/autonomous_agent.yaml`, `src/kumc_agent/apps/worker/app.py` |
| 再検索/再計画 | `max_replans`、`max_steps`、`max_search_calls`、`max_latency_seconds` を使うループを実装 | `src/kumc_agent/features/autonomous_agent/service.py` |
| 履歴とidempotency | run開始時に `AutomationRun(status="running")` を予約し、最終statusで更新。Postgres upsertも更新型に変更 | `src/kumc_agent/features/autonomous_agent/service.py`, `src/kumc_agent/infra/automation/repository.py` |
| 通知・承認候補永続化 | `AgentRun.metadata` と `AutomationRun.metadata` に通知候補、承認申請、候補参照を保存 | `src/kumc_agent/features/autonomous_agent/service.py` |
| 構造化副作用契約 | `side_effects`、`master_write_count`、`external_delivery_count`、`server_execute_count` をadapter結果に付与し、Verifierで禁止値を拒否 | `src/kumc_agent/features/autonomous_agent/integrated_input.py`, `src/kumc_agent/features/autonomous_agent/verifier.py` |
| payload方針 | worker/autonomous payloadの副作用・診断情報を `metadata` 配下に保持 | `src/kumc_agent/apps/worker/app.py`, `src/kumc_agent/domain/models/autonomous_agent.py` |
| runbook | CLI、worker、status対応、副作用境界を記述 | `docs/runbooks/autonomous-agent.md` |
| テスト | RAG差分、LLM guard、構造化副作用拒否、dry-run三値、schedule ruleを追加検証 | `tests/unit/test_autonomous_agent.py` |

## 仕様との差分

| 重要度 | 旧差分 | 再調査結果 |
| --- | --- | --- |
| High | 1日にn回の自動起動が未実装 | 解消。`schedule_times` からAutomation default ruleを生成し、worker jobへ接続した。 |
| High | RAG差分collectorが未実装 | 解消。ingestion active chunksを正データソースにし、直近差分sourceをcitation付きsnapshotへ入れる。 |
| High | 統合入力受付が標準経路になっていない | 解消。`build_autonomous_agent_app_context()` で統合入力受付を標準経路にした。 |
| High | 専用LLM Planner / Verifierが未実装 | 解消。PlannerとVerifierはそれぞれ独自にGemini/OpenAI providerとモデルを設定できる。APIキーは既存設定を使う。 |
| High | 再検索/再計画ループがない | 解消。VERIFYが `retry_search` の場合、budget内で再PLAN/TOOL/VERIFYを行う。 |
| High | 承認申請・候補の永続化契約が弱い | 解消。初期永続化先を `AgentRun.metadata` と `AutomationRun.metadata` に固定した。 |
| High | 安全性検証が仕様より浅い | 解消。構造化副作用契約、secret-like payload、内部IP、招待URL、個人連絡先をVerifierで検出する。 |
| Medium | Automation/Retrieval/Server adapterが未実装 | 解消。限定adapterを追加した。 |
| Medium | duplicate抑制が完全ではない | 解消。run開始時予約、Postgres upsert更新、`duplicate_suppression_hours` による通知対象抑制を実装した。 |
| Medium | AuditEventの粒度が不足 | 解消。duplicate、blocked、proposal、verifyの判断情報、search/replan/elapsedを保存する。 |
| Medium | output schemaが仕様と一部不一致 | 解消。`proposals`、`task_candidates`、`event_candidates`、`automation_runs`、`server_operations` を外部payloadの安定フィールドに追加し、run objectは外部payloadではmetadata参照に限定した。 |
| Medium | worker payload方針違反 | 解消。副作用情報は `metadata.side_effects` に保持する。 |
| Medium | `blocked` statusが実質使われない | 解消。設定無効時に `blocked` を返す経路を実装した。 |
| Low | `task_management.due_soon_notice_days` との関係が曖昧 | 解消。自律エージェントは独立設定 `autonomous_agent.lookahead_days.tasks` を優先する仕様に明記した。 |
| Low | docs/runbooks等の更新が未完了 | 解消。`docs/runbooks/autonomous-agent.md` を追加した。 |

## 仕様改善点の反映状況

| No. | 改善点 | 反映状況 |
| --- | --- | --- |
| 1 | 「初期実装」と「完全実装」の境界を明示する | ユーザー指示により今回はパス。 |
| 2 | scheduler仕様を具体化する | 自動インデックス更新に寄せ、Automation default rule + worker job起動を正とした。 |
| 3 | RAG差分のデータソースを定義する | ingestion repository active chunksを正とし、metadata時刻によるlookback抽出に決定した。 |
| 4 | dry-runの意味を三値で定義する | `None=config`、`true=強制dry-run`、`false=候補保存許可` に変更した。 |
| 5 | system actorの権限設定を追加する | `autonomous_agent.access` を追加し、worker admin defaultを廃止した。 |
| 6 | 承認申請と通知候補の保存先を確定する | 初期保存先を `AgentRun.metadata` と `AutomationRun.metadata` に固定した。 |
| 7 | 外部payloadと内部response objectを分ける | 外部payloadでは内部run objectを出さず、`metadata.run_id` / `metadata.trace_id` を返す方針にした。 |
| 8 | 専用LLM Planner / Verifierを完全実装要件として定義する | config、prompt、provider別LLM、schema validation、決定的guard、fallbackを実装し、設計にも明記した。 |
| 9 | 副作用検証をmarkerではなく構造化契約にする | adapter resultに構造化side effect fieldsを必須化し、Verifierで禁止値を拒否する。 |
| 10 | 再検索/再計画の予算仕様を具体化する | `max_steps`、`max_search_calls`、`max_replans`、`max_latency_seconds`、elapsed/search/replan/cost記録を実装した。 |
| 11 | duplicate/idempotencyの競合耐性を仕様化する | run開始時予約とPostgres upsert更新を実装した。 |
| 12 | 評価・テストマトリクスを完了条件に直結させる | unit testにschedule、RAG、LLM、side effect、dry-run三値を追加し、設計にも追記した。 |

## 検証

実行した検証:

```bash
python3 -m compileall -q src/kumc_agent/features/autonomous_agent src/kumc_agent/apps/autonomous_agent.py src/kumc_agent/apps/automation.py src/kumc_agent/apps/worker/app.py src/kumc_agent/features/automation/service.py src/kumc_agent/infra/automation/repository.py src/kumc_agent/domain/models/autonomous_agent.py src/kumc_agent/infra/llm/openai.py tests/unit/test_autonomous_agent.py
python3 -m unittest tests.unit.test_autonomous_agent tests.unit.test_config_loading
KUMC_DISCORD_BOT_TOKEN=x KUMC_GEMINI_API_KEY=x KUMC_DRIVE_FOLDER_ID=x python3 -m unittest tests.unit.test_automation_hardening
PYTHONPATH=src KUMC_DISCORD_BOT_TOKEN=x KUMC_GEMINI_API_KEY=x KUMC_DRIVE_FOLDER_ID=x app/.venv/bin/python -m unittest tests.unit.test_autonomous_agent tests.unit.test_config_loading tests.unit.test_automation_hardening
PYTHONPATH=src KUMC_DISCORD_BOT_TOKEN=x KUMC_GEMINI_API_KEY=x KUMC_DRIVE_FOLDER_ID=x app/.venv/bin/python -m unittest discover -s tests/unit
PYTHONPATH=src KUMC_DISCORD_BOT_TOKEN=x KUMC_GEMINI_API_KEY=x KUMC_DRIVE_FOLDER_ID=x app/.venv/bin/python - <<'PY'
from kumc_agent.apps.autonomous_agent import build_autonomous_agent_app_context
app = build_autonomous_agent_app_context()
print(type(app.autonomous_agent).__name__)
print(app.autonomous_agent.adapter.integrated_input is not None)
PY
rg "legacy" src/kumc_agent/features/autonomous_agent src/kumc_agent/apps/autonomous_agent.py src/kumc_agent/domain/models/autonomous_agent.py tests/unit/test_autonomous_agent.py
```

結果:

- compileall: OK。
- `python3 -m unittest tests.unit.test_autonomous_agent tests.unit.test_config_loading`: 14 tests / OK。
- `python3 -m unittest tests.unit.test_automation_hardening` は必須envを一時指定して 7 tests / OK。
- `.venv` での関連test: 21 tests / OK。
- `.venv` でのunit全体: 283 tests / OK。途中でHugging Face名前解決失敗のretryログなどが出たが、該当テストはfallback経路で完了。
- app context smoke test: `AutonomousAgentService` を構築し、統合入力受付が配線済みであることを確認。
- legacy import検索: 該当なし。
