# 自律エージェント runbook

## 設定

- 起動時刻、対象scope、dry-run、予算は `configs/main/autonomous_agent.yaml` で管理する。
- `enabled=false` の場合、Automation/schedule 経由のrunは `blocked` として記録される。手動CLI runは検証用途として実行できる。
- Planner / Verifier は `autonomous_agent.planner` / `autonomous_agent.verifier` で個別に `provider: gemini | openai` とモデルを指定する。APIキーは既存の `integrations.gemini_api_key` / `integrations.openai_api_key` を使う。

## 手動実行

```bash
kumc-agent autonomous --dry-run --slot manual --scope tasks --scope events
```

JSON payload の `metadata.run_id` で `AgentRun` を追跡する。`metadata.idempotency_key` が同じrunは二重実行されない。

## worker実行

```bash
kumc-agent worker --job-type autonomous_agent_run --payload-json '{"trigger":"worker","slot":"08:00","scopes":["tasks","events","rag_delta"],"dry_run":true}'
```

Automation default rule は `autonomous_agent.schedule_times` から生成され、action `autonomous_agent_run` として同じworker jobを呼ぶ。

## status対応

- `succeeded`: 通知候補のみ、または確認が正常終了した。
- `noop`: 対応対象なし。
- `needs_approval`: 通知候補、承認申請候補、Task/Event候補など承認フローへ渡す結果がある。
- `insufficient_evidence`: citation不足、再検索予算切れ、根拠不足。
- `blocked`: 設定無効または安全制約で実行不可。
- `duplicate`: 同じ `idempotency_key` のrunが既に記録済み。
- `failed`: 例外。`metadata.error_type` と audit log を確認する。

## 副作用境界

自律エージェントは承認前に外部投稿、サーバー操作、Task/Event正本更新を実行しない。TOOL結果には `metadata.side_effects`、`master_write_count`、`external_delivery_count`、`server_execute_count` が入り、禁止値は VERIFY で `noop` に落とす。
