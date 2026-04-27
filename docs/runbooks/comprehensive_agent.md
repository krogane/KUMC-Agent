# Comprehensive Agent Runbook

## 確認対象

- CLI / HTTP / Discord の `depth=deep` は総合エージェントへ接続する。
- `agent_runs` と `agent_steps` には `PLAN` / `TOOL` / `VERIFY` / `ANSWER` が保存される。
- 診断情報は外部payloadの `metadata` 配下に置く。

## insufficient_evidence

1. HTTP `GET /agent/runs/{run_id}` で `VERIFY` step の `missing` を確認する。
2. citation不足の場合はindex更新状況とsource filterを確認する。
3. queryが曖昧な場合は利用者へ追加情報を依頼する。

## needs_approval

1. 最終回答の承認待ち候補IDを確認する。
2. `/approval --action show` またはHTTP `/approval` で候補内容を確認する。
3. 承認前にTask / Event正本やserver executor結果が増えていないことを確認する。

## tool failure

1. `TOOL` step の `tool_name` と `status` を確認する。
2. workflow toolの場合は `/work` 単体で同じ `work_type` を実行して切り分ける。
3. secretや巨大contextはtraceに保存しない。必要な場合はsource id、candidate id、短い要約で追跡する。
