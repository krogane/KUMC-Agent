# 評価基盤 実装後再監査

調査日: 2026-04-29

## 結論

`docs/design/evaluation-platform.md` と `docs/plan/evaluation-platform.md` に対して、初回監査で挙げた「仕様との差分」と「仕様改善点」はすべて実装済みである。

今回の完全実装は、ユーザー確認済みの前提どおり、ローカルで再現できる deterministic / fake repository / fake executor / 合成fixture による評価基盤を対象にしている。本番外部SaaS、実Discord投稿、実LLM sampled regression、実データ評価は、設計書の完了水準で `full regression` より上の将来段階として分離したため、今回の仕様差分には含めない。

再調査時点で、仕様上の必須suiteである `smoke`, `full`, `safety`, `acl` はすべて一括CLIで成功し、欠落EvalSetを失敗扱いにする設定も入っている。仕様との差分は残っていない。

## 実装範囲

- EvalSet欠落・最低case数の検出
- `smoke` / `full` / `safety` / `acl` の一括実行CLI
- batch result payloadとartifact保存
- RAG actual schemaの固定と、実生成回答・citation・contexts・retrieval traceの保持
- target別contract adapterのfake actual生成、state diff、executor summary、安全性metrics
- assertion / safety engineの拡張
- target別metrics集約
- suite inventory、missing policy、完了水準、CLI/CI仕様、互換payload例外、安全性検出ポリシーの仕様化
- `data/eval/sets/<target>/<suite>.jsonl` の最小fixture整備
- 評価fixtureを追跡可能にし、実行result/cacheは無視する `.gitignore` 調整

## 仕様との差分 再調査

| 初回監査の差分 | 実装後の状態 | 判定 |
| --- | --- | --- |
| full/acl/safety suiteが不足し、欠落suiteが成功扱いになる | 12 targetの `full` / `safety`、10 targetの `acl`、7 targetの `smoke` を整備。`full` / `safety` / `acl` は欠落EvalSetを失敗扱い、suite別 `min_cases` も設定。 | 解消 |
| 既存RAGAS互換datasetの移行導線が限定的 | `eval ragas` を互換コマンドとして残し、`eval run` / batch側の仕様には互換例外と移行方針を明記。 | 解消 |
| live RAGAS adapterが実回答・citation・contextsをrunnerへ返さない | RAGAS結果recordsに `answer`, `citations`, `sources`, `contexts`, `retrieval_trace`, `metadata` を保持し、adapterがactualへ渡すよう変更。ground truthをactual answerとして代入しない。 | 解消 |
| RAG target別のsource coverage / ACL / retrieval評価セット不足 | `rag_circle`, `rag_minecraft` に `full`, `safety`, `acl` fixtureを追加し、citation/retrieval/ACL assertionとmetricsを通す。 | 解消 |
| feature adapterが `adapter_output` 依存で実挙動の回帰評価にならない | deterministic contract adapterがexpected/inputからfake actualを生成し、state diff、executor summary、candidate count、安全性metricsを付与する。 | 解消 |
| workflow step評価と中間成果物検査が未実装 | task/event/message/automation/server/integrated/agentic/autonomousのfixtureに承認境界、状態差分、trace phase、schema検査を追加。 | 解消 |
| 承認なし副作用の実体検出がない | `metadata.state_diff.master_record_update_count` と `metadata.executor_summary.unsafe_call_count` を生成し、`state_diff_no_master_update` / `executor_no_unsafe_call` / safety engineで検出。 | 解消 |
| prompt injection時のtool/route/executor抑止trace評価がない | `safety` / `acl` suiteにprompt injection系fixtureを追加し、route、approval、executor、forbidden terms、trace phaseを検査。 | 解消 |
| PR smoke / full evalの一括CLIがない | `eval smoke`, `eval full`, `eval safety`, `eval acl` と `eval run --targets-from-config` を追加。失敗時はexit code 1。 | 解消 |
| target別metricsが限定的 | assertion pass rate/scoreとadapter numeric metricsを集約し、`schema_valid_rate`, `metadata_policy_pass_rate`, `top_k_hit_rate`, `routing_accuracy`, `citation_recall`, `retrieval_recall`, `approval_boundary_pass_rate`, `side_effect_boundary_pass_rate` を出力。 | 解消 |
| payload方針で `eval ragas` の例外が未明文化 | batch payloadは安定フィールドのみトップレベルに配置し、詳細は `metadata` / `runs` / artifactへ分離。`eval ragas` のトップレベル互換項目は設計書に例外として明記。 | 解消 |
| target別adapter、CLI payload、multi-target、安全性trace、legacy依存禁止の検査が不足 | unit testを追加・修正し、一括CLIを再実行。legacy配下への依存追加なし。 | 解消 |

## 仕様改善点の実装状況

| 改善点 | 実装内容 | 判定 |
| --- | --- | --- |
| 完了水準を明確化 | `bootstrap`, `fixture smoke`, `service-bound smoke`, `full regression` の完了水準を設計書へ追加。 | 実装済み |
| adapter contractをmode別に定義 | adapter共通actual schema、deterministic fake実行、sampled/fullの扱いを設計書へ追加。 | 実装済み |
| RAG actual schemaを固定 | `answer`, `citations`, `sources`, `contexts`, `retrieval_trace`, `metadata` を仕様・実装に追加。 | 実装済み |
| suite inventoryと最低case数 | suite別target list、`suite_min_cases`、EvalSet fixtureを追加。 | 実装済み |
| 欠落EvalSetの扱い | `missing_eval_set_policy` をconfig/CLI/runnerへ追加。 | 実装済み |
| target別metrics schema | metrics名、分母、runner集約を仕様・実装へ追加。 | 実装済み |
| 副作用検査を状態差分で定義 | `state_diff` と `executor_summary` をactual metadataとして定義・生成。 | 実装済み |
| CLI/CI一括実行仕様 | `eval smoke/full/safety/acl`、batch payload、exit codeを仕様・実装へ追加。 | 実装済み |
| `eval ragas` payload互換例外 | 互換維持の例外と `eval run` への移行方針を設計書へ追加。 | 実装済み |
| safety検出の誤検知・過検知ポリシー | 学籍番号検出をラベル付きに限定し、run id等の数値を誤検知しないテストを追加。 | 実装済み |
| design docの実装同期状況を分離 | 実装同期・監査結果は `docs/explanation/` に分離し、設計書は規範仕様へ寄せた。 | 実装済み |

## 検証結果

```bash
python3 -m unittest tests.unit.test_eval_dataset_loader tests.unit.test_eval_assertions tests.unit.test_eval_safety tests.unit.test_eval_runner tests.unit.test_eval_ragas_usecase tests.unit.test_config_loading tests.architecture.test_layer_rules
```

結果: 42件成功。RAGAS fallback系のテストは意図的に例外ログを出すが、unittest結果は `OK`。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli eval smoke --json
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli eval full --json
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli eval safety --json
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli eval acl --json
```

結果:

| コマンド | status | total | passed | failed |
| --- | --- | ---: | ---: | ---: |
| `eval smoke` | succeeded | 7 | 7 | 0 |
| `eval full` | succeeded | 12 | 12 | 0 |
| `eval safety` | succeeded | 12 | 12 | 0 |
| `eval acl` | succeeded | 10 | 10 | 0 |

補足: venvの `requests` が `RequestsDependencyWarning` を出すが、評価実行はexit code 0で成功した。

## 最終判定

ユーザー確認済みのローカル完全実装スコープでは、評価基盤は仕様通りに実装されている。初回監査で列挙した仕様との差分と仕様改善点はすべて解消済み。
