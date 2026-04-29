# 評価基盤 実装監査

調査日: 2026-04-29

更新: 実装後の再監査結果は `docs/explanation/evaluation-platform-implementation-reaudit.md` に記録した。このファイルは初回監査時点の差分記録として残す。

## 結論

`docs/design/evaluation-platform.md` と `docs/plan/evaluation-platform.md` が求める「完全実装」には到達していない。現状は、設計書自身の「実装同期状況」にもある通り、汎用評価基盤の初期実装である。

実装済みなのは、EvalSet schema/loader、汎用runner、基本assertion、安全性スキャン、result artifact/EvalRun保存、`eval run` CLI、合成smoke fixtureである。一方、完全実装の中核である target別の実サービス/fixture repository結合adapter、full/acl/safety評価セット、RAGの実引用・検索trace評価、PR smoke/full evalの一括実行、CI/定期実行、業務workflow評価は未実装または骨組みに留まる。

## 調査対象

- 仕様: `docs/design/evaluation-platform.md`
- 実装計画: `docs/plan/evaluation-platform.md`
- 主な実装: `src/kumc_agent/usecases/eval/`
- CLI: `src/kumc_agent/cli.py`
- 設定: `configs/main/evaluation.yaml`
- 評価データ: `data/eval/sets/`
- テスト: `tests/unit/test_eval_*`

## 実装済みの範囲

| 領域 | 状況 |
| --- | --- |
| EvalCase schema | `EvalCase`, `EvalAssertion`, `EvalCaseResult`, `EvalRunResult` は実装済み。 |
| EvalSet loader | 新EvalSet JSONLとRAGAS互換 `question`/`query` + `ground_truth(s)` の変換に対応。 |
| 汎用runner | target/suite指定、limit、cancel、case latency、集約metrics、artifact保存、`EvalRun` 保存に対応。 |
| assertion | 回答包含、禁止語句、citation/source recall、retrieval recall、ACL、top-k、route、承認境界、副作用境界、metadata policyを実装。 |
| safety | secret、ACL flag、副作用、任意shell、metadata policyのゼロ許容判定を実装。 |
| adapter registry | 仕様対象targetは registry に登録済み。 |
| CLI | `eval run` と既存互換の `eval ragas` が存在する。 |
| 設定 | eval set/result path、smoke/full target、閾値、安全性、fixture/fake executor設定が追加済み。 |
| 評価データ | 各targetに合成 `smoke.jsonl` が1件ずつ、`agentic/safety.jsonl` が1件存在する。 |

## 仕様との差分

| 仕様・計画の要求 | 実装状況 | 差分 |
| --- | --- | --- |
| `data/eval/sets/<target>/<suite>.jsonl` のEvalSetを読み込む | 読み込み可能。存在しないEvalSetはrunnerが空集合として成功扱いにする。 | full/acl/safety suiteがほぼ無く、欠落suiteの成功扱いはCIでは危険。 |
| 既存 `data/eval/ragas.jsonl` 互換 | loaderと `eval ragas` は対応。 | `eval run` で既存RAGAS互換datasetを使うには明示pathとtarget/suiteが必要。移行導線は限定的。 |
| RAGAS metrics + 引用精度 + 検索recall + 権限違反 | assertion型はある。 | live RAGAS adapterは生成回答・citation・contextsをrunnerへ返さず、`exact_match > 0` の場合にground truthをanswerとして代入する。実RAGの引用精度・検索recall・ACLを完全には評価できない。 |
| サークルRAGとMinecraft Wiki RAGの分離評価 | target登録とsmoke fixtureはある。 | source coverage、multi-source synthesis、material search、history、recency、fast mode、ACL matrix、edition/version差分などの評価セットが不足。 |
| メンバー検索、画像検索、workflow系、サーバー、統合入力、agentic/autonomousの独自評価 | target登録と合成fixtureはある。 | adapterは `input.adapter_output` を読むcontract fixtureで、実サービス、fake repository、fake LLM、fake executorを起動していない。仕様が求める機能挙動の回帰評価ではない。 |
| 業務workflow評価 | 未確認。 | task/event lifecycle、image_to_task、member_assignment、server_operationのstep評価と中間成果物検査が未実装。 |
| 承認なし副作用のゼロ許容 | output上の `side_effects` やstatusは検査する。 | repositoryのbefore/after snapshot、外部executor呼び出し、Task/Event正本変更を実体として検出する仕組みはない。 |
| prompt injection安全性 | 入力中のmarker検出と危険output検出はある。 | prompt injectionを含むcaseで、どのtool/route/executorが抑止されたかを検査する実行trace評価はない。 |
| PR smoke / full eval | 設定にtarget listはある。 | 複数targetをまとめて実行するCLI/CIコマンド、full suite、定期実行、artifact保持方針は未実装。 |
| 成功率、失敗ケース、コスト、レイテンシ | runnerは pass rate、severity、latency、estimated_cost集計を持つ。 | target別metricsは限定的。`schema_valid_rate`, `metadata_policy_pass_rate`, `top_k_hit_rate`, `field_f1` など仕様記載metricsの多くは未集計。 |
| CLI/external payloadのmetadata方針 | `eval run` payloadは概ね準拠。 | `eval ragas` は互換維持のため `ragas_metrics` / `ragas_metadata` をトップレベル出力する。新payload方針の対象外として明文化が必要。 |
| unittestによる検証 | 評価系unit testは追加されている。 | target別adapter、CLI payload、multi-target smoke/full、repository snapshot、安全性trace、legacy依存禁止の検査が不足。現状の評価系unit testは1件失敗する。 |

## Phase別判定

| Phase | 判定 | メモ |
| --- | --- | --- |
| Phase 1: EvalSet schema/loader | ほぼ実装済み | 危険metadataはsanitizeするが、拒否/警告ポリシーは粗い。 |
| Phase 2: 汎用runner | 部分実装 | 基本runnerはあるが、欠落EvalSet成功扱いとtarget別metrics不足がある。 |
| Phase 3: RAGAS adapter統合 | 部分実装 | 既存RAGASは利用するが、runner側のactualが実回答・citation・contextsを保持しない。 |
| Phase 4: RAG独自assertion | 部分実装 | 型はあるが、実RAG traceとの接続が不足。 |
| Phase 5: 評価セット雛形 | 部分実装 | smokeのみ最小1件。full/acl/safetyが不足。 |
| Phase 6: Workflow系adapter | 未完了 | contract fixtureのみ。実workflow/fake repository未接続。 |
| Phase 7: Search系adapter | 未完了 | contract fixtureのみ。MemberProfile/Asset fixture repository未接続。 |
| Phase 8: Server/Integrated/Agentic adapter | 未完了 | fake executorやagent trace runnerの実体評価はない。 |
| Phase 9: 安全性engine | 部分実装 | output pattern検査中心。副作用境界の実体検証が不足。 |
| Phase 10: CLI整備 | 部分実装 | `eval run` はあるが、一括smoke/full、exit code、`--json`の挙動整理が不足。 |
| Phase 11: 設定追加 | 部分実装 | 基本項目はあるが、target/suite別閾値の実運用値は未整備。 |
| Phase 12: CI/定期実行 | 未完了 | コマンド定義、CI integration、artifact保持が不足。 |
| Phase 13: ドキュメント更新 | 部分実装 | 使い方メモはあるが、完全実装との差分や運用手順は不足していた。 |

## 検証結果

実行した確認:

```bash
python3 -m unittest tests.unit.test_eval_dataset_loader tests.unit.test_eval_assertions tests.unit.test_eval_safety tests.unit.test_eval_runner tests.unit.test_eval_ragas_usecase
```

結果: 29件中1件失敗。失敗は `tests.unit.test_eval_runner.EvalRunnerTests.test_runner_saves_artifact_and_eval_run` で、`TemporaryDirectory` を抜けた後にartifactの存在確認をしているため、テスト側の寿命管理に問題がある。

```bash
PYTHONPATH=src /Users/tatsuya.s/Documents/Documents/Programming/ChatBot/KUMC-Agent/app/.venv/bin/python -m kumc_agent.cli eval run --target task_management --suite smoke --json
```

結果: 成功。`task_management:smoke` は1件成功し、`data/eval/results/<run_id>.json` にartifactが保存された。

補足: system Pythonで `PYTHONPATH=src python3 -m kumc_agent.cli ...` を実行すると、`discord` 未導入でCLI import時に失敗した。ローカル検証は `app/.venv/bin/python` を使う必要がある。

## 仕様の改善点

1. 完了水準を明確化する

現在の仕様は「初期実装」と「完全実装」の境界が読み手によって揺れる。`bootstrap`, `fixture smoke`, `service-bound smoke`, `full regression` のように成熟度を分け、各段階の必須target、adapter、dataset、CI条件を明文化するべき。

2. adapter contractをmode別に定義する

`deterministic` では `adapter_output` を読むだけでよいのか、fake repositoryを起動する必要があるのかが曖昧。`sampled` と `full` では実サービスをどこまで呼び、どこからfake/dry-runに固定するかをtarget別に定義する必要がある。

3. RAG評価のactual schemaを固定する

RAG adapterは `answer`, `citations`, `sources`, `contexts`, `retrieval_trace`, `access_context`, `ragas_metrics`, `ragas_metadata` をどう返すかを仕様化するべき。特に、ground truthをactual answerに代入してはいけないこと、citation/retrieval assertionが実RAG出力に接続されることを明記する。

4. suite inventoryと最低case数を定義する

各targetに `smoke`, `full`, `acl`, `safety` のどれが必須か、各suiteに最低何ケース必要かを表で定義する。現状のようにfull targetが設定されていてもfull datasetがない状態を検出できる。

5. 欠落EvalSetの扱いを決める

runnerは存在しないEvalSetを空集合成功として扱う。CIやfull evalでは欠落は失敗にする、ローカル探索ではskipにする、などmode別のルールが必要。

6. target別metrics schemaを仕様化する

設計書には多数のmetrics名があるが、実装では `case_pass_rate` 中心である。`top_k_hit_rate`, `field_f1`, `schema_valid_rate`, `metadata_policy_pass_rate`, `routing_accuracy`, `approval_boundary_pass_rate` などの計算式、分母、threshold keyを定義するべき。

7. 副作用検査をartifactだけでなく状態差分で定義する

`side_effects` 自己申告だけでは不十分。Task/Event/Server/Message/Automationは、fake repositoryやexecutor spyのbefore/after、実行回数、idempotency key、承認状態を共通形式で記録し、assertionがその差分を読む仕様にする。

8. CLI/CIの一括実行仕様を追加する

`eval run --target ...` だけでなく、`eval smoke`, `eval full`, `eval safety` または `eval run --targets-from-config smoke` のような一括実行仕様、exit code、JSON/log分離、artifact directory、CI summaryを定義する。

9. `eval ragas` のpayload互換例外を明文化する

新payload方針では診断情報は `metadata` 配下だが、既存互換の `eval ragas` は `ragas_metadata` をトップレベルに出す。この互換を許す期間、非推奨化方針、`eval run` への移行例を仕様に書くべき。

10. safety検出の誤検知・過検知ポリシーを定義する

メール、電話番号、学籍番号らしき数値はsecret扱いだが、fixtureや公開データで許容される場合があり得る。許可list、hash fixture、expected leak fixture、検出severityの扱いを仕様化する必要がある。

11. design docの「実装同期状況」を分離する

詳細設計に実装同期状況を置くと、規範仕様と現状メモが混ざる。`docs/explanation/` または実装監査docへ移し、設計書は達成すべき仕様を中心にした方が差分管理しやすい。

## 次に必要な実装作業

完全実装へ進める場合の優先度は次の順が妥当。

1. 欠落EvalSetをmode別に失敗/skipへ変更し、`smoke_targets` / `full_targets` の一括実行を追加する。
2. RAG adapterを実回答・citation・contexts・retrieval traceを保持する形に修正する。
3. task/event/server/message/automationのfake repository/executor adapterを実装し、自己申告ではなく状態差分で副作用境界を検査する。
4. member/image search adapterを実サービスまたはfake indexに接続し、top-k/nDCG/evidence/ACL metricsを出す。
5. agentic/autonomous adapterでPLAN/TOOL/VERIFY traceと承認境界を評価する。
6. full/acl/safety suiteを追加し、target別metricsとthresholdを設定する。
7. CI用smoke/fullコマンドとartifact保持ルールを文書化・実装する。
