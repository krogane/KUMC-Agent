# 評価基盤 実装計画

## 1. 方針
`docs/design/kumc-agent.md` と `docs/design/evaluation-platform.md` に従い、評価基盤を実装する。

実装では `src/kumc_agent/infra/legacy` を参照・依存しない。既存の共通部品は `usecases.eval.ragas.EvaluateRagasUsecase`、`domain.models.operations.EvalRun`、`infra.operations.repository`、`runtime.container`、`cli.py`、`configs/ops/app.yaml` を優先して使う。現行実装と設計が矛盾する場合は `kumc-agent.md` を優先する。

現行の `eval ragas` は互換維持しつつ、評価セット、機能別scorer、安全性scorer、結果保存、PR/full evalを扱う共通評価基盤へ拡張する。

## 2. 完了条件
- 評価セットmanifestを読み込み、対象機能、実行モード、scorer、閾値を解決できる。
- RAGAS評価は現行の `EvaluateRagasUsecase` 互換を維持できる。
- サークル情報RAGとMinecraft Wiki RAGについて、回答正確性、引用精度、検索recall、権限違反、RAGAS指標を評価できる。
- メンバー検索、画像検索、タスク管理、イベント管理、メッセージ投稿、オートメーション、サーバー管理、統合入力受付を機能別評価できる。
- prompt injection、権限違反、秘密情報引用、危険操作、承認なし副作用を安全性評価できる。
- 重大な漏洩や危険操作はfail-fastでrun失敗にできる。
- PR小規模評価とfull evalを区別して実行できる。
- 評価結果に成功率、失敗ケース、コスト、レイテンシを含められる。
- `EvalRun` にsummaryを保存し、詳細artifactを `data/eval/runs/` に保存できる。
- CLIや外部連携payloadの診断情報が `metadata` 配下に入る。
- 主要動作を既存テスト方式で検証できる。

## 3. 実装ステップ
### Phase 1: 評価モデル追加
1. `src/kumc_agent/domain/models/evaluation.py` を追加する。
2. `EvalSet`、`EvalCase`、`EvalObservation`、`EvalCaseResult`、`EvalRunSummary` を定義する。
3. `EvalCase` は現行RAGAS互換の `question` / `query`、`ground_truth` / `ground_truths` を読み込めるようにする。
4. `metadata` 方針をモデル・payload helperのテストで固定する。
5. `domain.models.operations.EvalRun` はrun summary保存用として維持する。

検証:
- RAGAS互換JSONLを新モデルへ正規化できること。
- 診断情報がトップレベルへ出ないこと。
- 既存 `EvalRun` 保存payloadを壊さないこと。

### Phase 2: EvalSet loader
1. `configs/eval/` を追加する。
2. `configs/eval/rag-circle-smoke.yaml`、`configs/eval/minecraft-wiki-smoke.yaml`、`configs/eval/safety-smoke.yaml` を最小構成で追加する。
3. `data/eval/` 配下のcase fileをmanifestから参照する。
4. `EvalSetLoader` を追加し、manifestとJSONL/YAML caseを読み込む。
5. 不正schema、存在しないcase file、空case fileを明示的にエラーにする。

検証:
- manifestから評価ケースを読み込めること。
- 不正manifestが分かりやすく失敗すること。
- smoke/pr/full/manualのmode filterが効くこと。

### Phase 3: 共通EvalRunner
1. `src/kumc_agent/usecases/eval/runner.py` を追加する。
2. `EvaluateRequest` と `EvaluateResult` を定義する。
3. EvalSet、limit、mode、result path、cache設定、cancel eventを受け取る。
4. target adapterを呼び、caseごとに `EvalObservation` を作る。
5. scorerを実行し、case resultを集計する。
6. gate policyでrun statusを決定する。

検証:
- 複数caseを実行してsummaryを返せること。
- limit指定が効くこと。
- cancel eventで途中停止できること。

### Phase 4: RAGAS adapter統合
1. 現行 `EvaluateRagasUsecase` をRAGAS専用adapterとして残す。
2. `EvalRunner` からRAGAS評価セットを実行できるadapterを追加する。
3. `append_sources_to_response=False` と評価用履歴scopeの挙動を維持する。
4. RAGAS metric toggle、batch size、max workers、timeout、retryを既存ops設定から引き継ぐ。
5. RAGAS依存がない場合の扱いをmode別に分ける。

検証:
- 既存 `tests/unit/test_eval_ragas_usecase.py` が通ること。
- `eval ragas` CLI互換が維持されること。
- RAGAS skippedがPR/fullで期待通りに扱われること。

### Phase 5: RAG専用scorer
1. `exact_match` と `token_overlap` を共通scorerへ移す。
2. `retrieval_recall` scorerを追加する。
3. `citation_precision` scorerを追加する。
4. `permission_leak` scorerを追加する。
5. サークル情報RAGとMinecraft Wiki RAGのcase schemaに `expected_sources`、`expected_chunks`、`forbidden_sources` を追加する。

検証:
- 期待source/chunkがcontextに入るとpassすること。
- 権限外sourceが観測結果に含まれるとfailすること。
- Minecraft Wikiのedition/versionタグを評価できること。

### Phase 6: 機能別target adapter
1. `member_search` adapterを追加する。
2. `image_search` adapterを追加する。
3. `workflow` adapterを追加し、タスク管理・イベント管理・メッセージ投稿を評価できるようにする。
4. `automation` adapterを追加する。
5. `server_management` adapterを追加する。
6. `entry_router` adapterを追加する。
7. `agentic` adapterを追加し、内部で使う機能単位の観測結果を保存する。

検証:
- 各adapterが副作用をdry-runまたは承認待ち候補に閉じ込めること。
- 観測結果にlatencyとmetadataが入ること。
- 未実装adapterは明示的にskippedまたはunsupportedになること。

### Phase 7: 機能別scorer
1. メンバー検索用に権限、個人情報抑制、非断定表現、候補理由、根拠scorerを追加する。
2. 画像検索用に画像候補、OCR、類似画像、人物確認、権利確認を断定しないscorerを追加する。
3. タスク抽出用に担当、期限、状態、重複検出、承認前正本未登録scorerを追加する。
4. イベント管理用に日時、場所、状態、変更差分、承認フローscorerを追加する。
5. 統合入力受付用にintent、selected handler、payload schema scorerを追加する。

検証:
- 期待候補が返るとpassすること。
- 断定禁止表現や承認前副作用を検出できること。
- routeやselected handlerが `metadata` 配下にあること。

### Phase 8: 安全性評価
1. `prompt_injection` scorerを追加する。
2. `secret_leak` scorerを追加する。
3. `permission_violation` scorerを追加する。
4. `dangerous_action` scorerを追加する。
5. `unapproved_side_effect` scorerを追加する。
6. blocker/high severityの失敗をfail-fastできるようにする。

検証:
- prompt injectionケースで内部指示を漏らすと失敗すること。
- API key、token、招待URL、内部URLらしき値を検出できること。
- 承認なしの作成・変更・削除が失敗扱いになること。

### Phase 9: キャッシュと再現性
1. 評価回答キャッシュkeyを拡張する。
2. eval set id、case id、target、input hash、access context hash、index fingerprint、config profile、model id、prompt versionをkeyへ含める。
3. 現行 `ragas_answers.jsonl` は互換読み込みする。
4. PR小規模評価ではキャッシュ利用を許可し、full evalではrefreshを推奨する。
5. cache hit/missをrun metadataへ保存する。

検証:
- access contextが違うケースでcacheが共有されないこと。
- index/configが変わるとcache missになること。
- 既存cache fileを読み込めること。

### Phase 10: 結果保存
1. `EvalRun` を `infra.operations.repository` 経由で保存する。
2. `data/eval/runs/{run_id}/summary.json` を出力する。
3. `data/eval/runs/{run_id}/cases.jsonl` を出力する。
4. 失敗ケースだけを `failures.jsonl` に出力する。
5. raw prompt、検索context、secretを含む可能性がある値はartifact内部でもマスクまたは別管理にする。

検証:
- File repositoryでEvalRunを保存できること。
- Postgres repositoryでEvalRunを保存できること。
- CLI標準出力に大きなcontextやsecretが出ないこと。

### Phase 11: CLI拡張
1. 既存 `eval ragas` を維持する。
2. `eval run --set ... --mode ...` を追加する。
3. `--limit`、`--result-path`、`--refresh-cache`、`--fail-fast`、`--format json` を追加する。
4. CLI出力は `run_id`、`eval_set_id`、`status`、`total`、`passed`、`failed`、`metrics`、`artifact_dir`、`metadata` に揃える。
5. 既存 `eval ragas` のpayloadも将来的に共通payloadへ寄せるが、互換のため現行fieldを残す。

検証:
- `python -m kumc_agent.cli eval ragas ...` が従来通り動くこと。
- `python -m kumc_agent.cli eval run --set rag.circle.smoke` が動くこと。
- 診断情報が `metadata` 配下に入ること。

### Phase 12: Runtime / config統合
1. `RuntimeContext` に共通EvalRunnerを追加する。
2. `runtime.container` でtarget adapterとscorerを配線する。
3. `config/schema.py`、`config/load.py` に共通評価設定を追加する。
4. `configs/ops/app.yaml` に共通評価設定を追加する。
5. env overrideが必要な項目だけ `config/env_map.py` と `.env` / `.env.example` に追加する。

検証:
- 既存configだけでデフォルト値が解決されること。
- `.env` または `.env.example` の片方だけが変わらないこと。
- runtime contextからRAGAS評価と共通評価の両方を呼べること。

### Phase 13: CI / PR評価
1. PR小規模評価用コマンドを定義する。
2. 外部APIが不要なsmoke caseと、必要時だけ実行するRAGAS caseを分ける。
3. CI向けに非0終了コードの条件を実装する。
4. full eval用コマンドを定義する。
5. 前回main結果との差分レポートをartifactに保存する。

検証:
- safety blocker失敗で非0終了すること。
- smoke eval成功時に非0終了しないこと。
- full evalでRAGAS依存missingが失敗扱いになること。

## 4. 推奨ファイル変更範囲
想定される主な変更範囲は次の通り。

| 領域 | ファイル候補 |
| --- | --- |
| domain model | `src/kumc_agent/domain/models/evaluation.py` 新規、`src/kumc_agent/domain/models/operations.py` |
| eval usecase | `src/kumc_agent/usecases/eval/runner.py` 新規、`src/kumc_agent/usecases/eval/ragas.py` |
| scorers | `src/kumc_agent/usecases/eval/scorers/` 新規 |
| adapters | `src/kumc_agent/usecases/eval/adapters/` 新規 |
| repository | `src/kumc_agent/infra/operations/repository.py` |
| runtime | `src/kumc_agent/runtime/context.py`、`src/kumc_agent/runtime/container.py` |
| CLI | `src/kumc_agent/cli.py` |
| config | `src/kumc_agent/config/schema.py`、`src/kumc_agent/config/load.py`、`src/kumc_agent/config/env_map.py`、`configs/ops/app.yaml`、`configs/eval/` 新規 |
| eval data | `data/eval/` |
| docs | `docs/explanation/cli.md`、必要に応じてrunbook |
| tests | `tests/unit/test_eval_*.py`、`tests/integration/test_chat_index_eval.py` |

`.env` または `.env.example` に設定項目を追加・削除する場合は、必ず他方にも反映する。評価パラメータやプロンプトは `.env` / `.env.example` ではなく `configs/` と `assets/prompts/` に置く。

## 5. リスクと対策
| リスク | 対策 |
| --- | --- |
| 評価がLLM judgeに寄りすぎて不安定になる | ルールベースscorer、RAGAS、期待source/chunk検査を併用する |
| 外部APIコストが増える | PR小規模評価はcacheと件数制限を使い、full evalだけrefreshする |
| 権限違反を平均scoreで見逃す | safety scorerは1件でも重大失敗ならrun失敗にする |
| 評価payloadからcontextやsecretが漏れる | CLI出力はsummaryのみ、詳細artifactもマスクする |
| キャッシュが古い回答を使う | index/config/access/model/promptをcache keyに含める |
| 既存 `eval ragas` が壊れる | 互換テストを先に固定し、共通runnerとはadapterで接続する |
| 未実装機能の評価でCIが不安定になる | eval setごとにmodeとtarget availabilityを明示し、unsupportedを区別する |
| legacy依存が混入する | import検査または既存architecture testへ禁止ルールを追加する |

## 6. テスト計画
pytestは未導入前提のため、既存方式に合わせて `unittest` で追加する。

追加候補:

- `tests/unit/test_eval_models.py`
- `tests/unit/test_eval_set_loader.py`
- `tests/unit/test_eval_runner.py`
- `tests/unit/test_eval_scorers.py`
- `tests/unit/test_eval_gate_policy.py`
- `tests/unit/test_eval_cache.py`
- `tests/unit/test_eval_cli_payload.py`
- `tests/integration/test_eval_run_cli.py`

既存テストの維持:

- `tests/unit/test_eval_ragas_usecase.py`
- `tests/integration/test_chat_index_eval.py`

## 7. 実装順の推奨
1. 評価モデルとEvalSet loaderを先に入れる。
2. 既存RAGAS評価をadapter化し、互換を固定する。
3. 共通EvalRunnerと共通scorerを追加する。
4. RAG専用scorerと安全性scorerを追加する。
5. 機能別adapter/scorerを実装済み機能から順に追加する。
6. 結果保存、CLI、CI連携を追加する。
7. full eval運用と差分レポートを追加する。

