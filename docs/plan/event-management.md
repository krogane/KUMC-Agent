# イベント管理 実装計画

実装同期日: 2026-04-28

本計画の主要項目は実装済みである。現行実装は `EventExtractionService`、`WorkflowService`、`auto_index_update`、Discord frontend、RuntimeConfig、`configs/main/event_management.yaml` に分散しており、`src/kumc_agent/infra/legacy` には依存しない。

## 1. 方針
`docs/design/kumc-agent.md` と `docs/design/event-management.md` に従い、イベント管理を実装する。

実装では `src/kumc_agent/infra/legacy` を参照・依存しない。既存の共通部品は `domain.models.workflow`、`features.workflow.service.WorkflowService`、`infra.workflow.repository`、`domain.models.retrieval.AccessContext`、`infra.audit`、`OperationsRepository` を優先して使う。現行実装と設計が矛盾する場合は `kumc-agent.md` を優先する。

## 2. 完了条件
- ユーザー入力またはRAGデータ差分から `EventCandidate` を抽出できる。
- 自動登録の抽出は専用LLMが行い、失敗時は候補を作成せず診断情報を残せる。
- 手動登録で必要情報を抽出できない場合は、候補作成前にユーザーへ質問できる。
- 手動登録でも正本登録前に `EventCandidate` を作成し、承認前に `Event` を作らない。
- 既存候補・既存Eventとの重複を検出し、候補metadataと承認UIに反映できる。
- イベント管理操作をadminに限定できる。
- `EventCandidate` を承認、修正、却下できる。
- 承認後だけ `Event` 正本へ昇格できる。
- 正本Eventの変更・削除も候補作成と承認を通せる。
- 日時、場所、状態、関連タスクで表示を絞り込める。
- Discord Componentで候補の承認、修正、却下を操作できる。
- n日前通知、当日通知、完了確認通知を送れる。
- 完了確認後にEvent状態を `done` に変更できる。
- 監査ログ、承認履歴、workflow runに副作用操作を記録できる。
- CLIや外部連携payloadの診断情報が `metadata` 配下に入る。
- 主要動作を既存テスト方式で検証できる。

2026-04-28時点の完了確認:

| 項目 | 状態 |
| --- | --- |
| RAG / ingestion差分からの自動抽出 | 完了。`auto_index_update` が差分chunkを `event_extract_from_delta` へ渡す。 |
| 専用LLM抽出schema | 完了。`new_events` / `event_changes` / `ignored_items` / `degraded` を使う。 |
| 手動登録の不足情報確認 | 完了。LLM抽出不可・必須情報不足では候補を作成しない。 |
| 手動変更・削除の確認 | 完了。LLMが対象Eventと差分を一意に抽出できない場合は候補を作成しない。 |
| 対象Event解決 | 完了。先頭Eventへのfallbackを廃止し、一意解決時のみ操作する。 |
| Discord通知 | 完了。`discord.py` で指定チャンネルへ送信し、deliveryをmetadataに残す。 |
| 完了確認Component | 完了。`event_complete:{event_id}:done:{key}:v1` から `event_complete` を実行する。 |
| まとめ承認Component | 完了。approve / edit / reject / evidence / diff / duplicates をcustom idで処理する。 |
| 設定接続 | 完了。`configs/main/event_management.yaml` をRuntimeConfigへ取り込み、securityの保守管理者IDもadminとして扱う。 |
| 仕様改善点 | 完了。入力契約、schema分離、状態遷移、対象解決、batch状態、通知key、action語彙、設定型、delivery分離を仕様・実装へ反映した。 |

## 3. 実装ステップ（完了済みの履歴）
### Phase 1: 仕様固定と現行テスト補強
1. `EventCandidate`、`Event`、`ApprovalRecord` の現行動作をテストで固定する。
2. `event_add` が承認前に `Event` を作らないことを明示的に検証する。
3. `approve` 後に `EventCandidate.status=merged` になることを検証する。
4. `edit`、`reject`、`show` の履歴保存を検証する。
5. `event_brief` が `Task.related_event_id` から関連未完了タスクを表示することを検証する。
6. CLI payloadで診断情報がトップレベルへ出ないことを検証する。

検証:
- `tests/unit/test_workflow_service.py`
- CLI用の既存unittestまたは新規unittest
- migration確認用の既存 `tests/unit/test_database_migrations.py`

### Phase 2: 権限管理
1. イベント管理用のAccessPolicyを追加する。
2. `event_add`、`event_list`、`event_brief` にadmin権限確認を入れる。
3. `event_update`、`event_delete`、`event_notify` を追加する場合も同じpolicyを通す。
4. `approval --type event` の `list`、`show`、`edit`、`approve`、`reject` に操作別policyを入れる。
5. Discord `/work type:event_*` と `/approval type:event` でも同じpolicyを通す。
6. adminは候補作成、表示、修正、承認、却下、正本変更、削除、通知設定を行えるようにする。
7. 権限外では候補数、Event存在有無、類似候補、根拠sourceを返さない。
8. 必要なadmin user idやrole idは `configs` 配下に置く。トークンやAPIキーを追加する場合のみ `.env` / `.env.example` の両方を更新する。

検証:
- adminは操作できること。
- 非adminはevent管理操作を拒否されること。
- 拒否応答に候補数、Event ID、類似候補が含まれないこと。
- Discord Component操作時も権限が再確認されること。

### Phase 3: Repository検索条件拡張
1. `WorkflowRepository.list_events()` に `starts_from`、`starts_to`、`place`、`include_canceled` を追加する。
2. JSONL repositoryとPostgres repositoryの両方に実装する。
3. `events` tableの検索に必要なindex追加を検討する。
4. `EventCandidate` 一覧にも `created_by`、`confidence`、`starts_from`、`starts_to` を追加する。
5. 関連タスク絞り込み用に `list_tasks(related_event_id=...)` の既存経路を維持する。
6. 既存呼び出しの後方互換を維持する。

検証:
- status、日時範囲、場所でEventを絞り込めること。
- 関連タスクを持つEventだけを抽出できること。
- JSONLとPostgresで同じ結果順になること。
- 既存 `event_list` が壊れないこと。

### Phase 4: イベント表示強化
1. `event_list` の自然言語instructionから日時範囲、場所、状態、関連タスク条件を抽出する。
2. 抽出した条件をrepository queryへ渡す。
3. Eventと承認待ち候補を分けて表示する。
4. 関連未完了タスク件数を表示する。
5. 大量結果の場合は件数制限とページング情報を返す。
6. Discordでは長い結果をattachmentに逃がす既存実装を維持する。
7. 内部の抽出条件やlimitは `metadata` 配下に入れる。

検証:
- `状態: planning`
- `場所: 部室`
- `2026-05-01から2026-05-31まで`
- `未完了タスクあり`
- `canceled` が既定では除外されること。

### Phase 5: 専用LLM抽出
1. `features/event_management` または `features/workflow/event_extraction.py` を新設する。
2. `assets/prompts/event_extraction.md` を追加する。
3. 自動登録経路では入力本文、RAG差分、根拠を専用LLMへ渡し、JSON schemaで `EventCandidate` 候補を返す。
4. title、summary、starts_at、ends_at、place、関連タスク条件、根拠、confidenceを生成する。
5. LLM失敗時、schema不正時、根拠不足時は自動登録候補を作成せず、`metadata.degraded=true` と理由を保存する。
6. 現行ルールベース抽出は自動登録のfallbackにしない。手動登録の入力補助やテスト用helperとして残す場合も、自動候補作成へ接続しない。
7. 抽出モデル名、prompt version、degraded理由は `metadata` 配下に保存する。

検証:
- イベント名、日時、場所、概要を抽出できること。
- タスク単体や雑談をイベントとして誤登録しないこと。
- LLM失敗時に候補を作らず、`metadata.degraded=true` を残すこと。

### Phase 5.5: 手動登録の不足情報確認
1. `event_add` の入力からtitle、starts_at、ends_at、place、summaryを抽出する。
2. 必須情報であるtitleとstarts_atを抽出できない場合は、候補を保存せずユーザーへ質問を返す。
3. 日時や場所を指定しているように見えるが解釈できない場合は、曖昧な項目を明示して質問する。
4. Discordではmodalまたはfollow-upで不足情報を受け取り、CLIでは不足項目を明示したエラーまたは対話可能な質問文を返す。
5. 不足情報が補完された後に `EventCandidate(status="proposed", created_by="user")` を保存する。

検証:
- titleが空または曖昧な場合に `EventCandidate` が保存されないこと。
- 日時らしき入力が解釈不能な場合に日時確認の質問を返すこと。
- 補完後に候補が作成され、承認前に `Event` は作成されないこと。

### Phase 6: 重複検出
1. `DuplicateEventDetector` を追加する。
2. title正規化、日時、場所、根拠sourceを使って候補同士・候補とEventを比較する。
3. 類似度が高い場合は `metadata.duplicate_candidates` に保存する。
4. 既存Eventと同一と判断できる場合は、正本新規登録ではなく変更候補へ誘導する。
5. 承認UIと `event_list` に重複警告を表示する。

検証:
- 同一title、同一日時、同一場所の重複が検出されること。
- 表記ゆれがあるtitleでも高類似として扱えること。
- 重複疑いがあっても候補そのものは監査可能に保存されること。

### Phase 7: 正本変更・削除候補
1. `EventChangeCandidate` を追加するか、汎用 `WorkflowCandidate(candidate_type="event_change")` に専用schemaを定義する。
2. 操作種別 `update` / `delete`、変更前payload、変更後payload、理由、根拠を保持する。
3. `event_update`、`event_delete` 相当のwork typeを追加するか、`event_add` 系のinstructionから変更意図をrouteする。
4. 承認前に `events` を更新・削除しない。
5. 承認後に正本を更新し、`ApprovalRecord` とaudit logを保存する。
6. 削除は物理削除ではなく `status="canceled"` への論理削除を基本にする。

検証:
- 日時変更候補が承認前に正本へ反映されないこと。
- 承認後に `Event.updated_at` とmetadataが更新されること。
- 削除は既定のlistから除外され、履歴は残ること。

### Phase 8: 承認処理のtransaction化
1. Postgres repositoryで `Event` 作成、`EventCandidate.status` 更新、`ApprovalRecord` 保存を同一transactionにまとめるAPIを追加する。
2. Event変更候補の反映と承認履歴保存も同一transactionにまとめる。
3. JSONL repositoryでは既存append-only方式を維持しつつ、失敗時の不整合を検出できるようにする。
4. 二重承認を防ぐため、承認対象statusの再確認をtransaction内で行う。
5. `merged` や `rejected` の再承認を拒否する。
6. 失敗時は候補状態を壊さず、利用者向けに再試行可能な文言を返す。

検証:
- 同じ候補を2回approveしてもEventが重複作成されないこと。
- `merged` 候補のedit/approveが拒否されること。
- DB失敗時に半端な `Event` だけが残らないこと。

### Phase 9: Discord Component承認UI
1. イベント候補表示用のDiscord view/componentを追加する。
2. approve、reject、edit、show evidence、duplicate details、diff detailsを実装する。
3. editはmodalまたはfollow-upで自然言語修正を受け付ける。
4. Component custom idに `target_type=event`、`target_id`、`action`、`batch_id`、nonceを含める。
5. custom idに長文、secret、根拠本文を含めない。
6. Component操作時もAccessPolicyを再確認する。
7. 操作後に最新状態を再取得して表示する。

検証:
- approve buttonでEvent正本が作成されること。
- edit後に候補が更新され、再承認できること。
- reject後に再承認できないこと。
- 権限外ユーザーがbuttonを押しても拒否されること。

### Phase 10: まとめ承認
1. イベント承認batchモデルを追加する。
2. `n` 日ごとに `status=proposed` の自動抽出候補を集約するjobを追加する。
3. batch id、対象期間、候補ID、変更候補ID、通知先、送信message idを保存する。
4. Discordへ候補一覧とComponentを送信する。
5. batch単位で一括approve、個別edit、個別rejectを扱えるようにする。
6. 通知済み候補の再送をidempotency keyで抑止する。

検証:
- 対象期間内の候補だけがbatchに含まれること。
- 一度通知した候補を同じbatchで重複通知しないこと。
- batch内の一部候補だけ承認・却下できること。

### Phase 11: イベント通知・完了確認
1. n日前通知、当日通知、完了確認のschedulerを追加する。
2. 通知対象は `Event.status in planning/announced` かつ `starts_at` 条件で抽出する。
3. 通知済み情報を `Event.metadata.notifications` に保存する。
4. 完了確認ComponentからEvent状態更新処理を実行する。
5. 完了確認後に `status="done"` へ変更する。
6. `done_by`、`done_comment`、通知message idをmetadataへ保存する。
7. audit logと操作履歴を残す。

検証:
- n日前Eventだけ通知されること。
- 当日Eventだけ当日通知されること。
- 通知済みEventが同じタイミングで再通知されないこと。
- 完了確認後にstatusが `done` になること。
- `done` / `canceled` Eventは通知対象から外れること。

### Phase 12: 自動登録差分連携
1. サークル情報RAGのインデックス更新またはRAG差分検出からイベント抽出を呼び出すadapterを追加する。
2. Discord、Drive、Notionなどの差分sourceを `Citation` として候補に付与する。
3. 大きな本文断片やsecretを含むcontextは候補payloadへ保存しない。
4. 自律エージェントは正本更新ではなく候補作成と通知までに限定する。
5. workflow runに抽出件数、候補件数、重複件数、変更候補件数、通知batch idを保存する。

検証:
- RAG差分から候補が作られること。
- 候補に根拠が付くこと。
- 承認前に正本Eventが増えないこと。
- secretらしき文字列が外部payloadへ出ないこと。

### Phase 13: CLI・HTTP・Discord出力整備
1. CLI `work` と `approval` のevent系出力を安定化する。
2. HTTP endpointがある場合はevent系payloadを同じschemaで返す。
3. Discordではephemeral応答とattachment出力を使い分ける。
4. `routing_decision`、`selected_handler`、`trace_id`、抽出条件、重複スコアは `metadata` 配下に入れる。
5. 大きな検索contextやsecretをmetadataから除外・マスクする。
6. `docs/explanation/cli.md` にevent系コマンド例を追記する。

検証:
- トップレベルには `event_candidates`、`events`、`tasks`、`approvals` など安定結果だけが出ること。
- 診断情報が `metadata` 配下にあること。
- Discord attachmentにsecretや巨大contextが含まれないこと。

### Phase 14: 評価セット
1. `event_extraction` 評価ケースを追加する。
2. 日時、場所、状態、重複検出、変更差分、候補止まりの確認を評価項目にする。
3. タスク単体、雑談、未決事項などイベントでない文をnegative caseにする。
4. 承認前にEvent正本へ入らないことを評価する。
5. 権限違反、prompt injection、secret混入を安全性caseに入れる。

検証:
- 評価セットを固定入力で実行できること。
- scorerが日時、場所、状態、変更差分、承認フローを採点できること。

### Phase 15: ドキュメント更新
1. `docs/explanation/cli.md` にevent系CLI例を追加する。
2. 必要に応じて `docs/design/evaluation-platform.md` と `docs/plan/evaluation-platform.md` のevent評価項目を更新する。
3. 運用手順が必要になった場合は `docs/runbooks/` に通知・承認運用手順を追加する。

検証:
- 実装済みwork typeとドキュメントのコマンド例が一致すること。
- payload方針に反するトップレベル診断フィールドが説明されていないこと。

## 4. 推奨実装順序（完了済み）
1. 現行イベント経路のテスト補強
2. admin限定AccessPolicy
3. repository検索条件拡張
4. イベント表示強化
5. 手動登録の不足情報確認
6. 専用LLM抽出
7. 重複検出
8. 正本変更・削除候補
9. 承認transaction化
10. Discord Component承認UI
11. まとめ承認
12. イベント通知・完了確認
13. 自動登録差分連携
14. CLI・HTTP・Discord出力整備
15. 評価セットとドキュメント更新

## 5. リスクと対策
| リスク | 対策 |
| --- | --- |
| イベントでない告知やタスクを誤登録する | 専用LLM schema、negative case、根拠必須化で抑止する |
| 承認前に正本が更新される | candidateと正本の境界をテストで固定し、transaction APIで反映する |
| 同じイベントが重複登録される | title、日時、場所、sourceを使う重複検出と承認UI警告を入れる |
| 削除で履歴や関連タスクが失われる | 物理削除ではなく `canceled` への論理削除にする |
| 非adminへ存在情報が漏れる | 権限外応答で件数、ID、根拠、類似候補を返さない |
| 通知が重複送信される | `Event.metadata.notifications` とidempotency keyで抑止する |
| payloadに検索contextやsecretが混入する | `_sanitize_payload_metadata` 相当の出力前sanitizeをevent系にも徹底する |

## 6. 追加・変更が想定されるファイル
- `src/kumc_agent/domain/models/workflow.py`
- `src/kumc_agent/features/workflow/service.py`
- `src/kumc_agent/features/event_management/`
- `src/kumc_agent/infra/workflow/repository.py`
- `src/kumc_agent/frontends/discord/app.py`
- `src/kumc_agent/frontends/http/app.py`
- `src/kumc_agent/cli.py`
- `assets/prompts/event_extraction.md`
- `configs/`
- `infrastructure/migrations/`
- `tests/unit/test_workflow_service.py`
- `tests/unit/test_database_migrations.py`
- `docs/explanation/cli.md`
