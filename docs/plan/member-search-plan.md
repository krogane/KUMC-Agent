# メンバー検索 実装計画

## 1. 方針
`docs/design/kumc-agent.md` と `docs/design/member-search.md` に従い、メンバー検索を実装する。

実装では `src/kumc_agent/infra/legacy` を参照・依存しない。既存の共通部品は `domain.models.operations.MemberProfile`、`infra.operations.repository`、`features.workflow`、`features.rag`、`infra.indexing`、`domain.models.retrieval.AccessContext` を優先して使う。現行実装と設計が矛盾する場合は `kumc-agent.md` を優先する。

## 2. 完了条件
- 指定Guild IDからDiscordメンバー情報を取得し、検索用metadataへ正規化できる。
- サークル情報RAGを使い、メンバーごとの活動内容、得意分野、過去担当、興味分野の根拠を収集できる。
- 専用LLMで `MemberProfile` を生成し、個人情報・外部公開不可情報を除外またはマスクできる。
- `MemberProfile` に根拠、AccessScope、生成metadataを保存できる。
- `member_profiles` からDense index、通常Sparse検索、ステミングSparse転置インデックスを作成できる。
- 検索前に、指定Guild内チャットまたはadmin DMかを確認できる。
- 権限がない場合、候補や存在有無を返さず拒否できる。
- user id、表示名、ロールを検索条件として抽出・フィルタリングできる。
- Dense、通常Sparse、ステミングSparseをRRFで統合できる。
- 候補ごとにスキル、ロール、担当履歴、根拠、該当理由を返せる。
- 回答は候補提示であり、担当確定や参加意思を断定しない。
- CLIや外部連携payloadの診断情報が `metadata` 配下に入る。
- 主要動作を既存テスト方式で検証できる。

## 3. 実装ステップ
### Phase 1: ドメインモデルとRepository拡張
1. `MemberProfile` に `evidence` を追加する。
2. `metadata` に `profile_version`、`profile_status`、`source_fingerprint`、`generated_by` を保存できる方針をテストで固定する。
3. `infra.operations.repository` の JSONL payload と Postgres payload に `evidence` を追加する。
4. 既存JSONLで `evidence` がない場合も読み込める後方互換を入れる。
5. `search_member_profiles()` は暫定互換として残し、専用検索service導入後は薄いfallbackにする。

検証:
- 既存payloadを読み込めること。
- `evidence` 付きprofileを保存・再読込できること。
- 診断情報がトップレベルへ出ないこと。

### Phase 2: Discordメンバー取得
1. 指定Guild IDのメンバー一覧を取得する `MemberDirectoryConnector` またはDiscord loader拡張を追加する。
2. user id、display name、role id、role name、joined_at、bot判定を正規化する。
3. bot、退会済み、除外roleの扱いを設定化する。
4. 取得結果に `source_fingerprint` を付与し、差分検出できるようにする。
5. 取得失敗時は `IndexingRun` に失敗内容を保存する。

検証:
- botを除外できること。
- role変更とdisplay name変更を差分として検出できること。
- 対象Guild以外のメンバーが混入しないこと。

### Phase 3: RAG根拠収集
1. 各メンバーごとにRAG検索クエリを生成する。
2. 表示名、mention、user id、role、担当系キーワードを組み合わせて検索する。
3. サークル情報RAGの権限フィルタと出典情報を利用し、根拠単位の `access_scope` を保持する。
4. 根拠候補から個人情報やsecretを含む可能性がある本文断片を除外・マスクする。
5. 根拠数、検索クエリ、RAG routeなどの診断情報は `metadata` 配下に保存する。

検証:
- 根拠に `source_type`、`source_item_id`、`chunk_id`、`label`、`access_scope` が入ること。
- 閲覧不可sourceの根拠を回答時に使わないための情報が残ること。
- 大きなcontext本文をprofile payloadへ保存しないこと。

### Phase 4: プロフィール生成
1. `features/member_search` を新設し、プロフィール生成serviceを置く。
2. Discord情報とRAG根拠から `skills`、`interests`、`past_assignments`、`roles` を生成する専用プロンプトを追加する。
3. 根拠なし項目を生成しないようJSON schemaとvalidationを追加する。
4. 個人情報・外部公開不可情報のマスク処理を入れる。
5. LLM失敗時はDiscord情報のみのfallback profileを作る。
6. `operations.save_member_profile()` で保存する。

検証:
- 根拠なしスキルを追加しないこと。
- fallback profileに `metadata.profile_status=fallback` が入ること。
- 実名らしき情報やsecretが保存されないこと。

### Phase 5: メンバー検索index作成
1. `MemberProfile` から `profile_text` を構築する関数を追加する。
2. Dense indexへ `profile_text` を投入する。
3. 通常Sparse検索用のprofile corpusを作成する。
4. ステミングSparse転置インデックスを作成する。
5. index成果物の保存先と再構築単位を設定化する。
6. 自動インデックス更新の対象に `member_profiles` を追加する。

検証:
- display name、role、skill、past assignmentが検索対象に含まれること。
- Discord user idは完全一致フィルタで扱えること。
- Dense index未構築時にSparse fallbackできること。

### Phase 6: 権限確認
1. メンバー検索専用のAccessPolicyを追加する。
2. 指定Guild ID内チャットを許可する。
3. 指定admin user idのDMを許可する。
4. 現行の `organizer` role許可は、互換が必要な期間のみfeature flag配下へ隔離する。
5. 権限がない場合は候補数や存在有無を返さない。
6. 検索前と回答前の両方でAccessScopeを確認する。

検証:
- 対象Guild内では許可されること。
- admin DMでは許可されること。
- 対象外Guild、非admin DMでは拒否されること。
- 拒否応答に候補数や類似情報が含まれないこと。

### Phase 7: 検索条件抽出
1. Discord mentionとuser idをルールで抽出する。
2. role mention、role id、role nameをルールで抽出する。
3. display name候補を正規化して抽出する。
4. 抽出結果を内部 `MemberSearchConditions` として扱う。
5. 外部payloadでは `metadata.search_conditions` に入れる。

検証:
- `<@123>`、`123`、`@display` がuser/display条件になること。
- role条件で候補が絞り込まれること。
- 診断情報がトップレベルへ出ないこと。

### Phase 8: ハイブリッド検索
1. `MemberSearchService.search()` を追加する。
2. 検索前にAccessPolicyと条件フィルタを適用する。
3. 通常Sparse検索を実行する。
4. ステミングSparse検索を実行する。
5. Dense検索を実行する。
6. RRFでrankを統合する。
7. 完全一致条件がある候補を上位補正する。
8. 回答前に候補profileとevidenceを再度AccessScopeでフィルタする。

検証:
- Dense、通常Sparse、ステミングSparseがそれぞれ候補を返せること。
- RRFで統合rankが安定すること。
- Dense unavailable時に `metadata.degraded=true` で継続すること。
- 閲覧不可evidenceが回答候補から除外されること。

### Phase 9: 回答生成
1. メンバー検索専用回答生成componentを追加する。
2. LLMプロンプトに「候補提示であり担当確定ではない」「個人能力・参加意思を断定しない」を明記する。
3. 候補ごとに表示名、ロール、スキル、過去担当、該当理由、根拠を出す。
4. LLM利用不可時のテンプレート回答を追加する。
5. 個人情報と閲覧不可根拠の最終マスクを行う。

検証:
- 断定表現が出ないこと。
- 根拠なし項目を補完しないこと。
- 候補ごとに確認が必要である旨が含まれること。

### Phase 10: workflow・統合入力受付連携
1. `features.workflow.service.member_search()` を専用 `MemberSearchService` 呼び出しへ置き換える。
2. `WorkResponse.member_profiles` を維持する。
3. `detail_markdown` に候補理由と根拠を含める。
4. 統合入力受付でメンバー検索intentを `member_search` へルーティングする。
5. 担当候補作成が必要な場合は `WorkflowCandidate(candidate_type="member_assignment")` を承認待ちで作る。

検証:
- 既存workflow APIの戻り値が壊れないこと。
- `member_assignment` が承認前に正本へ入らないこと。
- routeやselected handlerが `metadata` 配下に入ること。

### Phase 11: CLI・HTTP・Discord出力
1. CLIで `member_search` routeのpayloadを整える。
2. HTTP `/ask` または該当endpointで `member_profiles` を返せるようにする。
3. Discordではmember情報をephemeralまたは権限付きチャンネルへ返す。
4. 長い結果はthreadまたはattachmentに分離する。
5. 検索スコア、内部rank、trace id、検索条件は `metadata` 配下に入れる。
6. 大きなcontext本文やsecretを含む可能性があるmetadataを出力前に除外・マスクする。

検証:
- payloadトップレベルが安定フィールドだけであること。
- Discord応答に候補提示の注意書きが含まれること。
- 権限外では存在有無を漏らさないこと。

### Phase 12: 運用・自動更新
1. 自動インデックス更新に `member_profiles` を追加する。
2. Guild member差分、role変更、display name変更、退会を検出する。
3. 差分があるprofileだけ再生成する。
4. 関連RAG根拠の更新時にprofile再生成対象を推定する。
5. `indexing_runs` に処理件数、失敗件数、差分情報を保存する。
6. 重大失敗時のrollback手順をrunbookに追記する。

検証:
- 差分更新で全profileを不要に再生成しないこと。
- 削除・退会ユーザーが検索結果から除外されること。
- 失敗時に前回indexを維持できること。

## 4. 推奨ファイル変更範囲
想定される主な変更範囲は次の通り。

| 領域 | ファイル候補 |
| --- | --- |
| domain model | `src/kumc_agent/domain/models/operations.py` |
| workflow response | `src/kumc_agent/domain/models/workflow.py` 必要に応じて |
| repository | `src/kumc_agent/infra/operations/repository.py` |
| member search feature | `src/kumc_agent/features/member_search/` 新規 |
| Discord member source | `src/kumc_agent/infra/loaders/discord.py` または `src/kumc_agent/infra/connectors/` 新規 |
| RAG根拠収集 | `src/kumc_agent/features/rag/service.py` の公開API利用、または薄いadapter |
| indexing | `src/kumc_agent/features/indexing/service.py`、`src/kumc_agent/infra/indexing/` |
| retrieval | `src/kumc_agent/features/member_search/retrieval.py` 新規候補 |
| workflow | `src/kumc_agent/features/workflow/service.py` |
| entry routing | `src/kumc_agent/features/rag/components/entry_routing.py` または統合入力受付側 |
| CLI | `src/kumc_agent/cli.py` |
| HTTP | `src/kumc_agent/frontends/http/app.py` |
| Discord | `src/kumc_agent/frontends/discord/app.py` |
| config | `src/kumc_agent/config/schema.py`、`src/kumc_agent/config/load.py`、`src/kumc_agent/config/env_map.py` |
| prompts | `assets/prompts/member_profile_generation.md`、`assets/prompts/member_search_answer.md` 新規候補 |
| tests | `tests/unit/test_member_search_*.py` |

`.env` または `.env.example` に設定項目を追加する場合は、必ず他方にも反映する。

## 5. リスクと対策
| リスク | 対策 |
| --- | --- |
| 個人情報がprofileや回答に混入する | 生成前、保存前、回答前の3段階でマスクする |
| 権限外ユーザーに存在有無が漏れる | 検索前に拒否し、件数や類似候補を返さない |
| 閲覧不可根拠由来の候補理由が出る | evidence単位のAccessScopeを回答前に再フィルタする |
| LLMが根拠なしスキルを生成する | JSON schema validationと根拠参照チェックを入れる |
| 現行のroleベース許可が仕様とズレる | feature flagで隔離し、指定Guild/admin DMポリシーを本経路にする |
| Dense index未構築で検索不能になる | Sparse fallbackと `metadata.degraded` を用意する |
| legacy依存が混入する | import検査または静的テストで `infra.legacy` 参照を禁止する |
| 担当確定と誤認される | 回答テンプレートとLLMプロンプトに確認必須文を固定する |

## 6. テスト計画
pytestは未導入前提のため、既存方式に合わせて `unittest` で追加する。

追加候補:

- `tests/unit/test_member_profile_repository.py`
- `tests/unit/test_member_profile_generation.py`
- `tests/unit/test_member_search_access.py`
- `tests/unit/test_member_search_conditions.py`
- `tests/unit/test_member_search_retrieval.py`
- `tests/unit/test_member_search_answer.py`
- `tests/unit/test_member_search_payload.py`
- `tests/unit/test_member_search_no_legacy_import.py`

優先度の高い検証項目:

- `MemberProfile.evidence` の保存・読込
- 既存payloadとの後方互換
- 権限外拒否で存在有無を漏らさないこと
- 指定Guild内チャットとadmin DMの許可
- user id、表示名、role条件抽出
- AccessScopeによるprofile/evidence除外
- Dense、通常Sparse、ステミングSparse、RRF
- Dense fallback
- 根拠なしプロフィール項目を生成しないこと
- 個人情報・secretマスク
- 非断定の回答文
- payload metadata方針

## 7. 推奨実装順
1. `MemberProfile` とrepositoryの後方互換拡張
2. メンバー検索AccessPolicy
3. 検索条件抽出
4. 既存profileに対するSparse検索service
5. workflowの `member_search` を専用serviceへ移行
6. Discordメンバー取得
7. RAG根拠収集
8. プロフィール生成LLM
9. Dense / ステミングSparse index
10. RRF統合と回答生成
11. CLI・HTTP・Discord payload整備
12. 自動更新・評価・runbook
