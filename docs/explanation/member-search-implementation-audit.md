# メンバー検索 実装後再調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/member-search.md`
- `docs/plan/member-search.md`
- 補助参照: `docs/design/kumc-agent.md` の「3. メンバー検索」

## 結論

前回調査で確認した仕様との差分は実装済みです。現時点のメンバー検索は、プロフィール作成、RAG根拠収集、LLMプロフィール生成、Postgres/JSONL保存・読込、Dense/通常Sparse/ステミングSparse index作成、保存済みindex優先の検索runtime、権限確認、AccessScopeフィルタ、条件抽出、除外条件、RRF統合、非断定回答、workflow/統合入力/CLI/HTTP/Discord/admin更新経路、自動インデックス更新連携を備えています。

仕様改善点のうち、項目4「検索runtimeのindex利用方針」は、保存済みindexをdefaultとして使い、index未構築・旧schema・破損時のみオンメモリfallbackし `metadata.degraded=true` を返す方針で実装しました。項目5「権限設定の空値 semantics」は、`allowed_guild_ids=[]` をdefault denyとして実装しました。

## 実装確認

| 仕様項目 | 実装後の状態 | 主な実装箇所 |
| --- | --- | --- |
| `MemberProfile` の evidence / AccessScope / metadata | 実装済み。evidenceには `evidence_id` も保存 | `src/kumc_agent/domain/models/operations.py`, `src/kumc_agent/features/member_search/service.py` |
| JSONL保存・後方互換読込 | 実装済み | `src/kumc_agent/infra/operations/repository.py` |
| Postgres保存・読込・検索 | 解消。`list_member_profiles()` / `search_member_profiles()` をPostgresへ接続 | `src/kumc_agent/infra/operations/repository.py` |
| Discord Guildメンバー取得 | 実装済み | `src/kumc_agent/infra/connectors/discord_members.py` |
| bot / inactive / 除外role | 実装済み。除外roleは `features.member_search.exclude_role_names` から設定 | `configs/main/features.yaml`, `src/kumc_agent/config/schema.py`, `src/kumc_agent/apps/workflow.py` |
| RAG根拠収集 | 実装済み。RAG citationの `access_scope` をevidenceへ継承 | `src/kumc_agent/features/retrieval/context.py`, `AskServiceEvidenceSource` |
| LLMプロフィール生成 | 実装済み。termごとに `evidence_id` を検証し、根拠なしtermは破棄 | `MemberProfileGenerator`, `assets/prompts/member_profile_generation.md` |
| 個人情報・secretマスク | 実装済み | `mask_sensitive_text()`, `sanitize_evidence()` |
| Dense index作成 | 実装済み。Dense本文からDiscord user idを除外 | `MemberProfileIndexService` |
| 通常Sparse / ステミングSparse index作成 | 実装済み。BM25設定値を使う保存済みkeyword indexへ移行 | `MemberProfileIndexService` |
| 検索時の保存済みindex利用 | 解消。新schemaの保存済みindexを優先し、fallback時はdegraded metadataを返す | `MemberSearchService._sparse_rank()`, `_dense_rank()` |
| 古い/安全性不明なindexの扱い | 実装済み。`member_index_metadata.json` のschemaが一致しないindexは使わない | `_member_index_metadata_valid()` |
| 権限確認 | 解消。指定Guildまたはadmin DMのみ許可。allowed guild未設定はdefault deny | `MemberSearchService._is_authorized()` |
| user id / display name / role抽出 | 実装済み | `extract_conditions()` |
| 除外条件抽出 | 解消。`除外ロール:`、`exclude_role:`、`除外ユーザー:`、`-role:` などをサポート | `extract_conditions()` |
| AccessScopeによるprofile/evidenceフィルタ | 解消。検索前に閲覧可能evidenceだけのprofile viewを作り、ranking/reason/answerを統一 | `MemberSearchService.search()`, `_filter_profile_for_response()` |
| 閲覧不可evidenceのindex投入 | 解消。indexにはpublic/guild相当の安全なevidenceだけを投入 | `_profile_for_index()` |
| Dense / Sparse / StemmingのRRF統合 | 実装済み | `_rank()`, `_rrf()` |
| Dense unavailable時のSparse fallback | 実装済み | `_dense_rank()` |
| 非断定回答 | 実装済み。LLM回答にも断定表現の後処理を適用 | `_candidate_safe_answer()`, `assets/prompts/member_search_answer.md` |
| 同一人物重複profile | 解消。`guild_id + discord_user_id` 単位で最新profileを採用 | `_dedupe_profiles()` |
| 監査ログ | 解消。workflow member_searchで実行者、認可結果、件数、検索条件、degradedを記録。query全文は記録しない | `WorkflowService.member_search()` |
| workflow連携 | 実装済み | `WorkflowService.member_search()` |
| 統合入力受付連携 | 実装済み | `src/kumc_agent/usecases/integrated_input/entry.py` |
| CLI / HTTP / Discord admin rebuild | 実装済み | `src/kumc_agent/cli.py`, `src/kumc_agent/frontends/http/app.py`, `src/kumc_agent/frontends/discord/app.py` |
| 自動インデックス更新連携 | 実装済み | `src/kumc_agent/usecases/indexing/auto_update.py` |

## 差分再調査

| 前回差分 | 再調査結果 |
| --- | --- |
| Postgres repositoryの読込経路が未実装 | 解消。Postgres版 `list_member_profiles()` / `search_member_profiles()` を追加し、保存後検索をテストで確認 |
| 検索runtimeが保存済みindexを使っていない | 解消。Denseは `FaissLikeIndex.search()`、Sparse/Stemmingは保存済みkeyword indexを優先利用 |
| Dense index本文にDiscord user idが混入 | 解消。Dense用本文は `include_user_id=False`、Sparse用本文は `include_user_id=True` に分離 |
| evidence単位AccessScopeが伝播しない | 解消。`Citation.access_scope` を追加し、ContextPackerからRAG citation、member evidenceへ伝播 |
| ranking/reasonに閲覧不可evidenceが影響し得る | 解消。検索前に閲覧可能evidenceだけのprofile viewを作り、ranking/reason/answerを同じviewで処理 |
| LLM生成結果の根拠チェックが粗い | 解消。LLM出力をterm + evidence_id形式にし、存在しないevidence参照のtermを破棄 |
| 権限のデフォルトが仕様より緩い | 解消。allowed guild未設定ではGuild内検索を拒否するdefault denyへ変更 |
| 除外条件抽出が未実装 | 解消。固定構文の除外条件を実装 |
| Sparse設定値がrankingに使われていない | 解消。member sparse index作成でBM25 k1/bを使用し、検索runtimeがそのindexを使う |
| 重複profile選択と監査ログが弱い | 解消。dedupeとmember_search監査metadataを追加 |

## 仕様改善点の実装

### 項目4: 検索runtimeのindex利用方針

実装方針:

- default: 保存済みindexを使う。
- fallback: index missing、旧schema、破損、Dense unavailable時のみオンメモリ検索へ落とす。
- metadata: fallback時は `degraded=true` と `degraded_reasons`、各sourceの `rank_source_modes` を返す。
- safety: `member_index_metadata.json` のschemaが一致しないindexは使わない。

### 項目5: 権限設定の空値 semantics

実装方針:

- `allowed_guild_ids=[]` はdefault deny。
- Guild検索は `allowed_guild_ids` に含まれるGuild IDだけ許可。
- DM検索は `admin_user_ids` に含まれるuser idだけ許可。
- workflow fallbackのroleベース許可は無効化し、専用 `MemberSearchService` 未設定時は拒否する。

## 追加検証

追加・更新した主なテスト:

- Postgres保存後に `list_member_profiles()` / `search_member_profiles()` で読めること
- allowed guild未設定でdefault denyになること
- 除外role / 除外user条件を抽出できること
- 除外role条件で候補が除外されること
- 閲覧不可evidenceだけではranking・reason・answerに出ないこと
- 保存済みDense/Sparse/Stemming indexが検索runtimeで使われること
- LLM生成termが存在する `evidence_id` を参照しない場合に破棄されること
- LLM回答の断定表現が非断定表現へ補正されること
- 同一 `discord_user_id` の重複profileでは最新profileを採用すること

実行した検証:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_member_search
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_config_loading tests.unit.test_design_gap_foundation
PYTHONPATH=src app/.venv/bin/python -m unittest discover tests/unit
PYTHONPATH=src app/.venv/bin/python -m unittest tests.integration.test_chat_index_eval
```

結果:

- `tests.unit.test_member_search`: 16 tests / OK
- `tests.unit.test_config_loading tests.unit.test_design_gap_foundation`: 9 tests / OK
- unit discovery: 243 tests / OK
- `tests.integration.test_chat_index_eval`: 1 test / OK

## 残リスク

完全に仕様通りであることを確認しましたが、運用上の残リスクはあります。

- 既存環境に旧schemaのmember indexが残っている場合、検索runtimeは安全のため保存済みindexを使わずオンメモリfallbackします。次回 `member_profiles` rebuildで新schema indexへ更新されます。
- RAG citationにAccessScopeがない外部実装を接続した場合、そのevidenceは `admin_only` として保存され、Guild内回答には表示されません。安全側の挙動ですが、根拠表示量は減ります。
- Discordのmember intentやGuild member取得可否はDiscord Bot設定に依存します。取得失敗時は `IndexingRun.status=failed` として扱われます。
