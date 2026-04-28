# メンバー検索 実装調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/member-search.md`
- `docs/plan/member-search.md`
- 補助参照: `docs/design/kumc-agent.md` の「3. メンバー検索」

## 結論

メンバー検索は初期実装ではなく、プロフィール作成、RAG根拠収集、LLMプロフィール生成、JSONL保存、Dense/Sparse/Stemming index作成、権限確認、条件抽出、ハイブリッドランキング、workflow/CLI/HTTP/Discord/admin更新経路まで実装されています。

ただし、仕様通りの完全実装とは判定できません。特に次の差分が残っています。

- Postgres有効時に `member_profiles` へ保存はできるが、検索・一覧取得がPostgresから読めない。
- 検索時は保存済みDense/Sparse/Stemming indexを使わず、repository上のprofileから毎回オンメモリでランキングしている。
- evidence単位のAccessScopeがRAG出典から継承されず、ランキング・候補理由生成時にも閲覧不可evidenceが影響し得る。
- LLM生成項目が「個別の根拠に紐づく」ことまでは検証していない。
- 空の許可Guild設定をワイルドカードとして扱うため、仕様の「指定Guild内チャットのみ許可」より緩い。

運用上の最重要修正は、Postgres読込経路、evidence AccessScopeの正確な伝播と検索時フィルタ、永続indexを使う検索runtimeの接続です。

## 実装確認

| 仕様項目 | 実装状況 | 主な実装箇所 |
| --- | --- | --- |
| `MemberProfile` に evidence / AccessScope / metadata を保持 | 実装済み | `src/kumc_agent/domain/models/operations.py` |
| JSONL保存と後方互換読込 | 実装済み | `src/kumc_agent/infra/operations/repository.py` |
| Postgres保存 | 部分実装。migrationとsaveはあるが、list/searchが未接続 | `infrastructure/migrations/012_member_profiles.sql`, `src/kumc_agent/infra/operations/repository.py` |
| Discord Guildメンバー取得 | 実装済み。member intentでuser id、display name、role、joined_at、bot判定を取得 | `src/kumc_agent/infra/connectors/discord_members.py` |
| bot / inactive / 除外roleの除外 | bot/inactiveは実装済み。除外roleのconfig配線は不足 | `src/kumc_agent/features/member_search/service.py` |
| RAG根拠収集 | 実装済み。ただし出典別AccessScopeは単純なguild scopeに丸められる | `AskServiceEvidenceSource` |
| LLMプロフィール生成とfallback | 実装済み | `MemberProfileGenerator`, `assets/prompts/member_profile_generation.md` |
| 個人情報・secretマスク | 実装済み。ただし実名らしき情報の汎用検出は限定的 | `mask_sensitive_text()` |
| Dense index作成 | 実装済み | `MemberProfileIndexService`, `FaissLikeIndex` |
| 通常Sparse / ステミングSparse index作成 | 実装済み。JSON artifactとして保存 | `MemberProfileIndexService` |
| 検索前権限確認 | 実装済み。ただし空allowed guildを全Guild許可扱い | `MemberSearchService._is_authorized()` |
| user id / display name / role抽出 | 実装済み | `extract_conditions()` |
| 除外条件抽出 | 未実装 | 該当なし |
| AccessScopeによるprofile/evidenceフィルタ | profileと回答evidenceは実装済み。ranking/reasonでは不完全 | `MemberSearchService._can_view_profile()`, `_visible_evidence()` |
| Dense / Sparse / StemmingのRRF統合 | 実装済み。ただし保存済みindexではなくオンメモリ検索 | `_rank()`, `_rrf()` |
| Dense unavailable時のSparse fallback | 実装済み | `_dense_rank()`, `tests/unit/test_member_search.py` |
| 非断定の候補回答 | 実装済み | `_template_answer()`, `assets/prompts/member_search_answer.md` |
| workflow連携 | 実装済み。専用serviceがあればそれを使用 | `WorkflowService.member_search()` |
| 統合入力受付連携 | 実装済み | `src/kumc_agent/usecases/integrated_input/entry.py` |
| CLI / HTTP / Discord admin rebuild | 実装済み | `src/kumc_agent/cli.py`, `src/kumc_agent/frontends/http/app.py`, `src/kumc_agent/frontends/discord/app.py` |
| 自動インデックス更新連携 | 実装済み | `src/kumc_agent/usecases/indexing/auto_update.py` |
| unittestによる検証 | 主要経路は実装済み。完全性検証は不足 | `tests/unit/test_member_search.py` |

## 仕様と実装の差分

### 1. Postgres repositoryの読込経路が未実装

`PostgresOperationsRepository.save_member_profile()` は `member_profiles` tableへupsertしますが、`list_member_profiles()` と `search_member_profiles()` はoverrideされていません。そのためPostgres有効時でも、検索serviceは親クラスのJSONL読込を使います。

該当箇所:

- JSONL読込: `src/kumc_agent/infra/operations/repository.py:116`
- JSONL一覧: `src/kumc_agent/infra/operations/repository.py:143`
- Postgres保存のみ: `src/kumc_agent/infra/operations/repository.py:210`

影響:

- Postgres環境では、保存したプロフィールを検索できない可能性が高い。
- 自動更新でPostgresへ保存しても、同じprocessの検索がJSONL fallbackを見て空になる可能性がある。

完全実装に必要な対応:

- `PostgresOperationsRepository.list_member_profiles()` と `search_member_profiles()` を実装する。
- JSONB payloadから `MemberProfile` へ復元する共通mapperを使う。
- Postgres保存後の検索を確認する単体テストを追加する。

### 2. 検索runtimeが保存済みindexを使っていない

`MemberProfileIndexService` はDense/Sparse/Stemming indexを保存しますが、`MemberSearchService.search()` はrepositoryから全profileを読み、通常Sparse・ステミングSparse・Denseを毎回オンメモリで計算しています。

該当箇所:

- index作成: `src/kumc_agent/features/member_search/service.py:399`
- 検索時のrepository全件読込: `src/kumc_agent/features/member_search/service.py:489`
- オンメモリDense計算: `src/kumc_agent/features/member_search/service.py:616`

影響:

- 仕様の「作成済みDense index / 転置indexを検索に使う」状態ではない。
- profile数が増えると検索時に毎回embeddingを作るため遅い。
- index artifactの破損や未publishを検索時に検知しにくい。

完全実装に必要な対応:

- Denseは `FaissLikeIndex.search()` を使う検索adapterを追加する。
- Sparse/Stemmingは保存済みkeyword indexを読み込むadapterを追加する。
- index未構築時だけオンメモリfallbackにし、`metadata.degraded=true` と理由を明示する。

### 3. Dense index本文にDiscord user idが混入する

設計ではDiscord user idはDense embeddingの主要本文に含めず、完全一致フィルタまたはSparse用に保持するとされています。一方、index作成時は `build_profile_text(profile, include_user_id=True)` をDense/Sparse共通docとして使っています。

該当箇所:

- `src/kumc_agent/features/member_search/service.py:406`
- `src/kumc_agent/features/member_search/service.py:674`

影響:

- 保存済みDense indexを今後検索に接続した場合、user idが意味ベクトルに混入する。
- 現在のオンメモリDense検索では `include_user_id=False` のため、保存済みDense indexと検索時Denseの対象本文が一致しない。

完全実装に必要な対応:

- Dense用profile textとSparse/filter用profile textを分ける。
- index metadataに投入本文種別とschema versionを保存する。

### 4. evidence単位のAccessScopeが十分に伝播していない

RAG citationから保存するevidenceは、実出典のACLではなく `{"guild_ids": [member.guild_id]}` に固定されています。設計上は「根拠単位の可視範囲」を保持し、検索時に質問者が閲覧できない根拠は候補理由へ使わない必要があります。

該当箇所:

- `src/kumc_agent/features/member_search/service.py:196`

影響:

- RAG出典が管理者限定、別チャンネル限定、特定ユーザー限定でも、同一guild内では表示可能扱いになり得る。
- 後段の `_visible_evidence()` があっても、元のscopeが粗いと正しく除外できない。

完全実装に必要な対応:

- RAG `Citation` またはcontext metadataからsource/chunkのAccessScopeを取得できるようにする。
- evidence保存時にsource側AccessScopeを保持する。
- source scopeが取得できない根拠は回答表示不可、または低信頼metadataを付ける。

### 5. rankingと候補理由に閲覧不可evidenceが影響し得る

回答直前の `candidate.evidence` は `_visible_evidence()` で絞られます。しかしranking用profile textと候補理由は、絞り込み前の `profile.evidence` を参照しています。

該当箇所:

- ranking対象profile作成: `src/kumc_agent/features/member_search/service.py:489`
- candidate reason生成時に未フィルタprofileを渡す: `src/kumc_agent/features/member_search/service.py:498`
- evidence有無の理由付け: `src/kumc_agent/features/member_search/service.py:970`
- profile textのevidence要約: `src/kumc_agent/features/member_search/service.py:675`

影響:

- 閲覧不可evidence中の語がrankingに効き、候補順位へ影響する可能性がある。
- 表示されない根拠しかない候補にも「参照可能な根拠があります」という理由が付く可能性がある。

完全実装に必要な対応:

- 検索前に、profileごとに閲覧可能evidenceだけを残した検索用viewを作る。
- ranking、reason、answer payloadはそのviewだけを参照する。
- 閲覧不可evidenceだけでhitしないことをテストする。

### 6. LLM生成結果の根拠チェックが粗い

`MemberProfileGenerator` はevidenceが1件でもあれば、LLMの `skills` / `interests` / `past_assignments` を受け入れます。項目ごとに根拠citationへ紐づくことまでは検証していません。

該当箇所:

- `src/kumc_agent/features/member_search/service.py:252`
- `_clean_terms(..., require_evidence=bool(evidence))`

影響:

- LLMが根拠にないスキルを混ぜても、短い名詞句なら保存され得る。
- 「根拠なし項目を生成しない」という仕様をテストで十分に固定できていない。

完全実装に必要な対応:

- LLM出力schemaを `{term, evidence_ids}` 形式にする。
- evidence idが存在しないtermを破棄するvalidationを入れる。
- 根拠なしtermをLLMが返すテストを追加する。

### 7. 権限のデフォルトが仕様より緩い

`MemberSearchService._is_authorized()` は、`allowed_guild_ids` が空の場合、任意のguild id付きaccessを許可します。仕様は「指定Guild ID内チャット」または「指定admin user idのDM」のみ許可です。

該当箇所:

- `src/kumc_agent/features/member_search/service.py:520`

影響:

- 設定漏れ時にdefault denyにならない。
- `configs/main/security.yaml` は `discord_guild_allow_list` と `discord_member_profile_guild_ids` が空なので、runtime構成によっては許可条件が曖昧になる。

完全実装に必要な対応:

- 空の `allowed_guild_ids` はdeny扱いにする。
- 開発用に全Guild許可が必要なら明示的なfeature flagまたは `"*"` を使う。
- 設定未完了時のreadiness警告を強める。

### 8. 除外条件抽出が未実装

設計の検索条件抽出には「除外条件」がありますが、実装の `MemberSearchConditions` には除外条件がありません。

該当箇所:

- `src/kumc_agent/features/member_search/service.py:125`
- `src/kumc_agent/features/member_search/service.py:657`

影響:

- 「デザイン担当候補、運営ロール以外」などの検索を仕様通りに扱えない。

完全実装に必要な対応:

- `exclude_user_ids`、`exclude_role_ids`、`exclude_role_names`、`exclude_terms` を条件に追加する。
- ルール抽出できる構文を仕様に固定する。

### 9. Sparse設定値がrankingに使われていない

`MemberSearchConfig` には `sparse_bm25_k1` と `sparse_bm25_b` がありますが、現在の `_keyword_rank()` は単純なtoken overlapです。通常SparseとステミングSparseの比率設定もありません。

該当箇所:

- config: `src/kumc_agent/features/member_search/service.py:116`
- ranking: `src/kumc_agent/features/member_search/service.py:876`

影響:

- 設計のBM25/転置index前提と実際のranking挙動が異なる。
- 設定を変えても検索品質が変わらない項目がある。

完全実装に必要な対応:

- 保存済みSparse indexまたはBM25 retrieverを使う。
- normal/stemming/denseの重みまたはRRF投入上限を設定化する。

### 10. 重複profile選択と監査ログが弱い

設計では同一人物の重複profileは最新 `updated_at` と `source_fingerprint` を優先するとされています。JSONL repositoryはid単位のlatest化を行いますが、同一 `discord_user_id` でidが異なるprofileの重複解消はありません。

また、設計の「監査ログには実行者、guild、検索意図、結果件数を保存できる」は、workflow run metadataの件数記録はありますが、member_search専用の監査イベントとしては明確ではありません。

完全実装に必要な対応:

- `discord_user_id` + `guild_id` 単位のdedupeを検索前に行う。
- member_search専用のaudit metadataを定義し、query全文ではなく意図・条件・件数だけを保存する。

## 検証状況

既存テストで確認されている主な項目:

- `MemberProfile.evidence` のJSONL保存・後方互換読込
- mention / role / display name / bare idの条件抽出
- 対象外guild拒否、対象guild許可、admin DM許可
- evidenceの回答前AccessScopeフィルタ
- Dense失敗時のSparse fallback
- workflowが `MemberSearchService` のmetadataを使うこと
- プロフィール生成、bot skip、index artifact作成
- email/tokenのマスク
- 自動インデックス更新で `member_profiles` stageが呼ばれること

不足しているテスト:

- Postgres保存後に `list_member_profiles()` / `search_member_profiles()` で読めること
- 保存済みDense/Sparse/Stemming indexを検索runtimeが使うこと
- 閲覧不可evidenceだけがhit根拠になる場合にranking・reason・answerへ出ないこと
- LLMが根拠なしtermを返した場合に破棄されること
- allowed guild未設定時にdefault denyになること
- 除外条件抽出
- same `discord_user_id` の重複profile解消
- LLM回答が断定表現を含まないこと
- `infra.legacy` 依存がmember_search経路へ混入しないこと

## 仕様の改善点

### 1. 完全実装の受け入れ基準を明文化する

設計と計画には完了条件がありますが、どこまでを完全実装とするかの判定が曖昧です。少なくとも次をmust/shouldで分けると、実装監査が安定します。

- must: 権限外で存在有無を漏らさない
- must: evidence単位AccessScopeをranking/reason/answerの全段階で適用する
- must: Postgres有効時の保存・読込・検索が成立する
- must: 根拠なしLLM生成項目を保存しない
- should: 保存済みindexを優先利用し、オンメモリ検索はfallbackにする
- should: 除外条件、重複profile解消、監査ログ

### 2. evidence schemaに可視範囲と引用安全性を追加する

現行仕様のEvidenceは十分ですが、実装に落とすには次を追加した方がよいです。

- `source_access_scope`: 元source/chunkから継承したAccessScope
- `display_policy`: `show_quote` / `show_label_only` / `hidden`
- `redaction_status`: `clean` / `redacted` / `blocked`
- `evidence_id`: LLM生成termが参照する安定ID

### 3. 生成profileのtermとevidenceの対応関係を仕様化する

`skills` などを単なる文字列配列にすると、根拠との対応が検証しにくくなります。保存payloadは後方互換のため文字列配列を維持しつつ、`metadata.term_evidence` または新しい内部構造で次を保存する方が安全です。

```json
{
  "skills": ["イベント告知デザイン"],
  "metadata": {
    "term_evidence": {
      "イベント告知デザイン": ["doc-1:chunk-1"]
    }
  }
}
```

### 4. 検索runtimeのindex利用方針を明確にする

仕様ではindex作成と検索が書かれていますが、検索時に保存済みindexを必ず使うのか、profile数が少ない場合はオンメモリ検索を許すのかが未定義です。

改善案:

- default: 保存済みindexを使う
- fallback: index missing/corrupt時のみオンメモリ検索
- metadata: `degraded=true`, `degraded_reason=index_unavailable`
- quality: index artifactとrepository profile countの整合性を検査

### 5. 権限設定の空値 semantics を固定する

`allowed_guild_ids=[]` を「全許可」にするのか「全拒否」にするのかを仕様で明記すべきです。個人に紐づく検索なので、default denyを推奨します。

### 6. payload schemaに候補専用フィールドを定義する

現在は `member_profiles` と `detail_markdown` に候補理由・根拠が分散します。外部連携で安定的に使うなら、トップレベルに安定フィールドとして `member_candidates` を追加し、scoreやrankなどの診断情報はcandidate内の `metadata` またはresponse `metadata` に閉じ込める案がよいです。

例:

```json
{
  "text": "...",
  "route": "member_search",
  "member_candidates": [
    {
      "profile_id": "...",
      "display_name": "...",
      "reason": "...",
      "evidence": []
    }
  ],
  "metadata": {
    "search_conditions": {},
    "degraded": false
  }
}
```

### 7. 除外条件の文法を仕様化する

「除外条件」を実装するには、曖昧な自然文だけではテストしづらいため、まず決定的な構文を定義するとよいです。

- `除外ロール: 運営`
- `exclude_role: organizer`
- `除外ユーザー: <@123>`
- `-role:運営`

### 8. 評価セットを具体化する

評価項目は定義済みですが、テストデータと期待出力の粒度が不足しています。次のfixtureを用意すると、仕様逸脱を検出しやすくなります。

- visible evidenceだけでhitする候補
- hidden evidenceだけでhitしてはいけない候補
- 根拠なしLLM termを返すfake LLM
- 同名display nameの複数profile
- same discord user idの古いprofileと新しいprofile
- allowed guild未設定
- Postgres repository

## 推奨対応順

1. Postgres `list_member_profiles()` / `search_member_profiles()` を実装する。
2. 検索前に閲覧可能evidenceだけを残したprofile viewを作り、ranking/reason/answerを統一する。
3. RAG citation/source metadataからevidence AccessScopeを継承できるようにする。
4. Dense/Sparse/Stemmingの保存済みindexを検索runtimeへ接続する。
5. LLM生成schemaをterm + evidence idへ拡張し、根拠なしtermを破棄する。
6. allowed guild空値をdefault denyへ変更し、設定未完了をreadinessで明示する。
7. 除外条件、dedupe、member_search監査ログを追加する。
8. 上記の不足テストを `unittest` で追加する。
