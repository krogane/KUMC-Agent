# External Integration And Operations Backlog

作成日: 2026-04-25

この文書は、コードだけでは完了できず、外部サービス接続、権限付与、利用規約確認、または運用メンバーの手作業が必要な残タスクをまとめる。

## Required External Setup

| Area | Required Action | Owner / Input Needed | Blocking Feature |
|---|---|---|---|
| Discord production guild | production guild ID、admin user ID、role ID を設定する | Discord 管理者 | `/ask`, `/work`, `/approval`, `/automation`, `/admin` production 公開 |
| Discord bot | slash command sync、権限確認、staging bot と production bot の切替 | Discord 管理者 | staging / production rollout |
| Google Drive | service account / OAuth credential、対象 folder allowlist、共有権限確認 | Drive 管理者 | Drive backfill / changes sync |
| Notion | integration token、database IDs、ページ権限 | Notion 管理者 | Notion backfill / sync |
| X | API 利用可否、投稿/取得方針、利用規約確認 | X 管理者 / 運営 | X connector / external posting |
| Hatena Blog | 取得対象、robots/API 方針、再利用条件 | 運営 | Hatena connector production enable |
| Crafters Colony | 利用規約、作者ページ、再利用可否 | 運営 | Crafters connector / asset reuse |
| Minecraft Wiki | API policy、対象 page list、edition/version review | 運営 | Wiki production indexing |
| Minecraft server | 対象 host、compose path、executor isolation、backup path | サーバー管理者 | Minecraft write executor |
| Object storage | S3-compatible endpoint/bucket/credential | インフラ担当 | raw snapshot production 保存 |
| PostgreSQL | DB URL、migration 実行 | インフラ担当 | production repository |
| Redis / queue | Redis URL、worker deployment | インフラ担当 | background jobs / automation runner |
| Secret Manager | production secrets を `.env` から移行 | インフラ担当 | production readiness |

## Manual Gates

1. `kumc-agent db migrate` を production/staging DB で実行する。
2. `kumc-agent admin --action readiness` を確認する。
3. `kumc-agent admin --action permissions` で guild/user allowlist を確認する。
4. `/ask`, `/work`, `/approval`, `/automation`, `/admin` を staging guild で slash command sync する。
5. 1週間以上、例会準備・タスク抽出・安全 RAG を staging 運用する。
6. 運営メンバーが回答品質、TaskCandidate、Meeting draft をレビューする。
7. backup/restore を staging で実演し、結果を運用ログに残す。
8. rollback 手順を staging で実演し、結果を運用ログに残す。
9. prompt injection / secret redaction eval で重大失敗 0 件を確認する。

## Source Terms Review

各 connector で `terms_review_status` を次のいずれかに確定する。

| Status | Meaning | External Reuse |
|---|---|---|
| `pending` | 利用規約・権利確認前 | 不可 |
| `internal_only` | 内部検索・要約のみ許可 | 不可 |
| `approved` | 外部公開素材としての利用条件を確認済み | 条件付き可 |
| `rejected` | 外部利用不可 | 不可 |

現状の初期値:

- Google Drive / Discord / Notion: `internal_only`
- Hatena / X / Crafters Colony / Minecraft Wiki / unknown source: `pending`

## External Posting Gate

Discord 告知投稿、X 投稿、ブログ投稿などは、次を満たすまで executor を有効化しない。

- `external_posting=approval_required` を維持する。
- Announcement draft の fact check が完了している。
- source visibility が public または外部公開可能である。
- 投稿先 channel/account を user が明示する。
- 投稿 preview を approver が確認する。

## Minecraft Executor Gate

Minecraft 書き込み操作は、次を満たすまで dry-run のみとする。

- isolated executor の実行環境を構築する。
- rootless container / filesystem allowlist / network allowlist を設定する。
- stdout/stderr size limit と timeout を設定する。
- backup path と restore 手順を確認する。
- high risk は admin approval、critical は two-person approval または disabled にする。
- 実サーバーでなく staging server で rollback を実演する。

## Image Search Gate

画像検索は、次が必要。

- 画像 attachment / Drive image / X image / blog image の indexing pipeline
- caption/OCR/feature vector provider
- 投稿者、出典 URL、投稿日時、媒体、権限 metadata
- 検索結果が候補提示であり、再利用可否を判定しないことを確認する safety eval

## Member Search Gate

メンバー検索は、次が必要。

- `member_profiles` repository
- Discord role/profile import policy
- 本人確認が必要な属性の分類
- organizer/admin 限定 ACL
- 外部公開回答からの個人情報 redaction
- member privacy redaction eval
