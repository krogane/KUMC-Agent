# CLI 経由の `index update` ロジック

このドキュメントは、CLI から `index update` を実行したときに、プロジェクト内で何がどの順番で起きるかを説明するものです。

対象コマンドは次です。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index update
```

## 全体像

`index update` は、通常運用で検索 index を安全に更新するためのコマンドです。

CLI からの手動 index 更新入口は `index update` に一本化されています。このコマンドは次のような運用上の安全機構を含みます。

- 同時実行を避ける lock
- source の差分取り込み
- `data/index/staging/{run_id}` への一時構築
- 品質確認
- `data/index` への publish
- 失敗時の rollback
- 実行結果 `IndexingRun` の保存
- CLI 向け JSON payload の出力

大きく見ると、処理は次の順番で進みます。

1. CLI が `index update` の引数を読む
2. `AutoIndexUpdateRequest` を作る
3. `AutoIndexUpdateUsecase.execute()` を呼ぶ
4. schedule 条件と lock を確認する
5. source の差分を ingestion repository に取り込む
6. 変更がなければ skip する
7. staging directory に検索 index を作る
8. member profile / task event index も必要なら作る
9. staging の品質を確認する
10. staging の成果物を `data/index` に publish する
11. 必要なら event 差分抽出を実行する
12. `IndexingRun` を保存し、CLI が JSON を出力する

## CLI 入口

CLI のサブコマンド定義は `src/kumc_agent/cli.py` にあります。

```python
index_parser = subparsers.add_parser("index", help="Index operations")
index_sub = index_parser.add_subparsers(dest="index_command", required=True)
update_parser = index_sub.add_parser("update")
update_parser.add_argument("--no-refresh-sources", action="store_true")
update_parser.add_argument("--full-rebuild", action="store_true")
update_parser.add_argument("--stage", action="append", dest="stages", default=None)
```

`index update` で使える主なオプションは次です。

| オプション | 意味 |
| --- | --- |
| `--no-refresh-sources` | 外部 source の更新取り込みをスキップする |
| `--full-rebuild` | 差分有無に関係なく、強制的に再構築する |
| `--stage <name>` | 指定した chunk stage だけを実行する。複数回指定可能 |

実行時は、CLI が `AutoIndexUpdateRequest` を作って `context.auto_index_update.execute()` を呼びます。

```python
auto_result = context.auto_index_update.execute(
    AutoIndexUpdateRequest(
        trigger="manual",
        refresh_sources=not args.no_refresh_sources,
        force=bool(args.full_rebuild),
        full_rebuild=bool(args.full_rebuild),
        stage_selection=tuple(args.stages or ()) or None,
    )
)
```

通常実行では `refresh_sources=True` です。`--no-refresh-sources` を付けた場合だけ `False` になります。

`--full-rebuild` は `force=True` と `full_rebuild=True` の両方に反映されます。つまり、source 差分検出側にも「強制」、index 構築側にも「全体作り直し」として伝わります。

## RuntimeContext で組み立てられる部品

CLI は `RuntimeContext` を通して usecase を呼びます。

`src/kumc_agent/runtime/container.py` では、`AutoIndexUpdateUsecase` に主に次の部品が渡されます。

| 部品 | 役割 |
| --- | --- |
| `config` | directory、scheduler、品質確認、連携先などの設定 |
| `build_usecase` | staging directory に通常の本文検索 index を作る |
| `operations` | `IndexingRun` や通知用 `ActionRun` を保存する |
| `ingestion_service` | 外部 source から差分を取り込み、repository に保存する |
| `member_profile_builder` | Discord member profile の検索 index を作る |
| `task_event_indexer` | task / event 正本から検索 index を作る |
| `task_delta_extractor` | 変更された source から task 候補・変更候補を抽出する |
| `event_delta_extractor` | 変更された source から event 候補・変更候補を抽出する |
| `event_delta_chunk_source` | task/event 差分抽出に使う active chunks を読む |

初学者向けに言うと、CLI 自体は重い処理を持っていません。CLI は入力を request に変換し、実処理は `AutoIndexUpdateUsecase` とその下のサービス群に任せています。

## `AutoIndexUpdateRequest` の意味

`src/kumc_agent/usecases/indexing/auto_update.py` に request の定義があります。

主なフィールドは次です。

| フィールド | CLI 実行時の値 | 意味 |
| --- | --- | --- |
| `trigger` | `manual` | 何をきっかけに実行されたか |
| `source_filter` | 空 tuple | 対象 source の絞り込み。CLI の `index update` では指定されない |
| `force` | `--full-rebuild` と同じ | 差分取り込みを強制実行するか |
| `full_rebuild` | `--full-rebuild` と同じ | index 構築側で全体再構築するか |
| `quality_check_enabled` | 既定で `True` | publish 前の品質確認を行うか |
| `refresh_sources` | `--no-refresh-sources` の逆 | 外部 source を取り込むか |
| `stage_selection` | `--stage` の値 | 実行する chunk stage の指定 |
| `scheduled_at` | `None` | schedule / automation 実行時の予定時刻 |

CLI 経由の手動実行では `trigger="manual"` なので、schedule 時刻の gate は基本的に関係しません。

## run id と初期 metadata

`AutoIndexUpdateUsecase.execute()` の最初で run id が作られます。

形式はおおむね次です。

```text
auto-index-{UTC timestamp}-{trigger}
```

例:

```text
auto-index-20260429T101530123456Z-manual
```

同時に、`IndexingRun.metadata` に入れる基本情報も作ります。

```python
metadata = {
    "trigger": request.trigger,
    "source_filter": list(request.source_filter),
    "force": request.force,
    "full_rebuild": request.full_rebuild,
    "quality_check_enabled": request.quality_check_enabled,
}
```

この metadata は、成功・失敗・skip のどの結果でも、後から実行内容を追跡するための基本情報になります。

## schedule gate

まず `_schedule_skip_reason()` で、定期実行として動かしてよいタイミングかを確認します。

ただし、CLI の `index update` は `trigger="manual"` なので、ここでは通常 skip されません。

この gate が効くのは主に `trigger` が `schedule` または `automation` の場合です。その場合は `configs/main/scheduler.yaml` の次の設定を見ます。

| 設定 | 意味 |
| --- | --- |
| `scheduler.auto_index_enabled` | 自動更新が有効か |
| `scheduler.auto_index_weekdays` | 実行してよい曜日 |
| `scheduler.auto_index_time` | 実行してよい時刻 |
| `scheduler.auto_index_timezone` | 時刻判定に使う timezone |

条件を満たさない場合は `IndexingRun(status="skipped")` として保存され、処理はそれ以上進みません。

## lock による同時実行防止

次に `build_indexing_lock(config)` で lock を作り、`lock.acquire(run_id=run_id)` を呼びます。

目的は、複数の index 更新が同時に走って `data/index` を壊さないようにすることです。

lock の実装は設定に応じて複数あります。

| lock | 使われる場面 |
| --- | --- |
| Postgres lock | Postgres が使える場合 |
| Redis lock | Redis が使える場合 |
| file lock | fallback。主に `data/index/.auto_index.lock` |

lock を取得できなかった場合は、run は `status="skipped"` になります。metadata には lock の理由や backend 情報が入ります。

lock を取得できた場合は、まず `IndexingRun(status="running")` を保存し、本処理に進みます。処理の最後では `finally` で必ず `lock.release()` が呼ばれます。

## source の差分取り込み

次に `_ingest_sources(request)` が呼ばれます。

ここで行うのは、外部 source の最新状態を ingestion repository に取り込む処理です。本文検索 index を直接作るのではなく、まず検索 index の元になる正規化済みデータを repository に保存します。

`request.refresh_sources=False` の場合、または `ingestion_service` がない場合は、取り込みは行われず空の結果になります。

取り込み対象は `source_filter` で絞り込めます。ただし、CLI の `index update` では `source_filter` は空なので、基本的に利用可能な source 全体が対象です。

実際には次のように `IngestionService.backfill_many()` が呼ばれます。

```python
self._ingestion_service.backfill_many(
    source_kinds=source_kinds,
    scope=BackfillScope(force=bool(request.force or request.full_rebuild)),
)
```

`--full-rebuild` が指定されている場合は `BackfillScope(force=True)` になり、差分検出側にも強制更新として伝わります。

## ingestion 結果の集計

`backfill_many()` の戻り値は source ごとの `IngestionResult` です。

`AutoIndexUpdateUsecase` はこれを集計して、次の件数を作ります。

| 件数 | 意味 |
| --- | --- |
| `seen` | 確認した source item 数 |
| `changed` | 新規・更新・権限変更など、index に反映すべき件数 |
| `skipped` | 差分なしとしてスキップした件数 |
| `deleted` | 削除または検索除外になった件数 |

source 別の結果は `metadata.source_results` に入ります。

一部 source が失敗した場合は、`metadata.degraded=True` と `metadata.failed_sources` が入ります。全 source が失敗した場合は `status="failed"` になり、staging build には進みません。

## 変更なし skip

次の条件をもとに、index を作り直す必要があるかを判定します。

```python
has_changes = bool(
    changed
    or deleted
    or request.force
    or request.full_rebuild
    or member_profile_refresh_planned
    or task_event_refresh_planned
)
```

つまり、本文 source に変更がなくても、member profile や task / event index の更新予定があれば更新は続行されます。

逆に、次の条件をすべて満たす場合は publish せず skip します。

- `refresh_sources=True`
- `ingestion_service` がある
- 差分がない
- 強制実行ではない
- member profile / task event の更新予定もない

この場合、成功扱いの rebuild ではなく、`status="skipped"`、`metadata.reason="no_source_changes"` として保存されます。

source 失敗があり、かつ publish できる変更がない場合は `status="failed"`、`metadata.reason="source_failed_without_publish"` になります。

## staging directory の作成

更新が必要な場合、まず publish 用の staging directory を決めます。

```python
staging_dir = self._publisher.staging_dir(run_id)
```

実体は次のようなパスです。

```text
data/index/staging/{run_id}
```

重要なのは、この時点ではまだ `data/index` 直下の公開中 index を直接書き換えないことです。新しい index は staging に作り、品質確認に通った後で公開します。

## 本文検索 index の staging build

staging directory に本文検索 index を作るため、`BuildIndexUsecase.execute()` を呼びます。

```python
BuildIndexRequest(
    refresh_sources=request.refresh_sources and self._ingestion_service is None,
    full_rebuild=request.full_rebuild,
    stage_selection=request.stage_selection,
    index_dir=staging_dir,
    prefer_ingestion_repository=self._ingestion_service is not None,
)
```

CLI からはこの staging build を直接起動せず、必ず `index update` 経由で実行します。

| 項目 | `index update` の staging build |
| --- | --- |
| 出力先 | `data/index/staging/{run_id}` |
| source refresh | 通常は `AutoIndexUpdateUsecase` 側で済ませるため、`BuildIndexUsecase` 側では再実行しない |
| 入力 chunk | `ingestion_repository` の active chunks を優先 |
| stage 指定 | CLI の `--stage` があれば引き継ぐ |

`prefer_ingestion_repository=True` の場合、`IndexingService.build()` は repository に保存済みの active chunks を優先して Dense / Sparse index を構築します。

これは `index update` の重要な考え方です。外部 source からの差分取り込みと、検索 index の構築を分けています。

## stage result の保存

本文検索 index の build が終わると、結果は `metadata.stage_results.index` に保存されます。

主な内容は次です。

| key | 意味 |
| --- | --- |
| `loaded_sources` | loader で読み込んだ source 数 |
| `documents` | build 対象 document 数 |
| `chunks` | build 対象 chunk 数 |
| `staging_dir` | 成果物を書いた staging directory |

`IndexBuildResult.stage_results` がある場合は、それも `metadata.stage_results` に追加されます。

## member profile index の更新

member profile の更新が必要な場合は、本文検索 index と同じ staging directory に対して `_rebuild_member_profiles()` が呼ばれます。

更新対象の Guild は `security.discord_member_profile_guild_ids` で決まります。未設定時は後方互換として `security.discord_guild_allow_list` が使われます。

各 Guild について `member_profile_builder.rebuild_guild()` が呼ばれ、その結果は次に入ります。

- `metadata.source_results`
- `metadata.stage_results.member_profiles`

どれかの Guild で失敗した場合は `status="failed"` になり、publish には進みません。

## task / event index の更新

task / event index の更新が必要な場合は、`task_event_indexer.rebuild(index_dir=staging_dir)` が呼ばれます。

これは workflow repository にある task / event の正本データを検索用 index に投影する処理です。

結果は次に入ります。

- `metadata.source_results`
- `metadata.stage_results.task_event`

失敗した場合は `status="failed"` になり、publish には進みません。

## 品質確認

`quality_check_enabled=True` の場合、publish 前に `IndexQualitySmokeChecker` で staging の品質を確認します。

主なチェックは次です。

| チェック | 失敗理由の例 |
| --- | --- |
| 必須 artifact が存在するか | `missing_artifact:dense_vectors.npy` |
| Dense vectors を load できるか | `dense_index_load_failed` |
| Dense vectors と chunks の件数が合うか | `dense_chunk_vector_mismatch` |
| Sparse index を load できるか | `sparse_index_load_failed` |
| chunk が 0 件ではないか | `chunk_count_zero` |
| 前回と比べて chunk 数が急減していないか | `chunk_count_ratio_below_threshold` |
| 削除・隔離・権限喪失 chunk が混入していないか | `disallowed_chunk_status_present` |
| smoke query が最低限 hit するか | `smoke_query_no_match` |
| feature index を load できるか | feature index 側の failure |

結果は `metadata.quality_check` に入ります。

品質確認に失敗した場合、新しい index は公開されません。run は `status="failed"` になり、失敗理由が `error` と `metadata.quality_check.critical_failures` に残ります。

## publish

品質確認に通ると、`IndexSnapshotPublisher.publish()` が呼ばれます。

publish の役割は、staging の成果物を検索 runtime が読む `data/index` 直下に反映することです。

処理は大きく次の順番です。

1. `data/index` がなければ作る
2. 既存の `data/index` 直下に index artifact があれば `data/index/previous/{snapshot_id}` に退避する
3. `data/index` 直下の古い artifact を削除する
4. staging directory の中身を `data/index` 直下にコピーする
5. `data/index/current.json` に今回 snapshot を記録する
6. `data/index/previous.json` に直前 snapshot を記録する
7. 古すぎる previous snapshot を削除する

publish 成功後、metadata には次が入ります。

| key | 意味 |
| --- | --- |
| `index_snapshot_id` | 今回 publish した snapshot id |
| `previous_snapshot_id` | 直前 snapshot id |
| `publish.current_pointer` | `current.json` のパス |
| `publish.previous_pointer` | `previous.json` のパス |

## publish 失敗と rollback

publish 中に例外が起きた場合、`rollback_to_latest_previous()` が呼ばれます。

rollback は `data/index/previous/{snapshot_id}` に退避していた直前成果物を `data/index` 直下に戻す処理です。

rollback が成功した場合、run は `status="rolled_back"` になります。rollback も失敗した場合は `status="failed"` になります。

結果は `metadata.rollback` に保存されます。

## workflow 差分抽出

publish に成功した後、必要なら `_run_workflow_delta_extraction()` が呼ばれます。

これは「更新された source の active chunks から、タスク/イベントの新規登録・変更・削除候補を抽出する」ための後続処理です。

実行条件は主に次です。

- `task_management.auto_extract_after_index_update=True` または `event_management.auto_extract_after_index_update=True`
- 成功した ingestion result の中に `changed` または `deleted` がある
- 対応する delta extractor と `event_delta_chunk_source` が設定されている

抽出に成功した場合は `metadata.workflow_extraction.task.status="succeeded"` または `metadata.workflow_extraction.event.status="succeeded"` になります。互換用に `metadata.task_delta_extraction` と `metadata.event_extraction` も同じ要約を保持します。

抽出に失敗しても、index publish 自体はすでに成功しているため、run 全体は失敗にしません。代わりに対応する抽出metadataへ `status="failed"` と `degraded=True` を残します。

## 実行結果の保存

最後に `_save_result()` が呼ばれます。

ここで行われることは次です。

1. 必要なら notification 用 `ActionRun` を保存する
2. `operations.save_indexing_run(run)` で `IndexingRun` を保存する
3. `AutoIndexUpdateResult` に変換して CLI に返す

失敗時や skip 時も、可能な限り `IndexingRun` として保存されます。これにより、後から「なぜ更新されなかったか」「どの source が失敗したか」「品質確認で何が落ちたか」を追えます。

## CLI の出力 payload

CLI は `AutoIndexUpdateResult.as_payload()` の結果を JSON で標準出力に出します。

トップレベルに出る安定フィールドは次です。

| フィールド | 意味 |
| --- | --- |
| `status` | `succeeded`, `failed`, `skipped`, `rolled_back` など |
| `run_id` | 実行 ID |
| `seen` | 確認した件数 |
| `changed` | 変更件数 |
| `skipped` | 差分なし件数 |
| `deleted` | 削除・検索除外件数 |
| `metadata` | 詳細情報 |

例:

```json
{
  "status": "succeeded",
  "run_id": "auto-index-20260429T101530123456Z-manual",
  "seen": 120,
  "changed": 3,
  "skipped": 117,
  "deleted": 0,
  "metadata": {
    "trigger": "manual",
    "source_results": [],
    "stage_results": {},
    "quality_check": {},
    "index_snapshot_id": "auto-index-20260429T101530123456Z-manual"
  }
}
```

このプロジェクトでは、CLI や外部連携向け payload のトップレベルには、利用者や連携先が主結果として扱う安定フィールドだけを置く方針です。

そのため、品質確認、source 別結果、snapshot、通知、rollback、event 抽出結果のような詳細情報は `metadata` 配下に入ります。

## CLI 入口の一本化

以前の CLI には直接 build 用の入口がありましたが、現在は `index update` だけを公開しています。

| 項目 | 現在の扱い |
| --- | --- |
| CLI 入口 | `index update` |
| 呼び出す usecase | `AutoIndexUpdateUsecase` |
| lock | あり |
| `IndexingRun` 保存 | あり |
| 変更なし skip | あり |
| staging | あり |
| 品質確認 | あり |
| publish / rollback | あり |
| ingestion repository | active chunks を優先 |
| CLI 出力 | `status`, `run_id`, 件数, `metadata` |

内部の staging build では `BuildIndexUsecase` を使いますが、これは `AutoIndexUpdateUsecase` から呼ばれる実装部品です。CLI 利用者は `index update` を使います。

## status 別の代表的な流れ

### `succeeded`

代表的な成功パターンです。

1. lock を取得する
2. source 差分を取り込む
3. 変更がある
4. staging に index を作る
5. 品質確認に通る
6. `data/index` に publish する
7. `IndexingRun(status="succeeded")` を保存する

### `skipped`

代表的な skip パターンは次です。

- schedule / automation の実行条件を満たさない
- lock を取得できない
- source を確認したが変更がない

skip でも `IndexingRun` は保存されます。理由は `metadata.reason` に入ります。

### `failed`

代表的な失敗パターンは次です。

- 全 source の取り込みに失敗した
- 変更なしだが source 失敗があり publish できなかった
- member profile index の rebuild に失敗した
- task / event index の rebuild に失敗した
- 品質確認に失敗した
- 想定外の例外が起きた

失敗時は `metadata.notification` が付くことがあり、通知用の `ActionRun` も保存されます。

### `rolled_back`

publish 中に失敗し、直前 snapshot への復元に成功した場合です。

この場合、新しい index の公開は失敗していますが、検索 runtime は直前の index に戻されます。rollback の詳細は `metadata.rollback` に入ります。

## 主要ファイル

| ファイル | 役割 |
| --- | --- |
| `src/kumc_agent/cli.py` | CLI 引数定義と `AutoIndexUpdateRequest` 作成 |
| `src/kumc_agent/runtime/container.py` | `AutoIndexUpdateUsecase` に渡す依存部品の組み立て |
| `src/kumc_agent/usecases/indexing/auto_update.py` | `index update` の中心ロジック |
| `src/kumc_agent/usecases/indexing/build.py` | staging build で呼ばれる通常 index 構築 usecase |
| `src/kumc_agent/features/indexing/service.py` | document / chunk / Dense / Sparse index の構築 |
| `src/kumc_agent/features/indexing/lock.py` | 同時実行防止 lock |
| `src/kumc_agent/features/indexing/quality.py` | publish 前の品質確認 |
| `src/kumc_agent/features/indexing/snapshot.py` | staging publish と rollback |
| `src/kumc_agent/features/ingestion/service.py` | source 差分取り込み |
| `src/kumc_agent/features/indexing/task_event.py` | task / event index 構築 |

## 読む順番のおすすめ

初めて読む場合は、次の順番がおすすめです。

1. `src/kumc_agent/cli.py` の `args.command == "index"` 分岐
2. `src/kumc_agent/usecases/indexing/auto_update.py` の `AutoIndexUpdateUsecase.execute()`
3. `src/kumc_agent/features/indexing/snapshot.py` の `IndexSnapshotPublisher.publish()`
4. `src/kumc_agent/features/indexing/quality.py` の `IndexQualitySmokeChecker.check()`
5. `docs/explanation/index-build-cli-logic.md`

特に `AutoIndexUpdateUsecase.execute()` は長いですが、処理は大きく「事前確認」「差分取り込み」「staging build」「品質確認」「publish」「結果保存」に分けて読むと理解しやすくなります。
