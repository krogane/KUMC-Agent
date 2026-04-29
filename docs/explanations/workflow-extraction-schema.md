# タスク抽出/イベント抽出形式統一

## 概要

タスク抽出とイベント抽出のLLM出力、内部Result、WorkflowResponse metadata、自動index差分metadataを `workflow_extraction.v1` に統一した。

抽出器の外部向け主結果は従来通り `task_candidates`、`task_change_candidates`、`event_candidates`、`event_change_candidates` に分けて返す。診断情報、schema version、抽出件数、degraded理由、ignored itemは `metadata.extraction` 配下に置く。

## 統一schema

LLMは次のJSONだけを返す。

```json
{
  "schema_version": "workflow_extraction.v1",
  "new_items": [],
  "change_items": [],
  "ignored_items": [],
  "degraded": false
}
```

- `new_items`: 新規候補。`item_type` は `task` または `event`。
- `change_items`: 既存正本の変更・削除候補。`target_id`、`operation`、`after`、`reason` を持つ。
- `ignored_items`: 候補化しなかった理由。長い本文断片やsecretは入れない。
- `degraded`: LLM未設定、schema不正、根拠不足などで候補保存を止める状態。

互換性のため、実装は旧キー `tasks`、`new_events`、`events`、`event_changes` も読み取れる。ただしプロンプトとテストは新schemaを正とする。

## 保存ルール

- `item_type="task"` の `new_items` は `TaskCandidate` に保存する。
- `item_type="task"` の `change_items` は `TaskChangeCandidate` に保存する。
- `item_type="event"` の `new_items` は `EventCandidate` に保存する。
- `item_type="event"` の `change_items` は `EventChangeCandidate` に保存する。
- change itemは既存Task/Eventを一意に特定できる場合だけ保存する。特定できない場合は候補化せず `ignored_items` として扱う。

## 自動差分metadata

`auto_index_update` 後の抽出結果は `metadata.workflow_extraction` に統一して保存する。

```json
{
  "workflow_extraction": {
    "task": {
      "status": "succeeded",
      "candidate_count": 1,
      "change_candidate_count": 0,
      "metadata": {"schema_version": "workflow_extraction.v1"}
    },
    "event": {
      "status": "succeeded",
      "candidate_count": 1,
      "change_candidate_count": 1,
      "metadata": {"schema_version": "workflow_extraction.v1"}
    }
  }
}
```

既存連携の互換性のため、`metadata.task_delta_extraction` と `metadata.event_extraction` も同じ要約を保持する。
