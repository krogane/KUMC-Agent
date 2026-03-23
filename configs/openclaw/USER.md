# KUMC OpenClaw User Contract

## 入力

入力は次のいずれかとして扱ってください。

1. KUMC ラッパー形式（JSON文字列）
   例: `{"kind":"kumc_user_query","query":"...","history_scope":"guild:...","user_context":{"question_author":"..."}}`
2. 生テキスト（Discord など）

## 通常質問フロー

1. `query` を決定する。
2. 可能なら `question_author` と `history_scope` を埋める（未設定時は空文字 / `default`）。
3. 検索クエリ `rag_query` を作成し、次を実行する。
   `PYTHONPATH="${KUMC_AGENT_PROJECT_SRC:-src}" python -m kumc_agent.cli tool rag --query "<rag_query>" --question-author "<question_author>" --history-scope "<history_scope>"`
4. `query` が `fast ` で始まる場合は `--force-fast-mode` を付与する（`fast` は取り除いて実行）。
5. RAG ツール結果をもとに、次を選ぶ。
   - そのまま最終回答にする。
   - 追加検索（新しい `rag_query` で再実行）や追加推論を行って最終回答にする。
6. 最終回答 JSON では `answer` を `text` に引き継ぐ。

## 出力 JSON スキーマ（必須）

```json
{
  "text": "string",
  "sources": [{"id": "string", "label": "string", "uri": "string"}],
  "fast_mode": false,
  "metadata": {}
}
```

- JSON 以外の文字を混在させないでください。
- エラー時は `metadata.error` に理由を入れてください。
- `route` / `routing_decision` は出力しないでください。
- RAG ツールを使った場合は、`metadata.rag_query` に最後に使った検索クエリを入れてください。
- 複数回検索した場合は、`metadata.rag_iterations` に実行回数を入れてください。
