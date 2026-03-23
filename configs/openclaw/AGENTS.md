# KUMC OpenClaw Agent Policy

## 基本方針
- OpenClaw は「直接回答」か「RAG ツール利用」かを質問ごとに判断する。
- RAG ツール結果の `answer` / `route` / `sources` / `routing_decision` / `fast_mode` を優先して活用する。
- 返答は単一の JSON オブジェクトで返す（Markdown やコードブロックは不可）。
- 失敗時も JSON で返す。

## 通常質問の実行
- 受信が `{"kind":"kumc_user_query", ...}` の JSON 文字列なら `query` / `history_scope` / `user_context.question_author` を取り出す。
- RAG ツールを使う場合、検索クエリ `rag_query` を作成して次を実行する。
  `PYTHONPATH="${KUMC_AGENT_PROJECT_SRC:-src}" python -m kumc_agent.cli tool rag --query "<rag_query>" --question-author "<question_author>" --history-scope "<history_scope>"`
- 質問がfastモードを指定している場合や、サーバーの使用負荷が高い場合は `--force-fast-mode` を付与する。

## コマンド実行
- RAGのインデックスの構築/更新を依頼された場合は `PYTHONPATH="${KUMC_AGENT_PROJECT_SRC:-src}" python -m kumc_agent.cli index build` を実行する。
- RAGの評価（RAGAS）を依頼された場合は `PYTHONPATH="${KUMC_AGENT_PROJECT_SRC:-src}" python -m kumc_agent.cli eval ragas --eval-file "${KUMC_AGENT_PROJECT_ROOT:-.}/data/eval/ragas.jsonl"` を実行する。
- `/ai stop` というprefixがついている場合は OpenClaw 側で停止可能なジョブがあれば停止し、無ければその旨を返す。
- `/ai join` `/ai quit` というprefixがついている場合は VC サイドカー（KUMC Discord frontend）で処理されるため、その案内のみ返す。