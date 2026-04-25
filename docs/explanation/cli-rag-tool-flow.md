# CLI経由のRAGツール出力が作られるまで

この資料は、`kumc-agent tool rag` を実行したときに、入力された質問がどの経路を通って最終的なJSON出力になるかを説明します。
プロジェクト全体をまだ把握していない人でも追えるように、処理の入口から順に見ます。

対象にしている主なコードは次のファイルです。

- `src/kumc_agent/cli.py`
- `src/kumc_agent/runtime/container.py`
- `src/kumc_agent/usecases/chat/answer.py`
- `src/kumc_agent/features/rag/service.py`
- `src/kumc_agent/features/rag/components/routing.py`
- `src/kumc_agent/features/rag/components/retrieval.py`
- `src/kumc_agent/features/rag/components/generation.py`

## 1. CLIの入力

RAGツールは次のように実行します。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli tool rag --query "KUMCの活動内容は？"
```

`--query` は複数回指定できます。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli tool rag \
  --query "KUMCの活動内容は？" \
  --query "次回の活動予定は？"
```

`src/kumc_agent/cli.py` の `_build_parser()` で、`tool rag` サブコマンドと次の引数が定義されています。

- `--query`: 質問文。`action="append"` なので複数回指定できる
- `--question-author`: 質問者情報。ルーティング用に渡される
- `--history-scope`: 履歴のスコープ。ただしこのCLI経路では履歴自体は無効化される
- `--force-fast-mode`: 高速モードを強制する

## 2. 実行前にRuntimeContextを作る

`main()` は一部のapp系コマンドを先に処理したあと、通常のCLI処理では `build_runtime_context()` を呼びます。

`RuntimeContext` は、RAG実行に必要な部品をまとめた入れ物です。
ここで設定ファイルや環境変数から設定を読み、次のような実体が組み立てられます。

- embedding: Geminiまたはローカルembedder
- dense index: FAISS系の検索index
- sparse index: Sudachi BM25検索index
- reranker: cross encoder reranker。有効な場合のみ
- prompt repository: `assets/prompts` 配下のプロンプト読み込み
- RAG回答用LLM、No-RAG回答用LLM
- `QueryRouter`
- `RetrievalComponent`
- `GenerationComponent`
- `RagService`
- `ChatAnswerUsecase`

CLI本体はRAGの詳細処理を直接持ちません。
`context.chat_answer` に入っている `ChatAnswerUsecase` が、RAG処理への窓口になります。

## 3. `tool rag` 分岐でChatRequestを作る

`args.command == "tool"` かつ `args.tool_command == "rag"` の場合、CLIは `args.query` をリストとして扱います。
質問が1件だけなら1回、複数なら質問ごとに同じ処理を繰り返します。

このとき作られる `ChatRequest` では、ツール連携用に次の指定が入ります。

- `query`: 入力された質問文
- `question_author`: CLI引数の値
- `history_scope`: CLI引数の値
- `force_fast_mode`: `--force-fast-mode` の有無
- `disable_history=True`
- `routing_history_override=[]`
- `generation_history_override=[]`
- `force_disable_additional_memory=True`

重要なのは、履歴と追加メモリを明示的に切っている点です。
通常のチャット経路では過去の会話をプロンプトへ入れる可能性がありますが、`tool rag` は外部ツールから安定して呼び出せるよう、質問単体を処理する経路になっています。

## 4. UsecaseはRagServiceへ渡すだけ

`ChatAnswerUsecase.execute()` は薄い層です。
`ChatRequest` の各フィールドを `RagService.answer()` に渡し、返ってきた `Answer` をそのまま返します。

ここから先がRAGの本体です。

## 5. RagServiceで質問を正規化し、ルーティングする

`RagService.answer()` はまず `query.strip()` で空白を削ります。
空文字なら検索やLLM呼び出しはせず、`route="none"` の空回答を返します。

空でなければ `QueryRouter.route()` を呼び、質問をどう扱うかを判定します。
現在の現行実装のルーティング結果は主に次のフィールドです。

- `recency_mode`: `off` / `soft` / `hard`
- `material_names`: 特定資料名が抽出された場合の資料名リスト
- `use_additional_memory`: 追加履歴が必要か
- `additional_queries`: 検索に足す追加クエリ
- `include_capabilities_info`: 回答プロンプトへ機能説明を入れるか

`tool rag` では `force_disable_additional_memory=True` が渡されるため、ルーティングが追加メモリを必要と判断しても無効化されます。
また `--force-fast-mode` が指定されると、`material_names` と `additional_queries` は空に置き換えられます。

## 6. QueryRouterの内部

`QueryRouter` はルーティングが有効なら、複数の小さな判定タスクをLLMで並列実行します。
対象タスクは次の4つです。

- `use_additional_memory`
- `additional_queries`
- `material_names`
- `recency_mode`

それぞれのタスクはGeminiで実行します。
GeminiにはJSON MIMEを指定して、JSONを返しやすくしています。

ルーティングLLMの出力が壊れている、またはAPIエラーが起きた場合はリトライします。
最終的に失敗した場合は安全側のデフォルトとして、`recency_mode="off"`、資料名なし、追加クエリなし、追加メモリなしの判定になります。

## 7. 検索クエリを決める

ルーティング後、`RagService` は検索に進みます。

`material_names` が空でなければ資料名検索寄りの `material_search` 経路になります。
この場合は `material_catalog.json` などから資料名に合う資料を探し、DiscordメッセージやX投稿のような会話系sourceを除外して、対象資料に絞ったchunkを作ろうとします。
資料名に合うものが見つからない場合は、通常検索へフォールバックします。

`material_names` が空なら通常のRAG検索です。
追加クエリがあり、fast modeでなければ、元の質問と追加クエリを並列で検索し、chunk IDで重複排除して統合します。
`tool rag` では通常、履歴は使いませんが、ルーティングが生成した追加検索語は使われます。

## 8. RetrievalComponentでdense検索とsparse検索を統合する

通常検索は `RetrievalComponent.retrieve()` に渡ります。
ここでは設定値に応じて次の検索を行います。

- dense検索: 質問をembeddingし、FAISS系indexから近いchunkを探す
- sparse検索: Sudachi BM25やkeyword indexでキーワードに合うchunkを探す

denseとsparseの両方が有効な場合、2つは並列に実行されます。
片方が失敗してもログを出して空扱いにし、もう片方の結果で続行します。

検索結果はdense側の順位とsparse側の順位をRRFで融合し、`Chunk` のリストとして `RagService` に戻ります。

## 9. RagServiceでrerank、recency、MMR、親chunk追加を行う

検索直後のchunkはそのまま回答生成に渡されません。
`RagService._rank_and_select_chunks()` で次の処理が入ります。

- rerankerが有効でfast modeでなければ、cross encoderで質問とchunkの関連度を再スコアリングする
- `recency_mode` が `soft` または `hard` なら、chunk metadataの日付を使って新しさのスコアを混ぜる
- 同じ親chunkから取りすぎないように上限をかける
- rerank pool sizeで候補数を絞る
- fast modeでなければMMRで多様性を加味して並べ替える
- `top_k` 件に絞る
- 設定が有効なら、second recursive chunkの親にあたる本文・要約chunkを追加する

この段階で、回答プロンプトに入るcontextが決まります。

## 10. chunkがない場合はNo-RAG回答になる

検索・選別後にchunkが0件なら、`GenerationComponent.generate_no_rag()` が呼ばれます。
この経路では検索contextを使わず、質問、履歴欄、必要なら機能説明、出力形式プロンプトからLLM入力を作ります。

返り値は `Answer` で、主な中身は次の通りです。

- `text`: 回答本文
- `route`: `no_rag`
- `sources`: 空リスト
- `metadata.raw`: LLMの生出力
- `metadata.answer_payload_is_json`: JSONとして読めたか
- `metadata.llm_prompt`: 実際にLLMへ渡したsystem/user prompt

## 11. chunkがある場合はRAG回答になる

chunkが1件以上ある場合は、`GenerationComponent.generate_rag_answer()` が呼ばれます。
この関数は次の順でプロンプトを作ります。

1. chunkを `[1]`, `[2]` のような番号付きcontext文字列に整形する
2. Discord chunkの場合は行ごとに `[1-1]` のようなsub source番号を付ける
3. source種別ごとに、Drive path、channel名、ブログURL、日付などのmetadataをcontextに追加する
4. `assets/prompts/answer_rag.md` などの回答形式プロンプトを読み込む
5. 質問、履歴、サークル基本情報、context、機能説明、追加指示、出力形式を結合してuser promptにする
6. system promptとuser promptをLLMへ渡す

LLMにはJSON形式の回答が期待されています。
`GenerationComponent` はLLM出力から次を取り出します。

- `answer`: 回答本文
- `sources`: 回答で使ったsource番号

JSONとして読めない場合でも、`answer` フィールドだけを復元できる場合は復元します。
設定されたリトライ回数内で再試行し、最後までJSONとして読めない場合はbest effortの本文を使います。

## 12. source一覧を作り、回答本文へ出典を付ける

LLMが選んだsource番号は、`_sources_from_chunks()` で `Source` オブジェクトに変換されます。

sourceのlabel/uriはchunk metadataから作られます。
たとえば次のような情報が優先して使われます。

- DiscordメッセージURL
- X URL
- Hatena Blog URL
- Crafters Colony URL
- Notion URL
- Google Drive URL
- VC transcriptのラベル

通常のRAG回答では、選ばれたsourceだけが出典になります。
資料名検索の `material_search` 経路では `force_all_sources=True` になり、選別されたchunk全体が出典候補として扱われます。

`append_sources_to_response=True` がデフォルトなので、最終的な `Answer.text` には回答本文に加えて整形済みの出典表示も入ります。

## 13. RagServiceがmetadataを追加してAnswerを確定する

生成直後の `Answer` は `RagService._finalize_answer()` を通ります。
ここで次の処理が行われます。

- fast modeなら、回答本文の先頭にfast mode noticeを付ける
- `metadata.routing_decision` にルーティング結果を入れる
- `metadata.fast_mode` にfast modeかどうかを入れる
- `disable_history=False` の場合だけ会話履歴に記録する

`tool rag` では `disable_history=True` なので、ここでも履歴には保存されません。

## 14. CLI用payloadへ変換する

`RagService` から戻った `Answer` は、CLIの `_build_tool_rag_payload()` でJSON出力用の辞書に変換されます。

トップレベルに置かれるのは、外部ツールが主結果として扱いやすい安定フィールドだけです。

```json
{
  "answer": "回答本文...",
  "route": "rag",
  "sources": [
    {
      "id": "chunk-id:1",
      "label": "https://example.com/source",
      "uri": "https://example.com/source"
    }
  ],
  "metadata": {
    "raw": "...",
    "answer_payload_is_json": true,
    "llm_prompt": {
      "system_prompt": "...",
      "user_prompt": "..."
    },
    "routing_decision": {
      "recency_mode": "soft",
      "material_names": [],
      "include_capabilities_info": false,
      "use_additional_memory": false,
      "additional_queries": []
    },
    "fast_mode": false
  }
}
```

ここで `metadata.contexts` は削除されます。
`contexts` には検索で集めた本文断片が入り、サイズが大きくなりやすく、外部ツールの主結果としては扱いづらいためです。

この構造は、このリポジトリのpayload方針に沿っています。
つまり、`answer`、`route`、`sources` のような安定した主結果だけをトップレベルに置き、ルーティング判断、fast mode、LLM生出力、プロンプトなどの診断情報は `metadata` 配下に置きます。

## 15. 複数query指定時の出力

`--query` が1件の場合、上記payloadがそのまま標準出力へ出ます。

`--query` が複数件の場合は、質問ごとのpayloadに `query` フィールドを追加し、次の形で標準出力へ出ます。

```json
{
  "query_count": 2,
  "results": [
    {
      "query": "KUMCの活動内容は？",
      "answer": "...",
      "route": "rag",
      "sources": [],
      "metadata": {}
    },
    {
      "query": "次回の活動予定は？",
      "answer": "...",
      "route": "rag",
      "sources": [],
      "metadata": {}
    }
  ]
}
```

複数queryは一括でRAGに渡されるわけではありません。
CLIが1件ずつ `ChatAnswerUsecase` を呼び、結果を配列にまとめています。

## 16. 全体の流れ

```text
CLI引数
  |
  v
src/kumc_agent/cli.py
  - argparseで tool rag を認識
  - ChatRequestを作成
  |
  v
RuntimeContext.chat_answer
  |
  v
ChatAnswerUsecase.execute()
  |
  v
RagService.answer()
  - query正規化
  - QueryRouterでルーティング
  - RetrievalComponentでdense/sparse検索
  - rerank / recency / MMR / parent chunk追加
  - GenerationComponentで回答生成
  - routing_decisionなどをmetadataへ追加
  |
  v
Answer
  |
  v
_build_tool_rag_payload()
  - answer / route / sources / metadata に整形
  - metadata.contexts を除外
  |
  v
標準出力JSON
```

## 17. 通常のchatコマンドとの違い

`chat` コマンドは `ChatEntryUsecase` を通り、OpenClaw連携なども含む入口判定を使います。
出力も回答本文だけです。

一方、`tool rag` は `ChatAnswerUsecase` を直接呼びます。
そのため、OpenClaw入口判定を通さず、ローカルRAGの結果をJSONで返すツールブリッジとして動きます。

また、`tool rag` は履歴・追加メモリを切っているため、同じindex・設定・外部LLM応答であれば、通常チャットよりも入力質問に閉じた結果になりやすい経路です。
