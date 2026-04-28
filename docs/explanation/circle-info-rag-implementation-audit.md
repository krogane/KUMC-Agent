# サークル情報RAG 実装後再調査結果

調査日: 2026-04-28

参照仕様:

- `docs/design/circle-info-rag.md`
- `docs/design/kumc-agent.md`

## 結論

前回調査で確認した仕様との差分は実装済み。現時点のサークル情報RAGは、コード上確認できる範囲では仕様通りに利用できる状態である。

主要入口である Discord `/ask`、HTTP `/ask`、CLI `chat` / `ask` / `tool rag` は、サークル情報RAGについて `ChatAnswerUsecase` / `RagService` 経路を使う。これにより、資料検索、通常検索、合成クエリ、サークル基本情報、回答JSON parse、回答フィルタリング、出典出力が同一実装に揃った。

運用上は、外部API認証、ネットワーク、実データの有無、Gemini/Drive/Discord等の接続状態に依存する。ただし、実装差分として残っていたブロッカーは解消済みである。

## 実装確認

| 仕様項目 | 実装後の状態 | 主な実装箇所 |
| --- | --- | --- |
| Discord / HTTP / CLI 入口 | `circle_rag` は `ChatAnswerUsecase` / `RagService` へ接続。`minecraft_wiki_rag` は既存専用経路を維持 | `usecases/integrated_input/entry.py`, `runtime/container.py`, `apps/bot/app.py`, `apps/api/app.py`, `cli.py` |
| Dense + Sparse RRF | DenseとSparseを並列実行してもSparse結果を保持し、RRF統合する | `features/rag/components/retrieval.py` |
| ingestion由来chunkの権限制御 | `access_scope` を優先し、`google_drive` / `discord` source_kindも保護対象として扱う | `features/rag/access.py` |
| 自動更新時のindex正本 | ingestion active chunksから第1/第2/Sparse/Summary、Dense/BM25、keyword、material catalogを同じ正本で構築 | `features/indexing/service.py` |
| Summary chunkのDense投入 | circle系summary chunkもDense/BM25対象へ含める | `features/indexing/service.py` |
| 資料名index | ingestion正本のmaterial catalogから資料名keyword indexを生成。Discord/Xは資料名検索対象外 | `features/indexing/service.py`, `features/rag/service.py` |
| routing task設定 | `needs_additional_query` を実行taskに含め、false時は追加クエリを使わない | `features/rag/components/routing.py` |
| fast mode | `fast_mode=true` の場合は資料検索routeをスキップし、通常検索で低負荷回答する | `features/rag/service.py` |
| answer filter fallback | filter LLM失敗・parse不能時は安全側で拒否する | `features/rag/components/answer_filter.py` |
| 資料検索出典 | `force_all_sources=True` の資料検索では `source_max_count` で打ち切らず全候補出典を付与する | `features/rag/components/generation.py` |
| CLI payload | RAG payloadの主結果はtop-level、診断情報は `metadata` 配下、context/raw/promptは除外 | `cli.py`, `features/foundation/payload_sanitizer.py` |

## 差分再調査

| 前回差分 | 再調査結果 |
| --- | --- |
| Discord / HTTP `/ask` / 通常CLI `chat` が `RagService` を通らない | 解消。`IntegratedInputUsecase` の `circle_rag` が `chat_answer_service.execute(ChatRequest(...))` を呼ぶ |
| Dense + Sparse 同時検索時にSparse結果が破棄される | 解消。例外時のみ `sparse_hits=[]` にし、通常成功時はRRFへ渡す |
| ingestion repository由来chunkのアクセス制御が不整合 | 解消。`access_scope` dictを優先判定し、`google_drive` / `discord` もprotected sourceとして扱う |
| 自動更新で ingestion repository chunkとlegacy成果物が混在する | 解消。repository chunkからstage chunks、keyword index、material catalogを再生成する |
| 統合入口の `AskService` は仕様上のサークル情報RAGではない | 解消。`circle_rag` は `AskService` ではなく `RagService` を使う。`AskService` はMinecraft Wiki等の既存経路で残す |
| `needs_additional_query` 設定が実行されない | 解消。routing task一覧とbool taskに追加 |
| material routeは fast modeでも資料検索をskipしない | 解消。effective fast mode時に `material_names` / `additional_queries` を空にして通常検索へ進む |
| 回答filter fallbackが allow | 解消。fallback時は `filter_fallback_refuse` で拒否する |
| 資料検索出典が `source_max_count` で打ち切られる | 解消。`force_all_sources=True` の場合は上限なしで出典化する |

## 主要入口ごとの利用可否

| 入口 | 実装後の状態 | 仕様利用 |
| --- | --- | --- |
| `python -m kumc_agent.cli tool rag ...` | `ChatAnswerUsecase` / `RagService` を通る | 可 |
| `python -m kumc_agent.cli chat ...` | `IntegratedInputUsecase` 経由で `RagService` を通る | 可 |
| `python -m kumc_agent.cli ask ...` | runtime contextの `IntegratedInputUsecase` 経由で `RagService` を通る | 可 |
| Discord `/ask` | runtime contextの `IntegratedInputUsecase` 経由で `RagService` を通る | 可 |
| HTTP `/ask` | runtime contextの `IntegratedInputUsecase` 経由で `RagService` を通る | 可 |
| 評価 `EvaluateRagasUsecase` | `ChatAnswerUsecase` を通る | 可 |

## 検証

実行した検証:

```bash
PYTHONPATH=src app/.venv/bin/python -m unittest tests.unit.test_rag_retrieval_rrf tests.unit.test_rag_access_filter tests.unit.test_answer_filter tests.unit.test_generation_component tests.unit.test_integrated_input tests.unit.test_query_router tests.unit.test_material_search_matching tests.unit.test_material_name_keyword_index tests.unit.test_indexing_repository_artifacts tests.unit.test_cli_tool_rag
PYTHONPATH=src app/.venv/bin/python -m unittest discover tests/unit
PYTHONPATH=src app/.venv/bin/python -m unittest tests.integration.test_chat_index_eval
git diff --check
```

結果:

- targeted unit: 42 tests / OK
- full unit discovery: 224 tests / OK
- integration `test_chat_index_eval`: 1 test / OK
- `git diff --check`: OK

補足:

- full unit中、既存reranker系テストでHugging Faceへの名前解決リトライが発生したが、テスト自体はfallbackして成功した。
- integration中、外部Gemini接続はネットワーク制限で失敗ログが出たが、既存fallbackによりテストは成功した。
