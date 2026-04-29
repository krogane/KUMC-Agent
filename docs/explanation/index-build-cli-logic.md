# CLI 経由の `index build` ロジック

このドキュメントは、CLI から `index build` を実行したときに、プロジェクト内で何がどの順番で起きるかを説明するものです。

対象コマンドは次です。

```bash
PYTHONPATH=src app/.venv/bin/python -m kumc_agent.cli index build
```

## 全体像

`index build` は、検索で使うインデックスを直接作り直すための低レベルなコマンドです。

大きく見ると、処理は次の順番で進みます。

1. CLI が `index build` の引数を読む
2. `BuildIndexUsecase` に処理を渡す
3. 必要なら外部・ローカルの source loader で ingestion source data を更新する
4. `IndexingService.build()` で ingestion source data から document / chunk / index を作る
5. 実行結果を JSON で標準出力に返す

`index build` は `index update` と違い、`AutoIndexUpdateUsecase` を通りません。そのため、lock、staging、snapshot publish、quality smoke check、変更なし skip、`IndexingRun` の保存は行いません。

## CLI 入口

CLI のサブコマンド定義は `src/kumc_agent/cli.py` にあります。

```python
index_parser = subparsers.add_parser("index", help="Index operations")
index_sub = index_parser.add_subparsers(dest="index_command", required=True)
build_parser = index_sub.add_parser("build")
build_parser.add_argument("--no-refresh-sources", action="store_true")
build_parser.add_argument("--full-rebuild", action="store_true")
build_parser.add_argument("--stage", action="append", dest="stages", default=None)
```

`index build` で使える主なオプションは次です。

| オプション | 意味 |
| --- | --- |
| `--no-refresh-sources` | ingestion source data の更新処理をスキップする |
| `--full-rebuild` | ingestion/chunk などの中間成果物を消してから作り直す |
| `--stage <name>` | 指定した chunk stage だけを実行する。複数回指定可能 |

実行時は、CLI が `BuildIndexRequest` を作って `context.build_index.execute()` を呼びます。

```python
result = context.build_index.execute(
    BuildIndexRequest(
        refresh_sources=not args.no_refresh_sources,
        full_rebuild=bool(args.full_rebuild),
        stage_selection=tuple(args.stages or ()) or None,
    )
)
```

つまり、通常実行では `refresh_sources=True` です。`--no-refresh-sources` を付けた場合だけ `False` になります。

## `BuildIndexUsecase` の役割

`BuildIndexUsecase` は、CLI と実際の index build 処理の間にある薄い調整役です。

実装は `src/kumc_agent/usecases/indexing/build.py` にあります。

処理は主に 2 つです。

1. `refresh_sources=True` の場合、各 source loader を順番に実行する
2. `IndexingService.build()` に index 構築を依頼する

loader の実行順は次です。

1. Discord loader
2. Google Drive loader
3. Hatena Blog loader
4. Crafters Colony loader
5. X posts loader
6. Notion loader
7. Minecraft Wiki source refresh

各 loader は、外部サービスや既存データから `data/ingestion` 配下の ingestion source data を更新する役割を持ちます。loader が未設定の場合は、その loader はスキップされます。

`--no-refresh-sources` を付けると、この loader 実行は行われません。その場合は、すでに `data/ingestion` 配下にある ingestion source data だけを使って index を作ります。

## `IndexingService.build()` の処理順

実際に document、chunk、検索 index を作る中心処理は `src/kumc_agent/features/indexing/service.py` の `IndexingService.build()` です。

処理順は次です。

### 1. 別の index directory が指定されているか確認する

`index build` の CLI から直接呼ぶ場合、通常は `data/index` が出力先です。

一方、他の usecase から `index_dir` が指定されると、一時的に runtime の `index_dir` を差し替えて、そのディレクトリに index を作ります。これは主に `index update` が staging directory を使うための仕組みです。

CLI の `index build` では、基本的にこの staging 用の分岐は使われません。

### 2. 実行対象 stage を整理する

`--stage` が指定されている場合、指定された stage だけを実行対象にします。

指定されていない場合は、利用可能な stage を一通り実行します。

主な stage 名は次です。

| stage | 内容 |
| --- | --- |
| `first_recursive` | ingestion document を大きめの chunk に分割する |
| `second_recursive` | first chunk をさらに検索向けの細かい chunk に分割する |
| `sparse_second_recursive` | BM25 など sparse search 向けの正規化 chunk を作る |
| `summary` | 要約 chunk を作る |

stage の有効・無効は `configs/main/indexing.yaml` の `indexing.stages` で管理されています。

### 3. `--full-rebuild` や refresh 設定に応じて中間データを削除する

`_apply_clear_flags()` が呼ばれます。

`--full-rebuild` の場合、ingestion source data や chunk data を作り直すため、関連ディレクトリの内容が削除対象になります。

`--full-rebuild` を指定していない場合でも、`configs/main/indexing.yaml` の `indexing.refresh.clear_*` 設定が true なら、対応するディレクトリが削除されます。

対象になる主なディレクトリは次です。

| 種類 | 主なディレクトリ |
| --- | --- |
| ingestion source data | `data/ingestion` |
| first recursive chunk | `data/chunks/first_rec_chunk` |
| second recursive chunk | `data/chunks/second_rec_chunk` |
| sparse second recursive chunk | `data/chunks/sparse_second_rec_chunk` |
| summary chunk | `data/chunks/summary_chunk` |

### 4. ingestion source data 用ディレクトリを作成する

`_ensure_ingestion_source_dirs()` が呼ばれ、source ごとの ingestion source data ディレクトリが存在する状態にします。

主な ingestion source data ディレクトリは次です。

| source | ディレクトリ |
| --- | --- |
| Google Docs など | `data/ingestion/docs` |
| Google Sheets など | `data/ingestion/sheets` |
| Discord messages | `data/ingestion/messages` |
| X posts | `data/ingestion/x` |
| VC transcript | `data/ingestion/vc` |
| Hatena Blog | `data/ingestion/hatenablog` |
| Crafters Colony | `data/ingestion/crafters_colony` |
| Notion | `data/ingestion/notion` |
| Minecraft Wiki | `data/ingestion/minecraft_wiki` |

### 5. ingestion source data から `Document` を作る

`_parse_documents_from_ingestion()` が `data/ingestion` 配下を読み取り、内部表現の `Document` に変換します。

読み取り対象は source ごとに決まっています。

| source | 対象拡張子 |
| --- | --- |
| docs | `.md` |
| sheets | `.csv` |
| messages | `.jsonl` |
| x_posts | `.jsonl` |
| vc_transcript | `.txt` |
| hatenablog | `.md` |
| crafters_colony | `.md` |
| notion | `.md` |

`.meta.json`、`.mtime.json`、`.state.json` のような sidecar file は本文としては扱いません。

作成された `Document` は `FileSystemStorage.save_documents()` で保存されます。

### 6. legacy chunk pipeline 用の設定を組み立てる

現在の実装では、chunk 作成の一部に `kumc_agent.infra.indexing` 側の既存処理を使っています。

そのため、RuntimeConfig から legacy indexing 用の `AppConfig` を作ります。

ここで反映される主な設定は次です。

- ingestion source data directory
- chunk directory
- index directory
- chunk size / overlap
- summary chunk の設定
- Gemini / embedding の設定
- sparse search の BM25 設定
- Google Drive / Discord など連携先の設定

また、summary chunk 用の prompt environment variable が未設定の場合は、デフォルト値も補完されます。

### 7. chunk の入力元を決める

`IndexingService.build()` には、chunk の入力元が 2 パターンあります。

1. ingestion repository の active chunks を使う
2. ingestion source data から legacy chunk pipeline で chunk を作る

CLI の `index build` では、通常 `prefer_ingestion_repository=False` なので、ingestion source data から legacy chunk pipeline を実行する経路になります。

一方、`index update` から staging build として呼ばれる場合は `prefer_ingestion_repository=True` になり、ingestion repository の active chunks を優先します。

### 8. ingestion source data から chunk を作る

CLI の `index build` で通常通るのは、ingestion source data から chunk を作る経路です。

通常 source の chunk pipeline は次の順番です。

1. `first_recursive`
2. `second_recursive`
3. `sparse_second_recursive`
4. `summary`

Minecraft Wiki は、通常 source とは別の chunking 設定を使って、次の順番で処理されます。

1. Minecraft Wiki 用 `first_recursive`
2. Minecraft Wiki 用 `second_recursive`
3. Minecraft Wiki 用 `sparse_second_recursive`
4. Minecraft Wiki 用 `summary`

この分離により、通常のサークル情報 RAG と Minecraft Wiki RAG で、chunk size や retrieval 設定を変えられます。

### 9. index に入れる chunk を読み込む

chunk pipeline が終わると、`_load_index_chunks_from_legacy_dirs()` で検索 index に入れる chunk を読み込みます。

`second_recursive` が有効な場合は、主に second recursive chunk が dense search の対象になります。

`second_recursive` が無効な場合は、first recursive chunk が使われます。

VC transcript は first recursive ではなく second recursive 側で直接 chunk 化されるため、second recursive chunk として扱われます。

### 10. chunk を保存する

読み込んだ index chunks は、`FileSystemStorage.save_chunks()` で `data/chunks/chunks.jsonl` に保存されます。

この `chunks.jsonl` は、検索処理やデバッグで参照しやすい、現在の index chunks のまとまった一覧です。

### 11. dense embedding を作る

各 chunk から embedding 用テキストを作り、embedder で vector に変換します。

```python
dense_texts = [self._chunk_embedding_text_for_dense(chunk) for chunk in index_chunks]
embeddings = self._embedder.embed_documents(dense_texts)
```

embedder は設定により Gemini または local embedder が使われます。

### 12. FAISS index を作る

dense embedding と chunk を使って、FAISS 互換の dense index を作ります。

```python
self._faiss_index.build(chunks=index_chunks, embeddings=embeddings)
```

この index は、意味的に近い文書を探す dense search で使われます。

### 13. BM25 index を作る

同じ chunk を使って、Sudachi BM25 の sparse index も作ります。

```python
self._bm25_index.build(index_chunks)
```

BM25 index は、キーワード一致に強い検索で使われます。

### 14. material catalog と keyword inverted index を作る

FAISS / BM25 とは別に、補助的な検索 artifact も作ります。

主な成果物は次です。

| artifact | 目的 |
| --- | --- |
| material catalog | 参照可能な資料・素材の一覧を作る |
| keyword inverted indexes | source や chunk 種別ごとのキーワード検索用 index を作る |
| material name keyword index | Minecraft などの素材名検索を強化する |

これらは retrieval の補助や、特定用途向け検索の精度改善に使われます。

### 15. image asset builder があれば画像系 artifact を作る

`image_asset_builder` が設定されている場合、ingestion source から画像検索用の artifact も作ります。

設定されていない場合、この処理はスキップされます。

### 16. 実行結果を返す

最後に `IndexBuildResult` が返ります。

含まれる主な値は次です。

| field | 内容 |
| --- | --- |
| `loaded_sources` | loader で読み込んだ source 件数 |
| `documents` | ingestion source data から作成した `Document` 件数 |
| `chunks` | index に入れた chunk 件数 |
| `index_dir` | index artifact の出力先 |
| `stage_results` | chunk 入力元や画像 build 結果などの補助情報 |

CLI はこの結果を次の JSON 形式で出力します。

```json
{
  "loaded_sources": 0,
  "documents": 0,
  "chunks": 0,
  "index_dir": "data/index",
  "metadata": {}
}
```

実際の数値は実行時の ingestion source data や loader の結果によって変わります。

## `index build` の出力先

通常の CLI 実行では、出力先は `configs/main/app.yaml` の `app.index_dir` です。

既定値は次です。

```yaml
app:
  index_dir: "data/index"
```

また、chunk の一覧は次に保存されます。

```yaml
app:
  chunks_path: "data/chunks/chunks.jsonl"
```

## `index build` と `index update` の違い

初学者が混乱しやすい点として、`index build` と `index update` は似ていますが役割が違います。

| 項目 | `index build` | `index update` |
| --- | --- | --- |
| 呼び出す usecase | `BuildIndexUsecase` | `AutoIndexUpdateUsecase` |
| 変更検出 | なし | あり |
| lock | なし | あり |
| staging directory | なし | あり |
| quality smoke check | なし | あり |
| snapshot publish | なし | あり |
| 変更なし skip | なし | あり |
| `IndexingRun` 保存 | なし | あり |
| member profiles 再構築 | なし | あり |
| task/event index 再構築 | なし | あり |

通常運用で安全に index を更新したい場合は `index update` を使います。

`index build` は、ingestion source data から index artifact を直接作る挙動を確認したいときや、chunk pipeline / embedding / FAISS / BM25 の構築を開発中に検証したいときに向いています。

## 主要ファイル

| ファイル | 役割 |
| --- | --- |
| `src/kumc_agent/cli.py` | CLI の `index build` コマンド入口 |
| `src/kumc_agent/runtime/container.py` | `BuildIndexUsecase` や loader の組み立て |
| `src/kumc_agent/usecases/indexing/build.py` | loader 実行と `IndexingService.build()` 呼び出し |
| `src/kumc_agent/features/indexing/service.py` | document / chunk / index artifact の作成本体 |
| `configs/main/app.yaml` | `data/ingestion`、`data/index` などの基本 path |
| `configs/main/indexing.yaml` | chunk size、stage 有効化、refresh 方針 |

## 読むときの目印

コードを追う場合は、次の順番で読むと理解しやすいです。

1. `src/kumc_agent/cli.py` の `args.command == "index"` の分岐
2. `src/kumc_agent/usecases/indexing/build.py` の `BuildIndexUsecase.execute()`
3. `src/kumc_agent/features/indexing/service.py` の `IndexingService.build()`
4. 同ファイルの `_run_legacy_chunk_pipeline()`
5. 同ファイルの `_run_minecraft_wiki_chunk_pipeline()`
6. 同ファイルの `_load_index_chunks_from_legacy_dirs()`

この順番で読むと、「CLI の引数がどの設定に変換され、どの ingestion source data から、どの chunk を作り、最終的にどの index artifact になるか」を追いやすくなります。
