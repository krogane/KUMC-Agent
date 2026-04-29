# Indexing 差分埋め込みの実装説明

## 概要

自動インデックス更新では、source差分の有無は従来どおり `features.ingestion` が `sync_cursors`、checksum、revision、ACL hashで判定する。今回の変更では、その後のDense index構築時に、最終 `index_chunks` のうちembedding textが未変更のchunkは既存vectorを再利用する。

公開artifactは従来どおり完全生成する。`dense_vectors.npy`、`dense_vectors.faiss`、`dense_chunks.jsonl` は全active chunkを含むため、検索runtime側の読み取り方式は変わらない。

## cache

cacheは `data/cache/index_embeddings/` に保存する。keyは次の組み合わせである。

- provider
- model
- dimensions
- chunk id
- embedding text hash

chunk metadataのchecksumだけではなく、実際にembedderへ渡す文字列のhashを使う。Minecraft Wikiでは記事名や見出しを本文にprefixするため、本文chunkのchecksumだけでは再利用判定に不十分である。

## full rebuild

`--full-rebuild` と admin `reindex` では、既定でcacheをbypassして全chunkを再埋め込みする。通常の `index update` とスケジュール更新ではcacheを利用する。

## publish後のcompact

publish成功後、今回のactive chunk keyだけを残すようにcacheをcompactする。壊れたcache行や次元不一致行は無視され、対象chunkは再埋め込みされる。cacheは最適化用なので、削除しても次回runで再作成できる。

## 確認方法

`IndexingRun.metadata.stage_results.embedding` を確認する。

- `embedded_chunks`: 実際に埋め込んだchunk数
- `reused_chunks`: cacheから再利用したchunk数
- `cache_misses`: cache欠損で再埋め込みしたchunk数
- `cache_invalid`: 壊れたcache record数
- `cache_compaction`: publish後compact結果

stagingおよび公開snapshotには `dense_embedding_manifest.jsonl` も出力される。これはchunkごとのhashとmodel情報の確認用であり、検索runtimeの必須入力ではない。
