# 画像検索 運用メモ

## 保存先
- Asset正本: `data/operations/assets.jsonl`
- 画像検索用画像: `data/image_search/images/`
- Dense index: `data/image_search/image_text_vectors.npy`, `data/image_search/image_assets.jsonl`
- 画像特徴量index: `data/image_search/image_feature_vectors.npy`

## 更新
通常のindex更新で、raw source取得後に画像Asset化、caption、OCR、Dense index、特徴量indexを更新する。

```bash
python3 -m kumc_agent.cli index update
```

Gemini API keyまたはOCRモデルが未設定の場合、caption/OCRはfallbackまたはskipされ、周辺テキストと既存metadataで検索を継続する。

## ロールバック
重大な誤indexや権限設定ミスがある場合は、直前のバックアップから次を戻す。

- `data/operations/assets.jsonl`
- `data/image_search/`

復旧後に `python3 -m kumc_agent.cli index update --no-refresh-sources --full-rebuild` を実行し、raw sourceを取り直さずindexだけ再構築する。

## 注意
検索結果は候補提示のみで、外部公開・転載・再利用の可否は判断しない。Drive/Discord画像は許可Guildまたはadmin DM以外へ返さない。
