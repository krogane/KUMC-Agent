from __future__ import annotations


def build_summary_prompt(
    *,
    transcript_text: str,
    previous_summaries: list[str],
    target_characters: int,
) -> str:
    history_section = ""
    if previous_summaries:
        joined = "\n".join(
            f"- {item.strip()}" for item in previous_summaries if item.strip()
        )
        history_section = (
            "\n直近の要約（文脈の維持にのみ使用。事実はTranscriptを優先）:\n"
            f"{joined}\n"
        )

    return (
        "あなたは会議要約アシスタントです。\n"
        "Transcriptのみを根拠に、重要事項を過不足なく簡潔に要約してください。\n"
        f"要約は日本語で約{target_characters}字にしてください。\n"
        "新しい情報の追加や推測は禁止です。要約文のみを出力してください。\n"
        f"{history_section}\n"
        "## Transcript\n"
        f"{transcript_text}\n\n"
        "## Output\n"
    )


def build_end_judgement_prompt(*, transcript_text: str) -> str:
    return (
        "あなたは例会終了判定アシスタントです。\n"
        "Transcriptを読み、例会が終了したと判断できる場合のみ true、"
        "それ以外は false を返してください。\n"
        "出力は厳密に true または false のみで、説明は不要です。\n\n"
        "## Transcript\n"
        f"{transcript_text}\n\n"
        "## Output\n"
    )


def build_final_summary_prompt(*, summary_chunks: list[str]) -> str:
    joined = "\n".join(f"- {item.strip()}" for item in summary_chunks if item.strip())
    return (
        "あなたは例会の議事要約アシスタントです。\n"
        "与えられた要約チャンク全体を統合し、重要事項を過不足なくまとめてください。\n"
        "例会の本筋とは関係ないような雑談は出力に含めないでください。\n"
        "決定事項・未決事項・次のアクションが分かるように日本語で出力してください。\n"
        "ただし、氏名（ユーザーネーム除く）・住所・パスワード・口座情報などの機密情報は絶対に回答には含めず、（非公開）と置き換えてください。\n\n"
        "## 要約チャンク\n"
        f"{joined}\n\n"
    )
