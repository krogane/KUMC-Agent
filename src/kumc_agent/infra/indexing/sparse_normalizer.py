from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import FrozenSet

from sudachipy import dictionary, tokenizer as sudachi_tokenizer

DEFAULT_STOPWORD_POS: FrozenSet[str] = frozenset(
    {"助詞", "助動詞", "補助記号"}
)
DEFAULT_LEMMA_POS: FrozenSet[str] = frozenset({"動詞", "形容詞"})


def is_symbol_only(text: str) -> bool:
    has_visible = False
    for ch in text:
        if ch.isspace():
            continue
        has_visible = True
        category = unicodedata.category(ch)
        if category.startswith(("L", "N", "M")):
            return False
    return has_visible


def _sudachi_mode(value: str) -> sudachi_tokenizer.Tokenizer.SplitMode:
    mode = (value or "B").upper()
    if mode == "A":
        return sudachi_tokenizer.Tokenizer.SplitMode.A
    if mode == "C":
        return sudachi_tokenizer.Tokenizer.SplitMode.C
    return sudachi_tokenizer.Tokenizer.SplitMode.B


@lru_cache(maxsize=1)
def _sudachi_tokenizer() -> sudachi_tokenizer.Tokenizer:
    return dictionary.Dictionary().create()


@dataclass(frozen=True)
class SparseNormalizerConfig:
    sudachi_mode: str = "B"
    use_normalized_form: bool = True
    remove_symbols: bool = True
    remove_stopwords: bool = False
    stopword_pos: FrozenSet[str] = DEFAULT_STOPWORD_POS
    lemma_pos: FrozenSet[str] = DEFAULT_LEMMA_POS


class SparseNormalizer:
    def __init__(self, *, config: SparseNormalizerConfig) -> None:
        self._config = config

    def normalize_tokens(self, text: str) -> list[str]:
        value = text.strip()
        if not value:
            return []

        tokenizer = _sudachi_tokenizer()
        mode = _sudachi_mode(self._config.sudachi_mode)
        tokens: list[str] = []
        for morph in tokenizer.tokenize(value, mode):
            pos = morph.part_of_speech()
            pos_major = pos[0] if pos else ""

            token = (
                morph.normalized_form()
                if self._config.use_normalized_form
                else morph.surface()
            )

            if pos_major in self._config.lemma_pos:
                lemma = morph.dictionary_form().strip()
                if lemma and lemma != "*":
                    token = lemma

            token = token.strip()
            if not token:
                continue
            if (
                self._config.remove_stopwords
                and pos_major in self._config.stopword_pos
            ):
                continue
            if self._config.remove_symbols and is_symbol_only(token):
                continue
            tokens.append(token)
        return tokens
