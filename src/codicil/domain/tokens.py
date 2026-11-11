"""Deterministic tokenisation for BM25 and for token budgets. Pure functions; no model needed."""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_WORD = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "of",
        "on",
        "or",
        "shall",
        "should",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "were",
        "which",
        "while",
        "will",
        "with",
        "would",
    ]
)
_stemmer = snowballstemmer.stemmer("english")


def words(text: str) -> list[str]:
    """Lower-cased word tokens; numbers like '1,00,000' and '2.5' stay whole."""
    return [m.group(0).lower() for m in _WORD.finditer(text)]


@lru_cache(maxsize=65536)
def _stem(w: str) -> str:
    return str(_stemmer.stemWord(w))


def analyse(text: str) -> list[str]:
    """Tokens for the sparse index: stop words removed, English stemming (Devanagari tokens are left unstemmed)."""
    out: list[str] = []
    for w in words(text):
        if w in STOPWORDS:
            continue
        out.append(w if _DEVANAGARI.search(w) or w[0].isdigit() else _stem(w))
    return out


def count_tokens(text: str) -> int:
    """Approximate budget unit: words plus a 1.3 factor for sub-word splitting (checked against the real tokenizer in tests)."""
    return int(len(words(text)) * 1.3) + 1


def sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by space and an upper-case letter, digit, bracket or quote; keeps abbreviations like 'No.' intact."""
    parts = re.split(r"(?<=[.;:!?])\s+(?=[A-Z0-9(‘'\"“])", text.strip())
    merged: list[str] = []
    for p in parts:
        if merged and re.search(r"\b(?:No|Rs|Sr|Dr|Ltd|vs|etc|viz|i\.e|e\.g|Art|Sec|Cl)\.$", merged[-1]):
            merged[-1] += " " + p
        else:
            merged.append(p)
    return [m for m in merged if m]
