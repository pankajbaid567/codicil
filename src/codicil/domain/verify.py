"""Support check and abstention features. Lexical, deterministic, cheap; an NLI model could replace `supported` behind the same signature."""

from __future__ import annotations

import re

from codicil.domain.tokens import analyse, words

_NUM = re.compile(r"\d[\d,.]*")
_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][A-Za-z]{2,}\b")


def numbers(text: str) -> set[str]:
    return {n.rstrip(".,").replace(",", "") for n in _NUM.findall(text)}


def support_score(sentence: str, evidence: str) -> float:
    """Content-word recall of the sentence in the evidence, forced to 0 if any number in the sentence is absent from it."""
    if not numbers(sentence) <= numbers(evidence):
        return 0.0
    s, e = set(analyse(sentence)), set(analyse(evidence))
    return len(s & e) / len(s) if s else 1.0


def supported(sentence: str, evidence: str, threshold: float = 0.7) -> bool:
    return support_score(sentence, evidence) >= threshold


def query_coverage(query: str, texts: list[str]) -> float:
    """Fraction of the query's content terms that occur in the given evidence texts."""
    q = set(analyse(query))
    if not q:
        return 0.0
    have = {t for x in texts for t in analyse(x)}
    return len(q & have) / len(q)


def evidence_words(text: str) -> set[str]:
    return set(words(text))
