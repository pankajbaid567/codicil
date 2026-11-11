"""Retrieval and abstention metrics, plus the evidence matcher and a bootstrap CI. Pure functions; golden-tested against hand values."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Sequence

from codicil.domain.tokens import words


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')).strip().lower()


def covers(chunk_text: str, quote: str, min_recall: float = 0.85) -> bool:
    """True if the chunk contains the gold quote verbatim (after normalisation) or at least `min_recall` of its words in order-free overlap.
    The word-overlap rule lets a quote that straddles two chunks still count for the chunk holding most of it."""
    c, q = _norm(chunk_text), _norm(quote)
    if q in c:
        return True
    qw = words(quote)
    if not qw:
        return False
    cw = set(words(chunk_text))
    return sum(w in cw for w in qw) / len(qw) >= min_recall


def hits_per_rank(retrieved: Sequence[tuple[str, str]], gold: Sequence[tuple[str, str]]) -> list[set[int]]:
    """For each retrieved (doc_id, text), the set of gold indices it covers. Gold items are (doc_id, quote)."""
    return [{gi for gi, (gd, gq) in enumerate(gold) if gd == doc and covers(text, gq)} for doc, text in retrieved]


def recall_at_k(hit_sets: Sequence[set[int]], n_gold: int, k: int) -> float:
    """Fraction of gold evidence items covered by the top k."""
    if n_gold == 0:
        return 0.0
    covered: set[int] = set().union(*hit_sets[:k]) if hit_sets[:k] else set()
    return len(covered) / n_gold


def success_at_k(hit_sets: Sequence[set[int]], k: int) -> float:
    return 1.0 if any(hit_sets[:k]) else 0.0


def reciprocal_rank(hit_sets: Sequence[set[int]], k: int = 10) -> float:
    for r, h in enumerate(hit_sets[:k], 1):
        if h:
            return 1.0 / r
    return 0.0


def ndcg_at_k(hit_sets: Sequence[set[int]], n_gold: int, k: int = 10) -> float:
    """nDCG with novelty: a rank earns gain 1 only for covering a gold item not covered by a higher rank, so repeated
    hits on the same evidence add nothing. The ideal ranking covers one new gold item per rank, so IDCG uses min(n_gold, k) ranks."""
    seen: set[int] = set()
    dcg = 0.0
    for i, h in enumerate(hit_sets[:k]):
        if h - seen:
            dcg += 1.0 / math.log2(i + 2)
            seen |= h
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(max(n_gold, 1), k)))
    return dcg / idcg


def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap over items."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    m = len(values)
    means = sorted(sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return (sum(values) / m, means[int(alpha / 2 * n)], means[min(n - 1, int((1 - alpha / 2) * n))])


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean diff a-b, lo, hi) over the same items."""
    return bootstrap_ci([x - y for x, y in zip(a, b, strict=True)], n, 0.05, seed)


def precision_recall(pred: Sequence[bool], truth: Sequence[bool]) -> tuple[float, float]:
    tp = sum(p and t for p, t in zip(pred, truth, strict=True))
    fp = sum(p and not t for p, t in zip(pred, truth, strict=True))
    fn = sum((not p) and t for p, t in zip(pred, truth, strict=True))
    return (tp / (tp + fp) if tp + fp else 1.0, tp / (tp + fn) if tp + fn else 1.0)


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1)]


Metric = Callable[[Sequence[set[int]]], float]
