"""Reciprocal Rank Fusion: combine ranked lists without comparing their (incomparable) scores."""

from __future__ import annotations

from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[int]], k: int = 60, weights: Sequence[float] | None = None, top_k: int | None = None) -> list[tuple[int, float]]:
    """Each ranking is a list of ids, best first. score(id) = sum_i w_i / (k + rank_i(id)), ranks starting at 1.
    Ties break on the smaller id so the result is deterministic."""
    w = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(w) != len(rankings):
        raise ValueError("weights and rankings differ in length")
    acc: dict[int, float] = {}
    for ranking, wi in zip(rankings, w, strict=True):
        seen: set[int] = set()
        for r, doc in enumerate(ranking, start=1):
            if doc in seen:  # a duplicate in one list must not count twice
                continue
            seen.add(doc)
            acc[doc] = acc.get(doc, 0.0) + wi / (k + r)
    out = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
    return out[:top_k] if top_k is not None else out
