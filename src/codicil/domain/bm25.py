"""Okapi BM25 with two weighted fields (header, body). Own implementation so every parameter and the index format are inspectable."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from codicil.domain.tokens import analyse


@dataclass(frozen=True, slots=True)
class BM25Params:
    k1: float = 1.2
    b: float = 0.75
    header_weight: float = 0.5


class BM25Index:
    """Postings are term -> list of (doc_ordinal, weighted_tf). Header tokens count `header_weight` times a body token."""

    def __init__(self, params: BM25Params, postings: dict[str, list[tuple[int, float]]], lengths: list[float]) -> None:
        self.params = params
        self.postings = postings
        self.lengths = lengths
        self.n = len(lengths)
        self.avg_len = sum(lengths) / self.n if self.n else 0.0
        self._idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()}

    @classmethod
    def build(cls, headers: Sequence[str], bodies: Sequence[str], params: BM25Params = BM25Params()) -> BM25Index:
        postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        lengths: list[float] = []
        for i, (h, b) in enumerate(zip(headers, bodies, strict=True)):
            tf: dict[str, float] = {}
            for t in analyse(b):
                tf[t] = tf.get(t, 0.0) + 1.0
            for t in analyse(h):
                tf[t] = tf.get(t, 0.0) + params.header_weight
            lengths.append(sum(tf.values()))
            for t, f in tf.items():
                postings[t].append((i, f))
        return cls(params, dict(postings), lengths)

    def score(self, query: str, top_k: int = 50, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        k1, b = self.params.k1, self.params.b
        acc: dict[int, float] = defaultdict(float)
        for t in dict.fromkeys(analyse(query)):  # unique query terms
            idf = self._idf.get(t)
            if idf is None:
                continue
            for doc, f in self.postings[t]:
                if allowed is not None and doc not in allowed:
                    continue
                norm = f + k1 * (1 - b + b * self.lengths[doc] / self.avg_len)
                acc[doc] += idf * f * (k1 + 1) / norm
        return sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

    def to_json(self) -> dict[str, Any]:
        return {
            "params": [self.params.k1, self.params.b, self.params.header_weight],
            "lengths": self.lengths,
            "postings": {t: [[d, f] for d, f in p] for t, p in sorted(self.postings.items())},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BM25Index:
        k1, b, hw = (float(x) for x in data["params"])
        post = {t: [(int(d), float(f)) for d, f in p] for t, p in data["postings"].items()}
        return cls(BM25Params(k1, b, hw), post, [float(x) for x in data["lengths"]])
