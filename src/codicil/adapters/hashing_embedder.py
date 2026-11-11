"""Deterministic, model-free embedder (hashed bag of stemmed words). Used in tests and as a no-download fallback; not a quality claim."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from codicil.application.ports import Matrix
from codicil.domain.tokens import analyse


class HashingEmbedder:
    model_id = "hashing-384"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vec(self, text: str) -> Matrix:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in analyse(text):
            h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def embed_queries(self, texts: Sequence[str]) -> Matrix:
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32) if texts else np.zeros((0, self.dim), np.float32)

    embed_passages = embed_queries
