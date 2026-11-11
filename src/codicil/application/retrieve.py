"""Hybrid retrieval over an `Index`: BM25 + dense, fused with RRF, then filtered."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from codicil.application.index import Index
from codicil.application.ports import Embedder
from codicil.domain.fusion import rrf
from codicil.domain.models import DocStatus, Scored


class DenseStore(Protocol):
    def search(self, qvec: np.ndarray, k: int, in_force_only: bool) -> list[tuple[int, float]]: ...


@dataclass(frozen=True, slots=True)
class RetrieveConfig:
    mode: str = "hybrid"  # bm25 | dense | hybrid
    pool: int = 50  # candidates taken from each retriever
    rrf_k: int = 60
    w_bm25: float = 1.0
    w_dense: float = 1.0
    include_superseded: bool = False
    entity_boost: float = 0.0  # multiplicative bonus for chunks whose document entity matches a hint
    dense_only_languages: tuple[str, ...] = ()  # BM25 cannot match a non-Latin query lexically; route these to dense retrieval only


class Retriever:
    def __init__(self, index: Index, embedder: Embedder, store: DenseStore | None = None) -> None:
        index.check_embedder(embedder)
        self.index, self.embedder, self.store = index, embedder, store
        self._status_ok = np.array([index.docs[c.doc_id].status is DocStatus.IN_FORCE for c in index.chunks])

    def _allowed(self, cfg: RetrieveConfig) -> set[int] | None:
        if cfg.include_superseded or self._status_ok.all():
            return None
        return {int(i) for i in np.flatnonzero(self._status_ok)}

    def dense(self, qvec: np.ndarray, k: int, allowed: set[int] | None) -> list[tuple[int, float]]:
        if self.store is not None:
            return self.store.search(qvec, k, in_force_only=allowed is not None)
        sims = self.index.vectors @ qvec
        if allowed is not None:
            mask = np.full(len(sims), -np.inf, dtype=np.float32)
            mask[list(allowed)] = 0
            sims = sims + mask
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(int(i), float(sims[i])) for i in top if np.isfinite(sims[i])]

    def search(
        self,
        query: str,
        cfg: RetrieveConfig = RetrieveConfig(),
        top_k: int = 20,
        entity_hints: tuple[str, ...] = (),
        qvec: np.ndarray | None = None,
        language: str = "en",
    ) -> list[Scored]:
        if cfg.mode == "hybrid" and language in cfg.dense_only_languages:
            cfg = replace(cfg, mode="dense")
        allowed = self._allowed(cfg)
        lists: list[list[int]] = []
        weights: list[float] = []
        if cfg.mode in ("bm25", "hybrid"):
            b = self.index.bm25.score(query, cfg.pool, allowed)
            lists.append([i for i, _ in b])
            weights.append(cfg.w_bm25)
        if cfg.mode in ("dense", "hybrid"):
            v = qvec if qvec is not None else self.embedder.embed_queries([query])[0]
            d = self.dense(v, cfg.pool, allowed)
            lists.append([i for i, _ in d])
            weights.append(cfg.w_dense)
        fused = rrf(lists, cfg.rrf_k, weights)
        if cfg.entity_boost and entity_hints:
            hints = {h.lower() for h in entity_hints}
            fused = sorted(
                ((i, s * (1 + cfg.entity_boost) if self.index.docs[self.index.chunks[i].doc_id].entity.lower() in hints else s) for i, s in fused),
                key=lambda kv: (-kv[1], kv[0]),
            )
        src = "fused" if cfg.mode == "hybrid" else cfg.mode
        return [Scored(self.index.chunks[i].chunk_id, s, r, src) for r, (i, s) in enumerate(fused[:top_k], start=1)]
