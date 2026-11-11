#!/usr/bin/env python3
"""Does the Qdrant dense search return what the in-process search returns, and what does the network hop cost? Reads CODICIL_QDRANT_* from .env (never printed)."""

from __future__ import annotations

import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from run_retrieval import load_questions  # noqa: E402

from codicil.adapters.onnx_embedder import OnnxEmbedder  # noqa: E402
from codicil.adapters.qdrant_store import QdrantDenseStore  # noqa: E402
from codicil.application.index import load_index  # noqa: E402
from codicil.application.retrieve import Retriever  # noqa: E402
from codicil.settings import Settings  # noqa: E402


def main() -> None:
    s = Settings()
    if not s.qdrant_url:
        sys.exit("set CODICIL_QDRANT_URL / CODICIL_QDRANT_API_KEY in .env")
    idx = load_index(ROOT / s.index_dir)
    emb = OnnxEmbedder(ROOT / s.models_dir / s.embedder)
    local = Retriever(idx, emb)
    store = QdrantDenseStore.connect(s.qdrant_url, s.qdrant_api_key.get_secret_value(), s.qdrant_collection)
    qs = load_questions("dev") + load_questions("test")
    same10, same1, t_local, t_remote = 0, 0, [], []
    for q in qs:
        v = emb.embed_queries([q["question"]])[0]
        t = time.perf_counter()
        a = local.dense(v, 10, None)
        t_local.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        b = store.search(v, 10, in_force_only=False)
        t_remote.append((time.perf_counter() - t) * 1000)
        same10 += len({i for i, _ in a} & {i for i, _ in b}) / 10
        same1 += a[0][0] == b[0][0]
    n = len(qs)
    print(f"questions {n}: mean top-10 overlap {same10 / n:.3f}, top-1 identical {same1}/{n}")
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    print(
        f"dense search ms  in-process p50 {st.median(t_local):.1f} p95 {q(t_local, 0.95):.1f} | qdrant p50 {st.median(t_remote):.1f} p95 {q(t_remote, 0.95):.1f}"
    )


if __name__ == "__main__":
    main()
