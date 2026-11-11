"""python -m codicil.cli.build --embedder bge-small-en-v1.5 --out build/index-bge"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from codicil.adapters.onnx_embedder import OnnxEmbedder
from codicil.adapters.rbi_html import parse_meta
from codicil.application.index import build_index

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="bge-small-en-v1.5")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    man = json.loads((ROOT / "data/corpus/manifest.json").read_text())["documents"]
    pairs = []
    for d in man:
        html = (ROOT / "data/raw" / f"{d['doc_id']}.html").read_text(encoding="utf-8")
        pairs.append((parse_meta(d["doc_id"], d["title"], d["url"], html), html))
    emb = OnnxEmbedder(ROOT / "data/models" / a.embedder)
    t = time.time()
    idx = build_index(pairs, emb, a.out)
    print(f"{len(idx.docs)} docs, {len(idx.chunks)} chunks, {len(idx.parents)} parents, embedder {idx.embedder_id}, {time.time() - t:.0f}s, sha {idx.sha[:12]}")


if __name__ == "__main__":
    main()
