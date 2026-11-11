#!/usr/bin/env python3
"""Retrieval ablation: python eval/run_retrieval.py --split dev [--out eval/results/retrieval-dev.json]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from codicil.adapters.onnx_embedder import OnnxEmbedder  # noqa: E402
from codicil.adapters.onnx_reranker import OnnxReranker  # noqa: E402
from codicil.application.index import load_index  # noqa: E402
from codicil.application.retrieve import RetrieveConfig, Retriever  # noqa: E402
from codicil.domain import metrics as M  # noqa: E402
from codicil.domain.glossary import normalise_query  # noqa: E402


def load_questions(split: str) -> list[dict]:
    qs = [json.loads(line) for line in (ROOT / "eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return [q for q in qs if q["split"] == split or split == "all"]


def evaluate(embedder_name: str, cfg: RetrieveConfig, qs: list[dict], normalise: bool, rerank: bool | str = False) -> dict:
    idx = load_index(ROOT / "index" / embedder_name)
    rr = OnnxReranker(ROOT / "data/models/ms-marco-MiniLM-L-6-v2") if rerank else None
    emb = OnnxEmbedder(ROOT / "data/models" / embedder_name)
    r = Retriever(idx, emb)
    answerable = [q for q in qs if q["answerable"]]
    queries = [normalise_query(q["question"]) if normalise else None for q in answerable]
    texts = [(qq.normalised if qq else q["question"]) for q, qq in zip(answerable, queries, strict=True)]
    qv = emb.embed_queries(texts)
    rows: dict[str, list[dict]] = {}
    for q, text, v in zip(answerable, texts, qv, strict=True):
        hints = tuple(normalise_query(q["question"]).entity_hints) if normalise else ()
        lang = normalise_query(q["question"]).language
        hits = r.search(text, cfg, top_k=20, entity_hints=hints, qvec=v, language=lang)
        if rr is not None and (rerank != "en" or lang == "en"):
            sc = rr.score(text, [idx.chunks[idx.by_id[h.chunk_id]].indexed_text for h in hits])
            hits = [hits[i] for i in sorted(range(len(hits)), key=lambda i: (-sc[i], i))]
        ret = [(idx.chunks[idx.by_id[h.chunk_id]].doc_id, idx.chunks[idx.by_id[h.chunk_id]].text) for h in hits]
        gold = [(g["doc_id"], g["quote"]) for g in q["gold_evidence"]]
        hs = M.hits_per_rank(ret, gold)
        rows.setdefault(q["slice"], []).append(
            {
                "id": q["id"],
                "recall@1": M.recall_at_k(hs, len(gold), 1),
                "recall@5": M.recall_at_k(hs, len(gold), 5),
                "recall@10": M.recall_at_k(hs, len(gold), 10),
                "recall@20": M.recall_at_k(hs, len(gold), 20),
                "mrr@10": M.reciprocal_rank(hs, 10),
                "ndcg@10": M.ndcg_at_k(hs, len(gold), 10),
                "success@10": M.success_at_k(hs, 10),
            }
        )
    rows["ALL"] = [x for v in rows.values() for x in v]
    out = {}
    for sl, items in rows.items():
        out[sl] = {"n": len(items)}
        for metric in ("recall@1", "recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10", "success@10"):
            mean, lo, hi = M.bootstrap_ci([x[metric] for x in items])
            out[sl][metric] = [round(mean, 3), round(lo, 3), round(hi, 3)]
        out[sl]["_per_item"] = items
    return out


CONFIGS = {
    "bm25": ("bge-small-en-v1.5", RetrieveConfig(mode="bm25"), True),
    "dense-bge": ("bge-small-en-v1.5", RetrieveConfig(mode="dense"), True),
    "hybrid-bge": ("bge-small-en-v1.5", RetrieveConfig(mode="hybrid"), True),
    "dense-e5": ("multilingual-e5-small", RetrieveConfig(mode="dense"), True),
    "hybrid-e5": ("multilingual-e5-small", RetrieveConfig(mode="hybrid"), True),
    "hybrid-e5-routed": ("multilingual-e5-small", RetrieveConfig(mode="hybrid", dense_only_languages=("hi",)), True),
    "hybrid-bge-routed": ("bge-small-en-v1.5", RetrieveConfig(mode="hybrid", dense_only_languages=("hi",)), True),
    "hybrid-e5-routed-rerank": ("multilingual-e5-small", RetrieveConfig(mode="hybrid", dense_only_languages=("hi",)), True, True),
    "hybrid-e5-routed-rerank-en": ("multilingual-e5-small", RetrieveConfig(mode="hybrid", dense_only_languages=("hi",)), True, "en"),
    "hybrid-bge-rerank": ("bge-small-en-v1.5", RetrieveConfig(mode="hybrid"), True, True),
    "hybrid-bge-nonorm": ("bge-small-en-v1.5", RetrieveConfig(mode="hybrid"), False),
    "hybrid-e5-nonorm": ("multilingual-e5-small", RetrieveConfig(mode="hybrid"), False),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    qs = load_questions(a.split)
    results = {name: evaluate(CONFIGS[name][0], CONFIGS[name][1], qs, CONFIGS[name][2], *CONFIGS[name][3:]) for name in a.configs}
    print(f"split={a.split}  answerable={sum(q['answerable'] for q in qs)}")
    for sl in ("ALL", "single", "multi", "hinglish", "hi"):
        print(f"\n== {sl} ==  {'config':20} {'n':>3} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'MRR':>6} {'nDCG':>6}")
        for name, res in results.items():
            if sl in res:
                r = res[sl]
                print(
                    f"   {name:24} {r['n']:3d} " + " ".join(f"{r[m][0]:6.3f}" for m in ("recall@1", "recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10"))
                )
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(
            json.dumps(
                {"split": a.split, "results": {n: {sl: {k: v for k, v in d.items() if k != "_per_item"} for sl, d in r.items()} for n, r in results.items()}},
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
