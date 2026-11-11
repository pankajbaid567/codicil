#!/usr/bin/env python3
"""Answer-level evaluation and abstention tuning.
python eval/run_answer.py --split dev --signals eval/results/signals-dev.json     # collect per-question signals + answers
python eval/run_answer.py --split dev --tune                                       # grid-search thresholds on dev, print table
python eval/run_answer.py --split test --tau-d 0.80 --tau-c 0.5                    # frozen thresholds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from run_retrieval import load_questions  # noqa: E402

from codicil.adapters.onnx_embedder import OnnxEmbedder  # noqa: E402
from codicil.adapters.onnx_reranker import OnnxReranker  # noqa: E402
from codicil.application.answer import AnswerConfig, AnswerService  # noqa: E402
from codicil.application.index import load_index  # noqa: E402
from codicil.application.retrieve import RetrieveConfig  # noqa: E402
from codicil.domain import metrics as M  # noqa: E402

NO_SCORE_ABSTENTION = ("hinglish",)  # no usable threshold exists for Hinglish (docs/results.md)
EMB = os.environ.get("CODICIL_EVAL_EMB", "multilingual-e5-small")


def service(rerank: bool = True) -> AnswerService:
    idx = load_index(ROOT / "index" / EMB)
    return AnswerService(idx, OnnxEmbedder(ROOT / "data/models" / EMB), OnnxReranker(ROOT / "data/models/ms-marco-MiniLM-L-6-v2") if rerank else None)


def base_cfg(**kw: float) -> AnswerConfig:
    return AnswerConfig(retrieve=RetrieveConfig(mode="hybrid", dense_only_languages=("hi",)), **kw)


def collect(split: str) -> list[dict]:
    svc = service()
    rows = []
    for q in load_questions(split):
        t = time.perf_counter()
        a = svc.ask(q["question"], base_cfg(min_dense=0.0))  # thresholds are applied afterwards, so every answer is generated
        ms = (time.perf_counter() - t) * 1000
        sig = dict(a.signals)
        gold = [(g["doc_id"], g["quote"]) for g in q["gold_evidence"]]
        top = [(c.doc_id, c.quote) for c in a.closest]
        text = " ".join(s.text for s in a.sentences)
        rows.append(
            {
                "id": q["id"],
                "slice": q["slice"],
                "answerable": q["answerable"],
                "language": q["language"],
                "signals": sig,
                "abstained_untuned": a.abstained,
                "answer": text,
                "ms": ms,
                "answer_hit": any(M.covers(text, gq, 0.7) for _, gq in gold) if gold and text else False,
                "closest_hit": any(M.covers(t2, gq, 0.85) for d, t2 in top for gd, gq in gold if gd == d) if gold else False,
            }
        )
    return rows


def decide(row: dict, tau_d: float, tau_c: float, tau_r: float | None = None) -> bool:
    s = row["signals"]
    if tau_r is not None and s["reranked"]:
        weak = s["top_score"] < tau_r  # English routes to the reranker score
    else:
        weak = s["top_dense"] < tau_d and row["language"] not in NO_SCORE_ABSTENTION
    return weak or s["coverage"] < tau_c


def score(rows: list[dict], tau_d: float, tau_c: float, tau_r: float | None = None) -> dict:
    pred = [decide(r, tau_d, tau_c, tau_r) for r in rows]
    truth = [not r["answerable"] for r in rows]
    p, rec = M.precision_recall(pred, truth)
    ans = [r for r, ab in zip(rows, pred, strict=True) if r["answerable"] and not ab]
    n_ans = sum(r["answerable"] for r in rows)
    return {
        "abstain_precision": p,
        "abstain_recall": rec,
        "answerable_served": len(ans) / max(1, n_ans),
        "answer_hit_given_served": sum(r["answer_hit"] for r in ans) / max(1, len(ans)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--signals", type=Path)
    ap.add_argument("--from-signals", type=Path)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tau-d", type=float)
    ap.add_argument("--tau-c", type=float)
    a = ap.parse_args()
    rows = json.loads(a.from_signals.read_text()) if a.from_signals else collect(a.split)
    if a.signals:
        a.signals.parent.mkdir(parents=True, exist_ok=True)
        a.signals.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    if a.tune:
        best = []
        for td in [x / 100 for x in range(55, 95)]:
            for tc in [x / 20 for x in range(0, 17)]:
                s = score(rows, td, tc)
                best.append((td, tc, s))
        ok = [b for b in best if b[2]["abstain_precision"] >= 0.8]
        ok.sort(key=lambda b: (-b[2]["abstain_recall"], -b[2]["answerable_served"]))
        print("top 8 settings with abstain precision >= 0.8 (dev):")
        for td, tc, s in ok[:8]:
            print(f"  tau_d={td:.2f} tau_c={tc:.2f}  " + "  ".join(f"{k}={v:.3f}" for k, v in s.items()))
        un = [r for r in rows if not r["answerable"]]
        an = [r for r in rows if r["answerable"]]
        import statistics as st

        for k in ("top_dense", "coverage"):
            print(f"{k}: answerable median {st.median(r['signals'][k] for r in an):.3f}  unanswerable median {st.median(r['signals'][k] for r in un):.3f}")
    if a.tau_d is not None:
        print(json.dumps(score(rows, a.tau_d, a.tau_c or 0.0), indent=1))


if __name__ == "__main__":
    main()
