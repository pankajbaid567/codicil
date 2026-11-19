#!/usr/bin/env python3
"""How much do the headline numbers depend on labels the author has not confirmed?

python eval/review_sensitivity.py

Reads the author's verdicts (eval/reviews/reviews.jsonl) and recomputes the final configuration's retrieval quality and English
abstention on (a) every question and (b) only the questions whose label the author marked correct. If the two agree, unreviewed
labels are not driving the result; if they differ, the confirmed-only column is the one to trust. Writes eval/results/review-sensitivity.json.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from run_answer import score  # noqa: E402
from run_retrieval import CONFIGS, evaluate, load_questions  # noqa: E402

from codicil.domain import review as R  # noqa: E402

FINAL = "hybrid-e5-routed-rerank-en"
TAU_D, TAU_C, TAU_R = 0.84, 0.0, -0.18  # frozen on dev (docs/results.md)


def retrieval_means(per_item: list[dict], ids: set[str] | None) -> dict[str, float | int]:
    rows = [r for r in per_item if ids is None or r["id"] in ids]
    return {"n": len(rows), **{m: round(st.fmean(r[m] for r in rows), 3) for m in ("ndcg@10", "recall@10", "mrr@10")}} if rows else {"n": 0}


def abstention(rows: list[dict], ids: set[str] | None) -> dict[str, float | int]:
    sub = [r for r in rows if r["language"] == "en" and (ids is None or r["id"] in ids)]
    if not any(not r["answerable"] for r in sub) or not any(r["answerable"] for r in sub):
        return {"n": len(sub)}
    s = score(sub, TAU_D, TAU_C, TAU_R)
    return {"n": len(sub), "n_unanswerable": sum(not r["answerable"] for r in sub), **{k: round(v, 3) for k, v in s.items()}}


def main() -> int:
    log = ROOT / "eval/reviews/reviews.jsonl"
    reviews = R.latest(R.read_log(log.read_text(encoding="utf-8").splitlines())) if log.exists() else {}
    if not reviews:
        print("no reviews yet: run `make review` first")
        return 1
    questions = load_questions("all")
    summary = R.summarise(questions, reviews)
    confirmed = R.confirmed_ids(reviews)
    cfg = CONFIGS[FINAL]
    per_item = evaluate(cfg[0], cfg[1], questions, cfg[2], *cfg[3:])["ALL"]["_per_item"]
    signals = [r for split in ("dev", "test") for r in json.loads((ROOT / f"eval/results/signals-{split}.json").read_text())]
    out = {
        "reviewed": summary["reviewed"],
        "total": summary["total"],
        "by_verdict": summary["by_verdict"],
        "config": FINAL,
        "retrieval": {"all": retrieval_means(per_item, None), "confirmed_only": retrieval_means(per_item, confirmed)},
        "abstention_english": {"all": abstention(signals, None), "confirmed_only": abstention(signals, confirmed)},
    }
    (ROOT / "eval/results/review-sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
