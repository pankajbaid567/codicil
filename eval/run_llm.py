#!/usr/bin/env python3
"""Compare LLM phrasing models on answerable questions. Reads CODICIL_LLM_BASE_URL / CODICIL_LLM_API_KEY from .env or your shell (the key is never printed or saved).
python eval/run_llm.py --models Qwen/Qwen3.5-27B google/gemma-3-27b-it --split dev --n 30
Per question: did the LLM answer survive verification (mode llm vs fallback), is the gold sentence covered, how long it took."""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from run_answer import base_cfg, service  # noqa: E402
from run_retrieval import load_questions  # noqa: E402

from codicil.adapters.openai_compat import CircuitBreaker, OpenAICompatGenerator  # noqa: E402
from codicil.domain import metrics as M  # noqa: E402
from codicil.settings import Settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--language", nargs="*", help="only these languages, e.g. en hinglish hi (see questions.jsonl)")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between questions, to stay under a free-tier rate limit")
    a = ap.parse_args()
    s = Settings()
    if not (s.llm_base_url and s.llm_api_key.get_secret_value()):
        sys.exit("fill CODICIL_LLM_BASE_URL and CODICIL_LLM_API_KEY in .env first")
    qs = [q for q in load_questions(a.split) if q["answerable"] and (not a.language or q["language"] in a.language)][: a.n]
    svc = service()
    (ROOT / "eval/results").mkdir(exist_ok=True)
    for model in a.models:
        svc.generator = OpenAICompatGenerator(
            s.llm_base_url, s.llm_api_key.get_secret_value(), model, s.llm_timeout_s, s.llm_reasoning_effort, s.llm_send_temperature
        )
        rows = []
        for q in qs:
            svc.generator.breaker = CircuitBreaker()  # a rate-limit burst must not make later questions look like model failures
            time.sleep(a.pause)
            t = time.perf_counter()
            ans = svc.ask(q["question"], base_cfg(min_dense=0.0))
            base = svc.ask(q["question"], base_cfg(min_dense=0.0), use_llm=False)  # extractive answer to the same question, same metric
            ms = (time.perf_counter() - t) * 1000
            text = " ".join(x.text for x in ans.sentences)
            gold = [g["quote"] for g in q["gold_evidence"]]
            btext = " ".join(x.text for x in base.sentences)
            rows.append(
                {
                    "id": q["id"],
                    "language": q["language"],
                    "mode": ans.mode,
                    "abstained": ans.abstained,
                    "why": next((t.note for t in ans.timings if "llm failed" in t.note), ""),
                    "base_hit": any(M.covers(btext, g) for g in gold),
                    "hit": any(M.covers(text, g) for g in gold),
                    "ms": ms,
                    "answer": text,
                }
            )
        why = Counter(r["why"][:160] for r in rows if r["why"])
        used = [r for r in rows if r["mode"] == "llm" and not r["abstained"]]
        print(
            f"\n{model}: n={len(rows)} llm-served={len(used)} fell-back={sum(r['mode'] != 'llm' for r in rows)} abstained={sum(r['abstained'] for r in rows)}"
        )
        print(f"  gold covered (llm-served): {sum(r['hit'] for r in used)}/{len(used)}   p50 {st.median(r['ms'] for r in rows):.0f} ms")
        print(f"  same questions, extractive baseline: gold covered {sum(r['base_hit'] for r in used)}/{len(used)} (llm-served subset)")
        for reason, c in why.most_common(3):
            print(f"  fallback x{c}: {reason}")
        for lang in sorted({r["language"] for r in rows}):
            sub = [r for r in used if r["language"] == lang]
            print(f"  {lang}: served {len(sub)}, gold covered {sum(r['hit'] for r in sub)}")
        (ROOT / "eval/results" / f"llm-{model.replace('/', '_')}-{a.split}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
