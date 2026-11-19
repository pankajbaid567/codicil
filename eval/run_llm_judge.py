#!/usr/bin/env python3
"""Pairwise AI-graded comparison of LLM answers against extractive answers, on Amazon Bedrock.

Needs AWS credentials with bedrock:InvokeModel on both models (on EC2 the instance role is enough).

python eval/run_llm_judge.py --generator openai.gpt-oss-120b-1:0 --judge qwen.qwen3-32b-v1:0 --region ap-south-1

For every answerable question (dev and test, all languages) the same retrieval is answered twice, once in extractive mode and once in
LLM mode. A different model than the generator then reads the question, the authoritative clause text and the two answers (position
randomised by running every pair in both orders) and says which answer a reader should prefer and whether each is correct and
supported by the clause. A pair counts as a win or a loss only if both orders agree; otherwise it is a tie. These are AI grades, not
human grades. The generator never sees the gold evidence, and nothing here is used to tune a threshold.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from run_answer import base_cfg, service  # noqa: E402
from run_retrieval import load_questions  # noqa: E402

from codicil.adapters.bedrock_converse import BedrockConverseGenerator  # noqa: E402
from codicil.adapters.openai_compat import CircuitBreaker  # noqa: E402
from codicil.application.ports import GenerationError, GenerationRequest  # noqa: E402
from codicil.domain import metrics as M  # noqa: E402
from codicil.wiring import default_answer_config  # noqa: E402

JUDGE_SYSTEM = (
    "You grade two answers (A and B) to a question about Reserve Bank of India regulations. You are given the authoritative clause text. "
    "Judge ONLY against that clause. An answer is correct if it states what the clause says in response to the question. "
    "It is unsupported if it states anything the clause does not support or contradicts it. An answer that is cut off, or only an introduction "
    "or heading without the substance the question asks for, is not correct. Ignore style otherwise. Reply with JSON only: "
    '{"a_correct": true|false, "b_correct": true|false, "a_unsupported": true|false, "b_unsupported": true|false, "better": "A"|"B"|"tie"} '
    'where "better" is the answer a reader should prefer for this question, and "tie" means equally good.'
)


def reference_text(svc, gold: list[dict]) -> str:
    """The clause text for the gold evidence: the whole chunk that contains each gold quote, else the quote itself."""
    parts = []
    for g in gold:
        chunk = next((c for c in svc.index.chunks if c.doc_id == g["doc_id"] and M.covers(c.text, g["quote"], 0.95)), None)
        parts.append((chunk.text if chunk else g["quote"])[:1800])
    return "\n---\n".join(parts)


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    return d if d.get("better") in ("A", "B", "tie") else None


def judge(judge_llm: BedrockConverseGenerator, question: str, reference: str, a: str, b: str) -> dict | None:
    judge_llm.breaker = CircuitBreaker()
    user = f"Question: {question}\n\nAuthoritative clause text:\n{reference}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}"
    try:
        return parse(judge_llm.generate(GenerationRequest(JUDGE_SYSTEM, user, max_tokens=400, temperature=0.0)).text)
    except GenerationError:
        return None


def bootstrap_ci(values: list[float], n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(st.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="openai.gpt-oss-120b-1:0")
    ap.add_argument("--judge", default="qwen.qwen3-32b-v1:0")
    ap.add_argument("--region", default="ap-south-1")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=ROOT / "eval/results/llm-bedrock-judged.json")
    a = ap.parse_args()
    if a.generator == a.judge:
        sys.exit("the judge must be a different model than the generator")

    svc = service()
    gen = BedrockConverseGenerator.connect(a.generator, a.region, timeout=25.0, reasoning_effort=a.effort)
    judge_llm = BedrockConverseGenerator.connect(a.judge, a.region, timeout=40.0)
    svc.generator = gen
    questions = [q for split in ("dev", "test") for q in load_questions(split) if q["answerable"]]
    if a.limit:
        questions = questions[: a.limit]

    cfg = base_cfg(min_dense=0.0)  # answer quality given that an answer is served; abstention is checked separately below
    rows = []
    for q in questions:
        gen.breaker = CircuitBreaker()
        time.sleep(a.pause)
        t = time.perf_counter()
        llm = svc.ask(q["question"], cfg, use_llm=True)
        llm_ms = (time.perf_counter() - t) * 1000
        ext = svc.ask(q["question"], cfg, use_llm=False)
        row = {
            "id": q["id"], "split": q["split"], "language": q["language"], "slice": q["slice"],
            "llm_mode": llm.mode, "llm_abstained": llm.abstained, "ext_abstained": ext.abstained, "llm_ms": llm_ms,
            "llm_answer": " ".join(s.text for s in llm.sentences), "ext_answer": " ".join(s.text for s in ext.sentences),
        }  # fmt: skip
        if llm.mode == "llm" and not llm.abstained and not ext.abstained and row["llm_answer"] != row["ext_answer"]:
            ref = reference_text(svc, q["gold_evidence"])
            j1 = judge(judge_llm, q["question"], ref, row["ext_answer"], row["llm_answer"])  # A = extractive, B = LLM
            j2 = judge(judge_llm, q["question"], ref, row["llm_answer"], row["ext_answer"])  # A = LLM, B = extractive
            row["judge_ok"] = bool(j1 and j2)
            if j1 and j2:
                v1 = {"A": "ext", "B": "llm", "tie": "tie"}[j1["better"]]
                v2 = {"A": "llm", "B": "ext", "tie": "tie"}[j2["better"]]
                row["verdict_1"], row["verdict_2"] = v1, v2
                row["verdict"] = v1 if v1 == v2 else "tie"
                row["llm_correct"] = (bool(j1["b_correct"]) + bool(j2["a_correct"])) / 2
                row["ext_correct"] = (bool(j1["a_correct"]) + bool(j2["b_correct"])) / 2
                row["llm_unsupported"] = (bool(j1["b_unsupported"]) + bool(j2["a_unsupported"])) / 2
                row["ext_unsupported"] = (bool(j1["a_unsupported"]) + bool(j2["b_unsupported"])) / 2
        rows.append(row)
        print(f"{q['id']:5s} {row['llm_mode']:10s} {row.get('verdict', '-')}", flush=True)

    # unanswerable questions under the production configuration: does the LLM mode answer where the extractive one abstains?
    prod = default_answer_config()
    unans = [q for split in ("dev", "test") for q in load_questions(split) if not q["answerable"]]
    ua = {"n": len(unans), "extractive_answered": 0, "llm_answered": 0}
    for q in unans:
        gen.breaker = CircuitBreaker()
        time.sleep(a.pause)
        ua["extractive_answered"] += not svc.ask(q["question"], prod, use_llm=False).abstained
        ua["llm_answered"] += not svc.ask(q["question"], prod, use_llm=True).abstained

    judged = [r for r in rows if r.get("judge_ok")]
    out = {"generator": a.generator, "judge": a.judge, "rows": rows, "unanswerable": ua}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== SUMMARY ===")
    print(f"generator {a.generator}, judge {a.judge}; answerable questions: {len(rows)}")
    print(f"served by the LLM: {sum(r['llm_mode'] == 'llm' for r in rows)}, fell back to extractive: {sum(r['llm_mode'] != 'llm' for r in rows)}")
    same = sum(r["llm_mode"] == "llm" and r["llm_answer"] == r["ext_answer"] for r in rows)
    print(f"identical text in both modes (not judged): {same}; judged pairs: {len(judged)}; judge failures: {sum(r.get('judge_ok') is False for r in rows)}")
    if judged:
        for name in ("all", "dev", "test", "en", "hinglish", "hi"):
            sub = judged if name == "all" else [r for r in judged if name in (r["split"], r["language"])]
            if not sub:
                continue
            w, l, t_ = (sum(r["verdict"] == k for r in sub) for k in ("llm", "ext", "tie"))
            net = [(r["verdict"] == "llm") - (r["verdict"] == "ext") for r in sub]
            lo, hi = bootstrap_ci([float(x) for x in net]) if len(sub) > 1 else (0.0, 0.0)
            print(
                f"  {name:9s} n={len(sub):3d}  LLM wins {w}, ties {t_}, extractive wins {l}   net (wins-losses)/n = {st.fmean(net):+.2f} (95% CI {lo:+.2f} to {hi:+.2f})"
            )
        order = sum(r["verdict_1"] != r["verdict_2"] for r in judged)
        print(f"judge disagreed with itself when the order was swapped on {order}/{len(judged)} pairs (counted as ties)")
        for key in ("correct", "unsupported"):
            print(
                f"  {key:11s} LLM {st.fmean(r['llm_' + key] for r in judged):.2f}   extractive {st.fmean(r['ext_' + key] for r in judged):.2f}   (mean of both judge orders)"
            )
        print(
            f"answer length, median characters: LLM {st.median(len(r['llm_answer']) for r in judged):.0f}  extractive {st.median(len(r['ext_answer']) for r in judged):.0f}"
        )
    print(f"LLM-mode latency p50 {st.median(r['llm_ms'] for r in rows):.0f} ms")
    print(
        f"unanswerable questions answered under the production configuration: extractive {ua['extractive_answered']}/{ua['n']}, LLM {ua['llm_answered']}/{ua['n']}"
    )


if __name__ == "__main__":
    main()
