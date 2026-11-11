#!/usr/bin/env python3
"""Compile eval/authored/*.py question lists into eval/questions.jsonl.

An authored item is (id, question, doc_key, phrase) for answerable questions: `phrase` is a verbatim fragment of the clause that answers it.
The script locates the phrase in the named document, widens it to the full sentence, and stores that sentence as the gold evidence quote.
A phrase that is not found verbatim is a hard error, so a typo can never become an unanswerable 'answerable' question.
Unanswerable items are (id, question, None, None). Paraphrase items (Hinglish/Hindi) point at a source id and reuse its evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from codicil.adapters.rbi_html import parse_blocks  # noqa: E402
from codicil.domain.tokens import sentences  # noqa: E402

DOCS = {
    "rbc": "commercial-banks-responsible",
    "cc": "commercial-banks-credit-cards",
    "cir": "commercial-banks-credit-information",
    "kyc": "commercial-banks-know-your",
    "dep": "commercial-banks-interest-rate-on-deposits",
    "adv": "commercial-banks-interest-rates-on",
    "dps": "commercial-banks-digital-payment",
    "emandate": "digital-payments-e-mandate",
    "auth": "authentication-mechanisms",
    "ppi": "on-prepaid-payment",
    "aa": "non-banking-financial-companies-account-aggregator",
    "p2p": "non-banking-financial-companies-peer",
    "nbfcrbc": "non-banking-financial-companies-responsible",
    "ombud": "commercial-banks-internal-ombudsman",
    "nbfccc": "non-banking-financial-companies-credit-cards",
    "psl": "priority-sector",
    "da": "deposits-and-accounts",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_text(doc_id: str) -> str:
    blocks = parse_blocks((ROOT / "data/raw" / f"{doc_id}.html").read_text(encoding="utf-8"))
    return norm(" ".join(b.text for b in blocks))


def resolve(key: str) -> str:
    hits = [p.stem for p in (ROOT / "data/raw").glob("*.html") if p.stem.startswith(DOCS[key])]
    if len(hits) != 1:
        raise SystemExit(f"doc key {key!r} resolves to {hits}")
    return hits[0]


def expand(text: str, phrase: str) -> str:
    i = text.find(norm(phrase))
    if i < 0:
        raise KeyError(phrase)
    pos = 0
    for s in sentences(text):
        j = text.find(s, pos)
        if j < 0:
            continue
        pos = j + len(s)
        if j <= i < pos:
            return s
    return norm(phrase)


def split_of(qid: str) -> str:
    return "dev" if int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 2 == 0 else "test"


def main() -> int:
    items: list[dict] = []
    for f in sorted((ROOT / "eval/authored").glob("*.py")):
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        items += [dict(zip(("id", "question", "doc", "phrase", "slice", "lang", "source"), (*t, *([None] * 7))[:7], strict=False)) for t in mod.ITEMS]
    texts: dict[str, str] = {}
    out: dict[str, dict] = {}
    errors = 0
    for it in items:
        qid, slice_, lang = it["id"], it["slice"] or "single", it["lang"] or "en"
        rec = {
            "id": qid,
            "question": it["question"],
            "language": lang,
            "slice": slice_,
            "answerable": it["doc"] is not None or bool(it["source"] and out[it["source"]]["answerable"]),
            "gold_evidence": [],
            "split": None,
            "provenance": "drafted by an AI assistant from the clause text; author review recorded in eval/README.md",
        }
        if it["source"]:
            src = out[it["source"]]
            rec["gold_evidence"], rec["split"], rec["paraphrase_of"] = (
                src["gold_evidence"],
                src["split"],
                it["source"],
            )  # an unanswerable source stays unanswerable
        elif it["doc"]:
            docs = it["doc"] if isinstance(it["doc"], list) else [(it["doc"], it["phrase"])]
            for key, phrase in docs:
                doc_id = resolve(key)
                texts.setdefault(doc_id, load_text(doc_id))
                try:
                    rec["gold_evidence"].append({"doc_id": doc_id, "quote": expand(texts[doc_id], phrase)})
                except KeyError:
                    print(f"NOT FOUND [{qid}] {key}: {phrase!r}")
                    errors += 1
            rec["split"] = split_of(qid)
        else:
            rec["split"] = split_of(qid)
        out[qid] = rec
    if errors:
        return 1
    (ROOT / "eval/questions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out.values()) + "\n", encoding="utf-8")
    from collections import Counter

    print(len(out), "questions;", dict(Counter(r["slice"] for r in out.values())), dict(Counter(r["split"] for r in out.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
