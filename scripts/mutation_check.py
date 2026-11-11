#!/usr/bin/env python3
"""Inject known bugs into the source, one at a time, and confirm the test suite fails. A surviving mutant is a gap in the tests.

Usage: python scripts/mutation_check.py     (restores every file afterwards, even on Ctrl-C)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (description, file, exact old text, replacement)
MUTANTS = [
    ("RRF: rank offset off by one", "src/codicil/domain/fusion.py", "wi / (k + r)", "wi / (k + r + 1)"),
    (
        "RRF: duplicate ids in one list counted twice",
        "src/codicil/domain/fusion.py",
        "            if doc in seen:  # a duplicate in one list must not count twice\n                continue\n",
        "",
    ),
    (
        "BM25: idf sign flipped",
        "src/codicil/domain/bm25.py",
        "math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))",
        "-math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))",
    ),
    ("verify: numbers no longer checked", "src/codicil/domain/verify.py", "    if not numbers(sentence) <= numbers(evidence):\n        return 0.0\n", ""),
    ("abstention threshold ignored", "src/codicil/application/answer.py", "weak = top_dense < cfg.min_dense and", "weak = False and"),
    ("rerank abstention ignored", "src/codicil/application/answer.py", "weak = hits[0].score < cfg.min_rerank", "weak = False"),
    (
        "chunker: last child group never flushed",
        "src/codicil/domain/chunking.py",
        "            group.append(ctext)\n            used += ct\n        flush(path)\n",
        "            group.append(ctext)\n            used += ct\n",
    ),
    (
        "structure: unlabelled text starts a new paragraph instead of joining the clause",
        "src/codicil/domain/structure.py",
        "            attach_text(target, text, b.amendments)\n    return root.freeze()",
        "            container().children.append(_B(NodeKind.PARAGRAPH, '', '', text, i))\n    return root.freeze()",
    ),
    ("embedder: passage prefix dropped for e5", "src/codicil/adapters/onnx_embedder.py", '("query: ", "passage: ", "mean")', '("query: ", "query: ", "mean")'),
    ("cache key without index hash", "src/codicil/api/app.py", 'f"{index_sha}|{question.strip().lower()}|{llm}"', 'f"{question.strip().lower()}|{llm}"'),
    (
        "glossary: original words dropped from normalised query",
        "src/codicil/domain/glossary.py",
        'normalised=f"{canon} {text}" if canon != text.lower() else text',
        "normalised=canon",
    ),
    ("retrieval: superseded documents not filtered", "src/codicil/application/retrieve.py", "if cfg.include_superseded or self._status_ok.all():", "if True:"),
    ("circuit breaker never opens", "src/codicil/adapters/openai_compat.py", "if self._fails >= self.threshold:", "if False:"),
    ("rate limiter always allows", "src/codicil/api/app.py", "ok = tokens >= 1.0", "ok = True"),
    ("index hash not verified on load", "src/codicil/application/index.py", '    if h.hexdigest() != man["sha256"]:', "    if False:"),
]


def run_tests() -> tuple[bool, str]:
    r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--no-header"], cwd=ROOT, capture_output=True, text=True, timeout=900)
    failed = r.returncode != 0
    tail = [line for line in r.stdout.splitlines() if line.startswith("FAILED") or "Error" in line][:1]
    return failed, tail[0][:110] if tail else ""


def main() -> int:
    caught = 0
    originals: dict[Path, str] = {}
    try:
        for desc, rel, old, new in MUTANTS:
            path = ROOT / rel
            originals.setdefault(path, path.read_text())
            src = originals[path]
            if src.count(old) != 1:
                print(f"SKIP  {desc}: pattern found {src.count(old)} times in {rel}")
                continue
            path.write_text(src.replace(old, new))
            t = time.time()
            failed, where = run_tests()
            path.write_text(src)
            print(f"{'CAUGHT  ' if failed else 'SURVIVED'} {desc} ({time.time() - t:.0f}s) {where}")
            caught += failed
    finally:
        for p, s in originals.items():
            p.write_text(s)
    print(f"\n{caught}/{len(MUTANTS)} mutants caught")
    return 0 if caught == len(MUTANTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
