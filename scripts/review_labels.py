#!/usr/bin/env python3
"""Review the eval labels in your browser. Runs on your machine only (127.0.0.1); nothing is uploaded.

python scripts/review_labels.py            # then open http://127.0.0.1:8765
python scripts/review_labels.py --no-retrieval   # skip the 'closest passages' shown for unanswerable questions (faster start)

Every label in eval/questions.jsonl was drafted by an AI assistant. For each question the page shows the question, the clause that is
supposed to answer it (or, for a question that should be unanswerable, the passages the system finds closest), and you decide:
  1 correct         the evidence answers the question (or, for unanswerable: nothing in the 17 Directions answers it)
  2 wrong evidence  the question is fine but the quoted evidence does not answer it (optionally paste a better quote)
  3 bad question    the question is ambiguous, wrong, or not something a user would ask
  4 unsure          come back to it
Your verdicts are appended to eval/reviews/reviews.jsonl (the last verdict for a question wins), so you can stop and continue any time,
and commit the file when you are done. Then: make review-report
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from codicil.application.index import load_index  # noqa: E402
from codicil.domain import metrics as M  # noqa: E402
from codicil.domain import review as R  # noqa: E402

PAGE = ROOT / "eval/review/page.html"
LOG = ROOT / "eval/reviews/reviews.jsonl"
NEAREST = ROOT / "eval/work/review-nearest.json"
EMB = "multilingual-e5-small"
LOCK = threading.Lock()


def load_questions() -> list[dict]:
    return [json.loads(x) for x in (ROOT / "eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if x]


def nearest_passages(questions: list[dict], index_sha: str) -> dict[str, list[dict]]:
    """The three closest passages for each unanswerable question under the production retrieval, cached per index build."""
    if NEAREST.exists():
        cached = json.loads(NEAREST.read_text())
        if cached.get("index") == index_sha:
            return cached["items"]
    sys.path.insert(0, str(ROOT / "eval"))
    from run_answer import base_cfg, service

    from codicil.domain.glossary import normalise_query

    svc = service()
    out: dict[str, list[dict]] = {}
    todo = [q for q in questions if not q["answerable"]]
    print(f"finding the closest passages for {len(todo)} unanswerable questions (about a minute)...", flush=True)
    for q in todo:
        nq = normalise_query(q["question"])
        hits, _ = svc.retrieve(nq, base_cfg(min_dense=0.0))
        hits = svc.rerank(nq, hits, base_cfg(min_dense=0.0))[:3]
        out[q["id"]] = [
            {
                "document": svc.index.docs[c.doc_id].short_title,
                "path": c.path_str,
                "text": c.text[:900],
            }
            for c in (svc.index.chunks[svc.index.by_id[h.chunk_id]] for h in hits)
        ]
    NEAREST.parent.mkdir(parents=True, exist_ok=True)
    NEAREST.write_text(json.dumps({"index": index_sha, "items": out}, ensure_ascii=False))
    return out


def build_items(with_retrieval: bool) -> list[dict]:
    qs = load_questions()
    by_id = {q["id"]: q for q in qs}
    idx = load_index(ROOT / "index" / EMB)
    near = nearest_passages(qs, idx.sha) if with_retrieval else {}
    items = []
    for q in qs:
        gold = []
        for g in q["gold_evidence"]:
            chunk = next((c for c in idx.chunks if c.doc_id == g["doc_id"] and M.covers(c.text, g["quote"], 0.95)), None)
            gold.append(
                {
                    "document": idx.docs[g["doc_id"]].short_title,
                    "path": chunk.path_str if chunk else "",
                    "quote": g["quote"],
                    "clause": chunk.text[:2500] if chunk else "",
                    "found": chunk is not None,
                }
            )
        src = by_id.get(q.get("paraphrase_of", ""))
        items.append(
            {
                "id": q["id"],
                "split": q["split"],
                "language": q["language"],
                "slice": q["slice"],
                "answerable": q["answerable"],
                "question": q["question"],
                "paraphrase_of": {"id": src["id"], "question": src["question"]} if src else None,
                "gold": gold,
                "nearest": near.get(q["id"], []),
            }
        )
    return items


def read_reviews(log: Path = LOG) -> dict[str, R.Review]:
    if not log.exists():
        return {}
    return R.latest(R.read_log(log.read_text(encoding="utf-8").splitlines()))


def make_handler(items: list[dict], port: int, log: Path = LOG) -> type[BaseHTTPRequestHandler]:
    known = [i["id"] for i in items]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:  # quiet
            return

        def _local(self) -> bool:
            # only this machine, and only under the names the page is served on (blocks DNS-rebinding from another site)
            return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: object) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if not self._local():
                return self._json(403, {"error": "local only"})
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/items":
                return self._json(200, items)
            if self.path == "/api/reviews":
                return self._json(200, {k: v.as_json() for k, v in read_reviews(log).items()})
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._local() or self.headers.get("X-Review") != "1":
                return self._json(403, {"error": "forbidden"})
            if self.path != "/api/review":
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 20_000:
                    raise R.ReviewError("body too large or empty")
                rev = R.parse_review(json.loads(self.rfile.read(n)), known_ids=known)
            except (ValueError, R.ReviewError) as e:
                return self._json(400, {"error": str(e)})
            with LOCK:
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rev.as_json(), ensure_ascii=False) + "\n")
            self._json(200, {"ok": True, "review": rev.as_json()})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--log", type=Path, default=LOG, help="where verdicts are appended")
    a = ap.parse_args()
    items = build_items(with_retrieval=not a.no_retrieval)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(items, a.port, a.log))
    done = len(read_reviews(a.log))
    print(f"{len(items)} questions, {done} already reviewed. Open http://127.0.0.1:{a.port}  (Ctrl-C to stop; verdicts are saved as you go)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
