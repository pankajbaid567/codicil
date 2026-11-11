#!/usr/bin/env python3
"""Download the selected RBI Directions (HTML pages) into data/raw/ (gitignored) and write data/corpus/manifest.json.

Why HTML and not the PDFs: RBI serves its Master Directions as full-text HTML pages to ordinary clients, whereas the PDF host
sits behind an interactive bot challenge, which this project does not try to get around. The HTML also keeps the document's own
structure (chapters, numbered paragraphs, tables), which is better input for clause parsing than text pulled out of a PDF layout.

Polite by design: honest User-Agent, sequential requests, a pause between them, no retries. The manifest records URL, size, SHA-256
and retrieval time, so the corpus is reproducible and any change at the source is visible.
Usage: python scripts/fetch_corpus.py [--force]
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SELECTION = ROOT / "data" / "corpus" / "selection.json"
MANIFEST = ROOT / "data" / "corpus" / "manifest.json"
RAW = ROOT / "data" / "raw"
BASE = "https://rbi.org.in/Scripts/"
LISTING = BASE + "BS_ViewMasDirections.aspx"
UA = "codicil-research/0.1 (student project; +https://github.com/pankajbaid567/codicil)"
PAUSE_S = 2.0


def listing_index(client: httpx.Client) -> dict[str, str]:
    """Map PDF file stem -> detail page id, pairing the two links that share a table row of the listing page."""
    page = client.get(LISTING)
    page.raise_for_status()
    pairs: dict[str, str] = {}
    for row in re.findall(r"<tr>.*?</tr>", page.text, flags=re.S):
        detail = re.search(r"BS_ViewMasDirections\.aspx\?id=(\d+)", row)
        pdf = re.search(r"/PDFs/([^'\"/]+)\.PDF", row, flags=re.I)
        if detail and pdf:
            pairs[pdf.group(1).upper()] = detail.group(1)
    return pairs


def main(force: bool) -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    docs = json.loads(SELECTION.read_text())
    previous = {d["doc_id"]: d for d in json.loads(MANIFEST.read_text())["documents"]} if MANIFEST.exists() else {}
    out: list[dict[str, object]] = []
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=90) as client:
        index = listing_index(client)
        for i, d in enumerate(docs, 1):
            stem = d["url"].rsplit("/", 1)[-1].rsplit(".", 1)[0].upper()
            page_id = index.get(stem)
            if page_id is None:
                print(f"[{i}/{len(docs)}] NO PAGE for {d['doc_id']} (pdf stem {stem})", file=sys.stderr)
                continue
            path = RAW / f"{d['doc_id']}.html"
            if path.exists() and not force and d["doc_id"] in previous and hashlib.sha256(path.read_bytes()).hexdigest() == previous[d["doc_id"]]["sha256"]:
                print(f"[{i}/{len(docs)}] cached  {d['doc_id']}")
                out.append(previous[d["doc_id"]])
                continue
            url = f"{BASE}BS_ViewMasDirections.aspx?id={page_id}"
            print(f"[{i}/{len(docs)}] fetch   {d['doc_id']}  (id {page_id})")
            r = client.get(url)
            r.raise_for_status()
            if "TSPD" in r.text[:4000]:
                print("  bot challenge page returned; stopping (not attempting to bypass)", file=sys.stderr)
                return 2
            path.write_bytes(r.content)
            out.append(
                {
                    "doc_id": d["doc_id"],
                    "title": html.unescape(d["title"]),
                    "issuer": d["issuer"],
                    "url": url,
                    "pdf_url": d["url"],
                    "bytes": len(r.content),
                    "sha256": hashlib.sha256(r.content).hexdigest(),
                    "retrieved_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
            time.sleep(PAUSE_S)
    MANIFEST.write_text(json.dumps({"issuer": "Reserve Bank of India", "source": LISTING, "documents": out}, indent=2) + "\n")
    print(f"{len(out)} documents, {sum(int(d['bytes']) for d in out) / 1e6:.1f} MB; manifest written")
    return 0 if len(out) == len(docs) else 1


if __name__ == "__main__":
    raise SystemExit(main("--force" in sys.argv))
