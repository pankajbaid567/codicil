#!/usr/bin/env python3
"""Weekly corpus refresh: re-fetch the RBI pages, report what changed, and (with --apply) rebuild the index incrementally.

python scripts/refresh_corpus.py                     # check only: fetch, compare, write data/corpus/freshness.json
python scripts/refresh_corpus.py --apply             # also write the new pages and build a new index (only changed chunks are embedded)
python scripts/refresh_corpus.py --baseline          # no network: record the content hash of the pages already on disk

What counts as a change is the text of the parsed page, not its bytes (RBI's raw HTML differs on every request). Applying is refused when
more than --max-changed documents moved at once (a person should look), when a page parses to far fewer blocks than before, or when the new
index is much smaller than the old one. The previous index directory is kept, so rolling back is re-pointing `current`.

Exit codes: 0 nothing to apply, 10 changes found (not applied), 11 too many changes to apply unattended, 12 the new index failed its checks,
20 applied (a new index is at <index-root>/current), 2 some pages could not be fetched (the rest were still processed).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from codicil.adapters.rbi_html import parse_meta  # noqa: E402
from codicil.adapters.rbi_source import LISTING, RbiFetcher  # noqa: E402
from codicil.application import refresh  # noqa: E402
from codicil.application.index import build_index, load_index  # noqa: E402
from codicil.domain.freshness import content_hash, normalise  # noqa: E402


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def baseline(data_dir: Path) -> int:
    man = read_manifest(data_dir / "corpus/manifest.json")
    for d in man["documents"]:
        blocks = refresh.default_parse((data_dir / "raw" / f"{d['doc_id']}.html").read_text(encoding="utf-8"))
        d["content_sha256"], d["blocks"] = content_hash(blocks), len(normalise(blocks))
    refresh.write_json(data_dir / "corpus/manifest.json", man)
    print(f"recorded content hashes for {len(man['documents'])} documents")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data", help="holds corpus/ (selection, manifest, freshness) and raw/ (pages)")
    ap.add_argument("--index-root", type=Path, default=ROOT / "index")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "data/models")
    ap.add_argument("--embedder", default="multilingual-e5-small")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-changed", type=int, default=4)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()

    corpus, raw = a.data_dir / "corpus", a.data_dir / "raw"
    if a.baseline:
        return baseline(a.data_dir)
    for name in ("selection.json", "manifest.json"):  # first run on a fresh data directory: start from the snapshot shipped with the code
        if not (corpus / name).exists():
            corpus.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / "data/corpus" / name, corpus / name)
    selection = json.loads((corpus / "selection.json").read_text())
    man = read_manifest(corpus / "manifest.json")
    docs = {d["doc_id"]: d for d in man["documents"]}

    with RbiFetcher(pause_s=a.pause) as fetcher:
        report = refresh.check(selection, docs, raw, fetcher)
    for r in report.results:
        extra = f"  {r.diff.added}+ {r.diff.removed}- {r.diff.changed}~ blocks" if r.diff else (f"  {r.error}" if r.error else "")
        print(f"{r.status:9s} {r.doc_id}{extra}")
        for ex in (r.diff.examples if r.diff else ())[:3]:
            print(f"            {ex}")
    refresh.seed_missing_pages(report, raw)  # a check stores pages it has not seen yet; it never overwrites one
    snapshot = max(str(d.get("retrieved_at", "")) for d in man["documents"])
    refresh.write_json(corpus / "freshness.json", refresh.freshness_summary(report, snapshot, applied=False))
    problems = report.of("failed", "rejected")
    actionable = report.actionable
    baselines = report.of("baseline")

    if not actionable and not baselines:
        print(f"nothing changed ({len(report.results)} documents checked)" + (f"; {len(problems)} could not be checked" if problems else ""))
        return 2 if problems else 0
    if not a.apply:
        print(
            f"{len(actionable)} document(s) changed at the source; run with --apply to rebuild"
            if actionable
            else "baseline hashes available; run with --apply to record them"
        )
        return 10 if actionable else 0
    if len(actionable) > a.max_changed:
        print(f"{len(actionable)} documents changed (more than --max-changed {a.max_changed}); not applying unattended", file=sys.stderr)
        return 11

    # build the new index from memory first; only if it looks right are the pages and the manifest written
    new_html = {r.doc_id: r.html.decode("utf-8", errors="replace") for r in actionable}
    base_entry = lambda d: docs.get(d) or {"doc_id": d, **next(s for s in selection if s["doc_id"] == d)}  # noqa: E731
    entries = refresh.new_manifest_entries(report, base_entry)
    merged = {**docs, **entries}
    index_built = None
    if actionable:
        pairs = []
        for doc_id, d in merged.items():
            html = new_html.get(doc_id) or (raw / f"{doc_id}.html").read_text(encoding="utf-8")
            pairs.append((parse_meta(doc_id, d["title"], d["url"], html), html))
        from codicil.adapters.onnx_embedder import OnnxEmbedder

        old_dir = next(d for d in (a.index_root / "current", a.index_root / a.embedder, ROOT / "index" / a.embedder) if d.exists())
        old = load_index(old_dir)
        building = a.index_root / ".building"
        shutil.rmtree(building, ignore_errors=True)
        t = time.time()
        new = build_index(pairs, OnnxEmbedder(a.models_dir / a.embedder), building, reuse=old)
        if len(new.chunks) < 0.7 * len(old.chunks) or not new.docs:
            shutil.rmtree(building, ignore_errors=True)
            print(f"new index has {len(new.chunks)} chunks against {len(old.chunks)}; refusing to apply", file=sys.stderr)
            return 12
        final = a.index_root / f"{a.embedder}-{new.sha[:12]}"
        shutil.rmtree(final, ignore_errors=True)
        building.rename(final)
        tmp_link = a.index_root / ".current.tmp"
        tmp_link.unlink(missing_ok=True)
        tmp_link.symlink_to(final.name)
        os.replace(tmp_link, a.index_root / "current")
        index_built = (new, old, time.time() - t, final)
        print(f"new index {new.sha[:12]}: {new.build_stats} in {time.time() - t:.0f}s (previous {old.sha[:12]} kept at {old_dir})")

    # the build passed: now persist pages and manifest
    refresh.write_pages(report, raw)
    man["documents"] = list(merged.values())
    refresh.write_json(corpus / "manifest.json", man)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    refresh.write_json(corpus / "freshness.json", {**refresh.freshness_summary(report, stamp, applied=True), "applied_at": stamp})
    if actionable and index_built:
        new, old, secs, final = index_built
        lines = [
            f"\n## {stamp[:10]}: {len(actionable)} document(s) changed at the source",
            f"index {old.sha[:12]} -> {new.sha[:12]} ({new.build_stats['embedded']} chunks embedded, {new.build_stats['reused']} reused, {secs:.0f} s)\n",
        ]
        for r in actionable:
            d = r.diff
            lines.append(
                f"- `{r.doc_id}` ({r.status}): "
                + (f"{d.added} blocks added, {d.removed} removed, {d.changed} changed" if d else "no stored copy of the old text to compare")
            )
            lines += [f"  - {ex}" for ex in (d.examples if d else ())]
        log = a.data_dir / "corpus-changelog.md"
        head = "# Corpus changelog\n\nEvery refresh that changed a document, newest last. Source: " + LISTING + "\n" if not log.exists() else ""
        log.write_text((log.read_text() if log.exists() else head) + "\n".join(lines) + "\n")
    print(f"applied; {len(actionable)} document(s) updated")
    return 20 if actionable else 0


if __name__ == "__main__":
    raise SystemExit(main())
