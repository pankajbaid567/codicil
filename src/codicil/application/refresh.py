"""Keep the corpus current: compare each source page with the recorded snapshot, report what changed, apply it safely.

`check` fetches every selected document through a `Fetcher` and compares the content hash of its parsed text with the one in the
manifest. `apply` writes the new pages and manifest entries, but refuses a page that looks broken (far fewer blocks than before),
so a bot-challenge page or a layout change at the source can never replace good text. Nothing here talks to the network itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from codicil.domain.freshness import BlockDiff, content_hash, diff_blocks, normalise

MIN_BLOCKS = 20  # a regulation page parses to far more; fewer means the page is not the document
MIN_KEEP_RATIO = 0.5  # a refreshed page with less than half the old block count is rejected as broken


class SourceUnavailable(Exception):
    """The page could not be fetched (network error, bot challenge, document missing from the listing)."""


class Fetcher(Protocol):
    def fetch(self, entry: dict[str, Any]) -> bytes: ...


ParseFn = Callable[[str], list[str]]  # html -> block texts


def default_parse(html: str) -> list[str]:
    from codicil.adapters.rbi_html import parse_blocks  # imported lazily: the adapter layer sits above this one

    return [b.text for b in parse_blocks(html)]


@dataclass(frozen=True, slots=True)
class DocCheck:
    doc_id: str
    status: str  # unchanged | baseline | changed | new | failed | rejected
    content_sha256: str = ""
    blocks: int = 0
    diff: BlockDiff | None = None
    error: str = ""
    html: bytes = field(default=b"", repr=False)

    def as_json(self) -> dict[str, object]:
        out: dict[str, object] = {"doc_id": self.doc_id, "status": self.status, "blocks": self.blocks}
        if self.diff is not None:
            out["diff"] = self.diff.as_json()
        if self.error:
            out["error"] = self.error
        return out


@dataclass(frozen=True, slots=True)
class Report:
    checked_at: str
    results: tuple[DocCheck, ...]

    def of(self, *status: str) -> list[DocCheck]:
        return [r for r in self.results if r.status in status]

    @property
    def actionable(self) -> list[DocCheck]:
        return self.of("changed", "new")

    def as_json(self) -> dict[str, object]:
        return {
            "checked_at": self.checked_at,
            "documents": len(self.results),
            "changed": [r.doc_id for r in self.of("changed")],
            "new": [r.doc_id for r in self.of("new")],
            "baseline": [r.doc_id for r in self.of("baseline")],
            "failed": [r.doc_id for r in self.of("failed")],
            "rejected": [r.doc_id for r in self.of("rejected")],
            "per_doc": {r.doc_id: r.as_json() for r in self.results if r.status != "unchanged"},
        }


def _baseline_hash(entry: dict[str, Any], raw_dir: Path, parse: ParseFn) -> tuple[str, list[str]]:
    """The recorded content hash of the snapshot, plus its blocks when the old page is still on disk (for a readable diff)."""
    path = raw_dir / f"{entry['doc_id']}.html"
    old_blocks = parse(path.read_text(encoding="utf-8")) if path.exists() else []
    recorded = str(entry.get("content_sha256") or "")
    return (recorded or (content_hash(old_blocks) if old_blocks else "")), old_blocks


def check(
    selection: Sequence[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    raw_dir: Path,
    fetcher: Fetcher,
    parse: ParseFn = default_parse,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Report:
    results: list[DocCheck] = []
    for sel in selection:
        doc_id = str(sel["doc_id"])
        entry = manifest.get(doc_id)
        try:
            html = fetcher.fetch(entry or sel)
        except SourceUnavailable as e:
            results.append(DocCheck(doc_id, "failed", error=str(e)))
            continue
        blocks = parse(html.decode("utf-8", errors="replace"))
        new_hash = content_hash(blocks)
        n = len(normalise(blocks))
        if entry is None:
            results.append(DocCheck(doc_id, "new", new_hash, n, html=html))
            continue
        old_hash, old_blocks = _baseline_hash({**entry, "doc_id": doc_id}, raw_dir, parse)
        old_n = len(normalise(old_blocks)) or int(entry.get("blocks") or 0)
        if n < MIN_BLOCKS or (old_n and n < MIN_KEEP_RATIO * old_n):
            results.append(DocCheck(doc_id, "rejected", new_hash, n, error=f"page parses to {n} blocks (snapshot had {old_n or 'unknown'}); not trusted"))
        elif not old_hash:
            results.append(DocCheck(doc_id, "baseline", new_hash, n, html=html))  # nothing recorded to compare with: record this page as the baseline
        elif new_hash == old_hash:
            results.append(DocCheck(doc_id, "unchanged", new_hash, n, html=html))
        else:
            results.append(DocCheck(doc_id, "changed", new_hash, n, diff_blocks(old_blocks, blocks) if old_blocks else None, html=html))
    return Report(now().strftime("%Y-%m-%dT%H:%M:%SZ"), tuple(results))


def new_manifest_entries(
    report: Report, manifest_doc: Callable[[str], dict[str, Any]], now: Callable[[], datetime] = lambda: datetime.now(UTC)
) -> dict[str, dict[str, Any]]:
    """The manifest entries that a refresh changes: changed and new pages get a new hash, size and time; a baseline page only gets its content hash. No I/O."""
    stamp = now().strftime("%Y-%m-%dT%H:%M:%SZ")
    updated: dict[str, dict[str, Any]] = {}
    for r in report.results:
        if r.status == "baseline":
            updated[r.doc_id] = {**manifest_doc(r.doc_id), "content_sha256": r.content_sha256, "blocks": r.blocks}
        elif r.status in ("changed", "new"):
            entry = dict(manifest_doc(r.doc_id))
            entry.update(bytes=len(r.html), sha256=hashlib.sha256(r.html).hexdigest(), content_sha256=r.content_sha256, blocks=r.blocks, retrieved_at=stamp)
            updated[r.doc_id] = entry
    return updated


def write_pages(report: Report, raw_dir: Path) -> None:
    """Store the pages of changed and new documents. Unchanged and baseline pages are only stored if missing, so the next check can show a readable diff."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for r in report.results:
        path = raw_dir / f"{r.doc_id}.html"
        if r.status in ("changed", "new") or (r.status in ("unchanged", "baseline") and not path.exists() and r.html):
            path.write_bytes(r.html)


def freshness_summary(report: Report, snapshot_retrieved_at: str, applied: bool) -> dict[str, object]:
    """The small record the web service shows: when we last looked, and whether the source has moved since the snapshot."""
    pending = [] if applied else [r.doc_id for r in report.actionable]
    return {
        "checked_at": report.checked_at,
        "snapshot_retrieved_at": snapshot_retrieved_at,
        "documents": len(report.results),
        "changed_upstream": pending,
        "failed": [r.doc_id for r in report.of("failed", "rejected")],
    }


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
