from datetime import UTC, datetime
from pathlib import Path

import pytest

from codicil.application import refresh
from codicil.application.refresh import SourceUnavailable
from codicil.domain.freshness import content_hash

NOW = lambda: datetime(2026, 11, 19, 12, 0, tzinfo=UTC)  # noqa: E731


def page(*paragraphs: str) -> bytes:
    return ("\n".join(paragraphs)).encode()


def parse(html: str) -> list[str]:
    return html.split("\n")


BODY = [f"{i}. Paragraph number {i} of the direction." for i in range(1, 31)]


class FakeFetcher:
    def __init__(self, pages: dict[str, bytes | Exception]):
        self.pages = pages

    def fetch(self, entry):
        p = self.pages[entry["doc_id"]]
        if isinstance(p, Exception):
            raise p
        return p


def entry(doc_id: str, blocks: list[str], with_hash: bool = True) -> dict:
    e = {"doc_id": doc_id, "title": doc_id, "url": "u", "sha256": "x", "retrieved_at": "2026-11-11T00:00:00Z"}
    if with_hash:
        e["content_sha256"] = content_hash(blocks)
    return e


def run(pages, manifest, raw: Path, ids):
    sel = [{"doc_id": i} for i in ids]
    return refresh.check(sel, manifest, raw, FakeFetcher(pages), parse=parse, now=NOW)


def test_statuses_for_unchanged_changed_new_failed_and_broken_pages(tmp_path):
    changed = [*BODY[:10], "11. The limit is Rs 20,000.", *BODY[11:]]
    manifest = {i: entry(i, BODY) for i in ("same", "moved", "down", "broken")}
    (tmp_path / "moved.html").write_text("\n".join(BODY))  # the old page is on disk, so a readable diff is possible
    pages = {
        "same": page(*BODY),
        "moved": page(*changed),
        "fresh": page(*BODY),
        "down": SourceUnavailable("503"),
        "broken": page("Please verify you are human"),
    }
    rep = run(pages, manifest, tmp_path, ["same", "moved", "fresh", "down", "broken"])
    by = {r.doc_id: r for r in rep.results}
    assert {k: v.status for k, v in by.items()} == {"same": "unchanged", "moved": "changed", "fresh": "new", "down": "failed", "broken": "rejected"}
    assert by["moved"].diff and by["moved"].diff.changed == 1 and "20,000" in by["moved"].diff.examples[0]
    assert [r.doc_id for r in rep.actionable] == ["moved", "fresh"]
    j = rep.as_json()
    assert (
        j["changed"] == ["moved"]
        and j["new"] == ["fresh"]
        and j["failed"] == ["down"]
        and j["rejected"] == ["broken"]
        and j["checked_at"] == "2026-11-19T12:00:00Z"
    )


def test_a_page_with_far_fewer_blocks_is_not_trusted_even_if_long_enough(tmp_path):
    manifest = {"d": entry("d", BODY * 4)}
    (tmp_path / "d.html").write_text("\n".join(BODY * 4))
    rep = run({"d": page(*BODY[:25])}, manifest, tmp_path, ["d"])  # 25 blocks against 120
    assert rep.results[0].status == "rejected"


def test_missing_baseline_is_recorded_not_reported_as_a_change(tmp_path):
    rep = run({"d": page(*BODY)}, {"d": entry("d", BODY, with_hash=False)}, tmp_path, ["d"])
    assert rep.results[0].status == "baseline" and not rep.actionable
    entries = refresh.new_manifest_entries(rep, lambda d: {"doc_id": d, "sha256": "keep"}, NOW)
    assert entries["d"]["content_sha256"] == content_hash(BODY) and entries["d"]["sha256"] == "keep"


def test_new_manifest_entries_and_page_writing(tmp_path):
    changed = [*BODY[:5], "6. Changed.", *BODY[6:]]
    manifest = {"a": entry("a", BODY), "b": entry("b", BODY)}
    rep = run({"a": page(*BODY), "b": page(*changed)}, manifest, tmp_path, ["a", "b"])
    entries = refresh.new_manifest_entries(rep, lambda d: manifest[d], NOW)
    assert set(entries) == {"b"} and entries["b"]["content_sha256"] == content_hash(changed) and entries["b"]["retrieved_at"] == "2026-11-19T12:00:00Z"
    refresh.write_pages(rep, tmp_path)
    assert (tmp_path / "b.html").read_bytes() == page(*changed)  # changed page stored
    assert (tmp_path / "a.html").exists()  # unchanged page stored because it was missing
    (tmp_path / "a.html").write_text("kept")
    refresh.write_pages(rep, tmp_path)
    assert (tmp_path / "a.html").read_text() == "kept"  # an existing unchanged page is not rewritten


def test_freshness_summary_lists_pending_changes_until_applied(tmp_path):
    rep = run({"a": page(*BODY), "b": page(*BODY[:5], "x", *BODY[6:])}, {"a": entry("a", BODY), "b": entry("b", BODY)}, tmp_path, ["a", "b"])
    pending = refresh.freshness_summary(rep, "2026-11-11T00:00:00Z", applied=False)
    assert pending["changed_upstream"] == ["b"] and pending["documents"] == 2
    assert refresh.freshness_summary(rep, "2026-11-19T12:00:00Z", applied=True)["changed_upstream"] == []


def test_write_json_is_atomic_and_readable(tmp_path):
    refresh.write_json(tmp_path / "x" / "f.json", {"k": "é"})
    assert (tmp_path / "x" / "f.json").read_text().strip().startswith("{") and not list((tmp_path / "x").glob("*.tmp"))


@pytest.mark.parametrize("bad", [b"", b"short"])
def test_empty_or_tiny_pages_are_rejected(tmp_path, bad):
    rep = run({"d": bad}, {"d": entry("d", BODY)}, tmp_path, ["d"])
    assert rep.results[0].status == "rejected"
