"""Turn an RBI Directions HTML page into metadata and an ordered list of `RawBlock`s.

The pages are semi-structured: chapter titles are centred `<p class="head">`, section titles are `<p class="head">`, numbered
paragraphs and sub-items are plain `<p>`, and amendments are marked with `<sup title="Substituted w.e.f. ...">`. This adapter only
reads HTML and emits blocks; deciding what is a paragraph or an item is the domain's job (`domain.structure`).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from html import unescape
from html.parser import HTMLParser

from codicil.domain.models import Amendment, BlockKind, DocStatus, DocumentMeta, RawBlock

_SKIP = {"script", "style", "noscript", "head"}
_WS = re.compile(r"\s+")
_ACTION = re.compile(r"\b(Substituted|Inserted|Deleted|Omitted|Amended|Renumbered|Added|Replaced|Modified|Revised)\b", re.I)
_DATE = re.compile(r"(?:w\.e\.f\.?|with effect from|effective from|dated|on)\s+([A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4})")


def _parse_date(s: str) -> date | None:
    s = s.replace(".", "").strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_amendment(note: str) -> Amendment:
    note = _WS.sub(" ", unescape(note)).strip()
    m = _ACTION.search(note)
    d = _DATE.search(note)
    return Amendment(action=m.group(1).capitalize() if m else "Other", text=note, effective=_parse_date(d.group(1)) if d else None)


class _LayoutScan(HTMLParser):
    """First pass: find tables that wrap paragraphs (page layout) as opposed to data tables, by order of appearance."""

    def __init__(self) -> None:
        super().__init__()
        self.layout: set[int] = set()
        self._stack: list[int] = []
        self._n = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._stack.append(self._n)
            self._n += 1
        elif tag in {"p", "li"} and self._stack:
            self.layout.update(self._stack)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._stack:
            self._stack.pop()


class _Collector(HTMLParser):
    def __init__(self, layout: set[int] | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self._layout = layout or set()
        self._tables_seen = 0
        self._tstack: list[bool] = []  # True = layout table (transparent)
        self.blocks: list[RawBlock] = []
        self.title = ""
        self._skip = 0
        self._cur: list[str] | None = None
        self._cls = ""
        self._align = ""
        self._amend: list[Amendment] = []
        self._in_title = False
        self._sup = 0
        self._table_depth = 0
        self._rows: list[list[str]] = []
        self._cell: list[str] | None = None
        self._tab_amend: list[Amendment] = []

    # -- helpers
    def _flush_text(self) -> str:
        return _WS.sub(" ", "".join(self._cur or [])).strip()

    def _emit(self, kind: BlockKind, text: str, amendments: list[Amendment]) -> None:
        if text:
            self.blocks.append(RawBlock(kind=kind, text=text, amendments=tuple(amendments)))

    # -- parser callbacks
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag in _SKIP and tag != "head":
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
        if self._skip:
            return
        if tag == "table":
            is_layout = self._tables_seen in self._layout
            self._tables_seen += 1
            self._tstack.append(is_layout)
            if not is_layout:
                self._table_depth += 1
                if self._table_depth == 1:
                    self._rows, self._tab_amend = [], []
        elif self._tstack and self._tstack[-1] and tag in {"tr", "td", "th"}:
            pass  # layout table: transparent
        elif tag == "tr" and self._table_depth:
            self._rows.append([])
        elif tag in {"td", "th"} and self._table_depth:
            self._cell = []
        elif tag in {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6"} and not self._table_depth:
            self._cur, self._cls, self._align, self._amend = [], a.get("class", ""), a.get("align", ""), []
            if tag.startswith("h"):
                self._cls = "head"
        elif tag == "sup":
            self._sup += 1
            note = a.get("title", "")
            if note:
                (self._tab_amend if self._table_depth else self._amend).append(parse_amendment(note))
        elif tag == "br":
            if self._cell is not None:
                self._cell.append(" ")
            elif self._cur is not None:
                self._cur.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP and tag != "head":
            self._skip = max(0, self._skip - 1)
            return
        if tag == "title":
            self._in_title = False
        if self._skip:
            return
        if tag == "sup":
            self._sup = max(0, self._sup - 1)
        elif self._tstack and self._tstack[-1] and tag in {"tr", "td", "th"}:
            pass
        elif tag in {"td", "th"} and self._table_depth and self._cell is not None:
            self._rows[-1].append(_WS.sub(" ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "table" and self._tstack and self._tstack.pop():
            pass  # end of a layout table
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0:
                lines = [" | ".join(c for c in r if c) for r in self._rows]
                text = "\n".join(line for line in lines if line)
                if len(text) > 40:  # tiny tables are page chrome
                    self._emit(BlockKind.TABLE, text, self._tab_amend)
        elif tag in {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6"} and self._cur is not None and not self._table_depth:
            text = self._flush_text()
            cls = self._cls.split()
            kind = BlockKind.PARAGRAPH
            if "head" in cls:
                kind = BlockKind.CHAPTER if self._align == "center" else BlockKind.HEADING
            if "footnote" not in cls and "links" not in cls:
                self._emit(kind, text, self._amend)
            self._cur = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title += data
            return
        if self._sup:
            return  # footnote markers such as "1" are noise; the note itself is kept as an Amendment
        if self._cell is not None:
            self._cell.append(data)
        elif self._cur is not None:
            self._cur.append(data)


def parse_blocks(html: str) -> list[RawBlock]:
    """All blocks from the first centred heading (the first chapter) onwards; the table of contents and page chrome before it are dropped."""
    scan = _LayoutScan()
    scan.feed(html)
    scan.close()
    c = _Collector(scan.layout)
    c.feed(html)
    c.close()
    blocks = c.blocks
    start = next((i for i, b in enumerate(blocks) if b.kind is BlockKind.CHAPTER), 0)
    return blocks[start:]


_ENTITY = [
    (r"Commercial Banks", "Commercial Banks"),
    (r"Non-Banking Financial Compan", "NBFCs"),
    (r"Payments? Banks", "Payments Banks"),
    (r"Small Finance Banks", "Small Finance Banks"),
    (r"Prepaid Payment|PPIs?\b", "Prepaid Payment Instrument issuers"),
    (r"Payment Aggregator|Payment System", "Payment system operators"),
    (r"Digital Payments|digital payment transactions", "Regulated entities (digital payments)"),
]


def infer_entity(title: str) -> str:
    for pat, name in _ENTITY:
        if re.search(pat, title, re.I):
            return name
    return "Regulated entities"


def short_title(title: str) -> str:
    t = re.sub(r"\((?:Updated|updated) as on[^)]*\)", "", title)
    t = re.sub(r"^Reserve Bank of India\s*", "", t).strip()
    t = re.sub(r"^\((.*?)\)\s*", r"\1 ", t)
    return _WS.sub(" ", t).strip(" -–")


def parse_meta(doc_id: str, title: str, url: str, html: str, superseded_by: str | None = None) -> DocumentMeta:
    """Metadata from the manifest title and the page header (circular number, issue date, 'Updated as on' date)."""
    text = _WS.sub(" ", unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S))))
    upd = re.search(r"Updated as on ([A-Z][a-z]+\.? \d{1,2}, \d{4})", title, re.I)
    updated = _parse_date(upd.group(1)) if upd else None
    ref = re.search(r"\b(RBI/[A-Z0-9\-]+/\d{4}-\d{2,4}/\d+)", text)
    issued = None
    if ref:
        tail = text[ref.end() : ref.end() + 160]
        d = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", tail)
        issued = _parse_date(d.group(1)) if d else None
    status = DocStatus.SUPERSEDED if re.search(r"superseded by", title, re.I) or superseded_by else DocStatus.IN_FORCE
    return DocumentMeta(
        doc_id=doc_id,
        title=_WS.sub(" ", title).strip(),
        short_title=short_title(title),
        issuer="RBI",
        entity=infer_entity(title),
        url=url,
        reference=ref.group(1) if ref else None,
        issued_on=issued,
        updated_as_on=updated,
        status=status,
        superseded_by=superseded_by,
    )
