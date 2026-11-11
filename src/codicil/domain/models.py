"""Immutable value objects shared by every layer. No I/O, no clock, no randomness."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class BlockKind(StrEnum):
    CHAPTER = "chapter"  # centred "Chapter II - Title"
    HEADING = "heading"  # "A. Eligibility", "B.1 Policy and Procedures"
    PARAGRAPH = "paragraph"  # ordinary text, possibly starting with a marker such as "7." or "(1)"
    TABLE = "table"  # one table flattened to "a | b | c" lines


@dataclass(frozen=True, slots=True)
class Amendment:
    """An inline amendment note, e.g. 'Substituted w.e.f. October 1, 2026 ...' attached to a clause."""

    action: str  # Substituted, Inserted, Deleted, Amended, Renumbered, Other
    text: str  # the full note as printed
    effective: date | None = None


@dataclass(frozen=True, slots=True)
class RawBlock:
    """One block of source text in reading order, before any structure has been inferred."""

    kind: BlockKind
    text: str
    amendments: tuple[Amendment, ...] = ()


class DocStatus(StrEnum):
    IN_FORCE = "in_force"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class DocumentMeta:
    doc_id: str
    title: str
    short_title: str
    issuer: str
    entity: str  # who the Directions apply to: "Commercial Banks", "NBFCs", ...
    url: str
    reference: str | None = None  # "RBI/DOR/2025-26/169"
    issued_on: date | None = None
    updated_as_on: date | None = None
    status: DocStatus = DocStatus.IN_FORCE
    superseded_by: str | None = None


class NodeKind(StrEnum):
    DOCUMENT = "document"
    CHAPTER = "chapter"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    ITEM = "item"
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class Node:
    """A clause-tree node. `text` is the node's own text; the text of descendants is in `children`."""

    kind: NodeKind
    label: str  # "Chapter II", "B.1", "7", "(1)", "(a)", "" for the root
    title: str  # heading text for chapters and sections, else ""
    text: str
    children: tuple[Node, ...] = ()
    amendments: tuple[Amendment, ...] = ()
    block_index: int = -1  # index of the first source block, for traceability

    def walk(self) -> list[Node]:
        out = [self]
        for c in self.children:
            out.extend(c.walk())
        return out


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrieval unit: one clause or a run of sibling clauses with the stem that gives them meaning."""

    chunk_id: str
    doc_id: str
    path: tuple[str, ...]  # ("Chapter II", "B.1", "8")
    header: str  # deterministic contextual header, embedded and indexed with the text
    text: str
    parent_id: str | None
    n_tokens: int
    amendments: tuple[Amendment, ...] = ()

    @property
    def indexed_text(self) -> str:
        return f"{self.header}\n{self.text}"

    @property
    def path_str(self) -> str:
        return " > ".join(self.path)


@dataclass(frozen=True, slots=True)
class Parent:
    """Wider context returned to the reader for a retrieved child chunk."""

    parent_id: str
    doc_id: str
    path: tuple[str, ...]
    text: str
    n_tokens: int


@dataclass(frozen=True, slots=True)
class Query:
    raw: str
    normalised: str  # canonical English used for retrieval
    language: str  # "en", "hi" (Devanagari), "hinglish"
    entity_hints: tuple[str, ...] = ()
    matched_terms: tuple[tuple[str, str], ...] = ()  # (original, canonical) glossary hits


@dataclass(frozen=True, slots=True)
class Scored:
    chunk_id: str
    score: float
    rank: int
    source: str  # "bm25", "dense", "fused", "rerank"


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    doc_id: str
    doc_title: str
    path: str
    quote: str
    updated_as_on: date | None
    status: DocStatus


@dataclass(frozen=True, slots=True)
class AnswerSentence:
    text: str
    citation_ids: tuple[str, ...]
    supported: bool = True


@dataclass(frozen=True, slots=True)
class StageTiming:
    stage: str
    ms: float
    note: str = ""


@dataclass(frozen=True, slots=True)
class Answer:
    sentences: tuple[AnswerSentence, ...]
    citations: tuple[Citation, ...]
    abstained: bool
    mode: str  # "extractive" | "llm"
    index_sha: str
    abstain_reason: str = ""
    closest: tuple[Citation, ...] = ()
    timings: tuple[StageTiming, ...] = ()
    signals: tuple[tuple[str, float], ...] = ()
    notice: str = "Informational only. Not legal or financial advice; check the cited RBI text."


def stable_id(*parts: str, length: int = 12) -> str:
    """Deterministic short id from content, so rebuilding an index yields identical ids."""
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return h[:length]


@dataclass(frozen=True, slots=True)
class EvalHit:
    """Result of matching one retrieved chunk against gold evidence."""

    is_hit: bool
    gold_index: int = -1
    detail: dict[str, str] = field(default_factory=dict)
