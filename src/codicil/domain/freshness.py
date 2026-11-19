"""Detect real changes in a regulation page: hash the parsed text, not the HTML, and summarise what changed.

RBI's pages carry volatile bytes (a session-style token: the raw HTML differs by about 80 bytes on every request), so the raw
SHA-256 changes on every fetch. The text of the parsed blocks does not, which makes it the right thing to compare.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

_WS = re.compile(r"\s+")


def normalise(texts: Sequence[str]) -> list[str]:
    return [t for t in (_WS.sub(" ", x).strip() for x in texts) if t]


def content_hash(texts: Sequence[str]) -> str:
    """SHA-256 of the normalised block texts, one per line."""
    return hashlib.sha256("\n".join(normalise(texts)).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class BlockDiff:
    added: int = 0
    removed: int = 0
    changed: int = 0
    examples: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return self.added + self.removed + self.changed

    def as_json(self) -> dict[str, object]:
        return {"added": self.added, "removed": self.removed, "changed": self.changed, "examples": list(self.examples)}


def _clip(s: str, n: int = 160) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def diff_blocks(old: Sequence[str], new: Sequence[str], max_examples: int = 5) -> BlockDiff:
    """Block-level diff: a replaced block counts as changed, the surplus of an uneven replacement as added or removed."""
    a, b = normalise(old), normalise(new)
    added = removed = changed = 0
    examples: list[str] = []

    def note(line: str) -> None:
        if len(examples) < max_examples:
            examples.append(line)

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            pairs = min(i2 - i1, j2 - j1)
            changed += pairs
            added += (j2 - j1) - pairs
            removed += (i2 - i1) - pairs
            for k in range(pairs):
                note(f"changed: {_clip(a[i1 + k], 100)} -> {_clip(b[j1 + k], 100)}")
            for k in range(i1 + pairs, i2):
                note(f"removed: {_clip(a[k])}")
            for k in range(j1 + pairs, j2):
                note(f"added: {_clip(b[k])}")
        elif tag == "delete":
            removed += i2 - i1
            for k in range(i1, i2):
                note(f"removed: {_clip(a[k])}")
        elif tag == "insert":
            added += j2 - j1
            for k in range(j1, j2):
                note(f"added: {_clip(b[k])}")
    return BlockDiff(added, removed, changed, tuple(examples))
