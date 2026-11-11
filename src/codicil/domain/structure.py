"""Infer the clause tree of a regulatory document from its ordered `RawBlock`s.

Shape of an RBI Direction:  Chapter > Section (A., B.1) > numbered paragraph (7.) > (1) > (a) / (i) > ...
Unlabelled blocks ("Provided that ...", "Explanation: ...", a trailing sentence) belong to the clause that precedes them, because a
proviso is meaningless apart from the sentence it modifies. The parser never drops text: every block's text ends up in exactly one node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from codicil.domain.models import Amendment, BlockKind, Node, NodeKind, RawBlock

_CHAPTER = re.compile(r"^Chapter\s+([IVXLC]+|\d+)\s*[-–—:.]?\s*(.*)$", re.I)
_SECTION = re.compile(r"^([A-Z](?:\.\d+)*|\d+)[.)]?\s+(\S.*)$")
_PARA = re.compile(r"^(\d+[A-Z]?(?:\.\d+)*\.|\d+\.\d+(?:\.\d+)*)\s+(\S.*)$")
_ITEM = re.compile(r"^\(([A-Za-z0-9]{1,5})\)\s*(.*)$", re.S)
_ROMAN = re.compile(r"^(?:x{0,3})(?:ix|iv|v?i{0,3})$")
_AMBIGUOUS = {"i", "v", "x", "l", "c", "d", "m"}


def classify_item(label: str, last_alpha: str | None) -> str:
    """Return 'num', 'alpha' or 'roman' for the text inside (...). `last_alpha` is the previous lettered sibling in scope."""
    if label.isdigit():
        return "num"
    low = label.lower()
    if low in _AMBIGUOUS:
        # "(i)" after "(h)" is the letter i; otherwise it opens a roman list.  Same for (v), (x).
        if last_alpha is not None and len(last_alpha) == 1 and ord(last_alpha) + 1 == ord(low):
            return "alpha"
        return "roman"
    if len(low) == 1:
        return "alpha"
    if _ROMAN.match(low):
        return "roman"
    return "alpha" if label.isalpha() and len(low) <= 2 else "other"  # "(aa)" style double letters


@dataclass
class _B:
    """Mutable builder node; frozen into `Node` at the end."""

    kind: NodeKind
    label: str
    title: str
    text: str
    block_index: int
    amendments: list[Amendment] = field(default_factory=list)
    children: list[_B] = field(default_factory=list)
    cls: str = ""  # item class: num / alpha / roman
    level: int = 0  # section depth for sections

    def freeze(self) -> Node:
        return Node(
            kind=self.kind,
            label=self.label,
            title=self.title,
            text=self.text.strip(),
            children=tuple(c.freeze() for c in self.children),
            amendments=tuple(self.amendments),
            block_index=self.block_index,
        )


def build_tree(blocks: list[RawBlock], title: str = "") -> Node:
    root = _B(NodeKind.DOCUMENT, "", title, "", -1)
    chapter: _B | None = None
    section_stack: list[_B] = []  # sections inside the current chapter
    item_stack: list[_B] = []  # paragraph and items inside the current section
    last_alpha: dict[int, str] = {}  # depth in item_stack -> last lettered label

    def container() -> _B:
        if section_stack:
            return section_stack[-1]
        return chapter or root

    def attach_text(node: _B, text: str, amendments: tuple[Amendment, ...]) -> None:
        node.text = f"{node.text}\n{text}" if node.text else text
        node.amendments.extend(amendments)

    for i, b in enumerate(blocks):
        text = b.text.strip()
        if not text:
            continue
        if b.kind is BlockKind.CHAPTER:
            m = _CHAPTER.match(text)
            label = f"Chapter {m.group(1).upper()}" if m else text
            chapter = _B(NodeKind.CHAPTER, label, (m.group(2).strip() if m else ""), "", i, list(b.amendments))
            root.children.append(chapter)
            section_stack, item_stack, last_alpha = [], [], {}
            continue
        long_numbered = b.kind is BlockKind.HEADING and len(text) > 140 and _PARA.match(text) is not None
        if b.kind is BlockKind.HEADING and not long_numbered and text.endswith(":") and len(text) < 16:
            long_numbered = False
            target = item_stack[-1] if item_stack else None
            if target is not None:
                attach_text(target, text, b.amendments)
                continue
        if b.kind is BlockKind.HEADING and not long_numbered:
            m = _SECTION.match(text)
            label, name = (m.group(1), m.group(2).strip()) if m else ("", text)
            depth = label.count(".") + 1 if label else 1
            while section_stack and section_stack[-1].level >= depth:
                section_stack.pop()
            sec = _B(NodeKind.SECTION, label, name, "", i, list(b.amendments), level=depth)
            (section_stack[-1] if section_stack else (chapter or root)).children.append(sec)
            section_stack.append(sec)
            item_stack, last_alpha = [], {}
            continue
        if b.kind is BlockKind.TABLE:
            tbl = _B(NodeKind.TABLE, "table", "", text, i, list(b.amendments))
            (item_stack[-1] if item_stack else container()).children.append(tbl)
            continue
        # paragraph block (a heading that is really a long numbered paragraph falls through to here)
        pm = _PARA.match(text)
        im = _ITEM.match(text)
        if pm:
            node = _B(NodeKind.PARAGRAPH, pm.group(1).rstrip("."), "", pm.group(2), i, list(b.amendments))
            container().children.append(node)
            item_stack, last_alpha = [node], {}
        elif im and item_stack:
            label, rest = im.group(1), im.group(2)
            cls = classify_item(label, last_alpha.get(len(item_stack)) if item_stack else None)
            # find the nearest ancestor of the same class and become its sibling; otherwise nest under the top
            depth = next((d for d in range(len(item_stack) - 1, 0, -1) if item_stack[d].cls == cls), None)
            parent_depth = (depth - 1) if depth is not None else len(item_stack) - 1
            del item_stack[parent_depth + 1 :]
            for k in [k for k in last_alpha if k > parent_depth + 1]:
                del last_alpha[k]
            node = _B(NodeKind.ITEM, f"({label})", "", rest, i, list(b.amendments), cls=cls)
            item_stack[-1].children.append(node)
            item_stack.append(node)
            if cls == "alpha":
                last_alpha[len(item_stack) - 1] = label.lower()
        elif im and not item_stack:
            node = _B(NodeKind.PARAGRAPH, f"({im.group(1)})", "", im.group(2), i, list(b.amendments))
            container().children.append(node)
            item_stack = [node]
        else:
            target = item_stack[-1] if item_stack else None
            if target is None:
                target = _B(NodeKind.PARAGRAPH, "", "", "", i)
                container().children.append(target)
                item_stack = [target]
            attach_text(target, text, b.amendments)
    return root.freeze()
