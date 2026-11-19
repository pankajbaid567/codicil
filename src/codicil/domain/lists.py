"""Complete a quoted sentence that introduces a list.

RBI clauses often say "... in the following events:" and put the substance in numbered items on the
next lines. Quoting only the introduction gives an answer that stops at a colon. `complete_list` appends
the items that follow the sentence in the same clause text, so the quote stays verbatim and verifiable.
"""

from __future__ import annotations

import re

# (1) (1A) 1) (a) a) (i) i) 1. a. i. and bullets, at the start of a line
_ITEM = re.compile(r"^\s*(?:\(?(?:\d{1,2}[A-Za-z]?|[a-z]{1,2}|[ivxlc]{1,6})\)|(?:\d{1,2}|[a-z]|[ivxlc]{1,6})\.|[•\-–])\s+\S")


def is_item(line: str) -> bool:
    return bool(_ITEM.match(line))


def introduces_list(sentence: str) -> bool:
    return sentence.rstrip().endswith((":", ":-", ":—"))


def complete_list(sentence: str, text: str, max_items: int = 8, max_chars: int = 1200) -> str:
    """Return the sentence followed by the list items that come right after it in `text`.

    Unchanged if the sentence does not end with a colon, cannot be found in `text`, or is not followed by
    list items. Stops at the first line that is not an item. When the cap is reached, ends with "…".
    """
    if not introduces_list(sentence):
        return sentence
    at = text.find(sentence)
    if at < 0:
        return sentence
    items: list[str] = []
    size = len(sentence)
    truncated = False
    for raw in text[at + len(sentence) :].splitlines():
        line = raw.strip()
        if not line:
            continue
        if not is_item(line):
            break
        if len(items) >= max_items or size + len(line) > max_chars:
            truncated = True
            break
        items.append(line)
        size += len(line) + 1
    if not items:
        return sentence
    return "\n".join([sentence, *items, *(["…"] if truncated else [])])
