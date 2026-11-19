"""Author review of the eval labels: what a verdict is, how the append-only log is read, and what a review summary says.

Every label in eval/questions.jsonl was drafted by an AI assistant. The author reviews each one in a local tool and the
verdicts are appended to a JSON-lines log (the last verdict for a question wins). Nothing here changes a label; it only
records how much of the dataset a person has confirmed, so the results can be shown for confirmed labels alone.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

OK = "ok"
WRONG_EVIDENCE = "wrong_evidence"
BAD_QUESTION = "bad_question"
UNSURE = "unsure"
VERDICTS = (OK, WRONG_EVIDENCE, BAD_QUESTION, UNSURE)
MAX_NOTE = 2000


class ReviewError(ValueError):
    """A review record that cannot be accepted."""


@dataclass(frozen=True, slots=True)
class Review:
    id: str
    verdict: str
    note: str = ""
    better_quote: str = ""
    at: str = ""

    def as_json(self) -> dict[str, str]:
        return {"id": self.id, "verdict": self.verdict, "note": self.note, "better_quote": self.better_quote, "at": self.at}


def parse_review(obj: object, known_ids: Iterable[str] | None = None, now: datetime | None = None) -> Review:
    if not isinstance(obj, dict):
        raise ReviewError("a review is a JSON object")
    qid, verdict = obj.get("id"), obj.get("verdict")
    if not isinstance(qid, str) or not qid:
        raise ReviewError("missing question id")
    if known_ids is not None and qid not in set(known_ids):
        raise ReviewError(f"unknown question id {qid!r}")
    if verdict not in VERDICTS:
        raise ReviewError(f"verdict must be one of {', '.join(VERDICTS)}")
    note, better = obj.get("note", ""), obj.get("better_quote", "")
    if not isinstance(note, str) or not isinstance(better, str):
        raise ReviewError("note and better_quote must be text")
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return Review(qid, str(verdict), note.strip()[:MAX_NOTE], better.strip()[:MAX_NOTE], stamp)


def read_log(lines: Iterable[str]) -> list[Review]:
    """Parse a JSON-lines log. A damaged line is skipped rather than losing the rest of a long review session."""
    out: list[Review] = []
    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        try:
            j = json.loads(text)
            out.append(Review(str(j["id"]), str(j["verdict"]), str(j.get("note", "")), str(j.get("better_quote", "")), str(j.get("at", ""))))
        except (ValueError, KeyError, TypeError):
            continue
    return [r for r in out if r.verdict in VERDICTS]


def latest(reviews: Iterable[Review]) -> dict[str, Review]:
    """The last verdict per question (the log is append-only, so a changed mind is a new line)."""
    out: dict[str, Review] = {}
    for r in reviews:
        out[r.id] = r
    return out


def summarise(questions: Sequence[Mapping[str, object]], reviews: Mapping[str, Review]) -> dict[str, object]:
    ids = [str(q["id"]) for q in questions]
    by = Counter(reviews[i].verdict for i in ids if i in reviews)
    per_lang: dict[str, dict[str, int]] = {}
    for q in questions:
        lang = str(q["language"])
        d = per_lang.setdefault(lang, {"total": 0, "reviewed": 0, OK: 0})
        d["total"] += 1
        r = reviews.get(str(q["id"]))
        d["reviewed"] += r is not None
        d[OK] += r is not None and r.verdict == OK
    return {
        "total": len(ids),
        "reviewed": sum(i in reviews for i in ids),
        "by_verdict": {v: by.get(v, 0) for v in VERDICTS},
        "unreviewed": [i for i in ids if i not in reviews],
        "by_language": per_lang,
    }


def confirmed_ids(reviews: Mapping[str, Review]) -> set[str]:
    return {i for i, r in reviews.items() if r.verdict == OK}


def flagged_ids(reviews: Mapping[str, Review]) -> set[str]:
    """Labels the author found wrong or unclear (not just unreviewed)."""
    return {i for i, r in reviews.items() if r.verdict != OK}
