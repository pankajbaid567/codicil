from datetime import UTC, datetime

import pytest

from codicil.domain import review as R

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
QS = [
    {"id": "q1", "language": "en"},
    {"id": "q2", "language": "en"},
    {"id": "h1", "language": "hinglish"},
    {"id": "d1", "language": "hi"},
]


def test_parse_accepts_a_valid_review_and_trims_and_stamps_it():
    r = R.parse_review({"id": "q1", "verdict": "ok", "note": "  fine  ", "better_quote": ""}, known_ids=["q1"], now=NOW)
    assert (r.id, r.verdict, r.note, r.at) == ("q1", "ok", "fine", "2026-11-19T12:00:00Z")


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {"verdict": "ok"},
        {"id": "", "verdict": "ok"},
        {"id": "q1", "verdict": "great"},
        {"id": "zz", "verdict": "ok"},
        {"id": "q1", "verdict": "ok", "note": 5},
    ],
)
def test_parse_rejects_malformed_or_unknown_reviews(bad):
    with pytest.raises(R.ReviewError):
        R.parse_review(bad, known_ids=["q1"])


def test_note_is_capped():
    assert len(R.parse_review({"id": "q1", "verdict": "unsure", "note": "x" * 9000}).note) == R.MAX_NOTE


def test_log_reading_skips_damaged_lines_and_last_verdict_wins():
    lines = [
        '{"id": "q1", "verdict": "wrong_evidence", "note": "first"}',
        "not json",
        "",
        '{"id": "q2", "verdict": "ok"}',
        '{"id": "q1", "verdict": "ok", "note": "changed my mind"}',
        '{"id": "q3", "verdict": "nonsense"}',
        '{"verdict": "ok"}',
    ]
    got = R.latest(R.read_log(lines))
    assert set(got) == {"q1", "q2"} and got["q1"].verdict == "ok" and got["q1"].note == "changed my mind"


def test_summary_counts_by_verdict_and_language_and_lists_unreviewed():
    rev = R.latest([R.Review("q1", "ok"), R.Review("q2", "bad_question"), R.Review("h1", "ok"), R.Review("gone", "ok")])
    s = R.summarise(QS, rev)
    assert s["total"] == 4 and s["reviewed"] == 3 and s["unreviewed"] == ["d1"]
    assert s["by_verdict"] == {"ok": 2, "wrong_evidence": 0, "bad_question": 1, "unsure": 0}
    assert s["by_language"]["en"] == {"total": 2, "reviewed": 2, "ok": 1} and s["by_language"]["hi"] == {"total": 1, "reviewed": 0, "ok": 0}


def test_confirmed_and_flagged_are_disjoint_and_exclude_unreviewed():
    rev = R.latest([R.Review("q1", "ok"), R.Review("q2", "unsure"), R.Review("h1", "wrong_evidence")])
    assert R.confirmed_ids(rev) == {"q1"} and R.flagged_ids(rev) == {"q2", "h1"} and "d1" not in R.confirmed_ids(rev) | R.flagged_ids(rev)
