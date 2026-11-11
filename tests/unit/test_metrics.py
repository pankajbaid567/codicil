import math

from hypothesis import given
from hypothesis import strategies as st

from codicil.domain import metrics as m


def test_covers_verbatim_and_partial():
    assert m.covers("12 The bank shall pay ₹100 per day.", "The bank shall pay ₹100 per day.")
    assert m.covers("’quoted’ text", "'quoted' text")
    assert not m.covers("completely different words here", "The bank shall pay ₹100 per day.")


def test_hand_computed_ranking_metrics():
    hs = [set(), {0}, set(), {1}]  # gold has 2 items; hits at rank 2 and 4
    assert m.recall_at_k(hs, 2, 1) == 0.0 and m.recall_at_k(hs, 2, 2) == 0.5 and m.recall_at_k(hs, 2, 4) == 1.0
    assert m.reciprocal_rank(hs) == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(m.ndcg_at_k(hs, 2, 10), dcg / idcg)
    assert m.success_at_k(hs, 1) == 0.0 and m.success_at_k(hs, 2) == 1.0


def test_precision_recall():
    assert m.precision_recall([True, True, False, False], [True, False, True, False]) == (0.5, 0.5)
    assert m.precision_recall([False], [False]) == (1.0, 1.0)


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    vals = [0.0, 1.0] * 20
    mean, lo, hi = m.bootstrap_ci(vals)
    assert lo <= mean <= hi and (mean, lo, hi) == m.bootstrap_ci(vals)


@given(st.lists(st.sets(st.integers(0, 3), max_size=2), max_size=15))
def test_metrics_bounded(hs):
    for v in (m.reciprocal_rank(hs), m.ndcg_at_k(hs, 4), m.recall_at_k(hs, 4, 10), m.success_at_k(hs, 5)):
        assert 0.0 <= v <= 1.0 + 1e-9


def test_percentile():
    assert m.percentile([1, 2, 3, 4], 50) == 2 and m.percentile([1, 2, 3, 4], 95) == 4
