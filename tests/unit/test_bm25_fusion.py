import itertools
import math

from hypothesis import given
from hypothesis import strategies as st

from codicil.domain.bm25 import BM25Index
from codicil.domain.fusion import rrf
from codicil.domain.tokens import analyse, sentences, words


def test_analyse_stems_and_drops_stopwords():
    assert analyse("The charges levied on customers") == ["charg", "levi", "custom"]
    assert "1,00,000" in words("limit of Rs 1,00,000 only")


def test_bm25_ranks_exact_term_first_and_is_deterministic():
    bodies = ["interest rate on savings deposits", "unauthorised electronic banking transaction liability", "kyc customer due diligence"]
    idx = BM25Index.build(["", "", ""], bodies)
    r = idx.score("unauthorised transaction liability")
    assert r[0][0] == 1 and r == idx.score("unauthorised transaction liability")


def test_bm25_idf_matches_hand_computation():
    idx = BM25Index.build(["", ""], ["alpha beta", "beta"])
    # 'alpha' appears in 1 of 2 docs: idf = ln(1 + (2-1+.5)/(1+.5)) = ln 2
    assert math.isclose(idx._idf["alpha"], math.log(2))


def test_bm25_header_field_helps():
    idx = BM25Index.build(["credit card", "deposit"], ["rules", "rules"])
    assert idx.score("credit")[0][0] == 0


def test_bm25_json_roundtrip():
    idx = BM25Index.build(["h"], ["interest rate"])
    again = BM25Index.from_json(idx.to_json())
    assert again.score("interest") == idx.score("interest")


def test_rrf_hand_computed():
    out = dict(rrf([[1, 2, 3], [3, 1]], k=60))
    assert math.isclose(out[1], 1 / 61 + 1 / 62)
    assert math.isclose(out[3], 1 / 63 + 1 / 61)
    assert next(d for d, _ in rrf([[1, 2, 3], [3, 1]])) == 1


@given(st.lists(st.lists(st.integers(0, 20), unique=True, max_size=10), min_size=1, max_size=4))
def test_rrf_properties(rankings):
    out = rrf(rankings)
    ids = [d for d, _ in out]
    assert len(ids) == len(set(ids)) == len({d for r in rankings for d in r})
    assert all(a[1] >= b[1] for a, b in itertools.pairwise(out))
    # a document ranked first in every list can never be beaten
    if rankings and all(r for r in rankings) and len({r[0] for r in rankings}) == 1:
        assert out[0][0] == rankings[0][0]


def test_sentences_keeps_abbreviations():
    assert sentences("See Circular No. 5 of 2020. The bank shall act.") == ["See Circular No. 5 of 2020.", "The bank shall act."]


def test_rrf_counts_a_repeated_id_once_per_list():
    assert dict(rrf([[1, 1, 2]], k=60))[1] == 1 / 61
