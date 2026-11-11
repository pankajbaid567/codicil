from codicil.domain.glossary import detect_language, normalise_query
from codicil.domain.verify import numbers, support_score, supported


def test_numbers_must_appear_in_evidence():
    ev = "The bank shall credit the amount within 10 working days."
    assert supported("The bank shall credit the amount within 10 working days.", ev)
    assert not supported("The bank shall credit the amount within 7 working days.", ev)  # same words, wrong number
    assert numbers("limit of ₹1,00,000 and 2.5%") == {"100000", "2.5"}


def test_support_score_is_content_recall():
    assert support_score("bank shall credit amount", "The bank shall credit the amount.") == 1.0
    assert support_score("bank shall waive charges", "The bank shall credit the amount.") < 0.7


def test_language_detection():
    assert detect_language("What is the limit?") == "en"
    assert detect_language("बैंक ब्याज दर क्या है") == "hi"
    assert detect_language("mere account se paise kat gaye kya bank wapas karega") == "hinglish"


def test_normalisation_translates_but_keeps_original_words():
    q = normalise_query("bina permission mere account se paise kat gaye")
    assert "unauthorised" in q.normalised and "account" in q.normalised
    assert "permission" in q.normalised  # original words stay: English terms inside a Hinglish sentence still match lexically
    assert q.matched_terms


def test_english_query_is_untouched_and_entity_hints_found():
    q = normalise_query("Can an NBFC charge a foreclosure fee?")
    assert q.normalised == "Can an NBFC charge a foreclosure fee?" and q.entity_hints == ("NBFCs",)
