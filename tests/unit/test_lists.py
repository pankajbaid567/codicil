from codicil.domain.lists import complete_list, is_item
from codicil.domain.verify import supported

CLAUSE = (
    "67 A customer's entitlement to zero liability shall arise where the unauthorised transaction occurs in the following events:\n"
    "  (1) Contributory fraud / negligence / deficiency on the part of the bank.\n"
    "  (2) Third party breach where the customer notifies the bank within three working days.\n"
    "68 A customer shall be liable for the loss in the following cases."
)
INTRO = "A customer's entitlement to zero liability shall arise where the unauthorised transaction occurs in the following events:"


def test_intro_sentence_gets_its_items_and_stops_at_the_next_paragraph():
    out = complete_list(INTRO, CLAUSE)
    lines = out.split("\n")
    assert lines[0] == INTRO
    assert lines[1].startswith("(1) Contributory") and lines[2].startswith("(2) Third party")
    assert len(lines) == 3  # paragraph 68 is not an item


def test_completed_list_is_still_supported_by_the_clause():
    assert supported(complete_list(INTRO, CLAUSE), CLAUSE)


def test_unchanged_without_colon_or_items_or_match():
    assert complete_list("A plain sentence.", CLAUSE) == "A plain sentence."
    assert complete_list("Not in the text:", CLAUSE) == "Not in the text:"
    assert complete_list("Ends here:", "Ends here:\nNext paragraph without a marker.") == "Ends here:"


def test_caps_items_and_marks_truncation():
    text = "Intro:\n" + "\n".join(f"({c}) item {c}" for c in "abcdefghij")
    out = complete_list("Intro:", text, max_items=3)
    assert out.split("\n") == ["Intro:", "(a) item a", "(b) item b", "(c) item c", "…"]


def test_item_markers():
    for line in ["(1) x", "(1A) x", "1) x", "(a) x", "a) x", "(iv) x", "iv. x", "2. x", "• x", "- x"]:
        assert is_item(line), line
    for line in ["The bank shall", "1.5 per cent of", "Rs. 500", "(note"]:
        assert not is_item(line), line


def test_items_of_the_next_paragraph_are_not_pulled_in():
    text = "Intro:\n(1) first\n(2) second\n69 A different paragraph begins here.\n(1) belongs to paragraph 69"
    assert complete_list("Intro:", text).split("\n") == ["Intro:", "(1) first", "(2) second"]
