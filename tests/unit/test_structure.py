from codicil.domain.models import BlockKind, RawBlock
from codicil.domain.structure import build_tree, classify_item


def B(kind, text):
    return RawBlock(kind, text)


def test_classify_item_disambiguates_letter_i_from_roman():
    assert classify_item("1", None) == "num"
    assert classify_item("a", None) == "alpha"
    assert classify_item("i", None) == "roman"
    assert classify_item("i", "h") == "alpha"
    assert classify_item("iv", None) == "roman"
    assert classify_item("v", "u") == "alpha"


def test_nesting_and_proviso_attachment():
    blocks = [
        B(BlockKind.CHAPTER, "Chapter II – Conduct"),
        B(BlockKind.HEADING, "A. Eligibility"),
        B(BlockKind.PARAGRAPH, "7. Banks may do X:"),
        B(BlockKind.PARAGRAPH, "(1) first rule"),
        B(BlockKind.PARAGRAPH, "(i) sub one"),
        B(BlockKind.PARAGRAPH, "(a) deep"),
        B(BlockKind.PARAGRAPH, "(ii) sub two"),
        B(BlockKind.PARAGRAPH, "Provided that exceptions apply."),
        B(BlockKind.PARAGRAPH, "(2) second rule"),
        B(BlockKind.PARAGRAPH, "8. Next paragraph"),
    ]
    root = build_tree(blocks)
    ch = root.children[0]
    assert (ch.label, ch.title) == ("Chapter II", "Conduct")
    sec = ch.children[0]
    assert (sec.label, sec.title) == ("A", "Eligibility")
    p7, p8 = sec.children
    assert p7.label == "7" and p8.label == "8"
    r1, r2 = p7.children
    assert [c.label for c in r1.children] == ["(i)", "(ii)"]
    assert r1.children[0].children[0].label == "(a)"
    assert "Provided that" in r1.children[1].text  # proviso stays with the clause it modifies
    assert r2.label == "(2)"


def test_no_text_is_lost():
    blocks = [
        B(BlockKind.CHAPTER, "Chapter I – X"),
        B(BlockKind.PARAGRAPH, "Unlabelled start"),
        B(BlockKind.PARAGRAPH, "1. Real"),
        B(BlockKind.PARAGRAPH, "continuation words"),
    ]
    words = {w for n in build_tree(blocks).walk() for w in n.text.split()}
    assert {"Unlabelled", "start", "Real", "continuation", "words"} <= words


def test_numeric_headings_and_long_numbered_heading_is_paragraph():
    long = "86B. " + "word " * 40
    root = build_tree(
        [
            B(BlockKind.CHAPTER, "Chapter I – X"),
            B(BlockKind.HEADING, "1. Introduction"),
            B(BlockKind.HEADING, long.strip()),
        ]
    )
    sec = root.children[0].children[0]
    assert sec.label == "1" and sec.children[0].label == "86B"
