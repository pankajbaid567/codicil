from codicil.domain.freshness import content_hash, diff_blocks, normalise


def test_hash_ignores_whitespace_and_empty_blocks_but_not_words():
    a = ["1. A bank shall   publish its charges.", "", "  2. Interest is credited quarterly. "]
    b = ["1. A bank shall publish its charges.", "2. Interest is credited quarterly."]
    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(["1. A bank shall publish its charges.", "2. Interest is credited monthly."])
    assert normalise(a) == b


def test_diff_counts_changed_added_removed_with_readable_examples():
    old = ["p1", "limit is Rs 10,000", "p3", "p4"]
    new = ["p1", "limit is Rs 15,000", "p3", "new paragraph", "p5"]
    d = diff_blocks(old, new)
    assert (d.changed, d.added, d.removed) == (2, 1, 0) and d.total == 3
    assert any("10,000" in e and "15,000" in e for e in d.examples)


def test_identical_blocks_give_an_empty_diff_and_examples_are_capped():
    assert diff_blocks(["a", "b"], ["a", "b"]).total == 0
    big = diff_blocks([], [f"paragraph {i}" for i in range(50)], max_examples=3)
    assert big.added == 50 and len(big.examples) == 3
