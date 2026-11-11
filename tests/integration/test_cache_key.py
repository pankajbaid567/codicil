from codicil.api.app import cache_key


def test_cache_key_depends_on_index_hash_and_normalises_case():
    assert cache_key("aaa", "Q?", False) != cache_key("bbb", "Q?", False)
    assert cache_key("aaa", " Q? ", False) == cache_key("aaa", "q?", False)
    assert cache_key("aaa", "q", True) != cache_key("aaa", "q", False)
