import numpy as np

from codicil.adapters.hashing_embedder import HashingEmbedder
from codicil.application.index import build_index, load_index
from tests.fixtures.mini import HTML, META


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.seen: list[str] = []

    def embed_passages(self, texts):
        self.seen += list(texts)
        return super().embed_passages(texts)


def test_unchanged_corpus_embeds_nothing_and_gives_the_same_index(tmp_path):
    first = build_index([(META, HTML)], HashingEmbedder(), tmp_path / "a")
    emb = CountingEmbedder()
    second = build_index([(META, HTML)], emb, tmp_path / "b", reuse=load_index(tmp_path / "a"))
    assert emb.seen == [] and second.build_stats["embedded"] == 0 and second.build_stats["reused"] == len(second.chunks)
    assert second.sha == first.sha  # identical vectors and text: the content address does not move


def test_one_changed_paragraph_embeds_only_its_chunks_and_keeps_other_vectors(tmp_path):
    old = build_index([(META, HTML)], HashingEmbedder(), tmp_path / "a")
    new_html = HTML.replace("within 10 working days", "within 7 working days")
    emb = CountingEmbedder()
    new = build_index([(META, new_html)], emb, tmp_path / "b", reuse=load_index(tmp_path / "a"))
    assert 0 < new.build_stats["embedded"] < len(new.chunks) and new.build_stats["reused"] > 0
    assert len(emb.seen) == new.build_stats["embedded"] and all("7 working days" in t for t in emb.seen[:1])
    assert new.sha != old.sha
    # the vectors of untouched chunks are exactly the old ones
    old_by_text = {c.indexed_text: old.vectors[i] for i, c in enumerate(old.chunks)}
    kept = [i for i, c in enumerate(new.chunks) if c.indexed_text in old_by_text]
    assert kept and all(np.array_equal(new.vectors[i], old_by_text[new.chunks[i].indexed_text]) for i in kept)


def test_an_index_from_another_embedder_is_never_reused(tmp_path):
    class Other(HashingEmbedder):
        model_id = "other-embedder"

    old = build_index([(META, HTML)], Other(), tmp_path / "a")
    emb = CountingEmbedder()
    new = build_index([(META, HTML)], emb, tmp_path / "b", reuse=old)
    assert new.build_stats["reused"] == 0 and len(emb.seen) == len(new.chunks)
