from dataclasses import replace

import pytest

from codicil.adapters.hashing_embedder import HashingEmbedder
from codicil.application.index import load_index
from codicil.application.retrieve import RetrieveConfig, Retriever
from codicil.domain.errors import IndexMismatchError
from codicil.domain.models import DocStatus
from tests.fixtures.mini import mini_index


def test_index_roundtrip_and_hash_is_stable(tmp_path):
    built = mini_index(tmp_path / "a")
    loaded = load_index(tmp_path / "a")
    assert loaded.sha == built.sha == mini_index(tmp_path / "b").sha  # same input, same artifact
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in built.chunks]
    assert (loaded.vectors.shape[0], loaded.docs["mini"].short_title) == (len(built.chunks), "(Mini) Directions, 2026")


def test_tampered_index_is_rejected(tmp_path):
    mini_index(tmp_path / "a")
    f = tmp_path / "a" / "chunks.jsonl"
    f.write_text(f.read_text().replace("10 working days", "99 working days"))
    with pytest.raises(IndexMismatchError, match="hash"):
        load_index(tmp_path / "a")


def test_embedder_mismatch_is_rejected(tmp_path):
    idx = mini_index(tmp_path / "a")

    class Other(HashingEmbedder):
        model_id = "other"

    with pytest.raises(IndexMismatchError, match="other"):
        Retriever(idx, Other())


def test_superseded_documents_are_excluded_by_default(tmp_path):
    idx = mini_index(tmp_path / "a")
    idx.docs["mini"] = replace(idx.docs["mini"], status=DocStatus.SUPERSEDED)
    r = Retriever(idx, HashingEmbedder())
    assert r.search("unauthorised electronic transaction", RetrieveConfig()) == []
    assert r.search("unauthorised electronic transaction", RetrieveConfig(include_superseded=True))


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_each_mode_finds_the_matching_clause(tmp_path, mode):
    idx = mini_index(tmp_path / "a")
    top = Retriever(idx, HashingEmbedder()).search("interest on savings deposits calculated daily product basis", RetrieveConfig(mode=mode), top_k=1)[0]
    assert "daily product basis" in idx.chunks[idx.by_id[top.chunk_id]].text
