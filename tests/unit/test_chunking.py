import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from codicil.domain.chunking import ChunkingConfig, build_chunks
from codicil.domain.models import BlockKind, DocumentMeta, RawBlock
from codicil.domain.structure import build_tree
from codicil.domain.tokens import words

META = DocumentMeta("d", "Reserve Bank of India (X) Directions, 2025", "(X) Directions, 2025", "RBI", "Banks", "u")


def _doc(paras):
    blocks = [RawBlock(BlockKind.CHAPTER, "Chapter I – A")]
    blocks += [RawBlock(BlockKind.PARAGRAPH, p) for p in paras]
    return build_tree(blocks)


def test_short_clause_is_one_chunk_with_header_and_path():
    chunks, parents = build_chunks(META, _doc(["1. Banks shall do X."]))
    assert len(chunks) == 1 and chunks[0].path == ("Chapter I A", "1")
    assert "Directions, 2025" in chunks[0].header and chunks[0].parent_id == parents[0].parent_id


def test_long_clause_children_repeat_the_stem():
    items = [f"({chr(97 + i)}) " + "word " * 60 for i in range(6)]
    chunks, _ = build_chunks(META, _doc(["1. The bank shall ensure the following:", *items]))
    assert len(chunks) > 1
    assert all("The bank shall ensure the following" in c.text for c in chunks)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.text(alphabet="abcdefg hij.", min_size=5, max_size=700), min_size=1, max_size=6))
def test_chunking_is_lossless_and_bounded(paras):
    root = _doc([f"{i + 1}. {p}" for i, p in enumerate(paras)])
    chunks, _ = build_chunks(META, root, ChunkingConfig())
    src = {w for n in root.walk() for w in words(n.text)}
    assert src <= {w for c in chunks for w in words(c.text)}
    assert all(c.n_tokens <= 300 for c in chunks)


def test_chunk_ids_are_deterministic():
    a, _ = build_chunks(META, _doc(["1. Same text."]))
    b, _ = build_chunks(META, _doc(["1. Same text."]))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.mark.skipif(not RAW.exists() or not any(RAW.glob("*.html")), reason="corpus not fetched")
def test_real_corpus_is_lossless():
    from codicil.adapters.rbi_html import parse_blocks, parse_meta

    manifest = json.loads((RAW.parent / "corpus" / "manifest.json").read_text())["documents"]
    for d in manifest:
        html = (RAW / f"{d['doc_id']}.html").read_text(encoding="utf-8")
        root = build_tree(parse_blocks(html))
        chunks, _ = build_chunks(parse_meta(d["doc_id"], d["title"], d["url"], html), root)
        src = {w for n in root.walk() for w in words(n.text)}
        assert src <= {w for c in chunks for w in words(c.text)}, d["doc_id"]
