import warnings
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from codicil.adapters.qdrant_store import QdrantDenseStore, StaleCollectionError
from codicil.application.index import load_index
from codicil.application.retrieve import Retriever


def test_qdrant_local_matches_numpy_dense(tmp_path):
    idx = load_index(Path("index/multilingual-e5-small"))
    store = QdrantDenseStore(QdrantClient(path=str(tmp_path)), "t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # local mode ignores payload indexes; the server honours them
        store.upload(idx)
    assert store.count() == len(idx.chunks)
    reference = Retriever.__new__(Retriever)
    reference.index, reference.store = idx, None
    for i in np.random.default_rng(0).integers(len(idx.chunks), size=5):
        want = [j for j, _ in reference.dense(idx.vectors[i], 10, None)]
        got = [j for j, _ in store.search(idx.vectors[i], 10, in_force_only=False)]
        assert got[:5] == want[:5]


def test_collection_from_another_index_build_is_refused(tmp_path):
    idx = load_index(Path("index/multilingual-e5-small"))
    client = QdrantClient(path=str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        QdrantDenseStore(client, "t").upload(idx)
    assert QdrantDenseStore(client, "t", expected_sha=idx.sha).search(idx.vectors[0], 3, in_force_only=False)
    with pytest.raises(StaleCollectionError):
        QdrantDenseStore(client, "t", expected_sha="another-build").search(idx.vectors[0], 3, in_force_only=False)
