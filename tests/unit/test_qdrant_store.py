import warnings
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

from codicil.adapters.qdrant_store import QdrantDenseStore
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
