from pathlib import Path

import numpy as np
import pytest

from codicil.adapters.s3_vectors_store import S3VectorsDenseStore, StaleCollectionError, index_name_for
from codicil.application.index import load_index


class _NotFound(Exception):
    pass


class _FakeS3Vectors:
    """Just enough of the boto3 `s3vectors` client: an in-memory cosine index."""

    class exceptions:
        NotFoundException = _NotFound

    def __init__(self, sha_override: str | None = None):
        self.indexes: dict[str, dict] = {}
        self.put_batches: list[int] = []
        self.last_query: dict = {}
        self.sha_override = sha_override

    def get_index(self, vectorBucketName, indexName):
        if indexName not in self.indexes:
            raise _NotFound(indexName)

    def create_index(self, vectorBucketName, indexName, dataType, dimension, distanceMetric):
        assert (dataType, distanceMetric) == ("float32", "cosine")
        self.indexes[indexName] = {"dim": dimension, "vectors": {}}

    def put_vectors(self, vectorBucketName, indexName, vectors):
        assert len(vectors) <= 500
        self.put_batches.append(len(vectors))
        for v in vectors:
            self.indexes[indexName]["vectors"][v["key"]] = (np.array(v["data"]["float32"], dtype=np.float32), v["metadata"])

    def query_vectors(self, vectorBucketName, indexName, queryVector, topK, returnDistance, returnMetadata, filter=None):
        self.last_query = {"topK": topK, "filter": filter, "returnMetadata": returnMetadata}
        q = np.array(queryVector["float32"], dtype=np.float32)
        rows = []
        for key, (vec, meta) in self.indexes[indexName]["vectors"].items():
            if filter and meta["status"] != filter["status"]["$eq"]:
                continue
            sim = float(vec @ q / (np.linalg.norm(vec) * np.linalg.norm(q)))
            m = dict(meta)
            if self.sha_override:
                m["index_sha"] = self.sha_override
            rows.append({"key": key, "distance": 1.0 - sim, "metadata": m if returnMetadata else None})
        rows.sort(key=lambda r: r["distance"])
        return {"vectors": rows[:topK]}

    def list_vectors(self, vectorBucketName, indexName, maxResults, nextToken=None):
        return {"vectors": [{"key": k} for k in self.indexes[indexName]["vectors"]]}


@pytest.fixture(scope="module")
def idx():
    return load_index(Path("index/multilingual-e5-small"))


def test_upload_is_batched_idempotent_on_the_index_and_search_matches_numpy(idx):
    fake = _FakeS3Vectors()
    store = S3VectorsDenseStore(fake, "b", index_name_for(idx), idx.sha)
    assert store.upload(idx) == len(idx.chunks)
    assert max(fake.put_batches) <= 500 and sum(fake.put_batches) == len(idx.chunks)
    store.upload(idx)  # second upload reuses the existing vector index
    assert store.count() == len(idx.chunks)
    for i in np.random.default_rng(0).integers(len(idx.chunks), size=4):
        q = idx.vectors[i].astype(np.float32)
        want = np.argsort(-(idx.vectors @ q), kind="stable")[:5]
        got = store.search(q, 5, in_force_only=False)
        assert [int(j) for j, _ in got] == [int(j) for j in want]
        assert got[0][1] == pytest.approx(float((idx.vectors @ q)[want[0]]), abs=1e-2)  # score is cosine similarity, not distance


def test_in_force_filter_is_sent_only_when_asked(idx):
    fake = _FakeS3Vectors()
    store = S3VectorsDenseStore(fake, "b", index_name_for(idx), idx.sha)
    store.upload(idx)
    q = idx.vectors[0].astype(np.float32)
    store.search(q, 3, in_force_only=False)
    assert fake.last_query["filter"] is None
    store.search(q, 3, in_force_only=True)
    assert fake.last_query["filter"] == {"status": {"$eq": "in_force"}}


def test_vectors_from_another_build_are_refused(idx):
    fake = _FakeS3Vectors()
    S3VectorsDenseStore(fake, "b", index_name_for(idx), idx.sha).upload(idx)
    fake.sha_override = "another-build"
    with pytest.raises(StaleCollectionError):
        S3VectorsDenseStore(fake, "b", index_name_for(idx), idx.sha).search(idx.vectors[0].astype(np.float32), 3, in_force_only=False)


def test_index_name_follows_the_build(idx):
    assert index_name_for(idx) == f"codicil-{idx.sha[:12]}"
