"""Dense search backed by Amazon S3 Vectors. Optional: the file-based index stays the default and the only thing CI needs.

Credentials come from the standard AWS chain (on EC2, the instance role), so no key is stored. The vector index is
named after the index build (`codicil-<sha12>`), so a rebuilt corpus uploads to a new index instead of overwriting
the old one, and a server never queries vectors from a different build: each vector also carries `index_sha`, which is
checked on every query.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from codicil.application.index import Index

_BATCH = 500  # PutVectors limit per call


class StaleCollectionError(RuntimeError):
    """The vector index was uploaded from a different index build, so keys no longer match chunk positions."""


def index_name_for(index: Index) -> str:
    return f"codicil-{index.sha[:12]}"


class S3VectorsDenseStore:
    """Vector key = chunk position in the index (as a string), so results map straight back to `Index.chunks`.

    Scores are cosine similarity (1 - cosine distance), the same scale as the in-process dot product on the
    normalised vectors, so the abstention thresholds apply unchanged.
    """

    def __init__(self, client: Any, bucket: str, index_name: str, expected_sha: str | None = None) -> None:
        self.client, self.bucket, self.index_name, self.expected_sha = client, bucket, index_name, expected_sha

    @classmethod
    def connect(cls, bucket: str, index_name: str, region: str = "", expected_sha: str | None = None) -> S3VectorsDenseStore:
        import boto3  # optional dependency, imported only when configured
        from botocore.config import Config

        client = boto3.client("s3vectors", region_name=region or None, config=Config(connect_timeout=3, read_timeout=8, retries={"max_attempts": 2}))
        return cls(client, bucket, index_name, expected_sha)

    def upload(self, index: Index) -> int:
        """Create the vector index if it does not exist, then put every chunk's vector. Returns the number of vectors put."""
        dim = int(index.vectors.shape[1])
        try:
            self.client.get_index(vectorBucketName=self.bucket, indexName=self.index_name)
        except self.client.exceptions.NotFoundException:
            self.client.create_index(
                vectorBucketName=self.bucket,
                indexName=self.index_name,
                dataType="float32",
                dimension=dim,
                distanceMetric="cosine",
            )
        for lo in range(0, len(index.chunks), _BATCH):
            vectors = [
                {
                    "key": str(i),
                    "data": {"float32": [float(x) for x in index.vectors[i]]},
                    "metadata": {
                        "status": index.docs[index.chunks[i].doc_id].status.value,
                        "doc_id": index.chunks[i].doc_id,
                        "index_sha": index.sha,
                    },
                }
                for i in range(lo, min(lo + _BATCH, len(index.chunks)))
            ]
            self.client.put_vectors(vectorBucketName=self.bucket, indexName=self.index_name, vectors=vectors)
        return len(index.chunks)

    def search(self, qvec: np.ndarray, k: int, in_force_only: bool) -> list[tuple[int, float]]:
        args: dict[str, Any] = {
            "vectorBucketName": self.bucket,
            "indexName": self.index_name,
            "queryVector": {"float32": [float(x) for x in qvec]},
            "topK": k,
            "returnDistance": True,
            "returnMetadata": self.expected_sha is not None,
        }
        if in_force_only:
            args["filter"] = {"status": {"$eq": "in_force"}}
        res = self.client.query_vectors(**args)
        hits = res.get("vectors", [])
        if self.expected_sha is not None:
            stale = {str((h.get("metadata") or {}).get("index_sha")) for h in hits} - {self.expected_sha}
            if stale:
                raise StaleCollectionError(
                    f"vector index {self.index_name!r} holds build {sorted(stale)}, server has {self.expected_sha}; re-run scripts/upload_s3vectors.py"
                )
        return [(int(h["key"]), 1.0 - float(h["distance"])) for h in hits]

    def count(self) -> int:
        """Number of vectors in the index (pages through ListVectors; for checks, not for the request path)."""
        total, token = 0, None
        while True:
            kwargs: dict[str, Any] = {"vectorBucketName": self.bucket, "indexName": self.index_name, "maxResults": 500}
            if token:
                kwargs["nextToken"] = token
            res = self.client.list_vectors(**kwargs)
            total += len(res.get("vectors", []))
            token = res.get("nextToken")
            if not token:
                return total
