"""Dense search backed by Qdrant. Optional: the file-based index stays the default and the only thing CI needs."""

from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient, models

from codicil.application.index import Index


class StaleCollectionError(RuntimeError):
    """The collection was uploaded from a different index build, so point ids no longer match chunk positions."""


class QdrantDenseStore:
    """Point id = chunk position in the index, so results map straight back to `Index.chunks`.

    Positions are only meaningful for the index build that was uploaded. Each point carries that build's
    `index_sha`; when `expected_sha` is given, a hit from another build raises StaleCollectionError instead of
    silently returning the wrong chunks (the answer service then falls back to the in-process index).
    """

    def __init__(self, client: QdrantClient, collection: str, expected_sha: str | None = None) -> None:
        self.client, self.collection, self.expected_sha = client, collection, expected_sha

    @classmethod
    def connect(cls, url: str, api_key: str, collection: str, expected_sha: str | None = None) -> QdrantDenseStore:
        return cls(QdrantClient(url=url, api_key=api_key or None, timeout=10), collection, expected_sha)

    def upload(self, index: Index, batch: int = 256) -> None:
        dim = int(index.vectors.shape[1])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(self.collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        self.client.create_payload_index(self.collection, "status", models.PayloadSchemaType.KEYWORD)
        for lo in range(0, len(index.chunks), batch):
            pts = [
                models.PointStruct(
                    id=i,
                    vector=index.vectors[i].tolist(),
                    payload={
                        "chunk_id": index.chunks[i].chunk_id,
                        "doc_id": index.chunks[i].doc_id,
                        "status": index.docs[index.chunks[i].doc_id].status.value,
                        "index_sha": index.sha,
                    },
                )
                for i in range(lo, min(lo + batch, len(index.chunks)))
            ]
            self.client.upsert(self.collection, pts, wait=True)

    def search(self, qvec: np.ndarray, k: int, in_force_only: bool) -> list[tuple[int, float]]:
        flt = models.Filter(must=[models.FieldCondition(key="status", match=models.MatchValue(value="in_force"))]) if in_force_only else None
        check = self.expected_sha is not None
        res = self.client.query_points(self.collection, query=qvec.tolist(), limit=k, query_filter=flt, with_payload=["index_sha"] if check else False)
        if check:
            stale = {str((p.payload or {}).get("index_sha")) for p in res.points} - {self.expected_sha}
            if stale:
                raise StaleCollectionError(
                    f"collection {self.collection!r} holds index {sorted(stale)}, server has {self.expected_sha}; re-run scripts/upload_qdrant.py"
                )
        return [(int(p.id), float(p.score)) for p in res.points]

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)
