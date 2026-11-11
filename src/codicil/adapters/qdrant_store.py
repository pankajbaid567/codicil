"""Dense search backed by Qdrant. Optional: the file-based index stays the default and the only thing CI needs."""

from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient, models

from codicil.application.index import Index


class QdrantDenseStore:
    """Point id = chunk position in the index, so results map straight back to `Index.chunks`."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client, self.collection = client, collection

    @classmethod
    def connect(cls, url: str, api_key: str, collection: str) -> QdrantDenseStore:
        return cls(QdrantClient(url=url, api_key=api_key or None, timeout=10), collection)

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
        res = self.client.query_points(self.collection, query=qvec.tolist(), limit=k, query_filter=flt, with_payload=False)
        return [(int(p.id), float(p.score)) for p in res.points]

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)
