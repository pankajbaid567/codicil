#!/usr/bin/env python3
"""Upload the vectors of the file index to Amazon S3 Vectors.

Needs AWS credentials (an instance role, or your shell's profile) with s3vectors:CreateIndex, GetIndex, PutVectors, ListVectors
on the vector bucket, and CODICIL_S3VECTORS_BUCKET set. The vector index is named after the index build (codicil-<sha12>).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codicil.adapters.s3_vectors_store import S3VectorsDenseStore, index_name_for
from codicil.application.index import load_index
from codicil.settings import Settings


def main() -> None:
    s = Settings()
    if not s.s3vectors_bucket:
        sys.exit("set CODICIL_S3VECTORS_BUCKET (and AWS credentials) first")
    idx = load_index(s.index_dir)
    name = s.s3vectors_index or index_name_for(idx)
    store = S3VectorsDenseStore.connect(s.s3vectors_bucket, name, s.s3vectors_region, idx.sha)
    put = store.upload(idx)
    print(f"put {put} vectors into {s.s3vectors_bucket}/{name}; index now holds {store.count()} (build {idx.sha[:12]})")


if __name__ == "__main__":
    main()
