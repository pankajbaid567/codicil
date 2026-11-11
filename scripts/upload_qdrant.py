#!/usr/bin/env python3
"""Upload the vectors of the file index to Qdrant. Reads CODICIL_QDRANT_URL / CODICIL_QDRANT_API_KEY / CODICIL_QDRANT_COLLECTION from your shell."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codicil.adapters.qdrant_store import QdrantDenseStore
from codicil.application.index import load_index
from codicil.settings import Settings


def main() -> None:
    s = Settings()
    if not s.qdrant_url:
        sys.exit("set CODICIL_QDRANT_URL (and CODICIL_QDRANT_API_KEY) in your shell first")
    idx = load_index(s.index_dir)
    store = QdrantDenseStore.connect(s.qdrant_url, s.qdrant_api_key.get_secret_value(), s.qdrant_collection)
    store.upload(idx)
    print(f"uploaded {store.count()} points to collection '{s.qdrant_collection}' (index {idx.sha[:12]})")


if __name__ == "__main__":
    main()
