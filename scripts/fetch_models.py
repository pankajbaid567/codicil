#!/usr/bin/env python3
"""Download the pinned ONNX models into data/models/ and verify their SHA-256 against data/models.lock.json.

Usage: python scripts/fetch_models.py [name ...]     (default: the two models the service needs)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["multilingual-e5-small", "ms-marco-MiniLM-L-6-v2"]


def main(names: list[str]) -> int:
    lock = json.loads((ROOT / "data/models.lock.json").read_text())
    for name in names or DEFAULT:
        spec = lock[name]
        dest = ROOT / "data/models" / name
        snapshot_download(spec["repo"], revision=spec["revision"], allow_patterns=[*spec["files"], "config.json", "tokenizer_config.json"], local_dir=dest)
        sources = [spec, *spec.get("extra", [])]  # extra: files pinned from a second repo (e5's SentencePiece model)
        for src in sources[1:]:
            snapshot_download(src["repo"], revision=src["revision"], allow_patterns=list(src["files"]), local_dir=dest)
        for src in sources:
            for rel, want in src["files"].items():
                got = hashlib.sha256((dest / rel).read_bytes()).hexdigest()
                if got != want:
                    print(f"CHECKSUM MISMATCH {name}/{rel}", file=sys.stderr)
                    return 1
        print(f"ok  {name} @ {spec['revision'][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
