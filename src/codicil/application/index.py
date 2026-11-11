"""The immutable index artifact: build it once, load it read-only, address it by content hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from codicil.application.ports import Embedder, Matrix
from codicil.domain.bm25 import BM25Index, BM25Params
from codicil.domain.chunking import ChunkingConfig, build_chunks
from codicil.domain.errors import IndexMismatchError
from codicil.domain.models import Amendment, Chunk, DocStatus, DocumentMeta, Parent
from codicil.domain.structure import build_tree

SCHEMA = 1


@dataclass
class Index:
    docs: dict[str, DocumentMeta]
    chunks: list[Chunk]
    parents: dict[str, Parent]
    bm25: BM25Index
    vectors: Matrix
    embedder_id: str
    sha: str
    by_id: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    def check_embedder(self, embedder: Embedder) -> None:
        if embedder.model_id != self.embedder_id or embedder.dim != self.vectors.shape[1]:
            raise IndexMismatchError(f"index built with {self.embedder_id} ({self.vectors.shape[1]}-d), serving with {embedder.model_id} ({embedder.dim}-d)")


def _d(x: date | None) -> str | None:
    return x.isoformat() if x else None


def _meta_json(m: DocumentMeta) -> dict[str, object]:
    return {
        "doc_id": m.doc_id,
        "title": m.title,
        "short_title": m.short_title,
        "issuer": m.issuer,
        "entity": m.entity,
        "url": m.url,
        "reference": m.reference,
        "issued_on": _d(m.issued_on),
        "updated_as_on": _d(m.updated_as_on),
        "status": m.status.value,
        "superseded_by": m.superseded_by,
    }


def _meta_from(j: dict[str, Any]) -> DocumentMeta:
    def dt(k: str) -> date | None:
        v = j.get(k)
        return date.fromisoformat(str(v)) if v else None

    return DocumentMeta(
        str(j["doc_id"]),
        str(j["title"]),
        str(j["short_title"]),
        str(j["issuer"]),
        str(j["entity"]),
        str(j["url"]),
        j.get("reference") or None,
        dt("issued_on"),
        dt("updated_as_on"),
        DocStatus(str(j["status"])),
        j.get("superseded_by") or None,
    )


def _chunk_json(c: Chunk) -> dict[str, object]:
    return {
        "id": c.chunk_id,
        "doc": c.doc_id,
        "path": list(c.path),
        "header": c.header,
        "text": c.text,
        "parent": c.parent_id,
        "n": c.n_tokens,
        "amend": [[a.action, a.text, _d(a.effective)] for a in c.amendments],
    }


def _chunk_from(j: dict[str, object]) -> Chunk:
    am = tuple(Amendment(a, t, date.fromisoformat(e) if e else None) for a, t, e in j["amend"])  # type: ignore[attr-defined]
    return Chunk(str(j["id"]), str(j["doc"]), tuple(j["path"]), str(j["header"]), str(j["text"]), j["parent"], int(j["n"]), am)  # type: ignore[arg-type,call-overload]


def build_index(
    metas_and_html: list[tuple[DocumentMeta, str]], embedder: Embedder, out_dir: Path, cfg: ChunkingConfig = ChunkingConfig(), bm25: BM25Params = BM25Params()
) -> Index:
    from codicil.adapters.rbi_html import parse_blocks

    docs: dict[str, DocumentMeta] = {}
    chunks: list[Chunk] = []
    parents: dict[str, Parent] = {}
    for meta, html in metas_and_html:
        docs[meta.doc_id] = meta
        c, p = build_chunks(meta, build_tree(parse_blocks(html), meta.title), cfg)
        chunks += c
        parents.update({x.parent_id: x for x in p})
    vectors = embedder.embed_passages([c.indexed_text for c in chunks]).astype(np.float16)
    bm = BM25Index.build([c.header for c in chunks], [c.text for c in chunks], bm25)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "docs.json": json.dumps([_meta_json(m) for m in docs.values()], ensure_ascii=False, indent=1).encode(),
        "chunks.jsonl": "\n".join(json.dumps(_chunk_json(c), ensure_ascii=False) for c in chunks).encode(),
        "parents.jsonl": "\n".join(
            json.dumps({"id": p.parent_id, "doc": p.doc_id, "path": list(p.path), "text": p.text, "n": p.n_tokens}, ensure_ascii=False)
            for p in parents.values()
        ).encode(),
        "bm25.json": json.dumps(bm.to_json(), separators=(",", ":")).encode(),
        "vectors.f16": vectors.tobytes(),
    }
    h = hashlib.sha256()
    for name in sorted(files):
        (out_dir / name).write_bytes(files[name])
        h.update(name.encode() + hashlib.sha256(files[name]).digest())
    sha = h.hexdigest()
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "sha256": sha,
                "embedder": embedder.model_id,
                "dim": embedder.dim,
                "n_chunks": len(chunks),
                "n_parents": len(parents),
                "n_docs": len(docs),
            },
            indent=1,
        )
    )
    return Index(docs, chunks, parents, bm, vectors.astype(np.float32), embedder.model_id, sha)


def load_index(path: Path) -> Index:
    man = json.loads((path / "manifest.json").read_text())
    if man["schema"] != SCHEMA:
        raise IndexMismatchError(f"index schema {man['schema']}, code expects {SCHEMA}")
    h = hashlib.sha256()
    for name in sorted(["docs.json", "chunks.jsonl", "parents.jsonl", "bm25.json", "vectors.f16"]):
        h.update(name.encode() + hashlib.sha256((path / name).read_bytes()).digest())
    if h.hexdigest() != man["sha256"]:
        raise IndexMismatchError("index files do not match the recorded hash (corrupt or edited)")
    docs = {m.doc_id: m for m in (_meta_from(j) for j in json.loads((path / "docs.json").read_text()))}
    chunks = [_chunk_from(json.loads(line)) for line in (path / "chunks.jsonl").read_text().splitlines() if line]
    parents = {}
    for line in (path / "parents.jsonl").read_text().splitlines():
        j = json.loads(line)
        parents[j["id"]] = Parent(j["id"], j["doc"], tuple(j["path"]), j["text"], j["n"])
    vec = np.frombuffer((path / "vectors.f16").read_bytes(), dtype=np.float16).reshape(-1, man["dim"]).astype(np.float32)
    bm = BM25Index.from_json(json.loads((path / "bm25.json").read_text()))
    return Index(docs, chunks, parents, bm, vec, man["embedder"], man["sha256"])
