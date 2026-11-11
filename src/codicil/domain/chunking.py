"""Structure-aware, small-to-big chunking.

Children are the retrieval units: whole clauses, or runs of sibling clauses packed under their shared stem so a lettered item never
loses the sentence that gives it meaning. Parents are the enclosing numbered paragraphs, returned to the reader after retrieval.
`fixed_window_chunks` is the structure-blind baseline used only in the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

from codicil.domain.models import Amendment, Chunk, DocumentMeta, Node, NodeKind, Parent, stable_id
from codicil.domain.tokens import count_tokens, sentences, words


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    child_max_tokens: int = 220
    parent_max_tokens: int = 700
    stem_max_tokens: int = 70
    table_rows_per_chunk: int = 8


def flatten(node: Node, depth: int = 0) -> str:
    """Own text then descendants, labels kept, one clause per line. Tables are rendered inline."""
    head = f"{node.label} " if node.label and node.kind in (NodeKind.PARAGRAPH, NodeKind.ITEM) else ""
    lines = [f"{'  ' * depth}{head}{node.text}".rstrip()] if (node.text or head) else []
    for c in node.children:
        lines.append(flatten(c, depth + 1))
    return "\n".join(line for line in lines if line.strip())


def _amendments(node: Node) -> tuple[Amendment, ...]:
    return tuple(a for n in node.walk() for a in n.amendments)


def _header(meta: DocumentMeta, path: tuple[str, ...]) -> str:
    upd = f" | updated {meta.updated_as_on:%d %b %Y}" if meta.updated_as_on else ""
    return f"{meta.issuer} | {meta.short_title}{upd} | {' > '.join(path)}"


def _truncate_tokens(text: str, limit: int) -> str:
    if count_tokens(text) <= limit:
        return text
    out: list[str] = []
    used = 0
    for s in sentences(text):
        t = count_tokens(s)
        if used + t > limit:
            if not out:  # a single over-long sentence: cut on words rather than keep it whole
                return " ".join(s.split()[: max(1, int(limit / 1.3))])
            break
        out.append(s)
        used += t
    return " ".join(out)


def _split_long(text: str, limit: int) -> list[str]:
    """Pack sentences into pieces under `limit`; a single over-long sentence is split on word count as a last resort."""
    pieces: list[str] = []
    cur: list[str] = []
    used = 0
    for s in sentences(text):
        t = count_tokens(s)
        if t > limit:
            if cur:
                pieces.append(" ".join(cur))
                cur, used = [], 0
            ws = s.split()
            step = max(1, int(limit / 1.3))
            pieces.extend(" ".join(ws[i : i + step]) for i in range(0, len(ws), step))
            continue
        if used + t > limit and cur:
            pieces.append(" ".join(cur))
            cur, used = [], 0
        cur.append(s)
        used += t
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def build_chunks(meta: DocumentMeta, root: Node, cfg: ChunkingConfig = ChunkingConfig()) -> tuple[list[Chunk], list[Parent]]:
    chunks: list[Chunk] = []
    parents: list[Parent] = []
    seq = 0

    def emit(path: tuple[str, ...], text: str, parent_id: str, amend: tuple[Amendment, ...]) -> None:
        nonlocal seq
        text = text.strip()
        if not text:
            return
        cid = stable_id(meta.doc_id, str(seq), text)
        seq += 1
        chunks.append(
            Chunk(
                chunk_id=cid,
                doc_id=meta.doc_id,
                path=path,
                header=_header(meta, path),
                text=text,
                parent_id=parent_id,
                n_tokens=count_tokens(text),
                amendments=amend,
            )
        )

    def unit(node: Node, path: tuple[str, ...]) -> None:
        """A numbered paragraph (or a table / stray clause) becomes one parent and one or more children."""
        full = flatten(node)
        parent_text = _truncate_tokens(full, cfg.parent_max_tokens) if count_tokens(full) > cfg.parent_max_tokens else full
        pid = stable_id(meta.doc_id, "parent", "/".join(path), full[:80])
        parents.append(Parent(pid, meta.doc_id, path, parent_text, count_tokens(parent_text)))
        amend = _amendments(node)
        if node.kind is NodeKind.TABLE:
            group_rows: list[str] = []
            used_t = 0
            for row in node.text.split("\n"):
                rt = count_tokens(row)
                if group_rows and (used_t + rt > cfg.child_max_tokens or len(group_rows) >= cfg.table_rows_per_chunk):
                    emit(path, "\n".join(group_rows), pid, amend)
                    group_rows, used_t = [], 0
                if rt > cfg.child_max_tokens:  # one huge row: split on words
                    for piece in _split_long(row, cfg.child_max_tokens):
                        emit(path, piece, pid, amend)
                    continue
                group_rows.append(row)
                used_t += rt
            if group_rows:
                emit(path, "\n".join(group_rows), pid, amend)
            return
        if count_tokens(full) <= cfg.child_max_tokens:
            emit(path, full, pid, amend)
            return
        own = f"{node.label} {node.text}".strip()
        if count_tokens(own) > cfg.stem_max_tokens:
            # the clause's own text is long: keep all of it as chunks of its own, and use a short stem for its children
            for piece in _split_long(own, cfg.child_max_tokens):
                emit(path, piece, pid, amend)
            stem = _truncate_tokens(own, cfg.stem_max_tokens)
        else:
            stem = own
        budget = cfg.child_max_tokens - count_tokens(stem)
        group: list[str] = []
        used = 0

        def flush(label_path: tuple[str, ...]) -> None:
            nonlocal group, used
            if group:
                emit(label_path, "\n".join(([stem] if stem else []) + group), pid, amend)
            group, used = [], 0

        for child in node.children:
            ctext = flatten(child)
            ct = count_tokens(ctext)
            cpath = (*path, child.label) if child.label else path
            if ct > budget:
                flush(path)
                for piece in _split_long(ctext, budget):
                    emit(cpath, "\n".join(([stem] if stem else []) + [piece]), pid, _amendments(child))
                continue
            if used + ct > budget:
                flush(path)
            group.append(ctext)
            used += ct
        flush(path)

    for chapter in root.children:
        chap_label = f"{chapter.label} {chapter.title}".strip()
        base = (chap_label,) if chap_label else ()
        if chapter.text:
            unit(Node(NodeKind.PARAGRAPH, "", "", chapter.text), base)

        def walk(n: Node, path: tuple[str, ...]) -> None:
            for c in n.children:
                if c.kind is NodeKind.SECTION:
                    sec = f"{c.label} {c.title}".strip()
                    walk(c, (*path, sec))
                elif c.kind in (NodeKind.PARAGRAPH, NodeKind.TABLE):
                    unit(c, (*path, c.label) if c.label and c.kind is NodeKind.PARAGRAPH else path)
                else:
                    unit(c, path)

        walk(chapter, base)
    return chunks, parents


def fixed_window_chunks(meta: DocumentMeta, root: Node, size: int = 256, overlap: float = 0.15) -> tuple[list[Chunk], list[Parent]]:
    """Baseline: ignore structure, slide a word window over the whole document text. No header, no parent."""
    text = "\n".join(flatten(c) for c in root.children)
    ws = text.split()
    step_words = max(1, int(size / 1.3))
    stride = max(1, int(step_words * (1 - overlap)))
    out: list[Chunk] = []
    for i in range(0, max(1, len(ws)), stride):
        piece = " ".join(ws[i : i + step_words])
        if not piece:
            break
        out.append(Chunk(stable_id(meta.doc_id, "fw", str(i)), meta.doc_id, (f"words {i}",), meta.short_title, piece, None, count_tokens(piece)))
        if i + step_words >= len(ws):
            break
    return out, []


def coverage_words(chunks: list[Chunk]) -> set[str]:
    return {w for c in chunks for w in words(c.text)}
