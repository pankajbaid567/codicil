"""XLM-R style tokenizer on the original SentencePiece model, with the interface the embedder uses (`encode_batch` -> `.ids`, `.attention_mask`).

Why it exists: the Hugging Face `tokenizer.json` of multilingual-e5 keeps about 290 MB resident (250k-piece vocabulary), more than the ONNX model itself, which
does not fit a 512 MB host. SentencePiece needs a small fraction of that. `tests/unit/test_sp_tokenizer.py` pins the ids it produces; the same ids were checked
against the Hugging Face tokenizer on every indexed chunk and every eval question (2,425 texts, no mismatch)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm

BOS, PAD, EOS, UNK = 0, 1, 2, 3  # fairseq layout used by XLM-R models: piece ids are shifted by one, SentencePiece's <unk> becomes 3


@dataclass(frozen=True, slots=True)
class Encoding:
    ids: list[int]
    attention_mask: list[int]


class SentencePieceTokenizer:
    def __init__(self, model_file: Path, max_length: int = 512) -> None:
        self._sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self._max = max_length

    def encode_batch(self, texts: Sequence[str]) -> list[Encoding]:
        rows = [[BOS, *[UNK if i == 0 else i + 1 for i in self._sp.encode(t)][: self._max - 2], EOS] for t in texts]
        width = max(len(r) for r in rows)
        return [Encoding(r + [PAD] * (width - len(r)), [1] * len(r) + [0] * (width - len(r))) for r in rows]
