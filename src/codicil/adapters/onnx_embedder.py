"""ONNX Runtime sentence embedder (int8 BERT-family models). Owns the model-specific query/passage prefixes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from codicil.application.ports import Matrix

# model id -> (query prefix, passage prefix, pooling)
_SPECS: dict[str, tuple[str, str, str]] = {
    "bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", "", "cls"),
    "multilingual-e5-small": ("query: ", "passage: ", "mean"),
}

# the default arena grows with every new input shape (400 -> 670 MB RSS in the benchmark); this keeps it flat
_PROVIDERS = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


class OnnxEmbedder:
    def __init__(self, model_dir: Path, max_len: int = 512, batch_size: int = 16, threads: int = 2) -> None:
        self.model_id = model_dir.name
        if self.model_id not in _SPECS:
            raise ValueError(f"unknown embedding model {self.model_id!r}; known: {sorted(_SPECS)}")
        self._qp, self._pp, self._pool = _SPECS[self.model_id]
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_len)
        self._tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.enable_mem_pattern = False  # variable-length inputs make the shape-pattern cache useless
        self._sess = ort.InferenceSession(str(model_dir / "onnx" / "model_quantized.onnx"), opts, providers=_PROVIDERS)
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._batch = batch_size
        self.dim = int(self._sess.get_outputs()[0].shape[-1])

    def _encode(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[Matrix] = []
        for i in range(0, len(texts), self._batch):
            enc = self._tok.encode_batch(list(texts[i : i + self._batch]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._sess.run(None, feed)[0].astype(np.float32)
            if self._pool == "cls":
                vec = hidden[:, 0, :]
            else:
                m = mask[:, :, None].astype(np.float32)
                vec = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            out.append(vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12))
        return np.vstack(out).astype(np.float32)

    def embed_queries(self, texts: Sequence[str]) -> Matrix:
        return self._encode([self._qp + t for t in texts])

    def embed_passages(self, texts: Sequence[str]) -> Matrix:
        return self._encode([self._pp + t for t in texts])
