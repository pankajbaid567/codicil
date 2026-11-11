"""ONNX cross-encoder reranker (MS MARCO MiniLM, int8)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# the default arena grows with every new input shape (400 -> 670 MB RSS in the benchmark); this keeps it flat
_PROVIDERS = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


class OnnxReranker:
    def __init__(self, model_dir: Path, max_len: int = 320, batch_size: int = 8, threads: int = 2) -> None:
        self.model_id = model_dir.name
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_len)
        self._tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.enable_mem_pattern = False  # variable-length inputs make the shape-pattern cache useless
        self._sess = ort.InferenceSession(str(model_dir / "onnx" / "model_quantized.onnx"), opts, providers=_PROVIDERS)
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._batch = batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(passages), self._batch):
            enc = self._tok.encode_batch([(query, p) for p in passages[i : i + self._batch]])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64), "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            logits = self._sess.run(None, feed)[0]
            out += [float(x) for x in np.asarray(logits).reshape(len(enc), -1)[:, 0]]
        return out
