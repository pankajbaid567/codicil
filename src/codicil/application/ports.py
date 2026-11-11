"""Ports: the interfaces the application layer needs from the outside world. Adapters implement them; fakes in tests do too."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Maps text to unit-length vectors. Query and passage are separate entry points because many models need different prefixes."""

    model_id: str
    dim: int

    def embed_queries(self, texts: Sequence[str]) -> Matrix: ...

    def embed_passages(self, texts: Sequence[str]) -> Matrix: ...


@runtime_checkable
class Reranker(Protocol):
    """Scores (query, passage) pairs; higher is more relevant. Scores are only comparable within one call."""

    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    system: str
    user: str
    max_tokens: int = 700
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class GenerationError(Exception):
    """The provider failed, timed out, or its circuit is open. The caller degrades to extractive mode."""


@runtime_checkable
class Generator(Protocol):
    model_id: str

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class Clock(Protocol):
    def monotonic(self) -> float: ...
