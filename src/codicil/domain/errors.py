"""Typed errors. Expected failures are values (see `result.py`); these are for programmer errors and fatal configuration."""

from __future__ import annotations


class CodicilError(Exception):
    """Base class for every error this package raises on purpose."""


class IndexMismatchError(CodicilError):
    """The index was built with a different embedder than the one serving queries."""


class CorpusError(CodicilError):
    """A source document is missing, changed unexpectedly or cannot be parsed."""


class ConfigError(CodicilError):
    """Settings are invalid or inconsistent."""
