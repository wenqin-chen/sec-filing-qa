"""Shared plumbing for every embedding backend.

All backends subclass :class:`BaseEmbedder`, which owns the behaviour the
:class:`secqa.core.contracts.Embedder` protocol promises and that must not differ between
backends:

* input validation (a ``list[str]``, nothing else);
* batching by ``batch_size``;
* blank texts (empty or whitespace-only) are never sent to a backend and come back as a zero
  vector, so the hashing, local and OpenAI embedders agree on the one input none of them can
  embed meaningfully (OpenAI rejects ``""`` outright);
* the output is always ``float32``, shape ``(n, dim)`` and L2-normalised.

Subclasses implement only :meth:`BaseEmbedder._embed_batch`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import Literal, TypeVar

import numpy as np

from secqa.core.logging import get_logger

EmbedKind = Literal["query", "passage"]
UsageCallback = Callable[[int], None]
"""Called with the number of billable tokens after each backend request (OpenAI only)."""

_T = TypeVar("_T")
_log = get_logger(__name__)


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Return ``vectors`` (2-D) as float32 rows of unit L2 norm; all-zero rows stay zero.

    Zero rows are kept rather than turned into NaN so a blank text embeds to a vector that has
    cosine similarity 0 with everything, which is the honest answer.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    return np.ascontiguousarray(arr / safe, dtype=np.float32)


def batched(items: Sequence[_T], size: int) -> Iterator[Sequence[_T]]:
    """Yield consecutive slices of ``items`` with at most ``size`` elements each."""
    if size < 1:
        raise ValueError(f"batch_size must be >= 1, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def validate_texts(texts: object) -> list[str]:
    """Check ``texts`` is a list (or tuple) of ``str`` and return it as a list."""
    if isinstance(texts, str) or not isinstance(texts, list | tuple):
        raise TypeError(f"texts must be a list of str, got {type(texts).__name__}")
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{i}] must be str, got {type(text).__name__}")
    return list(texts)


class BaseEmbedder(ABC):
    """Template for embedders: validation, blank handling, batching and normalisation.

    Attributes ``name`` and ``dim`` are set by subclasses in ``__init__`` and satisfy the
    :class:`secqa.core.contracts.Embedder` protocol structurally.
    """

    name: str
    dim: int

    @abstractmethod
    def _embed_batch(self, texts: list[str], kind: EmbedKind) -> np.ndarray:
        """Embed one batch of non-blank texts; returns any float array of shape ``(len, dim)``.

        Normalisation and dtype conversion happen in :meth:`embed`; subclasses need not do it.
        """

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
        kind: EmbedKind = "passage",
    ) -> np.ndarray:
        """Embed ``texts`` and return a float32, L2-normalised array of shape ``(n, dim)``.

        Args:
            texts: The strings to embed. Blank strings embed to a zero vector.
            batch_size: Maximum number of texts per backend call.
            kind: ``'query'`` for search queries, ``'passage'`` for indexed chunks. Backends that
                distinguish the two (bge query instruction) apply it here; the others ignore it.
        """
        if kind not in ("query", "passage"):
            raise ValueError(f"kind must be 'query' or 'passage', got {kind!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        items = validate_texts(texts)
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        non_blank = [i for i, text in enumerate(items) if text.strip()]
        if not non_blank:
            return out

        started = time.perf_counter()
        for index_batch in batched(non_blank, batch_size):
            batch_texts = [items[i] for i in index_batch]
            vectors = np.asarray(self._embed_batch(batch_texts, kind), dtype=np.float32)
            if vectors.shape != (len(batch_texts), self.dim):
                raise RuntimeError(
                    f"{self.name}: backend returned shape {vectors.shape}, "
                    f"expected {(len(batch_texts), self.dim)}"
                )
            out[list(index_batch)] = vectors
        _log.debug(
            "embedded",
            embedder=self.name,
            n_texts=len(items),
            n_blank=len(items) - len(non_blank),
            kind=kind,
            batch_size=batch_size,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return l2_normalise(out)


__all__ = [
    "BaseEmbedder",
    "EmbedKind",
    "UsageCallback",
    "batched",
    "l2_normalise",
    "validate_texts",
]
