"""Key-free, download-free embedder built on scikit-learn's ``HashingVectorizer``.

This is the CI / offline default. It is *not* a semantic embedder: two texts are close when they
share words and word bigrams, nothing more. That is exactly what tests and the fixture-corpus
smoke evaluation need (deterministic, instant, no weights on disk) and it is never used for
published numbers, which are built with the local ``bge-small-en-v1.5`` model.

Why hashing and not TF-IDF: a ``HashingVectorizer`` is stateless, so the vector for a text does
not depend on which other texts were indexed. Queries embedded at answer time and chunks embedded
at ingest time therefore live in the same space without persisting a fitted vocabulary.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from secqa.core.logging import get_logger
from secqa.embeddings.base import BaseEmbedder, EmbedKind

_log = get_logger(__name__)


class HashingEmbedder(BaseEmbedder):
    """Deterministic word + bigram hashing embedder (sklearn ``HashingVectorizer``).

    Args:
        dim: Number of hash buckets, i.e. the vector width. Must be positive.
        seed: Salt appended to every token before hashing. Two embedders with different seeds
            produce different (but each internally consistent) spaces; the default ``0`` adds no
            salt so vectors equal plain ``HashingVectorizer`` output.
    """

    def __init__(self, dim: int = 384, seed: int = 0) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"seed must be an int, got {seed!r}")
        # Imported here so `secqa.embeddings` stays cheap to import for backends that don't need it.
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = f"hashing-{dim}"
        self.dim = dim
        self.seed = seed
        # Word + bigram analyzer with sklearn's defaults (lowercase, accent-insensitive tokens of
        # >= 2 alphanumerics). Numbers such as "1,577" tokenise as "1" and "577", which is fine for
        # a lexical stand-in.
        tokeniser: Callable[[str], list[str]] = HashingVectorizer(
            ngram_range=(1, 2), strip_accents="unicode", lowercase=True
        ).build_analyzer()
        if seed == 0:
            analyzer = tokeniser
        else:
            salt = f"\x1f{seed}"

            def analyzer(text: str) -> list[str]:
                return [token + salt for token in tokeniser(text)]

        # norm=None: BaseEmbedder normalises; alternate_sign spreads collisions across the sign so
        # colliding buckets partially cancel instead of always adding up.
        self._vectorizer = HashingVectorizer(
            n_features=dim,
            analyzer=analyzer,
            norm=None,
            alternate_sign=True,
            dtype=np.float32,
        )
        _log.debug("hashing embedder ready", embedder=self.name, seed=seed)

    def _embed_batch(self, texts: list[str], kind: EmbedKind) -> np.ndarray:
        """Hash one batch. ``kind`` is ignored: a lexical space has no query/passage asymmetry."""
        sparse = self._vectorizer.transform(texts)
        return np.asarray(sparse.toarray(), dtype=np.float32)


__all__ = ["HashingEmbedder"]
