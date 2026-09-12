"""Local open-source embedder via ``sentence-transformers`` (``uv sync --extra local``).

The default model, ``BAAI/bge-small-en-v1.5`` (384-d, MIT), is what the published index and
the retrieval rows in RESULTS.md are built with. Its model card asks for an instruction prefix
on *short queries* used for passage retrieval and none on the passages themselves; that
asymmetry is why :class:`secqa.core.contracts.Embedder.embed` takes ``kind``.

``sentence_transformers`` (and therefore torch) is imported lazily inside ``__init__`` so the
rest of the package imports without it; tests that need it are ``@pytest.mark.slow``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from secqa.core.errors import ConfigError
from secqa.core.logging import get_logger
from secqa.embeddings.base import BaseEmbedder, EmbedKind

_log = get_logger(__name__)

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"

# (query_prefix, passage_prefix) per model family, from the respective model cards.
# BGE v1 / v1.5 English: instruction on queries only. E5: "query: " / "passage: " on both sides.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_E5_PREFIXES = ("query: ", "passage: ")


def prefixes_for(model_name: str) -> tuple[str, str]:
    """Return ``(query_prefix, passage_prefix)`` recommended by the model card of ``model_name``.

    Unknown models get no prefix. Matching is on the lower-cased model id, so both
    ``'BAAI/bge-small-en-v1.5'`` and a local path ending in ``bge-base-en`` are recognised.
    """
    lowered = model_name.lower()
    base = lowered.rsplit("/", 1)[-1]
    if base.startswith("bge-") and "-en" in base and "bge-m3" not in base:
        return (_BGE_QUERY_INSTRUCTION, "")
    if base.startswith(("e5-", "multilingual-e5-")):
        return _E5_PREFIXES
    return ("", "")


class LocalEmbedder(BaseEmbedder):
    """Sentence-transformers embedder running on CPU (or ``device``) with no network at query time.

    Args:
        model_name: Hugging Face model id or a local directory containing the weights.
        device: ``'cpu'`` (default, what the container runs), ``'cuda'``, ``'mps'`` ...
        cache_dir: Where to keep downloaded weights (``None`` = the Hugging Face default cache).

    Raises:
        ConfigError: ``sentence-transformers`` is not installed.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ConfigError("local embedder model_name must not be empty")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching
            raise ConfigError(
                "the local embedder needs sentence-transformers; install it with "
                "`uv sync --extra local` (or choose embedder='hashing' / 'openai')"
            ) from exc

        self.model_name = model_name.strip()
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.query_prefix, self.passage_prefix = prefixes_for(self.model_name)

        kwargs: dict[str, Any] = {"device": device}
        if self.cache_dir is not None:
            kwargs["cache_folder"] = str(self.cache_dir)
        self._model = SentenceTransformer(self.model_name, **kwargs)

        dim = self._model.get_sentence_embedding_dimension()
        if not isinstance(dim, int) or dim < 1:
            raise ConfigError(
                f"could not determine the embedding dimension of {self.model_name!r} (got {dim!r})"
            )
        self.dim = dim
        # 'BAAI/bge-small-en-v1.5' -> 'bge-small-en-v1.5' (what index_manifest records).
        self.name = self.model_name.rstrip("/").rsplit("/", 1)[-1]
        _log.info(
            "local embedder loaded",
            embedder=self.name,
            model=self.model_name,
            dim=self.dim,
            device=device,
            query_prefix=bool(self.query_prefix),
        )

    def _embed_batch(self, texts: list[str], kind: EmbedKind) -> np.ndarray:
        """Encode one batch with the model-card prefix for ``kind`` applied."""
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        inputs = [prefix + text for text in texts] if prefix else texts
        vectors = self._model.encode(
            inputs,
            batch_size=len(inputs),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


__all__ = ["DEFAULT_LOCAL_MODEL", "LocalEmbedder", "prefixes_for"]
