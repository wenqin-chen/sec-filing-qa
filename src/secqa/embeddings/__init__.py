"""Text -> L2-normalised float32 vectors (``secqa.core.contracts.Embedder``).

Backends: :class:`HashingEmbedder` (key-free, CI), :class:`LocalEmbedder` (bge-small, default for
real indexes), :class:`OpenAIEmbedder` (``text-embedding-3-small`` ablation). Pick one from a spec
string with :func:`get_embedder`.
"""

from secqa.embeddings.base import BaseEmbedder, EmbedKind, UsageCallback, l2_normalise
from secqa.embeddings.hashing import HashingEmbedder
from secqa.embeddings.local import DEFAULT_LOCAL_MODEL, LocalEmbedder
from secqa.embeddings.openai_embed import DEFAULT_OPENAI_MODEL, OpenAIEmbedder
from secqa.embeddings.registry import get_embedder, parse_spec

__all__ = [
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "BaseEmbedder",
    "EmbedKind",
    "HashingEmbedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "UsageCallback",
    "get_embedder",
    "l2_normalise",
    "parse_spec",
]
