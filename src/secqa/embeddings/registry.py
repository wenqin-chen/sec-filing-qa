"""Resolve an embedder spec string (``Settings.embedder`` / ``--embedder``) to an instance.

Accepted specs::

    hashing            HashingEmbedder(dim=384)              key-free, no downloads (CI default)
    hashing:<dim>      HashingEmbedder(dim=<dim>)
    local              LocalEmbedder('BAAI/bge-small-en-v1.5') needs `uv sync --extra local`
    local:<model>      LocalEmbedder(<model>)
    openai             OpenAIEmbedder('text-embedding-3-small', dim=384) needs OPENAI_API_KEY
    openai:<model>     OpenAIEmbedder(<model>, dim=384)

Anything else raises :class:`secqa.core.errors.ConfigError`. Unlike LLM providers there is no
Anthropic option: Anthropic has no embeddings API (SPEC 11), hence the local default.
"""

from __future__ import annotations

from secqa.core.contracts import Embedder
from secqa.core.errors import ConfigError
from secqa.core.logging import get_logger
from secqa.core.settings import Settings, get_settings
from secqa.embeddings.hashing import HashingEmbedder
from secqa.embeddings.local import DEFAULT_LOCAL_MODEL, LocalEmbedder
from secqa.embeddings.openai_embed import DEFAULT_OPENAI_MODEL, OpenAIEmbedder

_log = get_logger(__name__)

KINDS = ("hashing", "local", "openai")
DEFAULT_DIM = 384


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``'kind[:arg]'`` into ``(kind, arg)`` with ``kind`` lower-cased and validated."""
    if not isinstance(spec, str) or not spec.strip():
        raise ConfigError("embedder spec is empty; expected one of " + ", ".join(KINDS))
    kind, _, arg = spec.strip().partition(":")
    kind = kind.strip().lower()
    arg = arg.strip()
    if kind not in KINDS:
        raise ConfigError(
            f"unknown embedder {spec!r}; expected one of "
            + ", ".join(f"'{k}'" for k in KINDS)
            + " optionally followed by ':<model>' (':<dim>' for hashing)"
        )
    return kind, arg


def get_embedder(spec: str, settings: Settings | None = None) -> Embedder:
    """Build the embedder described by ``spec`` (see module docstring for the grammar).

    ``settings`` supplies the OpenAI key; ``None`` uses the process-wide :func:`get_settings`.
    """
    kind, arg = parse_spec(spec)
    embedder: Embedder
    if kind == "hashing":
        dim = DEFAULT_DIM
        if arg:
            try:
                dim = int(arg)
            except ValueError as exc:
                raise ConfigError(
                    f"hashing embedder takes an integer dim, got {arg!r} in {spec!r}"
                ) from exc
        try:
            embedder = HashingEmbedder(dim=dim)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    elif kind == "local":
        embedder = LocalEmbedder(model_name=arg or DEFAULT_LOCAL_MODEL)
    else:  # openai
        resolved = settings if settings is not None else get_settings()
        secret = resolved.openai_api_key
        key = secret.get_secret_value().strip() if secret is not None else ""
        if not key:
            raise ConfigError(f"embedder {spec!r} requires OPENAI_API_KEY but it is not set")
        embedder = OpenAIEmbedder(model=arg or DEFAULT_OPENAI_MODEL, dim=DEFAULT_DIM, api_key=key)
    _log.info("embedder selected", spec=spec, embedder=embedder.name, dim=embedder.dim)
    return embedder


__all__ = ["DEFAULT_DIM", "KINDS", "get_embedder", "parse_spec"]
