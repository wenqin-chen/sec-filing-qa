"""Grounding: deterministic citation verification shared by rag, agent, api and eval.

Public surface: :class:`CitationVerifier`, :func:`parse_ref`, :func:`quote_in_chunk`.
"""

from secqa.grounding.verifier import (
    CHUNK_PREFIX,
    SNIPPET_MAX_CHARS,
    XBRL_PREFIX,
    CitationVerifier,
    RefKind,
    parse_ref,
    quote_in_chunk,
)

__all__ = [
    "CHUNK_PREFIX",
    "SNIPPET_MAX_CHARS",
    "XBRL_PREFIX",
    "CitationVerifier",
    "RefKind",
    "parse_ref",
    "quote_in_chunk",
]
