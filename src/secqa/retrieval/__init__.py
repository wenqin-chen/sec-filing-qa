"""secqa.retrieval: the single retrieval entry point for rag, agent, api and eval.

Public surface: :class:`Retriever` (strategy switch, query embedding, ticker / doc / year / form
filters, timing) and :func:`to_hit_view` (the API / tool projection of a ``Hit``).
"""

from secqa.retrieval.retriever import SNIPPET_CHARS, STRATEGIES, Retriever, to_hit_view

__all__ = ["SNIPPET_CHARS", "STRATEGIES", "Retriever", "to_hit_view"]
