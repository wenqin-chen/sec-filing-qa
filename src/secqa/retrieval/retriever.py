"""The single retrieval entry point for rag, agent, api and eval.

:class:`Retriever` is a thin, explicit layer over :class:`secqa.store.DuckDBStore`:

* **strategy switch** -- ``'bm25'`` (FTS / Okapi), ``'dense'`` (brute-force cosine over the
  embedding column) or ``'hybrid'`` (reciprocal-rank fusion of the two, SPEC 4.2);
* **query embedding** -- the question is embedded once with ``kind='query'`` and only when the
  strategy needs a vector, so the BM25-only eval rows never touch the embedder;
* **filters** -- :class:`secqa.core.contracts.RetrievalFilters` (``ticker``, ``doc_names``,
  ``fiscal_year``, ``form``) are resolved to a concrete ``doc_names`` list through
  ``store.list_documents`` before searching, so the store only ever sees a ``doc_filter``;
* **timing** -- ``RetrievalResult.latency_ms`` covers embedding + search, which is what the
  harness reports as ``retrieval_ms`` separately from LLM time.

The retriever refuses to run against an index built with a different embedding width
(:class:`secqa.core.errors.IndexMismatch`) and, for the vector strategies, a different embedder
name -- a same-width vector from another model would silently return garbage.
"""

from __future__ import annotations

import time
from typing import get_args

from secqa.core.contracts import (
    Embedder,
    Hit,
    HitView,
    RetrievalFilters,
    RetrievalResult,
    RetrievalStrategy,
)
from secqa.core.errors import ConfigError, IndexMismatch
from secqa.core.logging import get_logger
from secqa.store import DuckDBStore

log = get_logger(__name__)

STRATEGIES: tuple[str, ...] = get_args(RetrievalStrategy)
"""The accepted ``strategy`` values, taken from the shared ``RetrievalStrategy`` literal."""

SNIPPET_CHARS = 1200
"""Maximum length of ``HitView.snippet`` (CONTRACTS: ``snippet <= 1200 chars``)."""


class Retriever:
    """Strategy-switching retrieval with filters and timing over one :class:`DuckDBStore`.

    Args:
        store: An initialised store (``init_schema`` already run, or opened from a built file).
        embedder: Query embedder; must match the store's embedding width, and its name for the
            ``dense`` / ``hybrid`` strategies.
        strategy: ``'bm25'``, ``'dense'`` or ``'hybrid'`` (default).
        k: Default number of hits returned by :meth:`retrieve`.
        k_each: Depth of each sub-ranking fused by the hybrid strategy.
        rrf_k: Reciprocal-rank-fusion constant (SPEC: 60).
    """

    def __init__(
        self,
        store: DuckDBStore,
        embedder: Embedder,
        strategy: RetrievalStrategy = "hybrid",
        k: int = 8,
        k_each: int = 30,
        rrf_k: int = 60,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ConfigError(
                f"unknown retrieval strategy {strategy!r}; expected one of {', '.join(STRATEGIES)}"
            )
        _check_positive("k", k)
        _check_positive("k_each", k_each)
        _check_positive("rrf_k", rrf_k)
        self.store = store
        self.embedder = embedder
        self.strategy: RetrievalStrategy = strategy
        self.k = k
        self.k_each = k_each
        self.rrf_k = rrf_k
        self._check_embedder_matches_store()

    # ---- public API -------------------------------------------------------------------------

    @property
    def uses_embeddings(self) -> bool:
        """True when the configured strategy needs a query vector (``dense`` or ``hybrid``)."""
        return self.strategy != "bm25"

    def retrieve(
        self,
        question: str,
        *,
        k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        """Return the top-``k`` chunks for ``question`` under the configured strategy.

        Args:
            question: Natural-language query. A blank question yields an empty result rather
                than an error so callers can record ``terminated_by='empty_retrieval'``.
            k: Overrides the retriever's default ``k`` for this call (must be >= 1).
            filters: Optional ``ticker`` / ``doc_names`` / ``fiscal_year`` / ``form`` filters.
                They are resolved to document names via ``store.list_documents``; a filter that
                matches no document yields an empty result.

        Returns:
            A :class:`RetrievalResult` whose ``hits`` are 1-ranked and whose ``latency_ms``
            covers query embedding and search.
        """
        top_k = self.k if k is None else k
        _check_positive("k", top_k)
        started = time.perf_counter()

        doc_filter = self._resolve_doc_filter(filters)
        hits: list[Hit]
        if not question.strip():
            log.warning("retrieve_blank_question", strategy=self.strategy)
            hits = []
        elif doc_filter == []:
            hits = []
        else:
            hits = self._search(question, top_k, doc_filter)

        latency_ms = (time.perf_counter() - started) * 1000.0
        log.info(
            "retrieve",
            strategy=self.strategy,
            k=top_k,
            n_hits=len(hits),
            n_docs_filtered=None if doc_filter is None else len(doc_filter),
            bm25_backend=self.store.bm25_backend,
            latency_ms=round(latency_ms, 1),
        )
        return RetrievalResult(
            query=question,
            strategy=self.strategy,
            k=top_k,
            hits=hits,
            filters=filters,
            latency_ms=latency_ms,
        )

    def pages_of(self, hits: list[Hit]) -> list[tuple[str, int]]:
        """Distinct ``(doc_name, page_num)`` pairs in rank order (first occurrence wins).

        This is the projection the page-recall metrics score against gold pages.
        """
        ordered = sorted(hits, key=lambda hit: hit.rank)
        return list(dict.fromkeys((hit.chunk.doc_name, hit.chunk.page_num) for hit in ordered))

    # ---- internals --------------------------------------------------------------------------

    def _search(self, question: str, k: int, doc_filter: list[str] | None) -> list[Hit]:
        """Dispatch to the store method for the configured strategy."""
        if self.strategy == "bm25":
            return self.store.search_bm25(question, k=k, doc_filter=doc_filter)
        qvec = self.embedder.embed([question], kind="query")[0]
        if self.strategy == "dense":
            return self.store.search_dense(qvec, k=k, doc_filter=doc_filter)
        return self.store.hybrid_search(
            question,
            qvec,
            k=k,
            k_each=self.k_each,
            rrf_k=self.rrf_k,
            doc_filter=doc_filter,
        )

    def _resolve_doc_filter(self, filters: RetrievalFilters | None) -> list[str] | None:
        """Turn ``filters`` into a list of document names, or ``None`` when unfiltered.

        ``ticker``, ``doc_names`` and ``fiscal_year`` are pushed down to
        ``store.list_documents``; ``form`` is compared in Python (case-insensitive, with or
        without the hyphen, so ``'10K'`` and ``'10-k'`` both match ``'10-K'``).
        """
        if filters is None:
            return None
        if (
            filters.ticker is None
            and filters.doc_names is None
            and filters.fiscal_year is None
            and filters.form is None
        ):
            return None
        documents = self.store.list_documents(
            ticker=filters.ticker,
            doc_names=filters.doc_names,
            fiscal_year=filters.fiscal_year,
        )
        if filters.form is not None:
            wanted = _normalise_form(filters.form)
            documents = [doc for doc in documents if _normalise_form(doc.form) == wanted]
        doc_names = [doc.doc_name for doc in documents]
        if filters.doc_names:
            missing = sorted(set(filters.doc_names) - set(doc_names))
            if missing:
                log.warning("retrieve_filter_unknown_docs", doc_names=missing)
        if not doc_names:
            log.warning(
                "retrieve_filter_matches_no_documents",
                ticker=filters.ticker,
                doc_names=filters.doc_names,
                fiscal_year=filters.fiscal_year,
                form=filters.form,
            )
        return doc_names

    def _check_embedder_matches_store(self) -> None:
        """Raise :class:`IndexMismatch` when the embedder cannot query this store."""
        store_dim = self.store.dim  # ConfigError before init_schema; that is the right error.
        if self.embedder.dim != store_dim:
            raise IndexMismatch(
                f"embedder {self.embedder.name!r} has dim {self.embedder.dim}; "
                f"store was built with dim {store_dim} ({self.store.embedder_name})"
            )
        store_name = self.store.embedder_name
        if store_name is not None and store_name != self.embedder.name:
            if self.uses_embeddings:
                raise IndexMismatch(
                    f"store was built with embedder {store_name!r}; "
                    f"strategy {self.strategy!r} cannot use {self.embedder.name!r}"
                )
            log.warning(
                "retriever_embedder_name_differs",
                store_embedder=store_name,
                embedder=self.embedder.name,
                strategy=self.strategy,
                note="bm25 does not use embeddings",
            )


def to_hit_view(hit: Hit, max_chars: int = SNIPPET_CHARS) -> HitView:
    """Project a :class:`Hit` to the API / tool-facing :class:`HitView`.

    The snippet is the chunk text truncated to ``max_chars`` (store-sourced, never model text).
    """
    _check_positive("max_chars", max_chars)
    return HitView(
        chunk_id=hit.chunk.chunk_id,
        doc_name=hit.chunk.doc_name,
        page_num=hit.chunk.page_num,
        section=hit.chunk.section,
        score=hit.score,
        snippet=hit.chunk.text[:max_chars],
    )


def _normalise_form(form: str) -> str:
    """``'10-k'`` / ``'10K'`` / ``' 10-K '`` -> ``'10K'`` for tolerant form comparison."""
    return form.strip().upper().replace("-", "")


def _check_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")


__all__ = ["SNIPPET_CHARS", "STRATEGIES", "Retriever", "to_hit_view"]
