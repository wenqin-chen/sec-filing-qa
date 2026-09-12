"""Retriever: strategy switch, k, filters, timing, empty index, embedder/store checks."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from secqa.core.contracts import Chunk, Hit, HitView, RetrievalFilters, RetrievalResult
from secqa.core.errors import ConfigError, IndexMismatch
from secqa.embeddings import HashingEmbedder
from secqa.retrieval import SNIPPET_CHARS, STRATEGIES, Retriever, to_hit_view
from secqa.store import DuckDBStore
from tests.retrieval.conftest import DIM, DOCS, EXACT_DOC, EXACT_PHRASE

QUESTION = "How did subscription revenue and net sales develop this fiscal year?"


# ---- construction -------------------------------------------------------------------------


def test_strategies_constant_matches_contract_literal() -> None:
    assert STRATEGIES == ("bm25", "dense", "hybrid")


def test_defaults(populated_store: DuckDBStore, embedder: HashingEmbedder) -> None:
    retriever = Retriever(populated_store, embedder)
    assert retriever.strategy == "hybrid"
    assert (retriever.k, retriever.k_each, retriever.rrf_k) == (8, 30, 60)
    assert retriever.uses_embeddings is True


def test_unknown_strategy_is_config_error(
    populated_store: DuckDBStore, embedder: HashingEmbedder
) -> None:
    with pytest.raises(ConfigError, match="unknown retrieval strategy"):
        Retriever(populated_store, embedder, strategy="semantic")  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["k", "k_each", "rrf_k"])
def test_non_positive_ints_rejected(
    populated_store: DuckDBStore, embedder: HashingEmbedder, field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        Retriever(populated_store, embedder, **{field: 0})


def test_dim_mismatch_raises_index_mismatch(store_factory: Callable[..., DuckDBStore]) -> None:
    store = store_factory(dim=DIM)
    with pytest.raises(IndexMismatch, match="dim"):
        Retriever(store, HashingEmbedder(dim=DIM * 2), strategy="bm25")


def test_embedder_name_mismatch_blocks_vector_strategies_only(
    store_factory: Callable[..., DuckDBStore], embedder: HashingEmbedder
) -> None:
    store = store_factory(dim=DIM, name="bge-small-en-v1.5")
    for strategy in ("dense", "hybrid"):
        with pytest.raises(IndexMismatch, match="bge-small-en-v1.5"):
            Retriever(store, embedder, strategy=strategy)  # type: ignore[arg-type]
    # BM25 never touches the embedding column, so a differently named embedder is tolerated.
    assert Retriever(store, embedder, strategy="bm25").uses_embeddings is False


def test_store_without_schema_is_config_error(embedder: HashingEmbedder) -> None:
    with DuckDBStore(":memory:") as store:
        with pytest.raises(ConfigError, match="init_schema"):
            Retriever(store, embedder)


# ---- strategy switch ------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategy_switch_sets_source_and_sub_ranks(
    populated_store: DuckDBStore, embedder: HashingEmbedder, strategy: str
) -> None:
    retriever = Retriever(populated_store, embedder, strategy=strategy, k=5)  # type: ignore[arg-type]
    result = retriever.retrieve(QUESTION)
    assert isinstance(result, RetrievalResult)
    assert result.strategy == strategy
    assert result.query == QUESTION
    assert result.hits, "the corpus shares vocabulary with the question; expected hits"
    assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))
    for hit in result.hits:
        assert hit.source == strategy
        if strategy == "bm25":
            assert hit.bm25_rank == hit.rank and hit.dense_rank is None
        elif strategy == "dense":
            assert hit.dense_rank == hit.rank and hit.bm25_rank is None
        else:
            assert hit.bm25_rank is not None or hit.dense_rank is not None


def test_bm25_finds_exact_phrase_first(
    populated_store: DuckDBStore, embedder: HashingEmbedder
) -> None:
    retriever = Retriever(populated_store, embedder, strategy="bm25", k=3)
    result = retriever.retrieve(EXACT_PHRASE)
    assert result.hits[0].chunk.doc_name == EXACT_DOC
    assert EXACT_PHRASE in result.hits[0].chunk.text


def test_hybrid_contains_bm25_top_hit(
    populated_store: DuckDBStore, embedder: HashingEmbedder
) -> None:
    bm25_top = Retriever(populated_store, embedder, strategy="bm25", k=1).retrieve(EXACT_PHRASE)
    hybrid = Retriever(populated_store, embedder, strategy="hybrid", k=5).retrieve(EXACT_PHRASE)
    hybrid_ids = {hit.chunk.chunk_id for hit in hybrid.hits}
    assert bm25_top.hits[0].chunk.chunk_id in hybrid_ids
    assert all(hit.source == "hybrid" for hit in hybrid.hits)


def test_bm25_strategy_never_calls_embedder(
    populated_store: DuckDBStore, embedder: HashingEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("embedder must not be called for bm25")

    monkeypatch.setattr(embedder, "embed", _boom)
    result = Retriever(populated_store, embedder, strategy="bm25", k=3).retrieve(QUESTION)
    assert result.hits


def test_query_is_embedded_as_query_kind(
    populated_store: DuckDBStore, embedder: HashingEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    original = embedder.embed

    def _spy(texts: list[str], *, batch_size: int = 64, kind: str = "passage"):  # type: ignore[no-untyped-def]
        seen.append(kind)
        return original(texts, batch_size=batch_size, kind=kind)  # type: ignore[arg-type]

    monkeypatch.setattr(embedder, "embed", _spy)
    Retriever(populated_store, embedder, strategy="dense", k=3).retrieve(QUESTION)
    assert seen == ["query"]


# ---- k ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_k_respected_default_and_override(
    populated_store: DuckDBStore, embedder: HashingEmbedder, strategy: str
) -> None:
    retriever = Retriever(populated_store, embedder, strategy=strategy, k=4)  # type: ignore[arg-type]
    default = retriever.retrieve(QUESTION)
    assert default.k == 4 and len(default.hits) == 4
    override = retriever.retrieve(QUESTION, k=2)
    assert override.k == 2 and len(override.hits) == 2
    assert [h.chunk.chunk_id for h in override.hits] == [h.chunk.chunk_id for h in default.hits[:2]]


def test_k_larger_than_corpus_returns_everything_available(
    populated_store: DuckDBStore, embedder: HashingEmbedder
) -> None:
    result = Retriever(populated_store, embedder, strategy="dense", k=100).retrieve(QUESTION)
    assert result.k == 100
    assert len(result.hits) == populated_store.counts()["chunks"]


def test_k_override_must_be_positive(
    populated_store: DuckDBStore, embedder: HashingEmbedder
) -> None:
    with pytest.raises(ValueError, match="k"):
        Retriever(populated_store, embedder).retrieve(QUESTION, k=0)


# ---- filters ---------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_ticker_filter_excludes_other_docs(
    populated_store: DuckDBStore, embedder: HashingEmbedder, strategy: str
) -> None:
    retriever = Retriever(populated_store, embedder, strategy=strategy, k=20)  # type: ignore[arg-type]
    unfiltered = retriever.retrieve(QUESTION)
    assert {h.chunk.doc_name for h in unfiltered.hits} == set(DOCS)
    filters = RetrievalFilters(ticker="acme")  # case-insensitive
    result = retriever.retrieve(QUESTION, filters=filters)
    assert result.hits
    assert {h.chunk.doc_name for h in result.hits} == {"ACME_2022_10K", "ACME_2023_10K"}
    assert result.filters == filters


def test_fiscal_year_filter(populated_store: DuckDBStore, embedder: HashingEmbedder) -> None:
    retriever = Retriever(populated_store, embedder, k=20)
    result = retriever.retrieve(QUESTION, filters=RetrievalFilters(fiscal_year=2023))
    assert {h.chunk.doc_name for h in result.hits} == {"ACME_2023_10K", "BOLT_2023_10Q"}


def test_ticker_and_year_combine(populated_store: DuckDBStore, embedder: HashingEmbedder) -> None:
    retriever = Retriever(populated_store, embedder, k=20)
    filters = RetrievalFilters(ticker="ACME", fiscal_year=2022)
    result = retriever.retrieve(QUESTION, filters=filters)
    assert {h.chunk.doc_name for h in result.hits} == {"ACME_2022_10K"}


@pytest.mark.parametrize("form", ["10-Q", "10q", " 10-q "])
def test_form_filter_is_tolerant_of_hyphen_and_case(
    populated_store: DuckDBStore, embedder: HashingEmbedder, form: str
) -> None:
    retriever = Retriever(populated_store, embedder, k=20)
    result = retriever.retrieve(QUESTION, filters=RetrievalFilters(form=form))
    assert {h.chunk.doc_name for h in result.hits} == {"BOLT_2023_10Q"}


def test_doc_names_filter(populated_store: DuckDBStore, embedder: HashingEmbedder) -> None:
    retriever = Retriever(populated_store, embedder, k=20)
    filters = RetrievalFilters(doc_names=["ACME_2023_10K", "NOT_A_DOC"])
    result = retriever.retrieve(QUESTION, filters=filters)
    assert {h.chunk.doc_name for h in result.hits} == {"ACME_2023_10K"}


def test_filter_matching_nothing_returns_empty_without_error(
    populated_store: DuckDBStore, embedder: HashingEmbedder
) -> None:
    retriever = Retriever(populated_store, embedder, k=5)
    for filters in (
        RetrievalFilters(ticker="ZZZZ"),
        RetrievalFilters(doc_names=[]),
        RetrievalFilters(fiscal_year=1999),
        RetrievalFilters(ticker="ACME", form="10-Q"),
    ):
        result = retriever.retrieve(QUESTION, filters=filters)
        assert result.hits == []
        assert result.filters == filters


def test_all_none_filters_behave_like_no_filter(
    populated_store: DuckDBStore, embedder: HashingEmbedder
) -> None:
    retriever = Retriever(populated_store, embedder, k=20)
    plain = retriever.retrieve(QUESTION)
    empty = retriever.retrieve(QUESTION, filters=RetrievalFilters())
    assert [h.chunk.chunk_id for h in empty.hits] == [h.chunk.chunk_id for h in plain.hits]
    assert plain.filters is None
    assert empty.filters == RetrievalFilters()


# ---- timing / empty cases ------------------------------------------------------------------


def test_latency_populated(populated_store: DuckDBStore, embedder: HashingEmbedder) -> None:
    result = Retriever(populated_store, embedder).retrieve(QUESTION)
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms > 0.0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_empty_index_returns_empty_result(
    empty_store: DuckDBStore, embedder: HashingEmbedder, strategy: str
) -> None:
    retriever = Retriever(empty_store, embedder, strategy=strategy, k=5)  # type: ignore[arg-type]
    result = retriever.retrieve(QUESTION)
    assert result.hits == []
    assert result.k == 5 and result.strategy == strategy
    assert result.latency_ms >= 0.0
    assert retriever.pages_of(result.hits) == []


@pytest.mark.parametrize("question", ["", "   \n\t"])
def test_blank_question_returns_empty_result(
    populated_store: DuckDBStore, embedder: HashingEmbedder, question: str
) -> None:
    result = Retriever(populated_store, embedder).retrieve(question)
    assert result.hits == []
    assert result.query == question


# ---- pages_of / to_hit_view ------------------------------------------------------------------


def _hit(doc: str, page: int, rank: int, text: str = "some chunk text") -> Hit:
    chunk = Chunk(
        chunk_id=f"{doc}-{page}-{rank}",
        doc_name=doc,
        page_num=page,
        chunk_idx=0,
        section="Item 7",
        text=text,
        n_tokens=3,
    )
    return Hit(chunk=chunk, score=1.0 / rank, rank=rank, source="hybrid")


def test_pages_of_is_distinct_in_rank_order(
    populated_store: DuckDBStore, embedder: HashingEmbedder
) -> None:
    retriever = Retriever(populated_store, embedder)
    hits = [
        _hit("A_2023_10K", 5, 1),
        _hit("A_2023_10K", 5, 2),  # same page again: deduplicated
        _hit("B_2023_10K", 1, 3),
        _hit("A_2023_10K", 2, 4),
        _hit("B_2023_10K", 1, 5),
    ]
    assert retriever.pages_of(hits) == [("A_2023_10K", 5), ("B_2023_10K", 1), ("A_2023_10K", 2)]
    # Order is by rank even if the list arrives shuffled.
    assert retriever.pages_of(list(reversed(hits))) == retriever.pages_of(hits)
    assert retriever.pages_of([]) == []


def test_pages_of_on_real_hits_matches_store_rows(
    populated_store: DuckDBStore, embedder: HashingEmbedder
) -> None:
    retriever = Retriever(populated_store, embedder, k=20)
    result = retriever.retrieve(QUESTION)
    pages = retriever.pages_of(result.hits)
    assert len(pages) == len(set(pages))
    assert set(pages) <= {(d, p) for d in DOCS for p in range(1, 4)}


def test_to_hit_view_truncates_snippet_and_copies_fields() -> None:
    long_text = "x" * (SNIPPET_CHARS + 50)
    view = to_hit_view(_hit("A_2023_10K", 3, 1, text=long_text))
    assert isinstance(view, HitView)
    assert (view.chunk_id, view.doc_name, view.page_num, view.section) == (
        "A_2023_10K-3-1",
        "A_2023_10K",
        3,
        "Item 7",
    )
    assert view.score == 1.0
    assert len(view.snippet) == SNIPPET_CHARS
    assert to_hit_view(_hit("A_2023_10K", 3, 1), max_chars=4).snippet == "some"
    with pytest.raises(ValueError, match="max_chars"):
        to_hit_view(_hit("A_2023_10K", 3, 1), max_chars=0)
