"""Retrieval-test fixtures: an in-memory store over a small synthetic corpus + HashingEmbedder.

Everything is offline and deterministic. The corpus has three documents from two tickers and
two forms so the ticker / fiscal-year / form / doc_names filters each exclude something.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime

import pytest

from secqa.core.contracts import Chunk, DocumentMeta, Page
from secqa.core.ids import chunk_id
from secqa.embeddings import HashingEmbedder
from secqa.store import DuckDBStore

DIM = 64
DOCS: dict[str, tuple[str, int, str]] = {
    # doc_name -> (ticker, fiscal_year, form)
    "ACME_2022_10K": ("ACME", 2022, "10-K"),
    "ACME_2023_10K": ("ACME", 2023, "10-K"),
    "BOLT_2023_10Q": ("BOLT", 2023, "10-Q"),
}
EXACT_PHRASE = "zirconium flywheel amortisation schedule"
EXACT_DOC = "BOLT_2023_10Q"

# Each document gets the same set of topical sentences so a query matches every document unless
# a filter excludes it; the wording differs per document only through the doc name prefix.
_TOPICS = (
    "Total net sales were strong this fiscal year driven by subscription revenue growth.",
    "Operating income increased while research and development expense declined slightly.",
    "Cash and cash equivalents at year end covered all long-term debt maturities.",
    "The company repurchased shares and paid a quarterly dividend to shareholders.",
    "Inventory and accounts receivable balances rose in line with revenue.",
    "Goodwill impairment testing found no impairment in any reporting segment.",
)


def make_document(doc_name: str) -> DocumentMeta:
    ticker, year, form = DOCS[doc_name]
    return DocumentMeta(
        doc_name=doc_name,
        company=f"{ticker.title()} Corp",
        ticker=ticker,
        cik=f"{len(ticker) * 111111111:010d}",
        form=form,
        fiscal_year=year,
        period_end=date(year, 12, 31),
        source_kind="fixture",
        source_url=f"https://example.invalid/{doc_name}.pdf",
        source_sha256="0" * 64,
        n_pages=3,
        ingested_at=datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC),
    )


def make_corpus() -> list[Chunk]:
    """3 docs x 3 pages x 2 chunks = 18 chunks; one chunk carries the exact BM25 phrase."""
    chunks: list[Chunk] = []
    for doc_name in DOCS:
        for page_num in range(1, 4):
            for chunk_idx in range(2):
                topic = _TOPICS[((page_num - 1) * 2 + chunk_idx) % len(_TOPICS)]
                text = f"{doc_name} page {page_num}. {topic}"
                if doc_name == EXACT_DOC and page_num == 2 and chunk_idx == 1:
                    text = f"Note 12. The {EXACT_PHRASE} is reviewed by the audit committee."
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id(doc_name, page_num, chunk_idx, text),
                        doc_name=doc_name,
                        page_num=page_num,
                        chunk_idx=chunk_idx,
                        section="Item 7" if page_num <= 2 else "Item 8",
                        text=text,
                        n_tokens=len(text.split()),
                    )
                )
    return chunks


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=DIM)


@pytest.fixture(scope="session")
def corpus() -> list[Chunk]:
    return make_corpus()


@pytest.fixture
def empty_store(embedder: HashingEmbedder) -> Iterator[DuckDBStore]:
    """Initialised in-memory store with no documents or chunks."""
    store = DuckDBStore(":memory:", embed_dim=DIM)
    store.init_schema(embedder.name, embedder.dim)
    yield store
    store.close()


@pytest.fixture
def populated_store(embedder: HashingEmbedder, corpus: list[Chunk]) -> Iterator[DuckDBStore]:
    """In-memory store with 3 documents, 9 pages, 18 chunks and a built FTS index."""
    store = DuckDBStore(":memory:", embed_dim=DIM)
    store.init_schema(embedder.name, embedder.dim)
    for doc_name in DOCS:
        store.upsert_document(make_document(doc_name))
    store.add_pages(
        [
            Page(doc_name=doc_name, page_num=p, text=f"{doc_name} full page {p} text")
            for doc_name in DOCS
            for p in range(1, 4)
        ]
    )
    store.add_chunks(corpus, embedder.embed([c.text for c in corpus]))
    store.rebuild_fts()
    yield store
    store.close()


@pytest.fixture
def store_factory(embedder: HashingEmbedder) -> Iterator[Callable[..., DuckDBStore]]:
    """Build initialised in-memory stores with a chosen embedder name / dim; closed at teardown."""
    opened: list[DuckDBStore] = []

    def _make(dim: int = DIM, name: str | None = None) -> DuckDBStore:
        store = DuckDBStore(":memory:", embed_dim=dim)
        store.init_schema(name or embedder.name, dim)
        opened.append(store)
        return store

    yield _make
    for store in opened:
        store.close()
