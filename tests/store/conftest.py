"""Store-test fixtures: an offline hashing embedder and a 200-chunk synthetic corpus.

The embedder here is deliberately local to the store tests (sklearn ``HashingVectorizer``, no
downloads) so these tests do not depend on the ``embeddings`` module being built first.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime

import numpy as np
import pytest

from secqa.core.contracts import Chunk, DocumentMeta, Page
from secqa.core.ids import chunk_id
from secqa.store import DuckDBStore

DIM = 64
DOCS = ("ACME_2022_10K", "ACME_2023_10K", "BOLT_2023_10K", "BOLT_2023_10Q")
EXACT_PHRASE = "zirconium flywheel amortisation schedule"
PARAPHRASE_TEXT = "Revenue from subscriptions grew strongly during the fiscal year."
PARAPHRASE_QUERY = "recurring billing income climbed sharply"

_WORDS = (
    "revenue income assets liabilities equity cash operating margin segment guidance "
    "capital expenditure depreciation amortization goodwill inventory receivable payable "
    "dividend repurchase share outstanding diluted earnings tax deferred lease pension "
    "research development marketing restructuring impairment foreign currency hedge"
).split()


class HashingEmbedder:
    """Deterministic, download-free embedder satisfying the ``Embedder`` protocol shape."""

    name = "hashing-test"
    dim = DIM

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            n_features=DIM, norm="l2", alternate_sign=False, ngram_range=(1, 2)
        )

    def embed(self, texts: list[str], *, batch_size: int = 64, kind: str = "passage") -> np.ndarray:
        matrix = self._vec.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


def make_document(doc_name: str, n_pages: int = 5) -> DocumentMeta:
    ticker, year, form = doc_name.split("_")
    return DocumentMeta(
        doc_name=doc_name,
        company=f"{ticker.title()} Corp",
        ticker=ticker,
        cik=f"{abs(hash(ticker)) % 10**10:010d}",
        form="10-K" if form == "10K" else "10-Q",
        fiscal_year=int(year),
        period_end=date(int(year), 12, 31),
        source_kind="fixture",
        source_url=f"https://example.invalid/{doc_name}.pdf",
        source_sha256="0" * 64,
        n_pages=n_pages,
        ingested_at=datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC),
    )


def make_corpus(seed: int = 0) -> tuple[list[Chunk], dict[str, str]]:
    """200 synthetic chunks over 4 documents; returns chunks and the ids of the seeded ones."""
    rng = random.Random(seed)
    chunks: list[Chunk] = []
    seeded: dict[str, str] = {}
    for doc_name in DOCS:
        for page_num in range(1, 6):
            for chunk_idx in range(10):
                words = rng.choices(_WORDS, k=40)
                text = f"{doc_name} page {page_num} chunk {chunk_idx}. " + " ".join(words) + "."
                cid = chunk_id(doc_name, page_num, chunk_idx, text)
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc_name=doc_name,
                        page_num=page_num,
                        chunk_idx=chunk_idx,
                        section="Item 7" if page_num <= 3 else "Item 8",
                        text=text,
                        n_tokens=len(text.split()),
                    )
                )
    # Seed one chunk with an exact phrase (BM25 target) and one with a paraphrase (dense target).
    exact_target = chunks[57]
    exact_text = f"Note 12. The {EXACT_PHRASE} is reviewed annually by the audit committee."
    chunks[57] = exact_target.model_copy(
        update={
            "text": exact_text,
            "chunk_id": chunk_id(
                exact_target.doc_name, exact_target.page_num, exact_target.chunk_idx, exact_text
            ),
        }
    )
    seeded["exact"] = chunks[57].chunk_id
    para_target = chunks[143]
    chunks[143] = para_target.model_copy(
        update={
            "text": PARAPHRASE_TEXT,
            "chunk_id": chunk_id(
                para_target.doc_name, para_target.page_num, para_target.chunk_idx, PARAPHRASE_TEXT
            ),
        }
    )
    seeded["paraphrase"] = chunks[143].chunk_id
    return chunks, seeded


@pytest.fixture(scope="session")
def corpus() -> tuple[list[Chunk], dict[str, str]]:
    return make_corpus()


def embed_corpus(
    chunks: list[Chunk], embedder: HashingEmbedder, seeded: dict[str, str]
) -> np.ndarray:
    """Hashing embeddings, with the paraphrase chunk's vector planted near its query vector.

    A bag-of-words hasher cannot represent a paraphrase, so the dense test plants the
    semantics: the paraphrase chunk is stored with the embedding of the *query* plus a little
    noise, which is exactly what a real sentence embedder would produce for a close paraphrase.
    """
    matrix = embedder.embed([c.text for c in chunks])
    idx = next(i for i, c in enumerate(chunks) if c.chunk_id == seeded["paraphrase"])
    rng = np.random.default_rng(0)
    planted = embedder.embed([PARAPHRASE_QUERY], kind="query")[0] + 0.05 * rng.standard_normal(
        DIM
    ).astype(np.float32)
    matrix[idx] = planted / np.linalg.norm(planted)
    return matrix


@pytest.fixture
def populated_store(
    embedder: HashingEmbedder, corpus: tuple[list[Chunk], dict[str, str]]
) -> Iterator[DuckDBStore]:
    """In-memory store with 4 documents, 20 pages, 200 chunks and a built FTS index."""
    chunks, seeded = corpus
    store = DuckDBStore(":memory:", embed_dim=DIM)
    store.init_schema(embedder.name, embedder.dim)
    for doc_name in DOCS:
        store.upsert_document(make_document(doc_name))
    store.add_pages(
        [
            Page(doc_name=doc_name, page_num=p, text=f"{doc_name} full page {p} text")
            for doc_name in DOCS
            for p in range(1, 6)
        ]
    )
    store.add_chunks(chunks, embed_corpus(chunks, embedder, seeded))
    store.rebuild_fts()
    yield store
    store.close()


@pytest.fixture
def store_factory(embedder: HashingEmbedder) -> Iterator[Callable[..., DuckDBStore]]:
    """Build initialised stores (in-memory by default) and close them at teardown."""
    opened: list[DuckDBStore] = []

    def _make(path: str = ":memory:", dim: int = DIM, name: str | None = None) -> DuckDBStore:
        store = DuckDBStore(path, embed_dim=dim)
        store.init_schema(name or embedder.name, dim)
        opened.append(store)
        return store

    yield _make
    for store in opened:
        store.close()
