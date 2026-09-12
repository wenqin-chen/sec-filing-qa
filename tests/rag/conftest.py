"""Fixtures for the rag tests: a small synthetic corpus in an in-memory store, a hybrid
retriever over the key-free HashingEmbedder, the deterministic MockProvider, and a recording
fake provider for the cases where the mock's behaviour is not what the test needs.

Everything is offline and deterministic; no dataset rows, no third-party text.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any

import pytest

from secqa.core.contracts import (
    Chunk,
    DocumentMeta,
    LLMResponse,
    Message,
    Page,
    StopReason,
    ToolSpec,
    Usage,
)
from secqa.core.ids import chunk_id
from secqa.embeddings import HashingEmbedder
from secqa.grounding import CitationVerifier
from secqa.providers.base import BaseProvider, Effort
from secqa.providers.mock_provider import MockProvider
from secqa.providers.pricing import PriceTable
from secqa.rag import RagPipeline
from secqa.retrieval import Retriever
from secqa.store import DuckDBStore

DIM = 64

# doc_name -> (ticker, fiscal_year, form)
DOCS: dict[str, tuple[str, int, str]] = {
    "FIXTURE_2023_10K": ("FIX", 2023, "10-K"),
    "OTHER_2022_10K": ("OTH", 2022, "10-K"),
}

# One chunk per page: (section, text). The page-1 text starts with the sentence that carries the
# numbers the "net sales" question asks for, so the extractive mock quotes exactly that sentence.
PAGE_TEXTS: dict[str, list[tuple[str, str]]] = {
    "FIXTURE_2023_10K": [
        (
            "Item 7. Management Discussion",
            "Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022. "
            "Growth was broad-based across segments.",
        ),
        (
            "Item 8. Financial Statements",
            "Operating income was $245 million and net income was $190 million. "
            "Capital expenditures totalled $80 million.",
        ),
        (
            "Item 9A. Controls and Procedures",
            "Cash and cash equivalents were $410 million at year end. "
            "Long-term debt was $1,200 million.",
        ),
    ],
    "OTHER_2022_10K": [
        ("Item 7. Overview", "Subscription revenue grew while hardware revenue declined in 2022."),
        ("Item 8. Statements", "Goodwill impairment testing found no impairment in any segment."),
        ("Item 9A. Controls", "The audit committee reviewed the internal control framework."),
    ],
}
TOP_DOC = "FIXTURE_2023_10K"
NET_SALES_QUESTION = "What were total net sales in fiscal 2023?"
NET_SALES_SENTENCE = (
    "Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022."
)

PRICES: dict[str, Any] = {
    "as_of": "2026-09-11",
    "models": {"openai": {"gpt-test": {"input": 4.0, "cached_input": 0.4, "output": 16.0}}},
}


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
        n_pages=len(PAGE_TEXTS[doc_name]),
        ingested_at=datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC),
    )


def make_pages(doc_name: str) -> list[Page]:
    return [
        Page(doc_name=doc_name, page_num=i, text=text)
        for i, (_section, text) in enumerate(PAGE_TEXTS[doc_name], start=1)
    ]


def make_chunks(doc_name: str) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=chunk_id(doc_name, i, 0, text),
            doc_name=doc_name,
            page_num=i,
            chunk_idx=0,
            section=section,
            text=text,
            n_tokens=len(text.split()),
        )
        for i, (section, text) in enumerate(PAGE_TEXTS[doc_name], start=1)
    ]


class FixedProvider(BaseProvider):
    """Return a fixed response and record every call (for cases the MockProvider can't script)."""

    def __init__(
        self,
        *,
        text: str = "",
        parsed: dict[str, Any] | None = None,
        stop_reason: StopReason = "end_turn",
        usage: Usage | None = None,
        provider: str = "openai",
        model: str = "gpt-test",
        latency_ms: float = 0.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self._text = text
        self._parsed = parsed
        self._stop_reason: StopReason = stop_reason
        self._usage = usage or Usage(input_tokens=1000, output_tokens=50)
        self._latency_ms = latency_ms
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        effort: Effort | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "system": system,
                "tools": tools,
                "json_schema": json_schema,
                "max_tokens": max_tokens,
                "effort": effort,
            }
        )
        response = LLMResponse(
            text=self._text,
            tool_calls=[],
            usage=self._usage,
            provider=self.provider,
            model=self.model,
            latency_ms=self._latency_ms,
            stop_reason=self._stop_reason,
            parsed=self._parsed,
        )
        self._log_response(response)
        return response


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=DIM)


@pytest.fixture(scope="session")
def corpus() -> list[Chunk]:
    return [chunk for doc_name in DOCS for chunk in make_chunks(doc_name)]


@pytest.fixture
def store(embedder: HashingEmbedder, corpus: list[Chunk]) -> Iterator[DuckDBStore]:
    """In-memory store with 2 documents, 6 pages, 6 chunks and a built FTS index."""
    db = DuckDBStore(":memory:", embed_dim=DIM)
    db.init_schema(embedder.name, embedder.dim)
    for doc_name in DOCS:
        db.upsert_document(make_document(doc_name))
        db.add_pages(make_pages(doc_name))
    db.add_chunks(corpus, embedder.embed([c.text for c in corpus]))
    db.rebuild_fts()
    yield db
    db.close()


@pytest.fixture
def retriever(store: DuckDBStore, embedder: HashingEmbedder) -> Retriever:
    return Retriever(store, embedder, strategy="hybrid", k=4)


@pytest.fixture
def verifier() -> CitationVerifier:
    return CitationVerifier()


@pytest.fixture
def prices() -> PriceTable:
    return PriceTable.from_dict(PRICES, source="tests/rag/conftest.py")


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def pipeline(
    retriever: Retriever,
    mock_provider: MockProvider,
    verifier: CitationVerifier,
    prices: PriceTable,
) -> RagPipeline:
    return RagPipeline(retriever, mock_provider, verifier, prices, k=4)
