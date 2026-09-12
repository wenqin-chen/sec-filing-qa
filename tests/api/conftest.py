"""Fixtures for the API tests.

A temporary DuckDB *file* is seeded with 2 synthetic documents (10 pages / 10 chunks each, 20
chunks total) plus the synthetic XBRL fixture, then closed; the app opens it read-only through
the real startup path (``create_app`` + lifespan). Providers are the deterministic
``MockProvider`` (server default) and, where a bill is needed, a scripted provider that
identifies as ``openai:gpt-test`` and is priced by an injected table. Everything is offline.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from secqa.api import AppState, create_app
from secqa.core.contracts import (
    Chunk,
    DocumentMeta,
    LLMResponse,
    Message,
    Page,
    ToolCall,
    ToolSpec,
    Usage,
)
from secqa.core.errors import ProviderError
from secqa.core.ids import chunk_id
from secqa.core.settings import Settings
from secqa.embeddings import HashingEmbedder
from secqa.providers.base import BaseProvider, Effort
from secqa.providers.pricing import PriceTable
from secqa.providers.scripted_provider import ScriptedProvider
from secqa.store import DuckDBStore
from secqa.xbrl import create_financials_view, load_companyfacts

DIM = 64
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
XBRL_FIXTURE = FIXTURES / "xbrl_companyfacts_small.json"

TICKER = "FIXT"
CIK = "0001234567"
TOP_DOC = "FIXTURE_2023_10K"
OTHER_DOC = "OTHER_2022_10K"
DOCS: dict[str, tuple[str, int, str]] = {  # doc_name -> (ticker, fiscal_year, form)
    TOP_DOC: (TICKER, 2023, "10-K"),
    OTHER_DOC: ("OTHR", 2022, "10-K"),
}
NET_SALES_QUESTION = "What were total net sales in fiscal 2023?"
NET_SALES_SENTENCE = (
    "Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022."
)

# Ten pages per document, one chunk per page: 20 chunks. Synthetic prose, no dataset text.
PAGE_TEXTS: dict[str, list[tuple[str, str]]] = {
    TOP_DOC: [
        ("Item 7. Management Discussion", NET_SALES_SENTENCE + " Growth was broad-based."),
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
        ("Item 1. Business", "The company designs industrial sensors sold through distributors."),
        ("Item 1A. Risk Factors", "Supply chain disruption for rare earth magnets remains a risk."),
        ("Item 2. Properties", "Manufacturing sites are located in Ohio and Bavaria."),
        ("Item 3. Legal Proceedings", "No material litigation was pending at fiscal year end."),
        (
            "Note 4. Inventories",
            "Inventories were $320 million, valued at lower of cost or market.",
        ),
        ("Note 6. Goodwill", "Goodwill of $510 million was tested with no impairment recorded."),
        ("Note 9. Income Taxes", "The effective tax rate was 21% compared with 23% a year ago."),
    ],
    OTHER_DOC: [
        ("Item 7. Overview", "Subscription revenue grew while hardware revenue declined in 2022."),
        ("Item 8. Statements", "Goodwill impairment testing found no impairment in any segment."),
        ("Item 9A. Controls", "The audit committee reviewed the internal control framework."),
        ("Item 1. Business", "The company operates a cloud analytics platform for retailers."),
        ("Item 1A. Risks", "Customer concentration: three customers made up 40% of billings."),
        ("Item 5. Market", "No dividends were declared during fiscal 2022."),
        ("Note 2. Revenue", "Deferred revenue was $88 million at the end of fiscal 2022."),
        ("Note 5. Leases", "Operating lease liabilities totalled $61 million."),
        ("Note 8. Debt", "The revolving credit facility of $150 million was undrawn."),
        ("Note 11. Equity", "Share repurchases amounted to $25 million in fiscal 2022."),
    ],
}

PRICES: dict[str, Any] = {
    "as_of": "2026-09-11",
    "models": {
        "openai": {"gpt-test": {"input": 4.0, "cached_input": 0.4, "output": 16.0}},
    },
}
PAID_SPEC = "openai:gpt-test"


def make_document(doc_name: str) -> DocumentMeta:
    ticker, year, form = DOCS[doc_name]
    return DocumentMeta(
        doc_name=doc_name,
        company=f"{ticker.title()} Corp",
        ticker=ticker,
        cik=CIK if ticker == TICKER else "0000000042",
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


def net_sales_chunk_id() -> str:
    return make_chunks(TOP_DOC)[0].chunk_id


# ---- providers -----------------------------------------------------------------------------


class PricedScripted(ScriptedProvider):
    """A scripted provider that bills as ``openai:gpt-test`` so budget rules can be exercised."""

    provider = "openai"

    def __init__(self, scenario: list[dict[str, Any]]) -> None:
        super().__init__(scenario)
        self.model = "gpt-test"


class FailingProvider(BaseProvider):
    """Raises ``ProviderError`` on every call (the 502 path)."""

    provider = "openai"
    model = "gpt-test"

    def __init__(self, *, retryable: bool = True) -> None:
        self.retryable = retryable

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
        raise ProviderError(
            "simulated upstream failure", retryable=self.retryable, provider="openai"
        )


class SlowAbstainingProvider(BaseProvider):
    """Sleeps ``delay_s`` per call; searches once, then abstains (drives the agent loop past
    its wall clock: the tool call keeps the loop alive so the clock is checked before turn 2)."""

    provider = "openai"
    model = "gpt-test"

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.calls = 0

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
        self.calls += 1
        time.sleep(self.delay_s)
        if self.calls == 1 and tools:
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(id="slow-1", name="search_filings", arguments={"query": "x"})],
                usage=Usage(input_tokens=10, output_tokens=5),
                provider=self.provider,
                model=self.model,
                latency_ms=self.delay_s * 1000.0,
                stop_reason="tool_use",
            )
        text = abstain_json()
        return LLMResponse(
            text=text,
            tool_calls=[],
            usage=Usage(input_tokens=10, output_tokens=5),
            provider=self.provider,
            model=self.model,
            latency_ms=self.delay_s * 1000.0,
            stop_reason="end_turn",
            parsed=json.loads(text) if json_schema is not None else None,
        )


def abstain_json() -> str:
    return json.dumps(
        {
            "answer": "INSUFFICIENT EVIDENCE",
            "value": None,
            "unit": None,
            "citations": [],
            "calculation": None,
            "abstain": True,
        }
    )


def search_turn(query: str, usage: dict[str, int] | None = None) -> dict[str, Any]:
    turn: dict[str, Any] = {
        "tool_calls": [
            {
                "name": "search_filings",
                "arguments": {
                    "query": query,
                    "ticker": None,
                    "fiscal_year": None,
                    "form": None,
                    "k": 3,
                },
            }
        ]
    }
    if usage:
        turn["usage"] = usage
    return turn


def rag_answer_turn(usage: dict[str, int]) -> dict[str, Any]:
    """A single-shot (json_schema) turn quoting the net-sales sentence with its real ref."""
    parsed = {
        "answer": NET_SALES_SENTENCE,
        "value": 1_577_000_000,
        "unit": "USD",
        "citations": [{"ref": f"chunk:{net_sales_chunk_id()}", "quote": NET_SALES_SENTENCE}],
        "abstain": False,
    }
    return {"text": json.dumps(parsed), "parsed": parsed, "usage": usage}


# ---- index and app fixtures ----------------------------------------------------------------


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=DIM)


@pytest.fixture(scope="session")
def companyfacts_doc() -> dict[str, Any]:
    return json.loads(XBRL_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def index_path(tmp_path: Path, embedder: HashingEmbedder, companyfacts_doc: dict[str, Any]) -> Path:
    """Build and close a DuckDB index file: 2 documents, 20 pages, 20 chunks, XBRL facts."""
    path = tmp_path / "index.duckdb"
    corpus = [chunk for doc_name in DOCS for chunk in make_chunks(doc_name)]
    with DuckDBStore(path, embed_dim=DIM) as db:
        db.init_schema(embedder.name, embedder.dim)
        for doc_name in DOCS:
            db.upsert_document(make_document(doc_name))
            db.add_pages(make_pages(doc_name))
        db.add_chunks(corpus, embedder.embed([c.text for c in corpus]))
        db.rebuild_fts()
        load_companyfacts(db, copy.deepcopy(companyfacts_doc), TICKER)
        create_financials_view(db)
        db.set_manifest(git_sha="f" * 40, inputs_sha256="0" * 64)
    return path


@pytest.fixture
def make_settings(
    settings_override: Callable[..., Settings], index_path: Path
) -> Callable[..., Settings]:
    """``make_settings(**overrides)`` -> Settings pointing at the seeded index, offline defaults."""

    def _make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "duckdb_path": index_path,
            "embedder": f"hashing:{DIM}",
            "provider": "mock",
            "rate_limit_per_min": 1000,
            "log_json": False,
        }
        values.update(overrides)
        return settings_override(**values)

    return _make


@pytest.fixture
def api_settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def app(api_settings: Settings) -> FastAPI:
    return create_app(api_settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """A client whose app has gone through startup (index loaded) and will be shut down."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def make_client() -> Callable[[FastAPI], TestClient]:
    """``make_client(app)`` -> a started client; closed when the test ends."""
    started: list[TestClient] = []

    def _make(application: FastAPI) -> TestClient:
        test_client = TestClient(application, raise_server_exceptions=False)
        test_client.__enter__()
        started.append(test_client)
        return test_client

    yield _make  # type: ignore[misc]  # generator fixture returning a factory
    for test_client in started:
        test_client.__exit__(None, None, None)


def state_of(app: FastAPI) -> AppState:
    state = app.state.secqa
    assert isinstance(state, AppState)
    return state


def install_paid_provider(app: FastAPI, provider: Any, spec: str = PAID_SPEC) -> None:
    """Register a priced provider under ``spec`` and make the price table know its model."""
    state = state_of(app)
    state.providers[spec] = provider
    state.prices = PriceTable.from_dict(PRICES, source="tests/api/conftest.py")
