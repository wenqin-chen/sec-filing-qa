"""EDGAR ticker ingest and companyfacts loading, with every HTTP call mocked by respx."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pytest
import respx

from secqa.core.errors import ConfigError
from secqa.edgar import EdgarClient, FilingRef, TickerNotFound
from secqa.edgar.models import TICKERS_URL, archive_url, companyfacts_url, submissions_url
from secqa.embeddings import HashingEmbedder
from secqa.indexing import Company, edgar_doc_name, ingest_ticker, load_xbrl_for_companies
from secqa.retrieval import Retriever
from secqa.store import DuckDBStore
from tests.indexing.conftest import FIXTURE_CIK, CountingEmbedder

ACME_CIK = "0007654321"

# Columnar submissions block exactly as EDGAR serves it (see secqa.edgar.client).
_FILINGS = [
    # accession, filingDate, reportDate, form, primaryDocument
    ("0001234567-24-000010", "2024-02-15", "2023-12-31", "10-K", "fixt-20231231.htm"),
    ("0001234567-23-000070", "2023-08-01", "2023-06-30", "10-Q", "fixt-20230630.htm"),
    ("0001234567-23-000030", "2023-03-01", "2022-12-31", "10-K/A", "fixt-20221231a.htm"),
    ("0001234567-23-000020", "2023-02-15", "2022-12-31", "10-K", "fixt-20221231.htm"),
    ("0001234567-22-000020", "2022-02-15", "2021-12-31", "10-K", "fixt-20211231.pdf"),
]


def _submissions() -> dict[str, Any]:
    cols = list(zip(*_FILINGS, strict=True))
    return {
        "cik": "1234567",
        "name": "Fixture Corp",
        "tickers": ["FIXT"],
        "filings": {
            "recent": {
                "accessionNumber": list(cols[0]),
                "filingDate": list(cols[1]),
                "reportDate": list(cols[2]),
                "form": list(cols[3]),
                "primaryDocument": list(cols[4]),
            },
            "files": [],
        },
    }


@pytest.fixture
def edgar_routes(
    respx_router: respx.MockRouter, edgar_html: bytes, companyfacts_json: dict[str, Any]
) -> dict[str, respx.Route]:
    routes = {
        "tickers": respx_router.get(TICKERS_URL).mock(
            return_value=httpx.Response(
                200, json={"0": {"cik_str": 1234567, "ticker": "FIXT", "title": "Fixture Corp"}}
            )
        ),
        "submissions": respx_router.get(submissions_url(FIXTURE_CIK)).mock(
            return_value=httpx.Response(200, json=_submissions())
        ),
        "companyfacts": respx_router.get(companyfacts_url(FIXTURE_CIK)).mock(
            return_value=httpx.Response(200, json=companyfacts_json)
        ),
        "companyfacts_acme": respx_router.get(companyfacts_url(ACME_CIK)).mock(
            return_value=httpx.Response(404)
        ),
    }
    for accession, _filed, report, _form, doc in _FILINGS:
        # Each filing gets distinct bytes so their source hashes differ.
        body = edgar_html.replace(b"December 31.", f"December 31 ({report}).".encode())
        routes[doc] = respx_router.get(archive_url(FIXTURE_CIK, accession, doc)).mock(
            return_value=httpx.Response(200, content=body, headers={"content-type": "text/html"})
        )
    return routes


# ---- edgar_doc_name -------------------------------------------------------------------------


def _ref(form: str, report: str | None, filed: str = "2024-02-15") -> FilingRef:
    return FilingRef(
        cik=FIXTURE_CIK,
        accession="0001234567-24-000010",
        form=form,
        filing_date=date.fromisoformat(filed),
        report_date=date.fromisoformat(report) if report else None,
        primary_doc="doc.htm",
        url="https://www.sec.gov/Archives/edgar/data/1234567/000123456724000010/doc.htm",
    )


@pytest.mark.parametrize(
    ("form", "report", "expected"),
    [
        ("10-K", "2023-12-31", "FIXT_2023_10-K"),
        ("10-K/A", "2023-12-31", "FIXT_2023_10-KA"),
        ("10-Q", "2023-06-30", "FIXT_2023Q2_10-Q"),
        ("10-Q", "2023-10-31", "FIXT_2023Q4_10-Q"),
        ("8-K", "2024-01-05", "FIXT_2024Q1_8-K"),
        ("10-K", None, "FIXT_2024_10-K"),  # no reportDate: filing year
    ],
)
def test_edgar_doc_name(form: str, report: str | None, expected: str) -> None:
    assert edgar_doc_name("fixt", _ref(form, report)) == expected


# ---- ingest_ticker --------------------------------------------------------------------------


def test_ingest_ticker_ingests_annual_filings(
    store: DuckDBStore,
    counting_embedder: CountingEmbedder,
    edgar_client: EdgarClient,
    edgar_routes: dict[str, respx.Route],
) -> None:
    names = ingest_ticker(
        store, counting_embedder, edgar_client, "fixt", forms=("10-K",), years=[2022, 2023]
    )
    assert names == ["FIXT_2022_10-K", "FIXT_2023_10-K"]  # filing-date order
    assert counting_embedder.calls == 2
    docs = {d.doc_name: d for d in store.list_documents(ticker="FIXT")}
    doc = docs["FIXT_2023_10-K"]
    assert (doc.company, doc.cik, doc.form, doc.fiscal_year) == (
        "Fixture Corp",
        FIXTURE_CIK,
        "10-K",
        2023,
    )
    assert doc.period_end == date(2023, 12, 31)
    assert doc.source_kind == "edgar_html"
    assert doc.source_url == archive_url(FIXTURE_CIK, "0001234567-24-000010", "fixt-20231231.htm")
    assert len(doc.source_sha256) == 64
    assert doc.n_pages >= 3  # the fixture has explicit page breaks
    assert docs["FIXT_2022_10-K"].source_sha256 != doc.source_sha256
    counts = store.counts()
    assert counts["documents"] == 2 and counts["chunks"] > 0
    assert store.manifest().n_documents == 2

    pages = store.get_pages("FIXT_2023_10-K", [1, 2, 3, 4])
    assert not any("HIDDEN_HEADER_MUST_NOT_APPEAR" in p.text for p in pages)
    assert any("Net sales | $1,577 | $1,408" in p.text for p in pages)

    retriever = Retriever(store, counting_embedder, strategy="bm25", k=2)  # type: ignore[arg-type]
    hits = retriever.retrieve("zirconium flywheel product line").hits
    assert hits and hits[0].chunk.doc_name.startswith("FIXT_")
    # No filing that was not requested was downloaded.
    assert edgar_routes["fixt-20230630.htm"].call_count == 0


def test_ingest_ticker_second_run_is_unchanged(
    store: DuckDBStore,
    counting_embedder: CountingEmbedder,
    edgar_client: EdgarClient,
    edgar_routes: dict[str, respx.Route],
) -> None:
    first = ingest_ticker(store, counting_embedder, edgar_client, "FIXT", years=[2023])
    calls = counting_embedder.calls
    second = ingest_ticker(store, counting_embedder, edgar_client, "FIXT", years=[2023])
    assert first == second == ["FIXT_2023_10-K"]
    assert counting_embedder.calls == calls
    assert store.counts()["documents"] == 1
    # The document bytes came from the disk cache: one archive request in total.
    assert edgar_routes["fixt-20231231.htm"].call_count == 1

    forced = ingest_ticker(
        store, counting_embedder, edgar_client, "FIXT", years=[2023], skip_unchanged=False
    )
    assert forced == ["FIXT_2023_10-K"] and counting_embedder.calls == calls + 1


def test_ingest_ticker_quarterly_and_amendment_names(
    store: DuckDBStore,
    embedder: HashingEmbedder,
    edgar_client: EdgarClient,
    edgar_routes: dict[str, respx.Route],
) -> None:
    assert ingest_ticker(store, embedder, edgar_client, "FIXT", forms=("10-Q",), years=[2023]) == [
        "FIXT_2023Q2_10-Q"
    ]
    assert ingest_ticker(
        store, embedder, edgar_client, "FIXT", forms=("10-K/A",), years=[2022]
    ) == ["FIXT_2022_10-KA"]
    forms = {d.doc_name: d.form for d in store.list_documents()}
    assert forms == {"FIXT_2023Q2_10-Q": "10-Q", "FIXT_2022_10-KA": "10-K/A"}


def test_ingest_ticker_skips_non_html_primary_document(
    store: DuckDBStore,
    embedder: HashingEmbedder,
    edgar_client: EdgarClient,
    edgar_routes: dict[str, respx.Route],
) -> None:
    assert ingest_ticker(store, embedder, edgar_client, "FIXT", years=[2021]) == []
    assert store.counts()["documents"] == 0
    assert edgar_routes["fixt-20211231.pdf"].call_count == 0


def test_ingest_ticker_unknown_ticker_raises(
    store: DuckDBStore,
    embedder: HashingEmbedder,
    edgar_client: EdgarClient,
    edgar_routes: dict[str, respx.Route],
) -> None:
    with pytest.raises(TickerNotFound):
        ingest_ticker(store, embedder, edgar_client, "NOPE")


def test_ingest_ticker_refuses_read_only_store(
    file_store_factory: Any, embedder: HashingEmbedder, edgar_client: EdgarClient, tmp_path: Any
) -> None:
    file_store_factory("ro.duckdb").close()
    with DuckDBStore(tmp_path / "ro.duckdb", read_only=True) as ro:
        with pytest.raises(ConfigError, match="read-only"):
            ingest_ticker(ro, embedder, edgar_client, "FIXT")


# ---- load_xbrl_for_companies ----------------------------------------------------------------


def test_load_xbrl_for_companies_loads_facts_and_view(
    store: DuckDBStore,
    edgar_client: EdgarClient,
    edgar_routes: dict[str, respx.Route],
    companies: list[Company],
) -> None:
    n = load_xbrl_for_companies(store, edgar_client, companies)
    assert n == 5  # 3 Revenues + 2 Assets entries in the fixture
    assert store.counts()["facts"] == 5
    assert store.manifest().n_facts == 5
    rows = store.conn.execute(
        "SELECT fiscal_year, revenue, total_assets FROM financials "
        "WHERE ticker = 'FIXT' ORDER BY fiscal_year"
    ).fetchall()
    assert rows == [(2022, 1408000000.0, 3100000000.0), (2023, 1577000000.0, 3350000000.0)]
    # ACME returned 404 and was skipped, not fatal.
    assert edgar_routes["companyfacts_acme"].call_count == 1
    # Re-running replaces rather than duplicates.
    assert load_xbrl_for_companies(store, edgar_client, companies) == 5
    assert store.counts()["facts"] == 5


def test_load_xbrl_for_companies_all_failures_is_config_error(
    store: DuckDBStore,
    edgar_client: EdgarClient,
    edgar_routes: dict[str, respx.Route],
    companies: list[Company],
) -> None:
    acme_only = [c for c in companies if c.ticker == "ACME"]
    with pytest.raises(ConfigError, match="could not be fetched for any"):
        load_xbrl_for_companies(store, edgar_client, acme_only)
    with pytest.raises(ValueError, match="must not be empty"):
        load_xbrl_for_companies(store, edgar_client, [])
