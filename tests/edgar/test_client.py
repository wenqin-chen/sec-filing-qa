"""Tests for secqa.edgar.client.EdgarClient (all HTTP mocked with respx; one live test)."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from secqa.core.errors import ConfigError
from secqa.core.settings import Settings
from secqa.edgar.client import RETRY_ATTEMPTS, EdgarClient, EdgarError, TickerNotFound
from secqa.edgar.models import (
    TICKERS_URL,
    archive_url,
    companyfacts_url,
    submissions_url,
)
from tests.edgar.conftest import FIXTURE_CIK, TEST_UA, FakeClock

# Captured at import time: the autouse env scrubber in tests/conftest.py removes SEC_USER_AGENT
# before any test body runs, so the live test reads it here instead.
_LIVE_UA = os.environ.get("SEC_USER_AGENT")


# ---- construction ----


def test_user_agent_without_email_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="contact email"):
        EdgarClient("just-a-name", cache_dir=tmp_path)


def test_invalid_rate_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        EdgarClient(TEST_UA, max_rps=0, cache_dir=tmp_path)


def test_from_settings_requires_sec_user_agent(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="SEC_USER_AGENT"):
        EdgarClient.from_settings(Settings(_env_file=None), cache_dir=tmp_path)
    settings = Settings(_env_file=None, sec_user_agent="Jane Doe jane@example.com")
    with EdgarClient.from_settings(settings, cache_dir=tmp_path) as client:
        assert client.user_agent == "Jane Doe jane@example.com"
        assert client.max_rps == 8.0


# ---- headers and rate limiting ----


def test_every_request_carries_user_agent_and_gzip(
    client: EdgarClient, edgar_routes: dict[str, respx.Route], respx_router: respx.MockRouter
) -> None:
    client.cik_for_ticker("FIXT")
    client.submissions(FIXTURE_CIK)
    client.companyfacts(FIXTURE_CIK)
    assert respx_router.calls.call_count == 3
    for call in respx_router.calls:
        assert call.request.headers["user-agent"] == TEST_UA
        assert "gzip" in call.request.headers["accept-encoding"]


def test_requests_pass_through_the_token_bucket(
    client_factory: Callable[..., EdgarClient],
    respx_router: respx.MockRouter,
    fake_clock: FakeClock,
) -> None:
    client = client_factory(max_rps=4.0)
    for i in range(6):
        respx_router.get(f"https://data.sec.gov/fixture/{i}").mock(
            return_value=httpx.Response(200, content=b"x")
        )
    t0 = fake_clock.now()
    for i in range(6):
        client.get_bytes(f"https://data.sec.gov/fixture/{i}")
    # 6 requests at 4 req/s with burst 1: the first is free, the next five wait 0.25 s each.
    assert fake_clock.now() - t0 == pytest.approx(5 * 0.25)


# ---- retries ----


@pytest.mark.parametrize("status", [429, 503])
def test_retryable_status_then_200_is_retried(
    client: EdgarClient, respx_router: respx.MockRouter, fake_clock: FakeClock, status: int
) -> None:
    url = "https://data.sec.gov/fixture/retry.json"
    route = respx_router.get(url).mock(
        side_effect=[httpx.Response(status), httpx.Response(200, json={"ok": True})]
    )
    assert client.get_json(url) == {"ok": True}
    assert route.call_count == 2
    # Backoff sleeps are the ones >= 1 s (bucket sleeps are 1/8 s); exactly one retry happened.
    backoff = [s for s in fake_clock.sleeps if s >= 1.0]
    assert len(backoff) == 1
    assert 1.0 <= backoff[0] <= 30.0


def test_network_error_then_200_is_retried(
    client: EdgarClient, respx_router: respx.MockRouter
) -> None:
    url = "https://data.sec.gov/fixture/net.json"
    route = respx_router.get(url).mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json={"ok": 1})]
    )
    assert client.get_json(url) == {"ok": 1}
    assert route.call_count == 2


def test_gives_up_after_five_attempts(
    client: EdgarClient, respx_router: respx.MockRouter, fake_clock: FakeClock
) -> None:
    url = "https://data.sec.gov/fixture/always429.json"
    route = respx_router.get(url).mock(return_value=httpx.Response(429))
    with pytest.raises(EdgarError) as excinfo:
        client.get_bytes(url)
    assert route.call_count == RETRY_ATTEMPTS == 5
    assert excinfo.value.status_code == 429
    assert excinfo.value.retryable is True
    backoff = [s for s in fake_clock.sleeps if s >= 1.0]
    assert len(backoff) == RETRY_ATTEMPTS - 1
    assert all(1.0 <= s <= 30.0 for s in backoff)
    assert not client.cache.has(url), "error responses are never cached"


def test_persistent_network_error_is_wrapped(
    client: EdgarClient, respx_router: respx.MockRouter
) -> None:
    url = "https://data.sec.gov/fixture/down.json"
    route = respx_router.get(url).mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(EdgarError, match="network error") as excinfo:
        client.get_bytes(url)
    assert route.call_count == RETRY_ATTEMPTS
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)


@pytest.mark.parametrize("status", [400, 403, 404, 500])
def test_non_retryable_status_fails_immediately(
    client: EdgarClient, respx_router: respx.MockRouter, fake_clock: FakeClock, status: int
) -> None:
    url = f"https://data.sec.gov/fixture/{status}.json"
    route = respx_router.get(url).mock(return_value=httpx.Response(status))
    with pytest.raises(EdgarError) as excinfo:
        client.get_bytes(url)
    assert route.call_count == 1
    assert excinfo.value.status_code == status
    assert excinfo.value.retryable is False
    assert [s for s in fake_clock.sleeps if s >= 1.0] == []


# ---- cache ----


def test_cache_hit_makes_no_network_call(
    client_factory: Callable[..., EdgarClient],
    respx_router: respx.MockRouter,
    fake_clock: FakeClock,
    tmp_path: Path,
) -> None:
    url = "https://data.sec.gov/fixture/cached.json"
    route = respx_router.get(url).mock(return_value=httpx.Response(200, json={"v": 1}))
    client = client_factory()
    assert client.get_json(url) == {"v": 1}
    assert route.call_count == 1
    assert client.cache.has(url)

    sleeps_before = list(fake_clock.sleeps)
    assert client.get_json(url) == {"v": 1}
    assert client.get_bytes(url) == b'{"v":1}'
    assert route.call_count == 1
    assert fake_clock.sleeps == sleeps_before, "cache hits do not consume rate-limit tokens"

    # A fresh client over the same cache directory is served from disk too.
    other = client_factory(cache_dir=tmp_path / "cache")
    assert other.get_json(url) == {"v": 1}
    assert route.call_count == 1


def test_corrupt_cache_entry_is_evicted_and_reported(
    client: EdgarClient, respx_router: respx.MockRouter
) -> None:
    url = "https://data.sec.gov/fixture/corrupt.json"
    respx_router.get(url).mock(return_value=httpx.Response(200, content=b"not json"))
    with pytest.raises(EdgarError, match="not valid JSON"):
        client.get_json(url)
    assert not client.cache.has(url)


def test_get_json_requires_an_object(client: EdgarClient, respx_router: respx.MockRouter) -> None:
    url = "https://data.sec.gov/fixture/list.json"
    respx_router.get(url).mock(return_value=httpx.Response(200, json=[1, 2]))
    with pytest.raises(EdgarError, match="not a JSON object"):
        client.get_json(url)


# ---- endpoints ----


def test_cik_for_ticker_pads_to_ten_digits(
    client: EdgarClient, edgar_routes: dict[str, respx.Route]
) -> None:
    assert client.cik_for_ticker("fixt") == "0001234567"
    assert client.cik_for_ticker("AAPL") == "0000320193"
    assert edgar_routes["tickers"].call_count == 1, "ticker table is cached after one fetch"


def test_unknown_ticker_raises_lookup_error(
    client: EdgarClient, edgar_routes: dict[str, respx.Route]
) -> None:
    with pytest.raises(TickerNotFound) as excinfo:
        client.cik_for_ticker("NOPE")
    assert isinstance(excinfo.value, LookupError)
    assert excinfo.value.url == TICKERS_URL
    with pytest.raises(ValueError):
        client.cik_for_ticker("  ")


def test_submissions_and_companyfacts_normalise_cik(
    client: EdgarClient, edgar_routes: dict[str, respx.Route]
) -> None:
    assert client.submissions("1234567")["name"] == "Fixture Corp"
    assert client.companyfacts("1234567")["entityName"] == "Fixture Corp"
    assert edgar_routes["submissions"].calls[0].request.url == submissions_url(FIXTURE_CIK)
    assert edgar_routes["companyfacts"].calls[0].request.url == companyfacts_url(FIXTURE_CIK)


def test_list_filings_default_forms(
    client: EdgarClient, edgar_routes: dict[str, respx.Route]
) -> None:
    refs = client.list_filings(FIXTURE_CIK)
    # 10-K and 10-Q only: the 8-K and the 10-K/A are excluded, the 10-Q with no primary
    # document is skipped, and the older page (FY2019/FY2020 10-Ks) is fetched when years=None.
    assert [(r.form, r.accession) for r in refs] == [
        ("10-K", "0001234567-24-000010"),
        ("10-Q", "0001234567-23-000090"),
        ("10-K", "0001234567-23-000020"),
        ("10-K", "0001234567-21-000010"),
        ("10-K", "0001234567-20-000010"),
    ]
    assert edgar_routes["submissions_page1"].call_count == 1
    first = refs[0]
    assert first.cik == FIXTURE_CIK
    assert first.filing_date == date(2024, 2, 15)
    assert first.report_date == date(2023, 12, 31)
    assert first.primary_doc == "fixt-20231231.htm"
    assert first.url == archive_url(FIXTURE_CIK, first.accession, first.primary_doc)
    assert first.url == (
        "https://www.sec.gov/Archives/edgar/data/1234567/000123456724000010/fixt-20231231.htm"
    )


def test_list_filings_filters_forms_and_years(
    client: EdgarClient, edgar_routes: dict[str, respx.Route]
) -> None:
    # years match the report period, not the filing date: FY2022 10-K was filed in 2023.
    fy2022 = client.list_filings(FIXTURE_CIK, forms=("10-K",), years=[2022])
    assert [r.accession for r in fy2022] == ["0001234567-23-000020"]
    assert edgar_routes["submissions_page1"].call_count == 0, "older page not needed for 2022"

    amendments = client.list_filings(FIXTURE_CIK, forms=("10-K/A",))
    assert [r.accession for r in amendments] == ["0001234567-23-000030"]

    eightk = client.list_filings(FIXTURE_CIK, forms=("8-k",), years=(2023,))  # case-insensitive
    assert [r.accession for r in eightk] == ["0001234567-23-000080"]

    assert client.list_filings(FIXTURE_CIK, years=[1999]) == []


def test_list_filings_fetches_older_page_only_when_years_need_it(
    client: EdgarClient, edgar_routes: dict[str, respx.Route]
) -> None:
    fy2020 = client.list_filings(FIXTURE_CIK, forms=("10-K",), years=[2020])
    assert [r.accession for r in fy2020] == ["0001234567-21-000010"]
    assert edgar_routes["submissions_page1"].call_count == 1


def test_list_filings_rejects_ragged_columns(
    client: EdgarClient, respx_router: respx.MockRouter
) -> None:
    respx_router.get(submissions_url(FIXTURE_CIK)).mock(
        return_value=httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "accessionNumber": ["0001234567-24-000010"],
                        "filingDate": ["2024-02-15"],
                        "reportDate": [],
                        "form": ["10-K"],
                        "primaryDocument": ["a.htm"],
                    }
                }
            },
        )
    )
    with pytest.raises(EdgarError, match="ragged"):
        client.list_filings(FIXTURE_CIK)


def test_fetch_primary_document_downloads_and_caches(
    client: EdgarClient, edgar_routes: dict[str, respx.Route], respx_router: respx.MockRouter
) -> None:
    ref = client.list_filings(FIXTURE_CIK, forms=("10-K",), years=[2023])[0]
    html = b"<html><body>Fixture Corp 10-K FY2023</body></html>"
    route = respx_router.get(ref.url).mock(
        return_value=httpx.Response(200, content=html, headers={"content-type": "text/html"})
    )
    assert client.fetch_primary_document(ref) == html
    assert client.fetch_primary_document(ref) == html
    assert route.call_count == 1
    meta = client.cache.get_meta(ref.url)
    assert meta is not None and meta.content_type == "text/html"


# ---- live ----


@pytest.mark.live
@pytest.mark.skipif(not _LIVE_UA, reason="SEC_USER_AGENT not set")
def test_live_ticker_lookup(tmp_path: Path) -> None:
    """One real request against sec.gov; opt in with ``-m live`` and SEC_USER_AGENT set."""
    assert _LIVE_UA is not None
    with EdgarClient(_LIVE_UA, cache_dir=tmp_path) as client:
        assert client.cik_for_ticker("AAPL") == "0000320193"
