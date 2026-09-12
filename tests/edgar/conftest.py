"""Fixtures for the edgar module tests: a fake clock, fixture JSON loaders, respx routes.

No test here touches the network: ``respx`` mocks every HTTP call and the client runs its rate
limiter and retry backoff against :class:`FakeClock`, so retries that would wait up to 30 s in
production complete instantly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from secqa.edgar.client import EdgarClient
from secqa.edgar.models import (
    TICKERS_URL,
    companyfacts_url,
    submissions_page_url,
    submissions_url,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE_CIK = "0001234567"
TEST_UA = "Test Runner test@example.com"


class FakeClock:
    """Monotonic clock whose ``sleep`` advances time instead of blocking."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise AssertionError(f"negative sleep {seconds}")
        self.sleeps.append(seconds)
        self.t += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


def load_fixture(name: str) -> dict[str, Any]:
    """Read ``tests/fixtures/<name>`` as JSON."""
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


@pytest.fixture
def tickers_json() -> dict[str, Any]:
    return load_fixture("edgar_company_tickers.json")


@pytest.fixture
def submissions_json() -> dict[str, Any]:
    return load_fixture("edgar_submissions_small.json")


@pytest.fixture
def submissions_page1_json() -> dict[str, Any]:
    return load_fixture("edgar_submissions_small_page1.json")


@pytest.fixture
def companyfacts_json() -> dict[str, Any]:
    return load_fixture("edgar_companyfacts_small.json")


@pytest.fixture
def client_factory(tmp_path: Path, fake_clock: FakeClock) -> Callable[..., EdgarClient]:
    """Build an ``EdgarClient`` with a per-test cache dir and the fake clock/sleep injected."""
    clients: list[EdgarClient] = []

    def _make(**overrides: Any) -> EdgarClient:
        kwargs: dict[str, Any] = {
            "user_agent": TEST_UA,
            "cache_dir": tmp_path / "cache",
            "sleep": fake_clock.sleep,
            "clock": fake_clock.now,
        }
        kwargs.update(overrides)
        client = EdgarClient(**kwargs)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.close()


@pytest.fixture
def client(client_factory: Callable[..., EdgarClient]) -> EdgarClient:
    return client_factory()


@pytest.fixture
def edgar_routes(
    respx_router: respx.MockRouter,
    tickers_json: dict[str, Any],
    submissions_json: dict[str, Any],
    submissions_page1_json: dict[str, Any],
    companyfacts_json: dict[str, Any],
) -> dict[str, respx.Route]:
    """Mock the four EDGAR endpoints used by the tests; returns the routes by name."""
    return {
        "tickers": respx_router.get(TICKERS_URL).mock(
            return_value=httpx.Response(200, json=tickers_json)
        ),
        "submissions": respx_router.get(submissions_url(FIXTURE_CIK)).mock(
            return_value=httpx.Response(200, json=submissions_json)
        ),
        "submissions_page1": respx_router.get(
            submissions_page_url("CIK0001234567-submissions-001.json")
        ).mock(return_value=httpx.Response(200, json=submissions_page1_json)),
        "companyfacts": respx_router.get(companyfacts_url(FIXTURE_CIK)).mock(
            return_value=httpx.Response(200, json=companyfacts_json)
        ),
    }
