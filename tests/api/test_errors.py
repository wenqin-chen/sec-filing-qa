"""Unit tests for the exception -> problem+json mapping on a bare FastAPI app."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from secqa.api.errors import (
    PROBLEM_MEDIA_TYPE,
    BudgetExceeded,
    NotReady,
    install_error_handlers,
    problem,
    problem_type,
)
from secqa.api.middleware import RequestContextMiddleware
from secqa.core.errors import ConfigError, IndexMismatch, ProviderError, SqlRejected
from secqa.xbrl import SqlExecutionError, SqlTimeout, SqlToolUnavailable


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)
    raisers = {
        "sql-timeout": SqlTimeout("query exceeded 5 s and was interrupted"),
        "sql-rejected": SqlRejected("DROP is not allowed"),
        "sql-execution": SqlExecutionError("query failed: Binder Error"),
        "sql-unavailable": SqlToolUnavailable("SQL tool is busy: 2 earlier queries ..."),
        "provider": ProviderError("rate limited", retryable=True, provider="anthropic"),
        "provider-permanent": ProviderError("bad key", retryable=False, provider="openai"),
        "config": ConfigError("provider needs a key"),
        "index": IndexMismatch("dim 64 vs 384"),
        "budget": BudgetExceeded("too expensive", extra={"answer": {"text": "partial"}}),
        "not-ready": NotReady("loading"),
        "boom": RuntimeError("secret internal detail"),
    }

    @app.get("/raise/{name}")
    def raise_named(name: str) -> None:
        raise raisers[name]

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("name", "status", "title"),
    [
        ("sql-timeout", 504, "SQL timeout"),
        ("sql-rejected", 400, "SQL rejected"),
        ("sql-execution", 400, "SQL rejected"),
        ("sql-unavailable", 503, "SQL tool unavailable"),
        ("provider", 502, "Upstream provider error"),
        ("provider-permanent", 502, "Upstream provider error"),
        ("config", 503, "Provider not configured"),
        ("index", 503, "Index mismatch"),
        ("budget", 402, "Cost budget exceeded"),
        ("not-ready", 503, "Service not ready"),
        ("boom", 500, "Internal server error"),
    ],
)
def test_mapping(client: TestClient, name: str, status: int, title: str) -> None:
    response = client.get(f"/raise/{name}", headers={"X-Request-ID": f"rid-{name}"})
    assert response.status_code == status
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    body = response.json()
    assert body["title"] == title and body["status"] == status
    assert body["type"] == problem_type(title)
    assert body["request_id"] == f"rid-{name}"
    assert response.headers["X-Request-ID"] == f"rid-{name}"


def test_sql_timeout_wins_over_its_parent(client: TestClient) -> None:
    assert client.get("/raise/sql-timeout").status_code == 504
    assert client.get("/raise/sql-rejected").status_code == 400


def test_sql_unavailable_is_retryable(client: TestClient) -> None:
    response = client.get("/raise/sql-unavailable")
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert response.json()["detail"].startswith("SQL tool is busy")


def test_provider_error_details(client: TestClient) -> None:
    retryable = client.get("/raise/provider")
    assert retryable.json()["retryable"] is True
    assert retryable.headers["Retry-After"] == "5"
    assert "[anthropic] rate limited (retryable)" == retryable.json()["detail"]
    permanent = client.get("/raise/provider-permanent")
    assert permanent.json()["retryable"] is False
    assert "Retry-After" not in permanent.headers


def test_extension_members_and_generic_500_detail(client: TestClient) -> None:
    assert client.get("/raise/budget").json()["answer"] == {"text": "partial"}
    boom = client.get("/raise/boom").json()
    assert "secret internal detail" not in boom["detail"]
    assert "request_id" in boom["detail"]


def test_problem_builder() -> None:
    response = problem(418, "I'm a teapot", "short and stout", "rid", extra={"status": 200, "x": 1})
    assert response.status_code == 418
    assert response.media_type == PROBLEM_MEDIA_TYPE
    body = response.body.decode()
    assert '"status":418' in body and '"x":1' in body, "extras never override required members"
    assert problem_type("I'm a teapot") == "urn:secqa:problem:i-m-a-teapot"
    assert problem_type("   ") == "urn:secqa:problem:error"
