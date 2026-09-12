"""Unit tests for request-id handling, Server-Timing merging and the rate-limit key."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from secqa.api.middleware import (
    RequestContextMiddleware,
    accept_request_id,
    client_ip,
    merge_server_timing,
    rate_limit_string,
)
from secqa.core.settings import Settings


@pytest.mark.parametrize("value", ["abc", "trace-1.2:3_x", "A" * 128])
def test_accept_request_id_keeps_well_formed_ids(value: str) -> None:
    assert accept_request_id(value) == value
    assert accept_request_id(f"  {value}  ") == value


@pytest.mark.parametrize("value", [None, "", "   ", "A" * 129, "has space", "bad\nnewline", "é"])
def test_accept_request_id_replaces_bad_ids(value: str | None) -> None:
    generated = accept_request_id(value)
    assert len(generated) == 32 and generated != value
    assert int(generated, 16)


def test_merge_server_timing() -> None:
    assert merge_server_timing(None, "app;dur=1.0") == "app;dur=1.0"
    assert merge_server_timing("llm;dur=2.0", "app;dur=1.0") == "llm;dur=2.0, app;dur=1.0"


def _request(
    headers: dict[str, str], client: tuple[str, int] | None = ("10.0.0.1", 1234)
) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope: dict[str, Any] = {"type": "http", "headers": raw, "client": client}
    return Request(scope)


def test_client_ip_prefers_first_forwarded_hop() -> None:
    assert client_ip(_request({"x-forwarded-for": "203.0.113.9, 10.0.0.2"})) == "203.0.113.9"
    assert client_ip(_request({})) == "10.0.0.1"
    assert client_ip(_request({}, client=None)) == "unknown"


def test_rate_limit_string() -> None:
    assert rate_limit_string(Settings(_env_file=None, rate_limit_per_min=7)) == "7/minute"


def test_middleware_sets_state_and_headers_on_a_bare_app() -> None:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/echo")
    def echo(request: Request) -> dict[str, str]:
        return {"request_id": request.state.request_id}

    with TestClient(app) as client:
        response = client.get("/echo", headers={"X-Request-ID": "given-1"})
        assert response.json() == {"request_id": "given-1"}
        assert response.headers["X-Request-ID"] == "given-1"
        assert response.headers["Server-Timing"].startswith("app;dur=")
        fresh = client.get("/echo")
        assert fresh.json()["request_id"] == fresh.headers["X-Request-ID"]
        assert len(fresh.headers["X-Request-ID"]) == 32
