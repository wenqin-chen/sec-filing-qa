"""Unit tests for request-id handling, Server-Timing merging and the rate-limit key."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from pydantic import ValidationError
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from secqa.api.middleware import (
    RequestContextMiddleware,
    accept_request_id,
    build_limiter,
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


PEER = "10.0.0.1"
REAL = "198.51.100.7"


def test_client_ip_ignores_forwarded_for_by_default() -> None:
    # A direct caller controls the whole header; with no trusted proxy it must not pick the key.
    assert client_ip(_request({"x-forwarded-for": "203.0.113.9, 10.0.0.2"})) == PEER
    assert client_ip(_request({})) == PEER
    assert client_ip(_request({}, client=None)) == "unknown"


@pytest.mark.parametrize(
    ("forwarded", "hops", "expected"),
    [
        # Cloud Run / Container Apps: the platform appends the real client after the caller's junk.
        (f"203.0.113.9, {REAL}", 1, REAL),
        (f"1.1.1.1, 2.2.2.2, {REAL}", 1, REAL),
        (f"{REAL}", 1, REAL),
        # Two appending proxies (e.g. an external LB in front of the platform proxy).
        (f"9.9.9.9, {REAL}, 10.0.0.2", 2, REAL),
        # Fewer entries than trusted proxies: nothing trustworthy in the header, use the peer.
        (f"{REAL}", 2, PEER),
        # Whitespace-only trusted entry: use the peer rather than an empty key.
        ("203.0.113.9, ", 1, PEER),
    ],
)
def test_client_ip_counts_trusted_hops_from_the_right(
    forwarded: str, hops: int, expected: str
) -> None:
    assert client_ip(_request({"x-forwarded-for": forwarded}), trusted_proxy_hops=hops) == expected


def test_client_ip_without_header_behind_a_proxy_uses_the_peer() -> None:
    assert client_ip(_request({}), trusted_proxy_hops=1) == PEER
    assert client_ip(_request({}, client=None), trusted_proxy_hops=1) == "unknown"


def test_client_ip_rejects_negative_hops() -> None:
    with pytest.raises(ValueError, match="trusted_proxy_hops"):
        client_ip(_request({}), trusted_proxy_hops=-1)


def _limited_app(settings: Settings) -> FastAPI:
    """A bare app with the real limiter on one route (no index, no providers)."""
    limiter = build_limiter(settings)
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/limited")
    @limiter.limit(rate_limit_string(settings))
    def limited(request: Request, response: Response) -> dict[str, str]:
        return {"ok": "yes"}

    return app


def test_spoofed_leftmost_hop_does_not_open_a_new_bucket_behind_a_proxy() -> None:
    # Simulates Cloud Run / Container Apps: the caller varies the leftmost entry on every request
    # and the platform appends the same real client. All six must land in ONE 2/min bucket.
    settings = Settings(_env_file=None, rate_limit_per_min=2, trusted_proxy_hops=1)
    with TestClient(_limited_app(settings)) as client:
        codes = [
            client.get("/limited", headers={"X-Forwarded-For": f"10.0.0.{i}, {REAL}"}).status_code
            for i in range(6)
        ]
    assert codes == [200, 200, 429, 429, 429, 429]


def test_forwarded_for_is_ignored_without_a_trusted_proxy() -> None:
    # Direct exposure (default): a caller-supplied header must not split the peer's bucket.
    settings = Settings(_env_file=None, rate_limit_per_min=2)
    with TestClient(_limited_app(settings)) as client:
        codes = [
            client.get("/limited", headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
            for i in range(4)
        ]
    assert codes == [200, 200, 429, 429]


def test_settings_trusted_proxy_hops_default_and_bounds() -> None:
    assert Settings(_env_file=None).trusted_proxy_hops == 0
    assert Settings(_env_file=None, trusted_proxy_hops=1).trusted_proxy_hops == 1
    with pytest.raises(ValidationError):
        Settings(_env_file=None, trusted_proxy_hops=-1)


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
