"""Request-scoped plumbing: request ids, structured access logs, ``Server-Timing``, rate limits.

:class:`RequestContextMiddleware` is a pure ASGI middleware (no ``BaseHTTPMiddleware``: that
wrapper buffers responses and breaks contextvar propagation in some Starlette versions). For
every HTTP request it

1. takes the caller's ``X-Request-ID`` when it is well-formed, otherwise generates one
   (:func:`secqa.core.ids.request_id`), stores it in ``scope['state']`` so handlers and error
   handlers can read ``request.state.request_id``;
2. binds ``request_id``, ``method`` and ``path`` to the structlog context, so every log line
   emitted while serving the request (including from the threadpool the sync endpoints run in)
   carries them;
3. echoes ``X-Request-ID`` and appends ``app;dur=<ms>`` to ``Server-Timing`` on the response
   (a handler may have set ``retrieval;dur`` / ``llm;dur`` already);
4. writes one ``http_request`` access log line with status and latency, then clears the context;
5. turns an exception no handler mapped into the generic 500 problem document (with the same
   headers) and re-raises it, so the server still logs it and test clients still see it.

The slowapi :class:`~slowapi.Limiter` built by :func:`build_limiter` keys buckets by client IP
(:func:`client_ip`): the socket peer by default, or, when ``SECQA_TRUSTED_PROXY_HOPS`` is set,
the ``X-Forwarded-For`` hop that many positions from the *right*. Cloud Run and Container Apps
append the real client to whatever ``X-Forwarded-For`` the caller sent, so the leftmost hop is
always caller-controlled and is never used.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from slowapi import Limiter
from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import Request

from secqa.api.errors import unhandled_problem
from secqa.core.ids import request_id as new_request_id
from secqa.core.logging import bind_context, clear_context, get_logger
from secqa.core.settings import Settings

log = get_logger("secqa.api.access")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

REQUEST_ID_HEADER = "X-Request-ID"
SERVER_TIMING_HEADER = "Server-Timing"
REQUEST_ID_MAX_CHARS = 128
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")


class RequestContextMiddleware:
    """See the module docstring."""

    def __init__(self, app: ASGIApp, header_name: str = REQUEST_ID_HEADER) -> None:
        self.app = app
        self.header_name = header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(self.header_name)
        rid = accept_request_id(incoming)
        scope.setdefault("state", {})["request_id"] = rid
        method = str(scope.get("method", "-"))
        path = str(scope.get("path", "-"))
        bind_context(request_id=rid, method=method, path=path)
        started = time.perf_counter()
        status: dict[str, int | None] = {"code": None}

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = int(message["status"])
                bind_context(status=status["code"])
                headers = MutableHeaders(scope=message)
                headers[self.header_name] = rid
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                headers[SERVER_TIMING_HEADER] = merge_server_timing(
                    headers.get(SERVER_TIMING_HEADER), f"app;dur={elapsed_ms:.1f}"
                )
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception as exc:
            # Nothing below mapped this exception. Render the 500 here so it still carries the
            # request id and timing headers, then re-raise so the server logs it / a test
            # client with raise_server_exceptions=True sees it (Starlette's outer
            # ServerErrorMiddleware sends nothing once a response has started).
            log.error("unhandled_exception", error=str(exc), exc_info=exc)
            if status["code"] is None:
                await unhandled_problem(rid)(scope, receive, send_with_headers)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            log.info(
                "http_request",
                status=status["code"] if status["code"] is not None else 500,
                latency_ms=round(elapsed_ms, 1),
            )
            clear_context()


def accept_request_id(incoming: str | None) -> str:
    """Return a safe request id: the caller's when well-formed, otherwise a fresh uuid4 hex.

    Only ``[A-Za-z0-9._:-]`` up to 128 characters are accepted so an id can be logged and
    echoed verbatim without header-injection or log-forging concerns.
    """
    if incoming is not None:
        candidate = incoming.strip()
        if _REQUEST_ID_RE.fullmatch(candidate):
            return candidate
        log.debug("request_id_rejected", length=len(candidate))
    return new_request_id()


def merge_server_timing(existing: str | None, metric: str) -> str:
    """Append one ``Server-Timing`` metric to an existing header value (comma separated)."""
    return f"{existing}, {metric}" if existing else metric


def client_ip(request: Request, trusted_proxy_hops: int = 0) -> str:
    """Rate-limit bucket key for ``request``.

    With ``trusted_proxy_hops == 0`` (the default) the key is the socket peer: ``X-Forwarded-For``
    is ignored because a direct caller can write anything into it. With ``trusted_proxy_hops =
    n > 0`` the key is the ``X-Forwarded-For`` entry ``n`` positions from the right, i.e. the
    address the outermost *trusted* proxy appended. Cloud Run and Azure Container Apps do not
    replace an incoming ``X-Forwarded-For``; they append the real client to it, so behind either
    the correct value is ``1`` and the leftmost entry (which the caller controls) is never used.
    When the header is missing or has fewer than ``n`` entries the socket peer is used, and
    ``'unknown'`` only when there is no peer either (some test transports).

    Args:
        request: The incoming request.
        trusted_proxy_hops: Number of proxies in front of the service that append to
            ``X-Forwarded-For`` (:attr:`Settings.trusted_proxy_hops`).

    Raises:
        ValueError: ``trusted_proxy_hops`` is negative.
    """
    if trusted_proxy_hops < 0:
        raise ValueError(f"trusted_proxy_hops must be >= 0, got {trusted_proxy_hops}")
    if trusted_proxy_hops > 0:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            hops = [hop.strip() for hop in forwarded.split(",")]
            if len(hops) >= trusted_proxy_hops:
                chosen = hops[len(hops) - trusted_proxy_hops]
                if chosen:
                    return chosen
    if request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


def rate_limit_string(settings: Settings) -> str:
    """slowapi limit string from :attr:`Settings.rate_limit_per_min` (``'10/minute'``)."""
    return f"{settings.rate_limit_per_min}/minute"


def build_limiter(settings: Settings) -> Limiter:
    """A per-app, in-memory slowapi limiter (per instance; documented in SPEC section 8).

    Buckets are keyed by :func:`client_ip` with :attr:`Settings.trusted_proxy_hops` bound; the
    wrapper is a plain function because slowapi inspects the key function's signature for a
    ``request`` parameter.
    """
    hops = settings.trusted_proxy_hops

    def key_func(request: Request) -> str:
        return client_ip(request, trusted_proxy_hops=hops)

    return Limiter(key_func=key_func, headers_enabled=True, enabled=settings.rate_limit_per_min > 0)


__all__ = [
    "REQUEST_ID_HEADER",
    "REQUEST_ID_MAX_CHARS",
    "SERVER_TIMING_HEADER",
    "RequestContextMiddleware",
    "accept_request_id",
    "build_limiter",
    "client_ip",
    "merge_server_timing",
    "rate_limit_string",
]
