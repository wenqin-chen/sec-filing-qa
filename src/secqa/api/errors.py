"""Error responses: ``application/problem+json`` everywhere (RFC 9457).

Two layers:

* :class:`ApiError` and its subclasses are raised by the routes / dependencies for the
  HTTP-level outcomes SPEC section 8 names (402 budget, 403 anonymous agent mode, 404, 422 for
  semantic validation, 503 not ready / provider not configured, 504 wall clock).
* :func:`install_error_handlers` maps *every* exception that can escape a route to a problem
  document: :class:`ApiError`, FastAPI validation errors (422), slowapi
  :class:`~slowapi.errors.RateLimitExceeded` (429), :class:`~secqa.xbrl.SqlTimeout` (504) before
  its parent :class:`~secqa.core.errors.SqlRejected` (400),
  :class:`~secqa.core.errors.ProviderError` (502), :class:`~secqa.core.errors.ConfigError` /
  :class:`~secqa.core.errors.IndexMismatch` (503) and Starlette's own ``HTTPException`` (404 /
  405 for unknown routes). Anything else becomes :func:`unhandled_problem` (500, generic detail
  so internals never leak), sent by the request-context middleware which also logs the traceback.

Every document carries the request id so a client can quote it against the server logs.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from secqa.core.errors import ConfigError, IndexMismatch, ProviderError, SqlRejected
from secqa.core.logging import get_logger
from secqa.xbrl import SqlTimeout, SqlToolUnavailable

log = get_logger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"
PROBLEM_TYPE_PREFIX = "urn:secqa:problem:"
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_UNKNOWN_REQUEST_ID = "unknown"


class ApiError(Exception):
    """An HTTP outcome the API chose deliberately; rendered by :func:`problem`.

    Subclasses fix ``status`` and ``title``; ``detail`` is the human-readable explanation and
    ``extra`` holds RFC 9457 extension members (for example the partial answer of a 402).
    """

    status: int = 500
    title: str = "Internal server error"

    def __init__(self, detail: str, *, extra: dict[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extra = dict(extra or {})


class NotReady(ApiError):
    """503: the index / embedder are not loaded (see ``/readyz`` for the reason)."""

    status = 503
    title = "Service not ready"


class ProviderUnavailable(ApiError):
    """503: the requested provider needs a key or price entry the server does not have."""

    status = 503
    title = "Provider not configured"


class Forbidden(ApiError):
    """403: agent mode / non-default providers need ``X-API-Key`` on this server."""

    status = 403
    title = "Forbidden"


class NotFound(ApiError):
    """404: unknown document or page."""

    status = 404
    title = "Not found"


class Unprocessable(ApiError):
    """422: the body is well-formed but semantically invalid (unknown provider, k > max_k)."""

    status = 422
    title = "Unprocessable request"


class BudgetExceeded(ApiError):
    """402: the per-request or per-instance daily cost cap would be (or was) exceeded."""

    status = 402
    title = "Cost budget exceeded"


class WallClockExceeded(ApiError):
    """504: the request ran past the wall-clock limit (partial answer attached)."""

    status = 504
    title = "Wall clock exceeded"


def problem_type(title: str) -> str:
    """Stable ``type`` URI for a title: ``'Rate limited'`` -> ``urn:secqa:problem:rate-limited``."""
    slug = _SLUG_RE.sub("-", title.strip().lower()).strip("-") or "error"
    return PROBLEM_TYPE_PREFIX + slug


def problem(
    status: int,
    title: str,
    detail: str,
    request_id: str,
    *,
    extra: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build an ``application/problem+json`` response.

    Args:
        status: HTTP status code.
        title: Short, stable summary of the problem class (also derives ``type``).
        detail: Occurrence-specific explanation, safe to show to a client.
        request_id: The request id echoed in ``X-Request-ID``.
        extra: RFC 9457 extension members merged into the document (must be JSON-able).
        headers: Additional response headers (``Retry-After`` for 429).
    """
    body: dict[str, Any] = {
        "type": problem_type(title),
        "title": title,
        "status": status,
        "detail": detail,
        "request_id": request_id,
    }
    for key, value in (extra or {}).items():
        body.setdefault(key, value)
    return JSONResponse(
        status_code=status, content=body, media_type=PROBLEM_MEDIA_TYPE, headers=headers
    )


def unhandled_problem(request_id: str) -> JSONResponse:
    """The 500 document for an exception nothing else mapped (generic detail, nothing leaks).

    Rendered by :class:`~secqa.api.middleware.RequestContextMiddleware`, not by a Starlette
    handler: the outermost ``ServerErrorMiddleware`` would send its response outside the
    request-context layer, and the document would lose ``X-Request-ID``.
    """
    return problem(
        500,
        "Internal server error",
        "an unexpected error occurred; quote the request_id when reporting it",
        request_id,
    )


def request_id_of(request: Request) -> str:
    """The request id bound by the middleware, or ``'unknown'`` when no middleware ran."""
    value = getattr(request.state, "request_id", None)
    return str(value) if value else _UNKNOWN_REQUEST_ID


def install_error_handlers(app: FastAPI) -> None:
    """Register every exception -> problem+json mapping on ``app`` (see the module docstring)."""

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        log.info("api_error", status=exc.status, title=exc.title, detail=exc.detail)
        return problem(exc.status, exc.title, exc.detail, request_id_of(request), extra=exc.extra)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = jsonable_encoder(exc.errors(), exclude={"input", "url", "ctx"})
        detail = "; ".join(_describe(error) for error in errors[:5]) or "invalid request"
        return problem(
            422,
            "Validation error",
            detail,
            request_id_of(request),
            extra={"errors": errors},
        )

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        response = problem(
            429,
            "Rate limited",
            f"rate limit exceeded: {exc.detail}",
            request_id_of(request),
        )
        limiter = getattr(request.app.state, "limiter", None)
        view_limit = getattr(request.state, "view_rate_limit", None)
        if limiter is not None and view_limit is not None:
            response = limiter._inject_headers(response, view_limit)  # noqa: SLF001 - slowapi API
        return response

    @app.exception_handler(SqlTimeout)
    async def _sql_timeout(request: Request, exc: SqlTimeout) -> JSONResponse:
        return problem(504, "SQL timeout", exc.reason, request_id_of(request))

    @app.exception_handler(SqlToolUnavailable)
    async def _sql_unavailable(request: Request, exc: SqlToolUnavailable) -> JSONResponse:
        log.warning("sql_tool_unavailable", reason=exc.reason)
        return problem(
            503,
            "SQL tool unavailable",
            exc.reason,
            request_id_of(request),
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(SqlRejected)
    async def _sql_rejected(request: Request, exc: SqlRejected) -> JSONResponse:
        return problem(400, "SQL rejected", exc.reason, request_id_of(request))

    @app.exception_handler(ProviderError)
    async def _provider_error(request: Request, exc: ProviderError) -> JSONResponse:
        log.warning("upstream_provider_error", error=str(exc), retryable=exc.retryable)
        headers = {"Retry-After": "5"} if exc.retryable else None
        return problem(
            502,
            "Upstream provider error",
            str(exc),
            request_id_of(request),
            extra={"retryable": exc.retryable},
            headers=headers,
        )

    @app.exception_handler(IndexMismatch)
    async def _index_mismatch(request: Request, exc: IndexMismatch) -> JSONResponse:
        return problem(503, "Index mismatch", str(exc), request_id_of(request))

    @app.exception_handler(ConfigError)
    async def _config_error(request: Request, exc: ConfigError) -> JSONResponse:
        return problem(503, "Provider not configured", str(exc), request_id_of(request))

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        title = _HTTP_TITLES.get(exc.status_code, "HTTP error")
        detail = str(exc.detail) if exc.detail else title
        headers = dict(exc.headers) if exc.headers else None
        return problem(exc.status_code, title, detail, request_id_of(request), headers=headers)


_HTTP_TITLES = {
    400: "Bad request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not found",
    405: "Method not allowed",
    413: "Payload too large",
    415: "Unsupported media type",
    429: "Rate limited",
}


def _describe(error: dict[str, Any]) -> str:
    """``'body.k: Input should be less than or equal to 20'`` from one validation error."""
    location = ".".join(str(part) for part in error.get("loc", ()))
    message = str(error.get("msg", "invalid"))
    return f"{location}: {message}" if location else message


__all__ = [
    "PROBLEM_MEDIA_TYPE",
    "ApiError",
    "BudgetExceeded",
    "Forbidden",
    "NotFound",
    "NotReady",
    "ProviderUnavailable",
    "Unprocessable",
    "WallClockExceeded",
    "install_error_handlers",
    "problem",
    "problem_type",
    "request_id_of",
    "unhandled_problem",
]
