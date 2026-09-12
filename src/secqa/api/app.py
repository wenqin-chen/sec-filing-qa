"""``create_app``: the FastAPI service over the shared pipeline objects (SPEC section 8).

Construction (:func:`create_app`) is side-effect light and synchronous: configure logging,
validate that the default provider has its key, build the price table / verifier / default
provider (:func:`~secqa.api.deps.build_state`), attach the limiter, middleware, error handlers
and routes. Opening the index happens in the lifespan (:func:`~secqa.api.deps.load_index`), off
the event loop, so ``/healthz`` is served immediately and a broken or missing index shows up as
a 503 with a reason on ``/readyz`` rather than as a crash loop.

``uvicorn secqa.api.app:app`` works through a lazily created module attribute (PEP 562) so
importing this module never reads settings or configures logging; ``uvicorn ... --factory
secqa.api.app:create_app`` is equivalent.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from secqa import __version__
from secqa.api.deps import build_state, close_index, load_index
from secqa.api.errors import install_error_handlers
from secqa.api.middleware import RequestContextMiddleware, build_limiter, rate_limit_string
from secqa.api.routes import build_router
from secqa.core.logging import configure_logging, get_logger
from secqa.core.settings import Settings, get_settings

log = get_logger(__name__)

TITLE = "sec-filing-qa"
DESCRIPTION = (
    "Grounded question answering over SEC 10-K/10-Q filings: hybrid retrieval with verified, "
    "page-level citations, a tool-using agent over local XBRL facts, and read-only SQL. "
    "Every error is an `application/problem+json` document carrying the `X-Request-ID`. "
    "Ingest is CLI-only; this service never writes to the index."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the index on startup (blocking I/O in a worker thread) and close it on shutdown."""
    state = get_state_of(app)
    await asyncio.to_thread(load_index, state)
    try:
        yield
    finally:
        close_index(state)
        log.info("api_shutdown")


def get_state_of(app: FastAPI) -> Any:
    """The :class:`~secqa.api.deps.AppState` attached to ``app`` (``app.state.secqa``)."""
    return app.state.secqa


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Explicit settings (tests, CLI ``secqa serve``); ``None`` reads the
            environment / ``.env`` through :func:`secqa.core.settings.get_settings`.

    Raises:
        ConfigError: the default provider needs a vendor key that is not set, or its model is
            not priced -- the process must not start in a state where every request fails.
    """
    resolved = settings if settings is not None else get_settings()
    configure_logging(resolved.log_json)
    state = build_state(resolved)
    limiter = build_limiter(resolved)

    app = FastAPI(
        title=TITLE,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.secqa = state
    app.state.limiter = limiter
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)
    app.include_router(build_router(limiter, rate_limit_string(resolved)))
    log.info(
        "api_created",
        provider=resolved.provider,
        embedder=resolved.embedder,
        duckdb_path=str(resolved.duckdb_path),
        rate_limit=rate_limit_string(resolved),
        api_key_required=resolved.api_key is not None,
    )
    return app


_lazy_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    """Provide ``secqa.api.app.app`` for ``uvicorn secqa.api.app:app`` without import-time work."""
    if name == "app":
        global _lazy_app
        if _lazy_app is None:
            _lazy_app = create_app()
        return _lazy_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DESCRIPTION", "TITLE", "create_app", "get_state_of", "lifespan"]
