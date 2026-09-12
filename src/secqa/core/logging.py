"""Structured logging via structlog.

``configure_logging(json=True)`` emits one JSON object per line (Cloud Run / ACA friendly);
``json=False`` gives a coloured console renderer for local development. Request-scoped fields
(``request_id``, ``provider``, ``model`` ...) are bound with :func:`bind_context` and cleared with
:func:`clear_context`; they ride on ``contextvars`` so they follow the request through threads
spawned by ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED: dict[str, bool | None] = {"json": None}


def configure_logging(json: bool, level: int | str = logging.INFO) -> None:
    """Configure structlog + stdlib logging. Idempotent; re-calling switches the renderer."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Third-party chatter that would otherwise flood JSON logs at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED["json"] = json


def is_configured() -> bool:
    """True once :func:`configure_logging` has run in this process."""
    return _CONFIGURED["json"] is not None


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger; configures console logging lazily if nothing configured yet."""
    if not is_configured():
        configure_logging(json=False)
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_context(**fields: Any) -> None:
    """Bind request-scoped fields (request_id, provider, model ...) to every subsequent log line."""
    structlog.contextvars.bind_contextvars(**fields)


def clear_context() -> None:
    """Drop all request-scoped fields."""
    structlog.contextvars.clear_contextvars()


__all__ = ["bind_context", "clear_context", "configure_logging", "get_logger", "is_configured"]
