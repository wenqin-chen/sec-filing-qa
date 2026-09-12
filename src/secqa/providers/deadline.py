"""Request-scoped deadline for vendor HTTP calls.

The agent wall clock (``AgentLoop.wall_clock_s``) and the API request timeout
(``Settings.request_timeout_s``) are checked *between* LLM calls; nothing there can interrupt a
call that is already in flight. Left alone, one ``complete()`` may legally block for
``(max_retries + 1) * timeout_s`` seconds - long past the platform request timeout, after which
Cloud Run / Container Apps answer 504 while the worker keeps running and paying for tokens.

This module closes that gap without touching the :class:`~secqa.core.contracts.LLMProvider`
contract. A caller opens ``with deadline(seconds):`` once per request; the two vendor adapters
call :func:`remaining_s` right before each SDK call and shrink that call's per-attempt timeout
and retry count with :func:`call_budget` so the whole call - retries included - finishes inside
what is left. The deadline rides on :mod:`contextvars`, exactly like the ``request_id`` the
logger binds (:func:`secqa.core.logging.bind_context`): it follows the request through the API's
worker thread, never enters a cassette key, and is invisible to the deterministic providers.

Outside any ``deadline()`` block :func:`remaining_s` returns ``None`` and the adapters use the
timeout and retry count they were constructed with.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_DEADLINE: ContextVar[float | None] = ContextVar("secqa_provider_deadline", default=None)


@contextmanager
def deadline(seconds: float) -> Iterator[None]:
    """Bound every vendor call made inside the block to ``seconds`` from now.

    Blocks nest: the innermost deadline wins while it is active and the outer one is restored on
    exit. ``seconds`` must be positive.
    """
    if seconds <= 0:
        raise ValueError(f"deadline seconds must be positive, got {seconds!r}")
    token = _DEADLINE.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def remaining_s() -> float | None:
    """Seconds left on the active deadline (``0.0`` once it passed), or ``None`` without one."""
    until = _DEADLINE.get()
    if until is None:
        return None
    return max(0.0, until - time.monotonic())


@dataclass(frozen=True)
class CallBudget:
    """Per-attempt HTTP timeout and retry count for one SDK call."""

    timeout_s: float
    max_retries: int

    @property
    def worst_case_s(self) -> float:
        """Longest the call can block, ignoring the SDK's short retry back-off."""
        return self.timeout_s * (self.max_retries + 1)


def call_budget(timeout_s: float, max_retries: int, remaining: float | None) -> CallBudget:
    """Fit ``(max_retries + 1)`` attempts of at most ``timeout_s`` each into ``remaining``.

    With no deadline (``remaining is None``) the configured values are returned unchanged. With
    one, each attempt is capped at the time left and retries are kept only while another full
    attempt still fits, so ``worst_case_s <= remaining``. ``remaining <= 0`` yields a zero-second
    budget; callers must refuse to make that call (see the vendor adapters).
    """
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries!r}")
    if remaining is None:
        return CallBudget(timeout_s=timeout_s, max_retries=max_retries)
    if remaining <= 0:
        return CallBudget(timeout_s=0.0, max_retries=0)
    attempt = min(timeout_s, remaining)
    attempts_that_fit = int(remaining // attempt)  # >= 1 because attempt <= remaining
    retries = min(max_retries, attempts_that_fit - 1)
    return CallBudget(timeout_s=attempt, max_retries=retries)


__all__ = ["CallBudget", "call_budget", "deadline", "remaining_s"]
