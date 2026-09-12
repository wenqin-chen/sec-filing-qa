"""Token-bucket rate limiter for SEC EDGAR fair-access compliance.

EDGAR allows at most 10 requests per second per client. The bucket refills continuously at
``rate`` tokens per second up to ``burst`` tokens; :meth:`TokenBucket.acquire` blocks (via the
injected ``sleep``) until one token is available. ``clock`` and ``sleep`` are injectable so tests
can drive a fake clock and assert on the schedule without waiting in real time.

Thread-safe: a lock guards the token count so a client shared between worker threads cannot
exceed the rate.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class TokenBucket:
    """Blocking token bucket: at most ``rate`` acquisitions per second in steady state.

    With the default ``burst=1`` consecutive acquisitions are spaced at least ``1 / rate``
    seconds apart, so any one-second window contains at most ``rate`` acquisitions.
    """

    def __init__(
        self,
        rate: float,
        burst: int = 1,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if burst < 1:
            raise ValueError(f"burst must be at least 1, got {burst}")
        self.rate = float(rate)
        self.burst = int(burst)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tokens = float(self.burst)
        self._last = self._clock()

    def _refill(self) -> None:
        """Add tokens earned since the last refill (caller holds the lock)."""
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(float(self.burst), self._tokens + elapsed * self.rate)

    @property
    def tokens(self) -> float:
        """Tokens currently available (after refilling to ``now``)."""
        with self._lock:
            self._refill()
            return self._tokens

    def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            # Sleep outside the lock so other threads can observe/refill; loop re-checks.
            self._sleep(wait)


__all__ = ["TokenBucket"]
