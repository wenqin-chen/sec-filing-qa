"""Tests for secqa.edgar.ratelimit.TokenBucket (driven by a fake clock, no real sleeping)."""

from __future__ import annotations

import pytest

from secqa.edgar.ratelimit import TokenBucket
from tests.edgar.conftest import FakeClock


def _max_in_any_window(times: list[float], window: float = 1.0) -> int:
    """Largest number of acquisitions falling inside any half-open window [t, t + window)."""
    best = 0
    for i, start in enumerate(times):
        count = 0
        for t in times[i:]:
            if t < start + window:
                count += 1
            else:
                break
        best = max(best, count)
    return best


@pytest.mark.parametrize("rate", [1.0, 8.0, 10.0])
def test_never_exceeds_rate_in_any_one_second_window(fake_clock: FakeClock, rate: float) -> None:
    bucket = TokenBucket(rate=rate, clock=fake_clock.now, sleep=fake_clock.sleep)
    times: list[float] = []
    for _ in range(int(rate * 10) + 7):
        bucket.acquire()
        times.append(fake_clock.now())
    assert _max_in_any_window(times) <= rate
    assert all(s > 0 for s in fake_clock.sleeps), "only sleeps when a token is unavailable"


def test_first_acquire_is_free_and_subsequent_are_spaced(fake_clock: FakeClock) -> None:
    bucket = TokenBucket(rate=8.0, clock=fake_clock.now, sleep=fake_clock.sleep)
    t0 = fake_clock.now()
    bucket.acquire()
    assert fake_clock.now() == t0  # burst of 1: first call needs no wait
    bucket.acquire()
    assert fake_clock.now() == pytest.approx(t0 + 1 / 8)
    bucket.acquire()
    assert fake_clock.now() == pytest.approx(t0 + 2 / 8)


def test_burst_allows_immediate_acquisitions_then_throttles(fake_clock: FakeClock) -> None:
    bucket = TokenBucket(rate=4.0, burst=3, clock=fake_clock.now, sleep=fake_clock.sleep)
    for _ in range(3):
        bucket.acquire()
    assert fake_clock.sleeps == []
    bucket.acquire()
    assert len(fake_clock.sleeps) == 1
    assert fake_clock.sleeps[0] == pytest.approx(0.25)


def test_idle_time_refills_up_to_burst_only(fake_clock: FakeClock) -> None:
    bucket = TokenBucket(rate=8.0, burst=2, clock=fake_clock.now, sleep=fake_clock.sleep)
    bucket.acquire()
    bucket.acquire()
    fake_clock.t += 60.0  # a long pause must not bank more than `burst` tokens
    assert bucket.tokens == pytest.approx(2.0)
    bucket.acquire()
    bucket.acquire()
    assert fake_clock.sleeps == []
    bucket.acquire()
    assert len(fake_clock.sleeps) == 1


def test_sleep_duration_is_exactly_the_shortfall(fake_clock: FakeClock) -> None:
    bucket = TokenBucket(rate=2.0, clock=fake_clock.now, sleep=fake_clock.sleep)
    bucket.acquire()
    fake_clock.t += 0.1  # 0.2 tokens earned; need 0.8 more at 2/s -> 0.4 s
    bucket.acquire()
    assert fake_clock.sleeps == [pytest.approx(0.4)]


def test_clock_going_backwards_is_tolerated(fake_clock: FakeClock) -> None:
    bucket = TokenBucket(rate=8.0, clock=fake_clock.now, sleep=fake_clock.sleep)
    bucket.acquire()
    fake_clock.t -= 5.0
    assert bucket.tokens == pytest.approx(0.0)
    bucket.acquire()  # must not raise or hang; it simply waits one interval
    assert fake_clock.sleeps[-1] == pytest.approx(1 / 8)


@pytest.mark.parametrize(("rate", "burst"), [(0.0, 1), (-1.0, 1), (8.0, 0)])
def test_invalid_parameters_rejected(rate: float, burst: int) -> None:
    with pytest.raises(ValueError):
        TokenBucket(rate=rate, burst=burst)


def test_default_clock_and_sleep_are_real_time() -> None:
    bucket = TokenBucket(rate=1000.0)
    bucket.acquire()
    assert 0.0 <= bucket.tokens <= 1.0
