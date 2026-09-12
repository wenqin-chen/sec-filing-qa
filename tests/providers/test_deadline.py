"""secqa.providers.deadline: the request-scoped deadline and the per-call budget arithmetic."""

from __future__ import annotations

import threading
import time

import pytest

from secqa.providers.deadline import CallBudget, call_budget, deadline, remaining_s


def test_no_deadline_outside_a_block() -> None:
    assert remaining_s() is None
    assert call_budget(45.0, 1, None) == CallBudget(timeout_s=45.0, max_retries=1)


def test_deadline_counts_down_and_is_reset_on_exit() -> None:
    with deadline(10.0):
        left = remaining_s()
        assert left is not None and 9.0 < left <= 10.0
    assert remaining_s() is None


def test_deadline_is_reset_when_the_block_raises() -> None:
    with pytest.raises(RuntimeError), deadline(10.0):
        raise RuntimeError("boom")
    assert remaining_s() is None


def test_nested_deadlines_restore_the_outer_one() -> None:
    with deadline(100.0):
        with deadline(1.0):
            inner = remaining_s()
            assert inner is not None and inner <= 1.0
        outer = remaining_s()
        assert outer is not None and outer > 90.0


def test_expired_deadline_reports_zero_never_negative() -> None:
    with deadline(0.01):
        time.sleep(0.02)
        assert remaining_s() == 0.0


def test_deadline_rejects_non_positive_seconds() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"), deadline(bad):
            pass  # pragma: no cover - the context manager raises before yielding


def test_deadline_is_scoped_to_the_thread_that_set_it() -> None:
    """contextvars: a worker thread started outside the block sees no deadline."""
    seen: list[float | None] = []

    def probe() -> None:
        seen.append(remaining_s())

    with deadline(10.0):
        worker = threading.Thread(target=probe)
        worker.start()
        worker.join()
    assert seen == [None]


@pytest.mark.parametrize(
    ("timeout_s", "max_retries", "remaining", "expected"),
    [
        # plenty of time: configured values unchanged
        (45.0, 1, 1000.0, CallBudget(45.0, 1)),
        # exactly two attempts fit
        (45.0, 1, 90.0, CallBudget(45.0, 1)),
        # one full attempt plus a partial one: the retry is dropped, the attempt keeps its size
        (45.0, 1, 60.0, CallBudget(45.0, 0)),
        # less than one attempt: the attempt shrinks to what is left
        (45.0, 1, 30.0, CallBudget(30.0, 0)),
        # retries are capped by the configured count, never raised
        (10.0, 0, 1000.0, CallBudget(10.0, 0)),
        (10.0, 2, 25.0, CallBudget(10.0, 1)),
        # nothing left: a zero budget the adapters refuse to spend
        (45.0, 1, 0.0, CallBudget(0.0, 0)),
    ],
)
def test_call_budget_never_exceeds_the_remaining_time(
    timeout_s: float, max_retries: int, remaining: float, expected: CallBudget
) -> None:
    budget = call_budget(timeout_s, max_retries, remaining)
    assert budget == expected
    assert budget.worst_case_s <= remaining
    assert budget.max_retries <= max_retries and budget.timeout_s <= timeout_s


def test_call_budget_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        call_budget(0.0, 1, None)
    with pytest.raises(ValueError, match="max_retries"):
        call_budget(45.0, -1, None)


def test_call_budget_follows_a_live_deadline() -> None:
    with deadline(20.0):
        budget = call_budget(45.0, 2, remaining_s())
    assert budget.max_retries == 0 and 19.0 < budget.timeout_s <= 20.0
