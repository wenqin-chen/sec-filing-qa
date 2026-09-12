"""Hand-computed reciprocal-rank fusion checks."""

from __future__ import annotations

import pytest

from secqa.store.fusion import rrf


def test_rrf_matches_hand_computation() -> None:
    bm25 = ["a", "b", "c"]
    dense = ["b", "d", "a"]
    fused = dict(rrf([bm25, dense], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)
    order = [item for item, _ in rrf([bm25, dense], k=60)]
    # b (1/62 + 1/61) > a (1/61 + 1/63) > d (1/62) > c (1/63)
    assert order == ["b", "a", "d", "c"]


def test_rrf_ties_broken_by_best_rank_then_id() -> None:
    # 'x' and 'y' both score exactly 1/61: tie -> best rank equal (1) -> id order.
    assert [item for item, _ in rrf([["y"], ["x"]])] == ["x", "y"]
    # q: rank 2 in one list + rank 1 in another = 1/3 + 1/2; p and z tie at 1/2 -> id order.
    fused = rrf([["p"], ["z", "q"], ["q"]], k=1)
    assert fused[0] == ("q", pytest.approx(1 / 3 + 1 / 2))
    assert fused[1] == ("p", pytest.approx(1 / 2))
    assert fused[2] == ("z", pytest.approx(1 / 2))


def test_rrf_counts_duplicate_within_one_list_once() -> None:
    assert rrf([["a", "a", "a"]], k=10) == [("a", pytest.approx(1 / 11))]


def test_rrf_empty_inputs_and_single_list() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []
    assert rrf([["a", "b"]], k=60) == [("a", pytest.approx(1 / 61)), ("b", pytest.approx(1 / 62))]


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        rrf([["a"]], k=0)
