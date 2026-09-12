"""Reciprocal-rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

``score(d) = sum over rankings r of 1 / (k + rank_r(d))`` for every ranking in which ``d``
appears. Scores from BM25 and cosine similarity live on incomparable scales, so fusing ranks
instead of scores needs no calibration; ``k=60`` is the value from the paper and the SPEC.
"""

from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one list of ``(id, score)`` sorted best-first.

    Args:
        rankings: ranked lists (best first) of item ids, e.g. ``[bm25_ids, dense_ids]``. An id
            repeated inside one list is counted once, at its best rank in that list.
        k: rank smoothing constant (must be positive); larger ``k`` flattens the contribution of
            top ranks.

    Returns:
        ``[(id, score), ...]`` sorted by score descending. Ties are broken deterministically by
        the item's best rank across lists, then by the id string, so results are reproducible.
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))


__all__ = ["rrf"]
