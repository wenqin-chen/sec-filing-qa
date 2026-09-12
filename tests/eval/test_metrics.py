"""Deterministic metrics: numeric_match, page recall, overlap recall, MRR, bootstrap, failures."""

from __future__ import annotations

from pathlib import Path

import pytest

from secqa.core.contracts import Chunk, Evidence, Hit
from secqa.core.errors import ConfigError
from secqa.eval.metrics import (
    bootstrap_ci,
    char_overlap,
    classify_failure,
    effective_label,
    evidence_overlap_recall,
    gold_page_mrr,
    judge_numeric_disagree,
    numeric_match,
    numeric_match_scale,
    page_recall_at_k,
    summarize,
)
from tests.eval.conftest import TOP_DOC, make_record, verified_citation, write_predictions

# ---- numeric_match ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pred", "gold", "expected"),
    [
        (1_577_000_000.0, "$1,577 million", True),
        (1_577_000_000.0, "$1577.00", True),  # gold in millions without saying so (scale)
        (1_577_000.0, "$1577.00", True),  # gold in thousands without saying so
        (1577.0, "$1,577 million", False),  # reverse direction: the model said $1,577, not $1,577M
        (1.577, "$1577.00", False),  # value / 1e3 == gold is a wrong answer, not a unit scale
        (0.01577, "$1577.00", False),  # nor value / 1e3 with the percent equivalence on top
        (1.577e15, "$1577.00", False),  # beyond the documented scales
        (1_580_000_000.0, "$1,577 million", True),  # within 1%
        (1_600_000_000.0, "$1,577 million", False),  # 1.5% off
        (0.12, "12%", True),
        (12.0, "12%", True),  # ratio/percent equivalence
        (0.12, "12.5%", False),
        (1.2e7, "12%", False),  # 1.2 million percent is not 12%
        (1.2e-5, "12%", False),
        (1200.0, "12%", False),  # 12 * 100 (percent) then * 1e3 (scale) must not chain
        (3.5, "3.5x", True),
        (35.0, "3.5x", False),  # x10 is neither a table-header scale nor the percent equivalence
        (-1577.0, "(1,577)", True),
        (1577.0, "(1,577)", False),  # sign matters
        (245.0, "$245 million", False),  # value is in base units by contract: 245 is $245
        (245.0, "Operating income was $245 million and net income was $190 million.", None),
        (245.0, "No", None),  # gold has no number
        (None, "$245 million", None),  # model gave no value
        (float("nan"), "$245 million", None),
    ],
)
def test_numeric_match_cases(pred: float | None, gold: str, expected: bool | None) -> None:
    assert numeric_match(pred, gold) is expected


def test_numeric_match_scale_can_be_disabled() -> None:
    assert numeric_match(1_577_000_000.0, "$1577.00", scales=(1.0,)) is False
    with pytest.raises(ValueError):
        numeric_match(1.0, "1", rel_tol=0.0)
    with pytest.raises(ValueError):
        numeric_match(1.0, "1", scales=(1.0, 0.0))


def test_numeric_match_accepts_only_the_documented_equivalences() -> None:
    """Regression: the old rule tried every scale in both directions and let the ratio/percent
    equivalence ride inside each scaled comparison, so every power of ten from 1.577e-3 to
    1.577e12 matched a gold of $1577.00 -- and overrode the judge. Exactly six values may match:
    the gold itself, the three one-way table-header scales, and the x100 ratio/percent
    equivalence of ``numbers_equal`` at scale 1 (never chained with a scale)."""
    gold = "$1577.00"
    scaled = {1577.0, 1_577_000.0, 1_577_000_000.0, 1_577_000_000_000.0}
    percent_at_scale_1 = {15.77, 157_700.0}
    probes = {1577.0 * 10.0**exp for exp in range(-9, 13)}
    for value in sorted(probes):
        expected = value in scaled | percent_at_scale_1
        assert numeric_match(value, gold) is expected, value


def test_numeric_match_scale_reports_the_matched_scale() -> None:
    assert numeric_match_scale(1_577_000_000.0, "$1577.00") == 1e6
    assert numeric_match_scale(1_577_000.0, "$1577.00") == 1e3
    assert numeric_match_scale(1_577_000_000.0, "$1,577 million") == 1.0
    assert numeric_match_scale(12.0, "12%") == 1.0  # percent equivalence lives at scale 1 only
    assert numeric_match_scale(1577.0, "$1,577 million") is None  # mismatch
    assert numeric_match_scale(None, "$1,577 million") is None  # undefined
    assert numeric_match_scale(1.0, "No") is None  # undefined
    assert numeric_match_scale(1_577_000_000.0, "$1577.00", scales=(1.0,)) is None


# ---- page recall / MRR --------------------------------------------------------------------


def test_page_recall_edges() -> None:
    gold = [(TOP_DOC, 3)]
    assert page_recall_at_k([], gold, 5) == 0.0
    assert page_recall_at_k([(TOP_DOC, 3)], [], 5) == 0.0  # no gold -> undefined -> 0
    ranked = [(TOP_DOC, 1), (TOP_DOC, 1), (TOP_DOC, 2), (TOP_DOC, 3)]  # duplicate page collapsed
    assert page_recall_at_k(ranked, gold, 2) == 0.0
    assert page_recall_at_k(ranked, gold, 3) == 1.0
    assert page_recall_at_k([("OTHER_2022_10K", 3)], gold, 10) == 0.0  # same page, other doc
    with pytest.raises(ValueError):
        page_recall_at_k(ranked, gold, 0)


def test_gold_page_mrr() -> None:
    gold = [(TOP_DOC, 2), (TOP_DOC, 3)]
    assert gold_page_mrr([(TOP_DOC, 1), (TOP_DOC, 1), (TOP_DOC, 3), (TOP_DOC, 2)], gold) == 0.5
    assert gold_page_mrr([(TOP_DOC, 2)], gold) == 1.0
    assert gold_page_mrr([(TOP_DOC, 9)], gold) == 0.0
    assert gold_page_mrr([], gold) == 0.0


# ---- overlap recall -----------------------------------------------------------------------


def _hit(text: str, page: int, rank: int) -> Hit:
    chunk = Chunk(
        chunk_id=f"{'0' * 39}{rank}",
        doc_name=TOP_DOC,
        page_num=page,
        chunk_idx=0,
        section=None,
        text=text,
        n_tokens=len(text.split()),
    )
    return Hit(chunk=chunk, score=1.0 / rank, rank=rank, source="hybrid")


def test_overlap_recall_on_offset_page_disagrees_with_page_recall() -> None:
    """The chunk carries the evidence text but claims page 2 (a 0/1-based offset bug)."""
    evidence = [Evidence(doc_name=TOP_DOC, page_num=1, text="Total net sales were $1,577 million")]
    hits = [
        _hit("Unrelated text about leases and other commitments.", 1, 1),
        _hit("Total net sales were $1,577 million in fiscal 2023, an increase of 12%.", 2, 2),
    ]
    assert evidence_overlap_recall(hits, evidence, k=10) == 1.0
    assert page_recall_at_k([(TOP_DOC, 2)], [(TOP_DOC, 1)], 10) == 0.0
    assert evidence_overlap_recall(hits, evidence, k=1) == 0.0  # cut before the matching chunk


def test_overlap_recall_threshold_and_edges() -> None:
    evidence = [Evidence(doc_name=TOP_DOC, page_num=1, text="alpha beta gamma delta epsilon zeta")]
    partial = [_hit("alpha beta gamma delta and then something else", 1, 1)]
    assert char_overlap(evidence[0].text, partial[0].chunk.text) == pytest.approx(
        len("alpha beta gamma delta ") / len("alpha beta gamma delta epsilon zeta")
    )
    assert evidence_overlap_recall(partial, evidence, k=5, min_overlap=0.5) == 1.0
    assert evidence_overlap_recall(partial, evidence, k=5, min_overlap=0.9) == 0.0
    assert evidence_overlap_recall([], evidence, k=5) == 0.0
    assert evidence_overlap_recall(partial, [], k=5) == 0.0
    blank = [Evidence(doc_name=TOP_DOC, page_num=1, text="  ")]
    assert evidence_overlap_recall(partial, blank, k=5) == 0.0
    with pytest.raises(ValueError):
        evidence_overlap_recall(partial, evidence, k=5, min_overlap=0.0)


def test_char_overlap_is_normalised() -> None:
    assert char_overlap("Net  Sales, $1,577\nmillion", "net sales $1,577 million!") == 1.0
    assert char_overlap("", "anything") == 0.0


# ---- bootstrap ----------------------------------------------------------------------------


def test_bootstrap_ci_shape_and_reproducibility() -> None:
    values = [1.0] * 60 + [0.0] * 40
    low, high = bootstrap_ci(values, n_boot=500, seed=0)
    assert 0.0 <= low <= 0.6 <= high <= 1.0
    assert (low, high) == bootstrap_ci(values, n_boot=500, seed=0)
    assert bootstrap_ci([0.7]) == (0.7, 0.7)
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], n_boot=0)


def test_bootstrap_ci_width_shrinks_with_n() -> None:
    small = bootstrap_ci([1.0] * 5 + [0.0] * 5, n_boot=300)
    large = bootstrap_ci([1.0] * 500 + [0.0] * 500, n_boot=300)
    assert (large[1] - large[0]) < (small[1] - small[0])


# ---- labels and failures ------------------------------------------------------------------


def test_effective_label_precedence() -> None:
    assert effective_label(make_record("a", abstained=True, judge_label="incorrect")) == "abstain"
    assert effective_label(make_record("b", numeric=True, judge_label="incorrect")) == "correct"
    assert effective_label(make_record("c", numeric=False, judge_label="correct")) == "incorrect"
    assert effective_label(make_record("d", numeric=None, judge_label="correct")) == "correct"
    assert effective_label(make_record("e", numeric=None, judge_label=None)) is None
    assert effective_label(make_record("f", numeric=True, error="provider: down")) is None


def test_judge_numeric_disagreement_flag() -> None:
    assert judge_numeric_disagree(make_record("a", numeric=True, judge_label="incorrect"))
    assert not judge_numeric_disagree(make_record("b", numeric=True, judge_label="correct"))
    assert not judge_numeric_disagree(make_record("c", numeric=None, judge_label="correct"))
    assert not judge_numeric_disagree(make_record("d", abstained=True, judge_label="abstain"))


def test_classify_failure_taxonomy() -> None:
    cite = [verified_citation()]
    assert classify_failure(make_record("ok", numeric=True, citations=cite)) == "none"
    assert classify_failure(make_record("abs", abstained=True)) == "none"
    assert classify_failure(make_record("err", error="provider: 502")) == "tool_error"
    assert (
        classify_failure(make_record("bud", numeric=False, terminated_by="budget", citations=cite))
        == "budget"
    )
    assert (
        classify_failure(make_record("t", numeric=False, terminated_by="error", citations=cite))
        == "tool_error"
    )
    miss = make_record("miss", numeric=False, retrieved=[(TOP_DOC, 3)], gold=[(TOP_DOC, 1)])
    assert classify_failure(miss) == "retrieval_miss"
    assert classify_failure(make_record("uncited", numeric=False)) == "unverified_citation"
    unverified = make_record("unv", numeric=False, citations=[verified_citation(False)])
    assert classify_failure(unverified) == "unverified_citation"
    assert (
        classify_failure(make_record("calc", numeric=False, grounded=True, citations=cite))
        == "calculation_error"
    )
    reasoning = make_record("reason", numeric=None, judge_label="incorrect", citations=cite)
    assert classify_failure(reasoning) == "reasoning_error"
    closed = make_record("cb", mode="closed_book", numeric=False, retrieved=[], gold=[(TOP_DOC, 1)])
    assert classify_failure(closed) == "reasoning_error"


def test_judge_error_keeps_the_record_scored(tmp_path: Path) -> None:
    """A failed judge call is not an answer failure: the record stays completed and scorable.

    Regression: judge failures used to be folded into ``error``, which unscored the record,
    shrank ``n_completed`` and misfiled a scorable answer as ``tool_error``.
    """
    cite = [verified_citation()]
    judged_out = make_record("j1", numeric=True, citations=cite, judge_error="judge: parse")
    assert effective_label(judged_out) == "correct"
    assert classify_failure(judged_out) == "none"
    wrong = make_record("j2", numeric=False, grounded=True, citations=cite, judge_error="judge: x")
    assert classify_failure(wrong) == "calculation_error"
    # No numeric and no verdict: unscored, but still completed (not an error).
    free_text = make_record("j3", numeric=None, judge_error="judge: parse")
    assert effective_label(free_text) is None
    assert classify_failure(free_text) == "none"

    records = [
        judged_out,
        wrong,
        free_text,
        make_record("j4", numeric=True, judge_label="correct", citations=cite),
        make_record("j5", error="provider: timeout"),
    ]
    summary = summarize(write_predictions(tmp_path / "run", records), n_boot=50, seed=0)
    assert summary.n == 5 and summary.n_completed == 4
    assert summary.n_dataset is None  # no config.json: the dataset size is unknown
    m = summary.metrics
    assert m["error_rate"] == pytest.approx(1 / 5)
    assert m["judge_error_rate"] == pytest.approx(3 / 4)
    assert m["n_scored"] == 3
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["numeric_match_rate"] == pytest.approx(2 / 3)
    assert summary.failures["tool_error"] == 1
    assert summary.failures["calculation_error"] == 1
    assert summary.failures["none"] == 3


# ---- summarize ----------------------------------------------------------------------------


def test_summarize_rates_and_ci(tmp_path: Path) -> None:
    cite = [verified_citation()]
    records = [
        make_record("q1", numeric=True, judge_label="correct", citations=cite),
        make_record("q2", numeric=False, judge_label="correct", citations=cite),  # override
        make_record("q3", abstained=True, judge_label="abstain"),
        make_record("q4", numeric=None, judge_label="incorrect", retrieved=[(TOP_DOC, 9)]),
        make_record("q5", question_type="novel-generated", error="provider: timeout"),
    ]
    pred = write_predictions(tmp_path / "run", records)
    summary = summarize(pred, n_boot=200, seed=0)
    assert summary.n == 5 and summary.n_completed == 4
    m = summary.metrics
    assert m["accuracy"] == pytest.approx(1 / 4)
    assert m["abstain_rate"] == pytest.approx(1 / 4)
    assert m["hallucination_rate"] == pytest.approx(2 / 3)
    assert m["numeric_match_rate"] == pytest.approx(1 / 2)
    assert m["numeric_coverage"] == pytest.approx(2 / 4)
    assert m["page_recall_10"] == pytest.approx(3 / 4)
    assert m["citation_verified_rate"] == pytest.approx(1.0)
    assert m["error_rate"] == pytest.approx(1 / 5)
    assert m["n_scored"] == 4
    assert summary.judge_numeric_disagreements == ["q2"]
    assert summary.failures["tool_error"] == 1
    assert summary.failures["retrieval_miss"] == 1
    assert summary.failures["calculation_error"] == 1
    assert set(summary.ci95) >= {"accuracy", "abstain_rate", "page_recall_10"}
    low, high = summary.ci95["accuracy"]
    assert 0.0 <= low <= m["accuracy"] <= high <= 1.0
    assert summary.cost_total_usd == pytest.approx(0.05)
    assert summary.cost_per_q_usd == pytest.approx(0.05 / 4)
    assert summary.latency_p50_ms == pytest.approx(100.0)
    assert summary.judge_model == "anthropic:claude-test"
    assert set(summary.by_question_type) == {"metrics-generated"}  # completed records only
    assert (tmp_path / "run" / "summary.json").is_file()
    assert summarize(pred, n_boot=200, seed=0) == summary  # byte-for-byte reproducible


def test_summarize_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        summarize(tmp_path / "nope.jsonl")
