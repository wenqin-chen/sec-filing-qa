"""Deterministic evaluation metrics, bootstrap confidence intervals, failure taxonomy, summary.

Every metric here is a pure function of its inputs (no model, no network) and is unit-tested
in ``tests/eval/test_metrics.py``. Definitions (SPEC section 7):

* ``page_recall@k`` -- 1.0 when any gold ``(doc_name, page_num)`` is among the distinct pages
  of the top-``k`` retrieved pages (pages, not chunks: several chunks of one page count once).
* ``evidence_overlap_recall@k`` -- 1.0 when at least ``min_overlap`` (default 50%) of the
  characters of some evidence text are covered by one top-``k`` chunk, regardless of which page
  the chunk claims to be on. It is deliberately page-offset-robust: a systematic disagreement
  with ``page_recall`` is the signature of the 0-/1-based page bug the SPEC warns about.
* ``gold_page_mrr`` -- reciprocal of the rank (1-based) of the first gold page; 0 when absent.
* ``numeric_match`` -- strict: the model's structured ``value`` against the *single* number in
  the gold answer; ``None`` (undefined) when the gold has zero or several numbers or the model
  gave no value. Tolerance is 1% relative. Two documented equivalences and nothing else: the
  ratio/percent equivalence of :func:`secqa.core.textnum.numbers_equal` (``0.12`` vs ``12``) at
  scale 1, and a one-way unit-scale equivalence -- ``value == gold * {1e3, 1e6, 1e9}`` --
  because FinanceBench gold answers quote table figures such as ``$1577.00`` without the
  "in millions" header the filing carries while ``value`` is in base units by contract. The
  reverse direction (``value * scale == gold``) is a wrong answer and is never accepted, and the
  percent equivalence is never composed with a scale; :func:`numeric_match_scale` reports which
  scale matched so every scaled acceptance is auditable.
* Judge accuracy, abstention and hallucination follow the tri-state label; when
  ``numeric_match`` is defined it overrides the judge and every override is listed in
  ``RunSummary.judge_numeric_disagreements``.
* ``bootstrap_ci`` -- percentile bootstrap of the mean (2000 resamples, seed 0 by default),
  reported as a 95% interval; with n = 150 that is roughly +-7-8 pp on a proportion.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from secqa.core.contracts import EvalRecord, Evidence, FailureClass, Hit, RunSummary
from secqa.core.errors import ConfigError
from secqa.core.logging import get_logger
from secqa.core.textnum import extract_numbers, normalize_text, numbers_equal

log = get_logger(__name__)

EffectiveLabel = Literal["correct", "incorrect", "abstain"]

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 0
CI_LEVEL = 0.95
SCALE_FACTORS: tuple[float, ...] = (1.0, 1e3, 1e6, 1e9)
"""Unit scales a gold answer may be *understated* by (a table figure quoted without its "in
thousands / millions / billions" header). Applied one way only: ``value == gold * scale``."""

PREDICTIONS_NAME = "predictions.jsonl"
SUMMARY_NAME = "summary.json"
CONFIG_NAME = "config.json"

_CI_METRICS: tuple[str, ...] = (
    "accuracy",
    "abstain_rate",
    "hallucination_rate",
    "numeric_match_rate",
    "faithfulness",
    "citation_verified_rate",
    "grounded_rate",
    "page_recall_10",
    "overlap_recall_10",
)


# ---------------------------------------------------------------------------------------------
# retrieval metrics
# ---------------------------------------------------------------------------------------------


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an int >= 1, got {k!r}")


def distinct_pages(pages: Iterable[tuple[str, int]]) -> list[tuple[str, int]]:
    """Distinct ``(doc_name, page_num)`` pairs in first-seen order."""
    return list(dict.fromkeys((str(doc), int(page)) for doc, page in pages))


def page_recall_at_k(
    pred_pages: Sequence[tuple[str, int]], gold: Sequence[tuple[str, int]], k: int
) -> float:
    """1.0 if any gold page is among the first ``k`` distinct predicted pages, else 0.0.

    ``pred_pages`` are in rank order (duplicates are collapsed before cutting at ``k``). With no
    gold pages the metric is undefined; this returns 0.0 and callers should skip such records
    (see :func:`summarize`).
    """
    _check_k(k)
    if not gold:
        return 0.0
    wanted = {(str(doc), int(page)) for doc, page in gold}
    top = distinct_pages(pred_pages)[:k]
    return 1.0 if any(page in wanted for page in top) else 0.0


def gold_page_mrr(pred_pages: Sequence[tuple[str, int]], gold: Sequence[tuple[str, int]]) -> float:
    """Reciprocal rank of the first gold page among the distinct predicted pages (0.0 if none)."""
    if not gold:
        return 0.0
    wanted = {(str(doc), int(page)) for doc, page in gold}
    for rank, page in enumerate(distinct_pages(pred_pages), start=1):
        if page in wanted:
            return 1.0 / rank
    return 0.0


def char_overlap(evidence_text: str, chunk_text: str) -> float:
    """Fraction of the (normalised) evidence covered by the longest common run inside the chunk.

    Both texts go through :func:`secqa.core.textnum.normalize_text` first, so line breaks,
    punctuation and case never matter. The overlap is the longest common substring measured in
    characters of the evidence, divided by the evidence length (0.0 for empty evidence). A
    dynamic-programming pass over ``len(evidence) x len(chunk)`` is fine at chunk scale
    (hundreds by a few thousand characters).
    """
    needle = normalize_text(evidence_text)
    haystack = normalize_text(chunk_text)
    if not needle or not haystack:
        return 0.0
    if needle in haystack:
        return 1.0
    best = _longest_common_substring(needle, haystack)
    return best / len(needle)


def _longest_common_substring(a: str, b: str) -> int:
    """Length of the longest common substring of ``a`` and ``b`` (O(len(a)*len(b)) memory-light)."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        char_a = a[i - 1]
        for j in range(1, len(b) + 1):
            if char_a == b[j - 1]:
                value = previous[j - 1] + 1
                current[j] = value
                if value > best:
                    best = value
        previous = current
    return best


def evidence_overlap_recall(
    hits: Sequence[Hit], evidence: Sequence[Evidence], k: int, min_overlap: float = 0.5
) -> float:
    """1.0 if some top-``k`` chunk covers at least ``min_overlap`` of some evidence text.

    Chunks are compared by text only (their ``page_num`` is ignored), which makes this metric
    robust to page-offset bugs; comparing it with :func:`page_recall_at_k` on the same records
    is how the harness detects them. Evidence entries with empty text are skipped; with no
    usable evidence the result is 0.0 (callers skip such records).
    """
    _check_k(k)
    if not 0.0 < min_overlap <= 1.0:
        raise ValueError(f"min_overlap must be in (0, 1], got {min_overlap}")
    texts = [item.text for item in evidence if item.text and item.text.strip()]
    if not texts or not hits:
        return 0.0
    top = sorted(hits, key=lambda hit: hit.rank)[:k]
    for text in texts:
        for hit in top:
            if char_overlap(text, hit.chunk.text) >= min_overlap:
                return 1.0
    return 0.0


# ---------------------------------------------------------------------------------------------
# numeric match
# ---------------------------------------------------------------------------------------------


def gold_number(gold_answer: str) -> float | None:
    """The single number in a gold answer, or ``None`` when it has zero or several numbers."""
    numbers = extract_numbers(gold_answer or "")
    if len(numbers) != 1:
        return None
    return numbers[0]


def _numeric_compare(
    pred_value: float | None, gold_answer: str, rel_tol: float, scales: Sequence[float]
) -> tuple[bool | None, float | None]:
    """``(match, scale)``: the verdict and the scale it was reached at (``None`` when undefined
    or no match). Shared by :func:`numeric_match` and :func:`numeric_match_scale`."""
    if rel_tol <= 0:
        raise ValueError(f"rel_tol must be positive, got {rel_tol}")
    for scale in scales:
        if scale <= 0:
            raise ValueError(f"scales must be positive, got {scale}")
    if pred_value is None:
        return None, None
    try:
        predicted = float(pred_value)
    except (TypeError, ValueError):
        return None, None
    if math.isnan(predicted) or math.isinf(predicted):
        return None, None
    gold = gold_number(gold_answer)
    if gold is None:
        return None, None
    for scale in scales:
        if scale == 1.0:
            # Plain match: 1% relative tolerance plus the ratio/percent equivalence, because
            # gold answers write percentages both ways (12% and 0.12).
            if numbers_equal(predicted, gold, rel_tol=rel_tol):
                return True, 1.0
        elif math.isclose(predicted, gold * scale, rel_tol=rel_tol):
            # Scaled match: the gold understates a table figure by exactly this factor. No
            # percent equivalence here and never the reverse direction, otherwise values up to
            # nine orders of magnitude apart would pass a metric that overrides the judge.
            return True, scale
    return False, None


def numeric_match(
    pred_value: float | None,
    gold_answer: str,
    rel_tol: float = 0.01,
    scales: Sequence[float] = SCALE_FACTORS,
) -> bool | None:
    """Strict structured-value match against the single number of the gold answer.

    Returns ``None`` (undefined) when the model gave no value, or the gold answer does not
    contain exactly one number. Otherwise ``True`` when, for some scale in ``scales`` (checked in
    order), the value agrees with ``gold * scale`` within ``rel_tol``: at scale ``1.0`` through
    :func:`secqa.core.textnum.numbers_equal` (ratio/percent equivalence included), at any other
    scale by plain relative tolerance only. The reverse relation ``value * scale == gold`` is
    never accepted: ``value`` is in base units by contract, so a model answering ``$1.577`` to a
    gold of ``$1577.00`` (millions) is wrong. Use :func:`numeric_match_scale` to learn which
    scale matched.
    """
    match, _ = _numeric_compare(pred_value, gold_answer, rel_tol, scales)
    return match


def numeric_match_scale(
    pred_value: float | None,
    gold_answer: str,
    rel_tol: float = 0.01,
    scales: Sequence[float] = SCALE_FACTORS,
) -> float | None:
    """The unit scale at which :func:`numeric_match` accepted the value, else ``None``.

    ``1.0`` is a plain match; ``1e3`` / ``1e6`` / ``1e9`` mean the gold answer understated a
    table figure by that factor. ``None`` when the match is undefined or ``False``. Meant for
    audit trails (the rule judge's rationale) so scaled acceptances are visible, not silent.
    """
    _, scale = _numeric_compare(pred_value, gold_answer, rel_tol, scales)
    return scale


# ---------------------------------------------------------------------------------------------
# labels and failure taxonomy
# ---------------------------------------------------------------------------------------------


def effective_label(rec: EvalRecord) -> EffectiveLabel | None:
    """The tri-state outcome of one record, ``None`` when nothing could score it.

    Order: an abstention is an abstention; a defined ``numeric_match`` overrides the judge;
    otherwise the judge's label; otherwise unscored (a rule-judged free-text answer).
    Records with an ``error`` (the answer itself failed) are unscored. A ``judge_error`` does
    not unscore a record: the answer is intact, so ``numeric_match`` and abstention still
    decide, and only a free-text answer without a verdict falls through to ``None``.
    """
    if rec.error:
        return None
    if rec.abstained:
        return "abstain"
    if rec.numeric_match is True:
        return "correct"
    if rec.numeric_match is False:
        return "incorrect"
    if rec.judge is not None:
        return rec.judge.label
    return None


def judge_numeric_disagree(rec: EvalRecord) -> bool:
    """True when both the judge and ``numeric_match`` are defined and disagree."""
    if rec.judge is None or rec.numeric_match is None or rec.abstained:
        return False
    numeric_label = "correct" if rec.numeric_match else "incorrect"
    return rec.judge.label != numeric_label


def classify_failure(rec: EvalRecord) -> FailureClass:
    """Assign one failure class to a record (``'none'`` unless it is scored incorrect).

    For an incorrect answer, in this order:

    1. ``budget`` -- the loop stopped on a budget / step limit (``terminated_by``).
    2. ``tool_error`` -- the run ended in an error state or recorded an exception.
    3. ``retrieval_miss`` -- gold pages exist and none was retrieved in the top 10
       (single-shot and agent modes that retrieve; closed_book has nothing to retrieve).
    4. ``unverified_citation`` -- the answer cites nothing verified or is not grounded: the
       number came from somewhere other than the evidence shown.
    5. ``calculation_error`` -- the answer is grounded in verified evidence yet its structured
       value disagrees with the gold number: the right passages, the wrong arithmetic or
       line item.
    6. ``reasoning_error`` -- everything else (grounded text judged wrong, wrong claim).
    """
    if rec.error:
        return "tool_error"
    if effective_label(rec) != "incorrect":
        return "none"
    if rec.terminated_by in ("budget", "max_steps"):
        return "budget"
    if rec.terminated_by == "error":
        return "tool_error"
    if rec.mode != "closed_book" and rec.gold_pages and rec.page_recall_10 == 0.0:
        return "retrieval_miss"
    verified_rate = rec.citation_verified_rate
    if rec.mode != "closed_book" and (not verified_rate or not rec.grounded):
        return "unverified_citation"
    if rec.numeric_match is False and rec.grounded and rec.mode != "closed_book":
        return "calculation_error"
    return "reasoning_error"


# ---------------------------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float], n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED
) -> tuple[float, float]:
    """Percentile-bootstrap 95% interval of the mean of ``values`` (deterministic in ``seed``).

    Raises:
        ValueError: on an empty sample or non-positive ``n_boot``.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    sample = np.asarray(list(values), dtype=np.float64)
    if sample.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if sample.size == 1:
        return float(sample[0]), float(sample[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sample.size, size=(n_boot, sample.size))
    means = sample[indices].mean(axis=1)
    alpha = (1.0 - CI_LEVEL) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


# ---------------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------------


def read_records(pred_path: Path) -> list[EvalRecord]:
    """Read ``predictions.jsonl``; blank lines are skipped, a malformed line is a ConfigError."""
    pred_path = Path(pred_path)
    if not pred_path.is_file():
        raise ConfigError(f"predictions file not found: {pred_path}")
    records: list[EvalRecord] = []
    with pred_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(EvalRecord.model_validate(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"{pred_path}:{line_no}: invalid EvalRecord: {exc}") from exc
    return records


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate(flags: Sequence[bool]) -> float | None:
    return _mean([1.0 if flag else 0.0 for flag in flags]) if flags else None


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def _indicators(records: Sequence[EvalRecord]) -> dict[str, list[float]]:
    """Per-record samples behind every rate, so CIs bootstrap exactly what the mean reports."""
    labels = [(rec, effective_label(rec)) for rec in records]
    scored = [(rec, label) for rec, label in labels if label is not None]
    decided = [label for _, label in scored if label in ("correct", "incorrect")]
    answered = [rec for rec in records if not rec.abstained]
    retrieving = [rec for rec in records if rec.mode != "closed_book" and rec.gold_pages]
    return {
        "accuracy": [1.0 if label == "correct" else 0.0 for _, label in scored],
        "abstain_rate": [1.0 if rec.abstained else 0.0 for rec in records],
        "hallucination_rate": [1.0 if label == "incorrect" else 0.0 for label in decided],
        "numeric_match_rate": [
            1.0 if rec.numeric_match else 0.0 for rec in records if rec.numeric_match is not None
        ],
        "faithfulness": [
            rec.faith.score
            for rec in records
            if rec.faith is not None and rec.faith.score is not None
        ],
        "citation_verified_rate": [
            rec.citation_verified_rate for rec in answered if rec.citation_verified_rate is not None
        ],
        "grounded_rate": [1.0 if rec.grounded else 0.0 for rec in answered],
        "page_recall_5": [rec.page_recall_5 for rec in retrieving],
        "page_recall_10": [rec.page_recall_10 for rec in retrieving],
        "page_recall_20": [rec.page_recall_20 for rec in retrieving],
        "overlap_recall_10": [rec.overlap_recall_10 for rec in retrieving],
        "gold_page_mrr": [rec.gold_page_mrr for rec in retrieving],
    }


def _metrics(records: Sequence[EvalRecord], n_total: int) -> dict[str, float | None]:
    samples = _indicators(records)
    metrics: dict[str, float | None] = {name: _mean(values) for name, values in samples.items()}
    n_completed = len(records)
    labels = [effective_label(rec) for rec in records]
    metrics["n_scored"] = float(sum(label is not None for label in labels))
    metrics["numeric_coverage"] = (
        _rate([rec.numeric_match is not None for rec in records]) if records else None
    )
    metrics["faith_coverage"] = (
        _rate([rec.faith is not None and rec.faith.score is not None for rec in records])
        if records
        else None
    )
    metrics["error_rate"] = float(n_total - n_completed) / n_total if n_total else None
    # Judge failures do not remove a record from ``records`` (the answer is scored); this rate
    # makes them visible next to ``faith_coverage`` and ``n_scored`` instead.
    metrics["judge_error_rate"] = (
        _rate([rec.judge_error is not None for rec in records]) if records else None
    )
    metrics["tool_calls_mean"] = _mean([float(rec.tool_calls) for rec in records])
    metrics["steps_mean"] = _mean([float(rec.steps) for rec in records])
    metrics["retrieval_ms_p50"] = _percentile([rec.retrieval_ms for rec in records], 50)
    metrics["llm_ms_p50"] = _percentile([rec.llm_ms for rec in records], 50)
    return metrics


def _by_question_type(records: Sequence[EvalRecord]) -> dict[str, dict[str, float | None]]:
    groups: dict[str, list[EvalRecord]] = {}
    for rec in records:
        groups.setdefault(rec.question_type, []).append(rec)
    out: dict[str, dict[str, float | None]] = {}
    for question_type in sorted(groups):
        group = groups[question_type]
        samples = _indicators(group)
        out[question_type] = {
            "n": float(len(group)),
            "accuracy": _mean(samples["accuracy"]),
            "abstain_rate": _mean(samples["abstain_rate"]),
            "hallucination_rate": _mean(samples["hallucination_rate"]),
            "numeric_match_rate": _mean(samples["numeric_match_rate"]),
            "page_recall_10": _mean(samples["page_recall_10"]),
            "faithfulness": _mean(samples["faithfulness"]),
        }
    return out


def _read_config(pred_path: Path) -> dict[str, Any]:
    config_path = Path(pred_path).parent / CONFIG_NAME
    if not config_path.is_file():
        return {}
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"corrupt {config_path}: {exc}") from exc
    return raw if isinstance(raw, dict) else {}


def _parse_datetime(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return fallback
    return fallback


def summarize(
    pred_path: Path, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED
) -> RunSummary:
    """Aggregate ``predictions.jsonl`` into a :class:`RunSummary` and write ``summary.json``.

    Records with an ``error`` count towards ``n`` but not ``n_completed`` and are excluded from
    every rate; records with only a ``judge_error`` are completed and scored like any other
    (their missing verdict shows up in ``n_scored``, ``faith_coverage`` and
    ``judge_error_rate``). Retrieval metrics average over records that have gold pages and are not
    ``closed_book``; ``citation_verified_rate`` and ``grounded_rate`` average over answered
    (non-abstained) records; ``faithfulness`` over records the faithfulness judge scored.
    ``ci95`` holds percentile-bootstrap intervals for every rate with at least two samples.
    Run-level provenance comes from ``config.json`` next to the predictions when present, else
    from the records themselves.

    Raises:
        ConfigError: when the predictions file is missing or malformed.
    """
    pred_path = Path(pred_path)
    all_records = read_records(pred_path)
    config = _read_config(pred_path)
    completed = [rec for rec in all_records if not rec.error]
    n_total = int(config.get("n_questions") or len(all_records))
    n_total = max(n_total, len(all_records))

    metrics = _metrics(completed, n_total)
    samples = _indicators(completed)
    ci95: dict[str, tuple[float, float]] = {
        name: bootstrap_ci(samples[name], n_boot=n_boot, seed=seed)
        for name in _CI_METRICS
        if len(samples[name]) >= 2
    }
    failures = Counter(classify_failure(rec) for rec in all_records)
    disagreements = sorted(rec.financebench_id for rec in completed if judge_numeric_disagree(rec))

    latencies = [rec.latency_ms for rec in completed]
    cost_total = float(sum(rec.cost_usd for rec in all_records))
    judge_cost = float(sum(rec.judge_cost_usd for rec in all_records))
    first = all_records[0] if all_records else None
    judged = next((rec.judge for rec in all_records if rec.judge is not None), None)
    now = datetime.now(UTC)
    timestamps = [rec.timestamp for rec in all_records]

    summary = RunSummary(
        config_name=str(
            config.get("config", {}).get("name") or (first.config_name if first else "")
        ),
        run_id=str(config.get("run_id") or (first.run_id if first else "")),
        n=n_total,
        n_completed=len(completed),
        metrics=metrics,
        ci95=ci95,
        by_question_type=_by_question_type(completed),
        failures={name: int(count) for name, count in sorted(failures.items())},
        judge_numeric_disagreements=disagreements,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        cost_total_usd=round(cost_total, 6),
        cost_per_q_usd=round(cost_total / len(completed), 6) if completed else 0.0,
        judge_cost_usd=round(judge_cost, 6),
        provider=str(config.get("provider") or (first.provider if first else "")),
        model=str(config.get("model") or (first.model if first else "")),
        embedder=str(config.get("embedder") or (first.embedder if first else "")),
        judge_model=str(judged.judge_model if judged else config.get("judge_model") or ""),
        judge_version=str(judged.judge_version if judged else config.get("judge_version") or ""),
        git_sha=str(config.get("git_sha") or (first.git_sha if first else "")),
        index_sha=str(config.get("index_sha") or (first.index_sha if first else "")),
        prompt_hashes=dict(config.get("prompt_hashes") or (first.prompt_hashes if first else {})),
        models_yaml_as_of=str(
            config.get("models_yaml_as_of") or (first.models_yaml_as_of if first else "")
        ),
        cassettes=(str(config["cassettes"]) if config.get("cassettes") else None),
        started_at=_parse_datetime(
            config.get("started_at"), min(timestamps) if timestamps else now
        ),
        finished_at=max(timestamps) if timestamps else now,
    )
    summary_path = pred_path.parent / SUMMARY_NAME
    summary_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log.info(
        "run_summarized",
        config=summary.config_name,
        run_id=summary.run_id,
        n=summary.n,
        n_completed=summary.n_completed,
        accuracy=metrics.get("accuracy"),
        abstain_rate=metrics.get("abstain_rate"),
        page_recall_10=metrics.get("page_recall_10"),
        cost_total_usd=summary.cost_total_usd,
        path=str(summary_path),
    )
    return summary


__all__ = [
    "CONFIG_NAME",
    "DEFAULT_N_BOOT",
    "DEFAULT_SEED",
    "PREDICTIONS_NAME",
    "SCALE_FACTORS",
    "SUMMARY_NAME",
    "EffectiveLabel",
    "bootstrap_ci",
    "char_overlap",
    "classify_failure",
    "distinct_pages",
    "effective_label",
    "evidence_overlap_recall",
    "gold_number",
    "gold_page_mrr",
    "judge_numeric_disagree",
    "numeric_match",
    "numeric_match_scale",
    "page_recall_at_k",
    "read_records",
    "summarize",
]
