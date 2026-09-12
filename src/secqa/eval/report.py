"""``RESULTS.md`` rendering from committed ``summary.json`` files (``secqa report``).

Honesty rules baked in (SPEC 7 and 9):

* The table shape is fixed by ``configs/*.yaml``, not by what has been run: a config without a
  completed run prints ``pending (not run)`` with the reason (missing key, missing index), and
  a run that stopped early prints ``partial (done/n)``, and a ``--limit`` run prints
  ``subset (done/dataset)``. Rows are never omitted.
* A row shows its widest run: a ``--limit`` smoke or pilot run finished after the full row can
  never displace it (``latest_summary`` orders by ``n``, then ``n_completed``, then time).
* Smoke configs (``mock`` / ``scripted`` providers) are excluded from the tables by design and
  listed in a footnote, so a CI number can never be mistaken for a benchmark number.
* Retrieval-only rows (``provider: mock:abstain``) show retrieval metrics only.
* Accuracy cells are marked provisional until a ``human_agreement.json`` next to the run's
  predictions reports Cohen's kappa >= 0.6.
* Judge error is shown, not assumed: the judge-agreement table prints the judge-swap kappa
  (``judge_swap_<provider>_<model>.json``) and the judge-vs-human kappa
  (``human_agreement.json``) with their ``n`` for every answering row, ``pending`` until the
  file exists (SPEC 7 and 13).
* The breakdown by ``question_type``, the failure taxonomy per incorrect answer and the
  judge / numeric side-by-side (match rate, coverage, disagreement count) come straight from
  ``summary.json``; the latency split (p50 / p95, retrieval vs LLM) and tool use likewise.
* Every number is traceable: the provenance table lists run id, git SHA, index hash, judge
  model / version, price-list date and the cassette directory for each completed row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from secqa.core.contracts import RunSummary
from secqa.core.errors import ConfigError
from secqa.core.logging import get_logger
from secqa.eval.judge import HUMAN_AGREEMENT_NAME, PROVISIONAL_KAPPA
from secqa.eval.metrics import SUMMARY_NAME
from secqa.eval.runner import EvalConfig

log = get_logger(__name__)

PROVISIONAL_MARK = "†"
DASH = "—"
_MODE_ORDER = {"closed_book": 0, "oracle": 1, "rag": 2, "agent": 3}
_STRATEGY_ORDER = {"bm25": 0, "dense": 1, "hybrid": 2}

RETRIEVAL_COLUMNS: tuple[str, ...] = (
    "Config",
    "Strategy",
    "k",
    "n",
    "Page recall@5",
    "Page recall@10",
    "Page recall@20",
    "Overlap recall@10",
    "Gold page MRR",
    "Retrieval p50 ms",
    "Status",
)
LLM_COLUMNS: tuple[str, ...] = (
    "Config",
    "Mode",
    "Provider:model",
    "n",
    "Accuracy (95% CI)",
    "Abstain",
    "Hallucination",
    "Faithfulness",
    "Citations verified",
    "Grounded",
    "Page recall@10",
    "p50 ms",
    "$/question",
    "Status",
)
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "Config",
    "Run id",
    "Git SHA",
    "Index SHA",
    "Judge",
    "Judge version",
    "Prices as of",
    "Cassettes",
)
AGREEMENT_COLUMNS: tuple[str, ...] = (
    "Config",
    "Judge",
    "Swap judge",
    "Swap n",
    "Swap kappa",
    "Swap agreement",
    "Human n",
    "Human kappa",
    "Human agreement",
)
QUESTION_TYPE_COLUMNS: tuple[str, ...] = (
    "Config",
    "Question type",
    "n",
    "Accuracy",
    "Abstain",
    "Hallucination",
    "Numeric match",
    "Page recall@10",
    "Faithfulness",
)
FAILURE_CLASSES: tuple[str, ...] = (
    "retrieval_miss",
    "reasoning_error",
    "calculation_error",
    "tool_error",
    "budget",
    "unverified_citation",
)
FAILURE_COLUMNS: tuple[str, ...] = (
    "Config",
    "Incorrect",
    *FAILURE_CLASSES,
    "Numeric match",
    "Numeric coverage",
    "Judge/numeric disagreements",
)
LATENCY_COLUMNS: tuple[str, ...] = (
    "Config",
    "p50 ms",
    "p95 ms",
    "Retrieval p50 ms",
    "LLM p50 ms",
    "Tool calls (mean)",
    "Steps (mean)",
    "Judge cost",
)
JUDGE_SWAP_GLOB = "judge_swap_*.json"
PENDING = "pending"


@dataclass(frozen=True)
class _Agreement:
    """The fields of an ``AgreementReport`` JSON file that the report shows.

    Reading is tolerant: a file written by an older version (or by hand for a test) may carry
    only ``kappa``; every other field then renders as a dash.
    """

    name_b: str
    n: int | None
    kappa: float | None
    agreement: float | None


@dataclass(frozen=True)
class _Row:
    cfg: EvalConfig
    summary: RunSummary | None
    run_dir: Path | None
    human: _Agreement | None
    swaps: tuple[_Agreement, ...]

    @property
    def human_kappa(self) -> float | None:
        """Judge-vs-human Cohen's kappa, or ``None`` when no usable ``human_agreement.json``."""
        return self.human.kappa if self.human is not None else None


# ---------------------------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------------------------


def load_configs(configs_dir: Path) -> list[EvalConfig]:
    """Every ``*.yaml`` under ``configs_dir`` as an :class:`EvalConfig`, sorted by file name."""
    configs_dir = Path(configs_dir)
    if not configs_dir.is_dir():
        raise ConfigError(f"configs directory not found: {configs_dir}")
    return [EvalConfig.from_yaml(path) for path in sorted(configs_dir.glob("*.yaml"))]


def latest_summary(results_dir: Path, name: str) -> tuple[RunSummary, Path] | None:
    """The ``summary.json`` under ``results/<name>/*/`` that the row should show (or ``None``).

    Widest run first (``n``: a ``--limit`` subset never displaces a full row), then the most
    completed, then the most recently finished; the run id breaks exact ties.
    """
    root = Path(results_dir) / name
    if not root.is_dir():
        return None
    found: list[tuple[int, int, datetime, str, RunSummary, Path]] = []
    for path in root.glob(f"*/{SUMMARY_NAME}"):
        try:
            summary = RunSummary.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            log.warning("summary_unreadable", path=str(path), error=str(exc))
            continue
        found.append(
            (
                summary.n,
                summary.n_completed,
                summary.finished_at,
                path.parent.name,
                summary,
                path.parent,
            )
        )
    if not found:
        return None
    found.sort(key=lambda item: item[:4])
    summary, run_dir = found[-1][4], found[-1][5]
    return summary, run_dir


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _read_agreement(path: Path, default_name: str) -> _Agreement | None:
    """Parse one agreement JSON file; ``None`` (with a warning) when it is missing or corrupt."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("agreement_unreadable", path=str(path), error=str(exc))
        return None
    if not isinstance(raw, dict):
        log.warning("agreement_unreadable", path=str(path), error="not a JSON object")
        return None
    name_b = raw.get("name_b")
    return _Agreement(
        name_b=name_b if isinstance(name_b, str) and name_b else default_name,
        n=_as_int(raw.get("n")),
        kappa=_as_float(raw.get("kappa")),
        agreement=_as_float(raw.get("agreement")),
    )


def _human_agreement(run_dir: Path | None) -> _Agreement | None:
    if run_dir is None:
        return None
    return _read_agreement(run_dir / HUMAN_AGREEMENT_NAME, "human")


def _judge_swaps(run_dir: Path | None) -> tuple[_Agreement, ...]:
    """Every ``judge_swap_<provider>_<model>.json`` next to the run's predictions, by file name."""
    if run_dir is None:
        return ()
    found: list[_Agreement] = []
    for path in sorted(run_dir.glob(JUDGE_SWAP_GLOB)):
        report = _read_agreement(path, path.stem.removeprefix("judge_swap_"))
        if report is not None:
            found.append(report)
    return tuple(found)


def _sort_key(cfg: EvalConfig) -> tuple[int, int, int, str]:
    kind = 0 if cfg.row_kind == "retrieval" else 1
    return (kind, _STRATEGY_ORDER.get(cfg.strategy, 9), _MODE_ORDER.get(cfg.mode, 9), cfg.name)


# ---------------------------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return DASH if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 0) -> str:
    return DASH if value is None else f"{value:,.{digits}f}"


def _usd(value: float | None) -> str:
    return DASH if value is None else f"${value:.4f}"


def _status(row: _Row) -> str:
    if row.summary is None:
        return f"pending (not run): {row.cfg.pending_reason}"
    n_dataset = row.summary.n_dataset
    if n_dataset is not None and row.summary.n < n_dataset:
        return f"subset ({row.summary.n_completed}/{n_dataset})"
    if row.summary.n_completed < row.summary.n:
        return f"partial ({row.summary.n_completed}/{row.summary.n})"
    return "complete"


def _accuracy_cell(row: _Row) -> str:
    summary = row.summary
    if summary is None:
        return DASH
    accuracy = summary.metrics.get("accuracy")
    if accuracy is None:
        return DASH
    ci = summary.ci95.get("accuracy")
    text = _pct(accuracy)
    if ci is not None:
        text += f" [{_pct(ci[0])}, {_pct(ci[1])}]"
    if row.human_kappa is None or row.human_kappa < PROVISIONAL_KAPPA:
        text += PROVISIONAL_MARK
    return text


def _table(columns: tuple[str, ...], rows: list[list[str]]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(" --- " for _ in columns) + "|"
    body = ["| " + " | ".join(cells) + " |" for cells in rows]
    return [header, divider, *body]


def _retrieval_cells(row: _Row) -> list[str]:
    metrics = row.summary.metrics if row.summary else {}
    return [
        f"`{row.cfg.name}`",
        row.cfg.strategy,
        str(row.cfg.k),
        str(row.summary.n_completed) if row.summary else DASH,
        _pct(metrics.get("page_recall_5")),
        _pct(metrics.get("page_recall_10")),
        _pct(metrics.get("page_recall_20")),
        _pct(metrics.get("overlap_recall_10")),
        _num(metrics.get("gold_page_mrr"), 3),
        _num(metrics.get("retrieval_ms_p50")),
        _status(row),
    ]


def _llm_cells(row: _Row) -> list[str]:
    summary = row.summary
    metrics = summary.metrics if summary else {}
    provider = f"{summary.provider}:{summary.model}" if summary else row.cfg.provider
    return [
        f"`{row.cfg.name}`",
        row.cfg.mode,
        f"`{provider}`",
        str(summary.n_completed) if summary else DASH,
        _accuracy_cell(row),
        _pct(metrics.get("abstain_rate")),
        _pct(metrics.get("hallucination_rate")),
        _pct(metrics.get("faithfulness")),
        _pct(metrics.get("citation_verified_rate")) if row.cfg.mode != "closed_book" else "n/a",
        _pct(metrics.get("grounded_rate")),
        _pct(metrics.get("page_recall_10")) if row.cfg.mode != "closed_book" else "n/a",
        _num(summary.latency_p50_ms) if summary else DASH,
        _usd(summary.cost_per_q_usd) if summary else DASH,
        _status(row),
    ]


def _provenance_cells(row: _Row) -> list[str]:
    summary = row.summary
    assert summary is not None
    return [
        f"`{row.cfg.name}`",
        f"`{summary.run_id}`",
        f"`{summary.git_sha[:7]}`" if summary.git_sha else DASH,
        f"`{summary.index_sha[:12]}`" if summary.index_sha else DASH,
        f"`{summary.judge_model}`" if summary.judge_model else DASH,
        summary.judge_version or DASH,
        summary.models_yaml_as_of or DASH,
        f"`{summary.cassettes}`" if summary.cassettes else "none",
    ]


def _kappa(value: float | None) -> str:
    return DASH if value is None else f"{value:.3f}"


def _agreement_cells(report: _Agreement | None) -> list[str]:
    """``n`` / kappa / agreement for one agreement file, or three ``pending`` cells."""
    if report is None:
        return [PENDING, PENDING, PENDING]
    return [_num(report.n), _kappa(report.kappa), _pct(report.agreement)]


def _agreement_rows(row: _Row) -> list[list[str]]:
    """One row per judge-swap file (one ``pending`` row when there is none), human kappa on each.

    A config that has not been run is still a row: every agreement cell reads ``pending``, so a
    missing kappa can never be mistaken for a measured one.
    """
    judge = row.summary.judge_model if row.summary and row.summary.judge_model else row.cfg.judge
    human = _agreement_cells(row.human)
    swaps: list[_Agreement | None] = list(row.swaps) or [None]
    return [
        [
            f"`{row.cfg.name}`",
            f"`{judge}`",
            f"`{swap.name_b}`" if swap is not None else PENDING,
            *_agreement_cells(swap),
            *human,
        ]
        for swap in swaps
    ]


def _question_type_rows(row: _Row) -> list[list[str]]:
    summary = row.summary
    assert summary is not None
    rows: list[list[str]] = []
    for question_type, group in summary.by_question_type.items():
        n = group.get("n")
        rows.append(
            [
                f"`{row.cfg.name}`",
                question_type,
                _num(n) if n is not None else DASH,
                _pct(group.get("accuracy")),
                _pct(group.get("abstain_rate")),
                _pct(group.get("hallucination_rate")),
                _pct(group.get("numeric_match_rate")),
                _pct(group.get("page_recall_10")),
                _pct(group.get("faithfulness")),
            ]
        )
    return rows


def _failure_cells(row: _Row) -> list[str]:
    summary = row.summary
    assert summary is not None
    counts = {name: int(count) for name, count in summary.failures.items()}
    incorrect = sum(count for name, count in counts.items() if name != "none")
    return [
        f"`{row.cfg.name}`",
        str(incorrect),
        *(str(counts.get(name, 0)) for name in FAILURE_CLASSES),
        _pct(summary.metrics.get("numeric_match_rate")),
        _pct(summary.metrics.get("numeric_coverage")),
        str(len(summary.judge_numeric_disagreements)),
    ]


def _latency_cells(row: _Row) -> list[str]:
    summary = row.summary
    assert summary is not None
    metrics = summary.metrics
    return [
        f"`{row.cfg.name}`",
        _num(summary.latency_p50_ms),
        _num(summary.latency_p95_ms),
        _num(metrics.get("retrieval_ms_p50")),
        _num(metrics.get("llm_ms_p50")),
        _num(metrics.get("tool_calls_mean"), 2),
        _num(metrics.get("steps_mean"), 2),
        _usd(summary.judge_cost_usd),
    ]


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------


def render_results_md(results_dir: Path, configs_dir: Path, *, now: datetime | None = None) -> str:
    """Render the results document (Markdown) for every config under ``configs_dir``.

    ``now`` fixes the generation timestamp (tests use it for a byte-stable golden file).
    """
    results_dir = Path(results_dir)
    configs = load_configs(configs_dir)
    rows: list[_Row] = []
    smoke: list[str] = []
    for cfg in sorted(configs, key=_sort_key):
        if cfg.row_kind == "smoke":
            smoke.append(cfg.name)
            continue
        found = latest_summary(results_dir, cfg.name)
        summary, run_dir = found if found else (None, None)
        rows.append(
            _Row(
                cfg=cfg,
                summary=summary,
                run_dir=run_dir,
                human=_human_agreement(run_dir),
                swaps=_judge_swaps(run_dir),
            )
        )

    retrieval_rows = [row for row in rows if row.cfg.row_kind == "retrieval"]
    llm_rows = [row for row in rows if row.cfg.row_kind == "llm"]
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    n_complete = sum(1 for row in rows if _status(row) == "complete")
    n_pending = sum(1 for row in rows if row.summary is None)

    lines: list[str] = [
        "# Results",
        "",
        f"Generated by `secqa report` on {stamp} from `{results_dir.name}/` "
        f"({n_complete} complete, {n_pending} pending of {len(rows)} rows).",
        "",
        "Every number below comes from a committed `summary.json`; pending rows have not been "
        "run, partial rows stopped early and subset rows were run with `--limit`. FinanceBench "
        "open set, 150 questions, one fixed "
        "judge per row. 95% intervals are percentile bootstraps (2000 resamples, seed 0): at "
        "n = 150 they span roughly ±7–8 pp, and overlapping intervals are not evidence of a "
        "difference.",
        "",
        "## Retrieval (no answering model)",
        "",
    ]
    if retrieval_rows:
        lines.extend(_table(RETRIEVAL_COLUMNS, [_retrieval_cells(row) for row in retrieval_rows]))
    else:
        lines.append("_No retrieval-only configs._")
    lines.extend(["", "## Question answering", ""])
    if llm_rows:
        lines.extend(_table(LLM_COLUMNS, [_llm_cells(row) for row in llm_rows]))
    else:
        lines.append("_No answering configs._")
    lines.extend(
        [
            "",
            f"{PROVISIONAL_MARK} provisional: judge-vs-human agreement (Cohen's kappa) is "
            f"missing or below {PROVISIONAL_KAPPA:.1f} for this run; see `human_agreement.json`.",
            "Hallucination = incorrect / (correct + incorrect). Faithfulness = supported claims "
            "/ claims, judged against cited passages only. `n/a` = not defined for the mode.",
            "",
            "## Judge agreement",
            "",
        ]
    )
    completed = [row for row in rows if row.summary is not None]
    completed_llm = [row for row in completed if row.cfg.row_kind == "llm"]
    if llm_rows:
        agreement_rows = [cells for row in llm_rows for cells in _agreement_rows(row)]
        lines.extend(_table(AGREEMENT_COLUMNS, agreement_rows))
    else:
        lines.append("_No answering configs._")
    lines.extend(
        [
            "",
            "Cohen's kappa between the row's effective labels (judge, numeric override "
            "included) and a second judge (`judge_swap_<provider>_<model>.json`, written by "
            "`secqa.eval.judge.judge_swap`) or the human-labelled subset "
            "(`human_agreement.json`, written by `secqa.eval.judge.human_agreement`); "
            "agreement = raw label agreement. `pending` = not yet computed for this row.",
            "",
            "## Breakdown by question type",
            "",
        ]
    )
    if completed:
        by_type = [cells for row in completed for cells in _question_type_rows(row)]
        lines.extend(_table(QUESTION_TYPE_COLUMNS, by_type))
    else:
        lines.append("_No completed runs._")
    lines.extend(["", "## Failure taxonomy and judge/numeric agreement", ""])
    if completed_llm:
        lines.extend(_table(FAILURE_COLUMNS, [_failure_cells(row) for row in completed_llm]))
        lines.extend(
            [
                "",
                "One failure class per incorrect answer (Incorrect = their sum). Numeric match "
                "= strict structured-value match where the gold answer has exactly one number "
                "(coverage = share of questions where it is defined); judge/numeric "
                "disagreements = questions where the judge label and `numeric_match` conflict "
                "(`numeric_match` wins; ids in `summary.json`).",
            ]
        )
    else:
        lines.append("_No completed answering runs._")
    lines.extend(["", "## Latency and tool use", ""])
    if completed_llm:
        lines.extend(_table(LATENCY_COLUMNS, [_latency_cells(row) for row in completed_llm]))
        lines.extend(
            [
                "",
                "Wall-clock per question (LLM cache off during timed runs); retrieval and LLM "
                "p50 are the split of the same questions. Judge cost is total per row and is "
                "not part of `$/question`.",
            ]
        )
    else:
        lines.append("_No completed answering runs._")
    lines.extend(["", "## Provenance", ""])
    if completed:
        lines.extend(_table(PROVENANCE_COLUMNS, [_provenance_cells(row) for row in completed]))
    else:
        lines.append("_No completed runs._")
    lines.extend(["", "## Excluded by design", ""])
    if smoke:
        names = ", ".join(f"`{name}`" for name in smoke)
        lines.append(
            f"Smoke configs run with a mock or scripted provider ({names}) exercise the harness "
            "in CI and never appear in the tables above."
        )
    else:
        lines.append("_No smoke configs._")
    lines.append("")
    return "\n".join(lines)


def write_results_md(results_dir: Path, configs_dir: Path, out_path: Path) -> Path:
    """Render and write ``RESULTS.md``; returns ``out_path``."""
    out_path = Path(out_path)
    out_path.write_text(render_results_md(results_dir, configs_dir), encoding="utf-8")
    log.info("results_md_written", path=str(out_path))
    return out_path


def summary_dict(summary: RunSummary) -> dict[str, Any]:
    """JSON-ready view of a summary (used by the CLI to print a run)."""
    return summary.model_dump(mode="json")


__all__ = [
    "AGREEMENT_COLUMNS",
    "DASH",
    "FAILURE_CLASSES",
    "FAILURE_COLUMNS",
    "JUDGE_SWAP_GLOB",
    "LATENCY_COLUMNS",
    "LLM_COLUMNS",
    "PENDING",
    "PROVENANCE_COLUMNS",
    "PROVISIONAL_MARK",
    "QUESTION_TYPE_COLUMNS",
    "RETRIEVAL_COLUMNS",
    "latest_summary",
    "load_configs",
    "render_results_md",
    "summary_dict",
    "write_results_md",
]
