"""``RESULTS.md`` rendering from committed ``summary.json`` files (``secqa report``).

Honesty rules baked in (SPEC 7 and 9):

* The table shape is fixed by ``configs/*.yaml``, not by what has been run: a config without a
  completed run prints ``pending (not run)`` with the reason (missing key, missing index), and
  a run that stopped early prints ``partial (done/n)``. Rows are never omitted.
* Smoke configs (``mock`` / ``scripted`` providers) are excluded from the tables by design and
  listed in a footnote, so a CI number can never be mistaken for a benchmark number.
* Retrieval-only rows (``provider: mock:abstain``) show retrieval metrics only.
* Accuracy cells are marked provisional until a ``human_agreement.json`` next to the run's
  predictions reports Cohen's kappa >= 0.6.
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


@dataclass(frozen=True)
class _Row:
    cfg: EvalConfig
    summary: RunSummary | None
    run_dir: Path | None
    human_kappa: float | None


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
    """The most recently finished ``summary.json`` under ``results/<name>/*/`` (or ``None``)."""
    root = Path(results_dir) / name
    if not root.is_dir():
        return None
    found: list[tuple[datetime, str, RunSummary, Path]] = []
    for path in root.glob(f"*/{SUMMARY_NAME}"):
        try:
            summary = RunSummary.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            log.warning("summary_unreadable", path=str(path), error=str(exc))
            continue
        found.append((summary.finished_at, path.parent.name, summary, path.parent))
    if not found:
        return None
    found.sort(key=lambda item: (item[0], item[1]))
    _, _, summary, run_dir = found[-1]
    return summary, run_dir


def _human_kappa(run_dir: Path | None) -> float | None:
    if run_dir is None:
        return None
    path = run_dir / HUMAN_AGREEMENT_NAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    kappa = raw.get("kappa") if isinstance(raw, dict) else None
    return float(kappa) if isinstance(kappa, int | float) else None


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
            _Row(cfg=cfg, summary=summary, run_dir=run_dir, human_kappa=_human_kappa(run_dir))
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
        "run and partial rows stopped early. FinanceBench open set, 150 questions, one fixed "
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
            "## Provenance",
            "",
        ]
    )
    completed = [row for row in rows if row.summary is not None]
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
    "DASH",
    "LLM_COLUMNS",
    "PROVENANCE_COLUMNS",
    "PROVISIONAL_MARK",
    "RETRIEVAL_COLUMNS",
    "latest_summary",
    "load_configs",
    "render_results_md",
    "summary_dict",
    "write_results_md",
]
