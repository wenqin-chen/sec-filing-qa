"""RESULTS.md rendering: golden file with every real row pending, mock rows excluded,
then a synthetic completed run rendering numbers, partial status and the provisional mark."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from secqa.core.errors import ConfigError
from secqa.eval.metrics import summarize
from secqa.eval.report import (
    PROVISIONAL_MARK,
    latest_summary,
    load_configs,
    render_results_md,
    write_results_md,
)
from tests.eval.conftest import FIXTURES, make_record, verified_citation, write_predictions

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
GOLDEN = FIXTURES / "eval_report_golden.md"
NOW = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def test_golden_all_pending(tmp_path: Path) -> None:
    """Empty results dir: every real row is pending with its reason, mock rows are footnoted."""
    rendered = render_results_md(tmp_path / "results", CONFIGS_DIR, now=NOW)
    expected = GOLDEN.read_text(encoding="utf-8")
    assert rendered == expected, (
        "report shape changed; regenerate tests/fixtures/eval_report_golden.md"
    )
    assert "pending (not run): requires ANTHROPIC_API_KEY" in rendered
    assert "pending (not run): requires OPENAI_API_KEY" in rendered
    assert "requires the FinanceBench index" in rendered
    assert "`rag_mock`" in rendered and "`agent_mock`" in rendered
    assert rendered.count("| `rag_mock`") == 0  # never a table row
    assert "complete" not in rendered.split("## Retrieval")[1].split("## Provenance")[0].replace(
        "0 complete", ""
    )


def test_completed_partial_and_provisional_rows(tmp_path: Path) -> None:
    results = tmp_path / "results"
    cite = [verified_citation()]
    records = [
        make_record("q1", numeric=True, judge_label="correct", citations=cite),
        make_record("q2", numeric=False, judge_label="incorrect", citations=cite),
        make_record("q3", abstained=True, judge_label="abstain"),
    ]
    run_dir = results / "rag_hybrid_gpt" / "abc1234_20260911-1200"
    pred = write_predictions(run_dir, records)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config": {"name": "rag_hybrid_gpt"},
                "run_id": "abc1234_20260911-1200",
                "n_questions": 4,
                "git_sha": "abc1234" + "0" * 33,
                "index_sha": "idx" + "0" * 61,
                "provider": "openai",
                "model": "gpt-test",
                "embedder": "hashing",
                "judge_model": "anthropic:claude-test",
                "judge_version": "v1",
                "models_yaml_as_of": "2026-09-11",
                "cassettes": "cassettes/abc1234_20260911-1200",
                "started_at": NOW.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    summarize(pred, n_boot=100)
    rendered = render_results_md(results, CONFIGS_DIR, now=NOW)
    row = next(line for line in rendered.splitlines() if line.startswith("| `rag_hybrid_gpt`"))
    assert "partial (3/4)" in row
    assert "`openai:gpt-test`" in row
    assert "33.3% [" in row and PROVISIONAL_MARK in row  # 1 correct of 3 scored, with CI
    assert "| 33.3% |" in row  # abstain
    assert "$0.0100" in row
    assert "| `abc1234_20260911-1200` | `abc1234` |" in rendered  # provenance row
    assert "`cassettes/abc1234_20260911-1200`" in rendered

    # a human agreement file with kappa >= 0.6 clears the provisional mark
    (run_dir / "human_agreement.json").write_text(json.dumps({"kappa": 0.75}), encoding="utf-8")
    rendered = render_results_md(results, CONFIGS_DIR, now=NOW)
    row = next(line for line in rendered.splitlines() if line.startswith("| `rag_hybrid_gpt`"))
    assert PROVISIONAL_MARK not in row

    # the latest run wins; an unreadable summary is skipped
    newer = results / "rag_hybrid_gpt" / "abc1234_20260911-1300"
    newer.mkdir()
    (newer / "summary.json").write_text("{not json", encoding="utf-8")
    found = latest_summary(results, "rag_hybrid_gpt")
    assert found is not None and found[1] == run_dir
    assert latest_summary(results, "never_run") is None

    out = write_results_md(results, CONFIGS_DIR, tmp_path / "RESULTS.md")
    assert out.read_text(encoding="utf-8").startswith("# Results")


def test_retrieval_row_shows_retrieval_metrics_only(tmp_path: Path) -> None:
    results = tmp_path / "results"
    records = [
        make_record("q1", abstained=True, retrieved=[("D", 1)], gold=[("D", 1)]),
        make_record("q2", abstained=True, retrieved=[("D", 2)], gold=[("D", 1)]),
    ]
    run_dir = results / "retrieval_bm25" / "run1"
    pred = write_predictions(run_dir, records)
    summarize(pred, n_boot=50)
    rendered = render_results_md(results, CONFIGS_DIR, now=NOW)
    row = next(line for line in rendered.splitlines() if line.startswith("| `retrieval_bm25`"))
    assert "| bm25 | 20 | 2 | 50.0% | 50.0% | 50.0% |" in row
    assert "complete" in row


def test_load_configs_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_configs(tmp_path / "nowhere")
    assert len(load_configs(CONFIGS_DIR)) == len(list(CONFIGS_DIR.glob("*.yaml")))
