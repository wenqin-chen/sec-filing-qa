"""Rescore: record a run with a priced provider into cassettes, then replay it with no
provider at all (a stub that raises if reached) and get identical metrics; a missing cassette
fails loudly with CassetteMiss; mock runs are simply re-run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from secqa.core.contracts import FBQuestion
from secqa.core.errors import CassetteMiss, ConfigError
from secqa.eval.metrics import read_records
from secqa.eval.rescore import PREVIOUS_PREDICTIONS_NAME, ReplayStub, read_run_config, rescore
from secqa.eval.runner import EvalConfig, run_eval
from secqa.providers.pricing import PriceTable
from secqa.store import DuckDBStore
from tests.eval.conftest import FixedProvider

STRIP = {"timestamp", "latency_ms", "retrieval_ms", "llm_ms"}


def _closed_book_cfg(**overrides: object) -> EvalConfig:
    base = {
        "name": "closed_book_test",
        "mode": "closed_book",
        "provider": "openai:gpt-test",
        "embedder": "hashing:64",
        "judge": "anthropic:claude-test",
        "cassette_mode": "record",
        "limit": 3,
    }
    base.update(overrides)
    return EvalConfig.model_validate(base)


def _providers() -> tuple[FixedProvider, FixedProvider]:
    parsed = {
        "answer": "Total net sales were $1,577 million.",
        "value": 1_577_000_000,
        "unit": "USD",
        "citations": [],
        "abstain": False,
    }
    answering = FixedProvider(
        parsed=parsed, text=json.dumps(parsed), provider="openai", model="gpt-test"
    )
    judge = FixedProvider(
        text='{"label": "correct", "rationale": "same figure"}',
        provider="anthropic",
        model="claude-test",
    )
    return answering, judge


def _records(run_dir: Path) -> list[dict]:
    return [rec.model_dump(exclude=STRIP) for rec in read_records(run_dir / "predictions.jsonl")]


def test_rescore_replays_cassettes_without_a_provider(
    store: DuckDBStore, questions: list[FBQuestion], prices: PriceTable, tmp_path: Path
) -> None:
    answering, judge = _providers()
    run_dir = run_eval(
        _closed_book_cfg(),
        questions,
        store,
        out_dir=tmp_path / "results",
        prices=prices,
        provider=answering,
        judge=judge,
        cassette_dir=tmp_path / "cassettes",
    )
    config = read_run_config(run_dir)
    cassettes = Path(config["cassettes"])
    assert cassettes == tmp_path / "cassettes" / run_dir.name
    assert len(list(cassettes.glob("*.json"))) >= 3  # answers + judge calls
    original = _records(run_dir)
    n_answer_calls, n_judge_calls = len(answering.calls), len(judge.calls)

    summary = rescore(run_dir, questions, store=store, prices=prices)
    assert summary.n == 3 and summary.n_completed == 3 and summary.run_id == run_dir.name
    assert _records(run_dir) == original  # every metric regenerated from cassettes
    assert (run_dir / PREVIOUS_PREDICTIONS_NAME).is_file()
    assert len(answering.calls) == n_answer_calls and len(judge.calls) == n_judge_calls
    assert read_run_config(run_dir)["rescored_at"]
    assert read_run_config(run_dir)["rescore_judge"] == "anthropic:claude-test"

    # a new judge is called for real (recorded into the same cassette dir)
    swap = FixedProvider(text='{"label": "incorrect", "rationale": "x"}', model="claude-swap")
    summary = rescore(run_dir, questions, judge=swap, store=store, prices=prices)
    assert len(swap.calls) == 3
    assert all(
        rec.judge and rec.judge.judge_model == "anthropic:claude-swap"
        for rec in read_records(run_dir / "predictions.jsonl")
    )
    assert summary.judge_model == "anthropic:claude-swap"


def test_rescore_missing_cassette_is_loud(
    store: DuckDBStore, questions: list[FBQuestion], prices: PriceTable, tmp_path: Path
) -> None:
    answering, judge = _providers()
    run_dir = run_eval(
        _closed_book_cfg(),
        questions,
        store,
        out_dir=tmp_path / "results",
        prices=prices,
        provider=answering,
        judge=judge,
        cassette_dir=tmp_path / "cassettes",
    )
    cassettes = Path(read_run_config(run_dir)["cassettes"])
    for entry in list(cassettes.glob("*.json"))[:1]:
        entry.unlink()
    with pytest.raises(CassetteMiss):
        rescore(run_dir, questions, store=store, prices=prices)
    shutil.rmtree(cassettes)
    with pytest.raises(ConfigError, match="cassette directory not found"):
        rescore(run_dir, questions, store=store, prices=prices)


def test_rescore_mock_run_reruns_deterministically(
    store: DuckDBStore, questions: list[FBQuestion], prices: PriceTable, tmp_path: Path
) -> None:
    cfg = EvalConfig.model_validate(
        {
            "name": "rag_mock",
            "mode": "rag",
            "provider": "mock",
            "embedder": "hashing:64",
            "k": 4,
            "judge": "rule",
            "cassette_mode": "off",
        }
    )
    run_dir = run_eval(cfg, questions, store, out_dir=tmp_path / "results", prices=prices)
    original = _records(run_dir)
    summary = rescore(run_dir, questions, store=store, prices=prices)
    assert _records(run_dir) == original and summary.n_completed == 6
    with pytest.raises(ConfigError, match="not in the loaded dataset"):
        rescore(run_dir, questions[:2], store=store, prices=prices)
    with pytest.raises(ConfigError, match="run config not found"):
        rescore(tmp_path / "nowhere", questions, store=store, prices=prices)


def test_replay_stub_never_answers() -> None:
    from secqa.core.errors import ProviderError

    stub = ReplayStub("openai", "gpt-test")
    with pytest.raises(ProviderError, match="cassettes only"):
        stub.complete([])
