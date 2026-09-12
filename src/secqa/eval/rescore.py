"""Offline re-scoring of a published run from its cassettes (``secqa rescore``).

A real run records every LLM and judge call under ``cassettes/<run_id>/``. Re-scoring rebuilds
the exact pipeline of the run (same config, same index, same prompts) with providers that only
*replay*: :class:`~secqa.providers.ReplayCacheProvider` in ``replay`` mode over a stub that can
never reach the network. Every metric, interval and the summary regenerate with zero API keys;
a missing recording raises :class:`~secqa.core.errors.CassetteMiss` loudly instead of paying.

Passing a ``judge`` re-judges with that model (recording new cassettes into the same directory)
-- the path a judge-prompt revision takes. Mock and scripted providers are deterministic and are
simply re-run.

Timings are the one thing a replay cannot regenerate: a cassette hit answers in microseconds, so
``latency_ms`` / ``retrieval_ms`` / ``llm_ms`` of a cassette-served rescore are carried over from
the previous ``predictions.jsonl`` (the original run's measurement) and ``config.json`` records
``rescore_timings = "carried_over"``. Mock and scripted re-runs are measured afresh
(``"remeasured"``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from secqa.core.contracts import (
    Embedder,
    EvalRecord,
    FBQuestion,
    LLMProvider,
    LLMResponse,
    Message,
    RunSummary,
    ToolSpec,
)
from secqa.core.errors import ConfigError, ProviderError
from secqa.core.logging import get_logger
from secqa.core.settings import get_settings, provider_vendor
from secqa.eval.judge import RULE_JUDGE, Judge, LLMJudge, RuleJudge
from secqa.eval.metrics import CONFIG_NAME, PREDICTIONS_NAME, SUMMARY_NAME, read_records, summarize
from secqa.eval.runner import EvalConfig, run_eval
from secqa.providers import ReplayCacheProvider, get_provider
from secqa.providers.base import BaseProvider, Effort
from secqa.providers.pricing import PriceTable
from secqa.store import DuckDBStore

log = get_logger(__name__)

PREVIOUS_PREDICTIONS_NAME = "predictions.previous.jsonl"
TIMING_FIELDS: tuple[str, ...] = ("latency_ms", "retrieval_ms", "llm_ms")
RescoreTimings = Literal["carried_over", "remeasured"]


class ReplayStub(BaseProvider):
    """A provider that must never be called: the cassette answers or the run fails."""

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        effort: Effort | None = None,
    ) -> LLMResponse:
        raise ProviderError(
            "replay stub reached: rescore must be served from cassettes only",
            retryable=False,
            provider=self.provider,
        )


def read_run_config(run_dir: Path) -> dict[str, Any]:
    """``config.json`` of a run directory (``ConfigError`` when missing or malformed)."""
    path = Path(run_dir) / CONFIG_NAME
    if not path.is_file():
        raise ConfigError(f"run config not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"corrupt {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("config"), dict):
        raise ConfigError(f"{path} does not look like a run config")
    return raw


def _split_spec(spec: str) -> tuple[str, str]:
    vendor = provider_vendor(spec)
    _, _, model = spec.partition(":")
    return vendor, model.strip()


def _replay_provider(vendor: str, model: str, spec: str, cassette_dir: Path | None) -> LLMProvider:
    if vendor in ("mock", "scripted"):
        return get_provider(spec, get_settings().model_copy(update={"cassette_mode": "off"}))
    if cassette_dir is None:
        raise ConfigError(f"rescoring {spec!r} needs the run's cassettes, but none were recorded")
    if not Path(cassette_dir).is_dir():
        raise ConfigError(f"cassette directory not found: {cassette_dir}")
    return ReplayCacheProvider(
        ReplayStub(vendor, model), cache_dir=Path(cassette_dir), mode="replay"
    )


def carry_over_timings(pred_path: Path, previous_path: Path) -> int:
    """Copy the per-question timings of ``previous_path`` onto ``pred_path`` in place.

    A replayed answer was never timed: the provider returned a cassette entry in microseconds, so
    the only real measurement of that question is the one the previous predictions hold. Records
    are matched by ``financebench_id``; a record without a completed counterpart (either side
    carries an ``error``) keeps its own timings. Returns the number of records updated.
    """
    previous = {
        rec.financebench_id: rec for rec in read_records(previous_path) if rec.error is None
    }
    records = read_records(pred_path)
    updated: list[EvalRecord] = []
    n_updated = 0
    for rec in records:
        prior = previous.get(rec.financebench_id)
        if prior is None or rec.error is not None:
            updated.append(rec)
            continue
        updated.append(
            rec.model_copy(update={name: getattr(prior, name) for name in TIMING_FIELDS})
        )
        n_updated += 1
    with pred_path.open("w", encoding="utf-8") as fh:
        for rec in updated:
            fh.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")
    log.info("rescore_timings_carried_over", n_updated=n_updated, n_records=len(records))
    return n_updated


def rescore(
    run_dir: Path,
    questions: list[FBQuestion],
    judge: LLMProvider | None = None,
    *,
    store: DuckDBStore | None = None,
    prices: PriceTable | None = None,
    cassette_dir: Path | None = None,
    embedder: Embedder | None = None,
) -> RunSummary:
    """Recompute ``predictions.jsonl`` and ``summary.json`` of ``run_dir`` from cassettes.

    Args:
        run_dir: ``results/<config>/<run_id>/`` of a finished (or partial) run.
        questions: The locally loaded dataset; the run's ``question_ids`` select and order them.
        judge: Optional new judge model. ``None`` replays the original judge from cassettes
            (or re-runs the rule judge).
        store: The index the run used; opened read-only from the config's ``index_path`` or
            ``Settings.duckdb_path`` when omitted.
        prices: Price table override (default ``models.yaml``).
        cassette_dir: Where this run's cassettes live (default: the path recorded in
            ``config.json``, e.g. after unpacking ``cassettes/<run_id>.tar.zst`` in place).
        embedder: Embedder override for the retrieving modes.

    The previous predictions are kept as ``predictions.previous.jsonl`` for diffing. When the
    answers are served from cassettes, the per-question timings (``latency_ms``,
    ``retrieval_ms``, ``llm_ms``) are carried over from them: a replay cannot re-measure a call it
    never made. ``config.json`` records which happened in ``rescore_timings``.

    Raises:
        ConfigError: missing config, index, cassettes or questions.
        CassetteMiss: a call that was never recorded (replay never touches the network).
    """
    run_dir = Path(run_dir)
    raw = read_run_config(run_dir)
    original = EvalConfig.model_validate(raw["config"])
    cfg = original.model_copy(update={"cassette_mode": "off"})  # providers below are pre-wrapped
    cassettes = (
        Path(cassette_dir)
        if cassette_dir is not None
        else (Path(raw["cassettes"]) if raw.get("cassettes") else None)
    )

    wanted_ids = [str(qid) for qid in raw.get("question_ids") or []]
    by_id = {q.id: q for q in questions}
    missing = [qid for qid in wanted_ids if qid not in by_id]
    if missing:
        raise ConfigError(
            f"rescore: {len(missing)} question id(s) not in the loaded dataset: {missing[:5]}"
        )
    selected = [by_id[qid] for qid in wanted_ids] if wanted_ids else list(questions)

    vendor, _ = _split_spec(original.provider)
    provider = _replay_provider(vendor, str(raw.get("model") or ""), original.provider, cassettes)
    resolved_judge: Judge
    if judge is not None:
        wrapped = judge
        if cassettes is not None and not isinstance(judge, ReplayCacheProvider):
            wrapped = ReplayCacheProvider(judge, cache_dir=cassettes, mode="record")
        resolved_judge = LLMJudge(wrapped)
    elif original.judge == RULE_JUDGE:
        resolved_judge = RuleJudge()
    else:
        judge_vendor, judge_model = _split_spec(str(raw.get("judge_model") or original.judge))
        resolved_judge = LLMJudge(
            _replay_provider(
                judge_vendor, judge_model, str(raw.get("judge_model") or original.judge), cassettes
            )
        )

    own_store = store is None
    if store is None:
        index_path = original.index_path or get_settings().duckdb_path
        store = DuckDBStore(index_path, read_only=True)
    try:
        previous = run_dir / PREDICTIONS_NAME
        if previous.is_file():
            previous.replace(run_dir / PREVIOUS_PREDICTIONS_NAME)
        run_eval(
            cfg,
            selected,
            store,
            out_dir=run_dir.parent.parent,
            resume=False,
            provider=provider,
            judge=resolved_judge,
            embedder=embedder,
            prices=prices,
            run_id=run_dir.name,
        )
    finally:
        if own_store:
            store.close()

    timings: RescoreTimings = "remeasured"
    if (
        isinstance(provider, ReplayCacheProvider)
        and (run_dir / PREVIOUS_PREDICTIONS_NAME).is_file()
    ):
        carry_over_timings(run_dir / PREDICTIONS_NAME, run_dir / PREVIOUS_PREDICTIONS_NAME)
        summarize(run_dir / PREDICTIONS_NAME, seed=original.seed)
        timings = "carried_over"

    summary = RunSummary.model_validate(
        json.loads((run_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    )
    config_path = run_dir / CONFIG_NAME
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["rescored_at"] = datetime.now(UTC).isoformat()
    raw["rescore_judge"] = resolved_judge.model
    raw["rescore_timings"] = timings
    config_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info(
        "run_rescored",
        run_dir=str(run_dir),
        n=summary.n,
        n_completed=summary.n_completed,
        judge=resolved_judge.model,
        timings=timings,
        accuracy=summary.metrics.get("accuracy"),
    )
    return summary


__all__ = [
    "PREVIOUS_PREDICTIONS_NAME",
    "TIMING_FIELDS",
    "ReplayStub",
    "carry_over_timings",
    "read_run_config",
    "rescore",
]
