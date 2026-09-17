"""The evaluation runner: one config x one question list -> ``results/<name>/<run_id>/``.

Every run writes three files (SPEC 4.2, 7, 9):

* ``config.json`` -- the :class:`EvalConfig` plus provenance: git SHA, index manifest and its
  hash, resolved provider / model / sampling params, embedder, judge, prompt hashes, price list
  date, question ids, cassette directory and timestamps.
* ``predictions.jsonl`` -- one :class:`~secqa.core.contracts.EvalRecord` per question, appended
  and flushed as soon as it is scored, so a killed run resumes where it stopped and a record
  never contains dataset text (ids, our answer, retrieved pages, verdicts, usage only).
* ``summary.json`` -- :func:`secqa.eval.metrics.summarize` over the predictions.

Retrieval metrics per mode: ``rag`` scores the top-``k`` passages the model saw, ``agent`` the
pages its ``search_filings`` calls returned, ``closed_book`` has none (recorded as 0 and skipped
by the summary), and ``oracle`` records the supplied gold pages, so its recall is 1.0 by
construction and is shown for completeness only.

Failure policy: a :class:`~secqa.core.errors.ProviderError` (vendor outage, bad request) is
recorded in ``EvalRecord.error`` and the run continues (the question counts towards ``n`` but not
``n_completed``); a judge failure (:class:`~secqa.eval.judge.JudgeParseError` or a provider error
on the judge call) keeps the answer and every deterministic metric, records the failure in
``EvalRecord.judge_error`` and leaves only that verdict ``None``, so the question stays completed
and scorable by ``numeric_match``; :class:`~secqa.core.errors.CassetteMiss` in replay mode and
every other exception propagate, because they signal a broken setup rather than a bad question.
Cassettes are recorded under ``<cassette_dir>/<run_id>/`` so nothing is paid twice and the run
can be re-scored offline.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import Field, field_validator

from secqa.agent import AgentLoop, ToolRuntime
from secqa.core.contracts import (
    Answer,
    Chunk,
    Citation,
    DocumentMeta,
    Embedder,
    EvalRecord,
    FBQuestion,
    Frozen,
    Hit,
    HitView,
    LLMProvider,
    Mode,
    Page,
    RetrievalFilters,
    RetrievalStrategy,
    Usage,
)
from secqa.core.errors import ConfigError, ProviderError
from secqa.core.ids import sha256_hex
from secqa.core.logging import get_logger
from secqa.core.settings import CassetteMode, Settings, get_settings, provider_vendor
from secqa.embeddings import get_embedder
from secqa.eval.judge import (
    JUDGE_VERSION,
    RULE_JUDGE,
    Judge,
    JudgeParseError,
    LLMJudge,
    RuleJudge,
    judge_prompt_hashes,
    make_judge,
)
from secqa.eval.metrics import (
    CONFIG_NAME,
    PREDICTIONS_NAME,
    classify_failure,
    distinct_pages,
    evidence_overlap_recall,
    gold_page_mrr,
    numeric_match,
    page_recall_at_k,
    read_records,
    summarize,
)
from secqa.grounding import CitationVerifier
from secqa.indexing import ingest_document
from secqa.providers import ReplayCacheProvider, get_provider
from secqa.providers.pricing import PriceTable
from secqa.rag import RagPipeline, answer_closed_book, answer_with_oracle_context, prompt_hashes
from secqa.retrieval import Retriever
from secqa.store import DuckDBStore, resolve_git_sha

log = get_logger(__name__)

MODES: tuple[str, ...] = get_args(Mode)
STRATEGIES: tuple[str, ...] = get_args(RetrievalStrategy)
EFFORTS: tuple[str, ...] = ("low", "medium", "high")
CASSETTE_MODES: tuple[str, ...] = get_args(CassetteMode)
MAX_K = 20
DEFAULT_OUT_DIR = Path("results")
DEFAULT_CASSETTE_DIR = Path("cassettes")
TOOL_RESULT_CHARS = 20_000  # CONTRACTS rule 12: three full pages per get_pages call
RETRIEVED_PAGES_KEPT = 20
RETRIEVAL_ONLY_PROVIDER = "mock:abstain"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


# ---------------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------------


class EvalConfig(Frozen):
    """One row of the evaluation matrix (``configs/<name>.yaml``).

    The contract fields come first; the three optional paths at the end let a config say where
    its questions and index live (mock / CI rows use the fixture corpus, real rows default to
    the FinanceBench cache and ``Settings.duckdb_path``).
    """

    name: str
    mode: Mode
    provider: str
    embedder: str
    strategy: RetrievalStrategy = "hybrid"
    k: int = Field(default=8, ge=1, le=MAX_K)
    doc_filter: bool = False
    judge: str = "anthropic:claude-sonnet-5"
    effort: str = "medium"
    max_cost_usd_per_q: float = Field(default=0.5, ge=0.0)
    max_total_cost_usd: float | None = Field(default=None, ge=0.0)
    limit: int | None = Field(default=None, ge=1)
    seed: int = 0
    cassette_mode: str = "record"
    questions_path: Path | None = None
    index_path: Path | None = None
    fixture_pages_path: Path | None = None

    @field_validator("name")
    @classmethod
    def _name_is_a_directory_name(cls, value: str) -> str:
        value = value.strip()
        if not _NAME_RE.match(value):
            raise ValueError(f"config name {value!r} must be a plain directory name")
        return value

    @field_validator("provider", "embedder", "judge")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("effort")
    @classmethod
    def _effort_known(cls, value: str) -> str:
        if value not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}; got {value!r}")
        return value

    @field_validator("cassette_mode")
    @classmethod
    def _cassette_mode_known(cls, value: str) -> str:
        if value not in CASSETTE_MODES:
            raise ValueError(
                f"cassette_mode must be one of {', '.join(CASSETTE_MODES)}; got {value!r}"
            )
        return value

    @classmethod
    def from_yaml(cls, path: Path) -> EvalConfig:
        """Load ``configs/<name>.yaml``; a missing ``name`` defaults to the file stem."""
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read eval config {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"eval config {path} must be a mapping")
        raw.setdefault("name", path.stem)
        try:
            return cls.model_validate(raw)
        except ValueError as exc:
            raise ConfigError(f"invalid eval config {path}: {exc}") from exc

    @property
    def vendor(self) -> str:
        """``'mock'`` / ``'scripted'`` / ``'openai'`` / ``'anthropic'``."""
        return provider_vendor(self.provider)

    @property
    def row_kind(self) -> str:
        """``'retrieval'`` (no answering model), ``'smoke'`` (mock/scripted) or ``'llm'``."""
        if self.provider.strip().lower() == RETRIEVAL_ONLY_PROVIDER:
            return "retrieval"
        if self.vendor in ("mock", "scripted"):
            return "smoke"
        return "llm"

    @property
    def pending_reason(self) -> str:
        """Why this row has no numbers yet (what the report prints for a pending cell)."""
        if self.row_kind == "retrieval":
            return "requires the FinanceBench index (secqa ingest financebench)"
        if self.vendor in ("openai", "anthropic"):
            return f"requires {self.vendor.upper()}_API_KEY and the FinanceBench index"
        return "not run"


def load_config(path: Path) -> EvalConfig:
    """Alias of :meth:`EvalConfig.from_yaml` for callers that prefer a function."""
    return EvalConfig.from_yaml(path)


# ---------------------------------------------------------------------------------------------
# run ids and directories
# ---------------------------------------------------------------------------------------------


def make_run_id(git_sha: str, now: datetime | None = None) -> str:
    """``<git_sha7>_<YYYYMMDD-HHMM>`` (UTC)."""
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%d-%H%M")
    short = (git_sha or "unknown")[:7] or "unknown"
    return f"{short}_{stamp}"


def find_resumable_run(out_dir: Path, cfg: EvalConfig, n_questions: int) -> Path | None:
    """Latest run directory of ``cfg`` whose config matches and which is not complete."""
    root = Path(out_dir) / cfg.name
    if not root.is_dir():
        return None
    wanted = cfg.model_dump(mode="json")
    candidates: list[tuple[str, Path]] = []
    for config_path in root.glob("*/config.json"):
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict) or raw.get("config") != wanted:
            continue
        pred_path = config_path.parent / PREDICTIONS_NAME
        n_done = _count_scored(pred_path) if pred_path.is_file() else 0
        if n_done >= n_questions:
            continue
        candidates.append((str(raw.get("started_at") or ""), config_path.parent))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _count_lines(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def write_records(path: Path, records: list[EvalRecord]) -> None:
    """Rewrite ``predictions.jsonl`` atomically with exactly ``records`` (drops failed rows)."""
    tmp = Path(path).with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")
    tmp.replace(path)


def _count_scored(path: Path) -> int:
    """Records that were answered AND judged (no ``error``/``judge_error``); others are retried."""
    n = 0
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += rec.get("error") is None and rec.get("judge_error") is None
    return n


def index_sha_of(store: DuckDBStore) -> str:
    """Identity of the index: ``inputs_sha256`` from the manifest, else a hash of the manifest."""
    manifest = store.manifest()
    if manifest.inputs_sha256:
        return manifest.inputs_sha256
    payload = json.dumps(
        {
            "embedder": manifest.embedder,
            "dim": manifest.dim,
            "n_documents": manifest.n_documents,
            "n_pages": manifest.n_pages,
            "n_chunks": manifest.n_chunks,
            "n_facts": manifest.n_facts,
            "dataset_revision": manifest.dataset_revision,
        },
        sort_keys=True,
    )
    return sha256_hex(payload)


# ---------------------------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------------------------


@dataclass
class _Harness:
    """Everything one run needs, built once; :meth:`answer` and :meth:`record` do the work."""

    cfg: EvalConfig
    store: DuckDBStore
    provider: LLMProvider
    judge: Judge
    prices: PriceTable
    verifier: CitationVerifier
    embedder_name: str
    run_id: str
    git_sha: str
    index_sha: str
    prompt_hashes: dict[str, str]
    retriever: Retriever | None = None
    rag: RagPipeline | None = None
    agent: AgentLoop | None = None

    def answer(self, q: FBQuestion) -> Answer:
        """Run the configured mode for one question (provider errors propagate)."""
        cfg = self.cfg
        filters = RetrievalFilters(doc_names=[q.doc_name]) if cfg.doc_filter else None
        if cfg.mode == "closed_book":
            return answer_closed_book(
                q.question, self.provider, self.prices, request_id=q.id, effort=cfg.effort
            )
        if cfg.mode == "oracle":
            wanted = sorted({item.page_num for item in q.evidence})
            pages: list[Page] = self.store.get_pages(q.doc_name, wanted) if wanted else []
            if len(pages) < len(wanted):
                log.warning(
                    "oracle_pages_missing",
                    financebench_id=q.id,
                    doc_name=q.doc_name,
                    wanted=wanted,
                    found=[page.page_num for page in pages],
                )
            return answer_with_oracle_context(
                q.question,
                pages,
                self.provider,
                self.verifier,
                self.prices,
                request_id=q.id,
                effort=cfg.effort,
            )
        if cfg.mode == "rag":
            assert self.rag is not None  # built in _build_harness for this mode
            return self.rag.answer(q.question, filters=filters, request_id=q.id)
        assert self.agent is not None  # built in _build_harness for this mode
        return self.agent.run(q.question, filters=filters, request_id=q.id)

    def record(self, q: FBQuestion, answer: Answer | None, error: str | None) -> EvalRecord:
        """Score one answer (deterministic metrics + judge) into an :class:`EvalRecord`.

        ``error`` is the answer-level failure (``answer`` is then ``None``) and is the only thing
        that makes a record unscored. A failing judge call is caught here and recorded in
        ``judge_error``; the answer, its ``numeric_match``, citations and retrieval metrics are
        kept, and the summary still counts the record as completed.
        """
        cfg = self.cfg
        gold_pages = distinct_pages((item.doc_name, item.page_num) for item in q.evidence)
        judge_error: str | None = None
        judge_cost = 0.0
        verdict = faith = None
        if answer is None:
            retrieved_pages: list[tuple[str, int]] = []
            hits: list[Hit] = []
        else:
            retrieved_pages = distinct_pages(
                (view.doc_name, view.page_num) for view in answer.retrieved
            )[:RETRIEVED_PAGES_KEPT]
            hits = self._hits(answer.retrieved)
            # The two judge calls are independent: a faithfulness failure must not discard a
            # correctness verdict already in hand, and vice versa. Either failure lands in
            # ``judge_error`` -- never in ``error`` -- so the answer stays scored (numeric_match,
            # citations, retrieval metrics) and the record stays in ``n_completed``.
            judge_failures: list[str] = []
            try:
                verdict = self.judge.correctness(q, answer)
            except (JudgeParseError, ProviderError) as exc:
                judge_failures.append(f"judge correctness: {exc}")
                log.warning(
                    "judge_failed", financebench_id=q.id, call="correctness", error=str(exc)
                )
            try:
                faith = self.judge.faithfulness(answer, store=self.store)
            except (JudgeParseError, ProviderError) as exc:
                judge_failures.append(f"judge faithfulness: {exc}")
                log.warning(
                    "judge_failed", financebench_id=q.id, call="faithfulness", error=str(exc)
                )
            judge_error = "; ".join(judge_failures) or None
            judge_usage = Usage()
            if verdict is not None:
                judge_usage = judge_usage + verdict.usage
            if faith is not None:
                judge_usage = judge_usage + faith.usage
            if not isinstance(self.judge, RuleJudge):
                judge_cost = self.prices.cost_usd(
                    self.judge.provider.provider, self.judge.provider.model, judge_usage
                )

        citations: list[Citation] = list(answer.citations) if answer else []
        verified_rate = (
            sum(1 for c in citations if c.verified) / len(citations) if citations else None
        )
        provisional = EvalRecord(
            financebench_id=q.id,
            question_type=q.question_type,
            config_name=cfg.name,
            run_id=self.run_id,
            git_sha=self.git_sha,
            index_sha=self.index_sha,
            prompt_hashes=dict(self.prompt_hashes),
            models_yaml_as_of=self.prices.as_of.isoformat(),
            provider=answer.provider if answer else self.provider.provider,
            model=answer.model if answer else self.provider.model,
            mode=cfg.mode,
            embedder=self.embedder_name,
            strategy=cfg.strategy,
            k=cfg.k,
            answer_text=answer.text if answer else "",
            value=answer.value if answer else None,
            unit=answer.unit if answer else None,
            abstained=answer.abstained if answer else False,
            grounded=answer.grounded if answer else False,
            citations=citations,
            retrieved_pages=retrieved_pages,
            gold_pages=gold_pages,
            page_recall_5=page_recall_at_k(retrieved_pages, gold_pages, 5),
            page_recall_10=page_recall_at_k(retrieved_pages, gold_pages, 10),
            page_recall_20=page_recall_at_k(retrieved_pages, gold_pages, 20),
            overlap_recall_10=evidence_overlap_recall(hits, q.evidence, 10),
            gold_page_mrr=gold_page_mrr(retrieved_pages, gold_pages),
            numeric_match=numeric_match(answer.value, q.answer) if answer else None,
            judge=verdict,
            faith=faith,
            citation_verified_rate=verified_rate,
            failure="none",
            usage=answer.usage if answer else Usage(),
            cost_usd=answer.cost_usd if answer else 0.0,
            judge_cost_usd=judge_cost,
            latency_ms=answer.latency_ms if answer else 0.0,
            retrieval_ms=answer.retrieval_ms if answer else 0.0,
            llm_ms=answer.llm_ms if answer else 0.0,
            steps=answer.steps if answer else 0,
            tool_calls=answer.tool_calls if answer else 0,
            terminated_by=answer.terminated_by if answer else "error",
            error=error,
            judge_error=judge_error,
            timestamp=datetime.now(UTC),
        )
        return provisional.model_copy(update={"failure": classify_failure(provisional)})

    def _hits(self, views: Sequence[HitView]) -> list[Hit]:
        """Rebuild ``Hit`` objects (full chunk text from the store when available)."""
        if not views:
            return []
        by_id = {
            chunk.chunk_id: chunk for chunk in self.store.get_chunks([v.chunk_id for v in views])
        }
        hits: list[Hit] = []
        for rank, view in enumerate(views, start=1):
            chunk = by_id.get(view.chunk_id) or Chunk(
                chunk_id=view.chunk_id,
                doc_name=view.doc_name,
                page_num=view.page_num,
                chunk_idx=0,
                section=view.section,
                text=view.snippet,
                n_tokens=0,
            )
            hits.append(Hit(chunk=chunk, score=view.score, rank=rank, source=self.cfg.strategy))
        return hits


def _wrap_cassettes(provider: LLMProvider, mode: str, cassette_dir: Path) -> LLMProvider:
    if mode == "off" or isinstance(provider, ReplayCacheProvider):
        return provider
    return ReplayCacheProvider(provider, cache_dir=cassette_dir, mode=mode)  # type: ignore[arg-type]


def _build_harness(
    cfg: EvalConfig,
    store: DuckDBStore,
    *,
    run_id: str,
    cassette_dir: Path,
    provider: LLMProvider | None,
    judge: LLMProvider | Judge | None,
    embedder: Embedder | None,
    prices: PriceTable | None,
) -> _Harness:
    settings: Settings = get_settings().model_copy(
        update={"cassette_mode": cfg.cassette_mode, "cassette_dir": cassette_dir}
    )
    resolved_provider = (
        _wrap_cassettes(provider, cfg.cassette_mode, cassette_dir)
        if provider is not None
        else get_provider(cfg.provider, settings)
    )
    resolved_judge: Judge
    if isinstance(judge, RuleJudge | LLMJudge):
        resolved_judge = judge
    elif cfg.judge == RULE_JUDGE:
        resolved_judge = make_judge(RULE_JUDGE)
    elif judge is not None:
        resolved_judge = make_judge(
            cfg.judge, _wrap_cassettes(judge, cfg.cassette_mode, cassette_dir)
        )
    else:
        resolved_judge = make_judge(cfg.judge, get_provider(cfg.judge, settings))
    resolved_prices = prices if prices is not None else PriceTable.load()

    retriever: Retriever | None = None
    embedder_name = cfg.embedder
    if cfg.mode in ("rag", "agent"):
        resolved_embedder = (
            embedder if embedder is not None else get_embedder(cfg.embedder, settings)
        )
        embedder_name = resolved_embedder.name
        retriever = Retriever(store, resolved_embedder, strategy=cfg.strategy, k=cfg.k)
    verifier = CitationVerifier()
    harness = _Harness(
        cfg=cfg,
        store=store,
        provider=resolved_provider,
        judge=resolved_judge,
        prices=resolved_prices,
        verifier=verifier,
        embedder_name=embedder_name,
        run_id=run_id,
        git_sha=resolve_git_sha(),
        index_sha=index_sha_of(store),
        prompt_hashes={**prompt_hashes(), **judge_prompt_hashes()},
        retriever=retriever,
    )
    if cfg.mode == "rag" and retriever is not None:
        harness.rag = RagPipeline(
            retriever, resolved_provider, verifier, resolved_prices, k=cfg.k, effort=cfg.effort
        )
    elif cfg.mode == "agent" and retriever is not None:
        runtime = ToolRuntime(store, retriever, max_result_chars=TOOL_RESULT_CHARS)
        harness.agent = AgentLoop(
            resolved_provider,
            runtime,
            verifier,
            resolved_prices,
            max_cost_usd=cfg.max_cost_usd_per_q,
            effort=cfg.effort,
        )
    return harness


def _provider_params(provider: LLMProvider) -> dict[str, Any]:
    params = getattr(provider, "params", None)
    try:
        return dict(params()) if callable(params) else {}
    except Exception:  # a provider's params() must never break a run's bookkeeping
        return {}


def _write_config(
    run_dir: Path,
    cfg: EvalConfig,
    harness: _Harness,
    questions: Sequence[FBQuestion],
    n_dataset: int,
    cassette_dir: Path | None,
    started_at: datetime,
) -> None:
    manifest = harness.store.manifest()
    payload: dict[str, Any] = {
        "config": cfg.model_dump(mode="json"),
        "run_id": harness.run_id,
        "git_sha": harness.git_sha,
        "index_sha": harness.index_sha,
        "index_manifest": manifest.model_dump(mode="json"),
        "bm25_backend": harness.store.bm25_backend,
        "provider": harness.provider.provider,
        "model": harness.provider.model,
        "provider_params": _provider_params(harness.provider),
        "embedder": harness.embedder_name,
        "judge_model": harness.judge.model,
        "judge_version": JUDGE_VERSION,
        "prompt_hashes": harness.prompt_hashes,
        "models_yaml_as_of": harness.prices.as_of.isoformat(),
        "n_questions": len(questions),
        "n_dataset": n_dataset,
        "question_ids": [q.id for q in questions],
        "cassettes": str(cassette_dir) if cassette_dir is not None else None,
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "n_completed": 0,
    }
    (run_dir / CONFIG_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _update_config(run_dir: Path, **fields: Any) -> None:
    path = run_dir / CONFIG_NAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.update(fields)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_eval(
    cfg: EvalConfig,
    questions: list[FBQuestion],
    store: DuckDBStore,
    out_dir: Path = DEFAULT_OUT_DIR,
    resume: bool = True,
    *,
    provider: LLMProvider | None = None,
    judge: LLMProvider | Judge | None = None,
    embedder: Embedder | None = None,
    prices: PriceTable | None = None,
    cassette_dir: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Evaluate ``cfg`` over ``questions`` and return the run directory.

    Args:
        cfg: The row to run (``cfg.limit`` truncates ``questions``; ``config.json`` keeps the
            untruncated count as ``n_dataset`` so a limited run reports as a subset).
        questions: FinanceBench questions (or the fixture questions) in the order to run.
        store: The index (read-only is fine); its manifest identifies the index in every record.
        out_dir: ``results/``; the run lands in ``<out_dir>/<cfg.name>/<run_id>/``.
        resume: Continue the latest incomplete run of this exact config instead of starting a
            new one (done ids are skipped, cassettes reused).
        provider: Answering model override (tests, scripted scenarios); by default resolved
            from ``cfg.provider`` through :func:`secqa.providers.get_provider`.
        judge: Judge override: an :class:`LLMProvider` (wrapped as an LLM judge) or a ready
            :class:`RuleJudge` / :class:`LLMJudge`; ``cfg.judge == 'rule'`` needs none.
        embedder: Embedder override (only used by the retrieving modes).
        prices: Price table override (default: ``src/secqa/eval/models.yaml``).
        cassette_dir: Root for cassettes (``<cassette_dir>/<run_id>/``); default ``cassettes/``.
        run_id: Force a run id (creates or resumes exactly that directory).

    Raises:
        ConfigError: bad config, missing keys, unpriced model, no questions.
        CassetteMiss: replay mode with a missing recording (never silently re-paid).
    """
    if not questions:
        raise ConfigError("run_eval needs at least one question")
    selected = list(questions[: cfg.limit] if cfg.limit else questions)
    out_dir = Path(out_dir)
    cassette_root = Path(cassette_dir) if cassette_dir is not None else DEFAULT_CASSETTE_DIR

    run_dir: Path | None = None
    if run_id is not None:
        if not _RUN_ID_RE.match(run_id):
            raise ConfigError(f"run_id {run_id!r} must be a plain directory name")
        run_dir = out_dir / cfg.name / run_id
    elif resume:
        run_dir = find_resumable_run(out_dir, cfg, len(selected))
        if run_dir is not None:
            run_id = run_dir.name
            log.info("eval_resuming", run_dir=str(run_dir))
    git_sha = resolve_git_sha()
    if run_id is None:
        run_id = make_run_id(git_sha)
        run_dir = out_dir / cfg.name / run_id
        suffix = 1
        while run_dir.exists():  # two runs within the same minute
            suffix += 1
            run_dir = out_dir / cfg.name / f"{run_id}-{suffix}"
        run_id = run_dir.name
    assert run_dir is not None
    run_dir.mkdir(parents=True, exist_ok=True)
    run_cassettes = cassette_root / run_id if cfg.cassette_mode != "off" else None

    harness = _build_harness(
        cfg,
        store,
        run_id=run_id,
        cassette_dir=run_cassettes or cassette_root / run_id,
        provider=provider,
        judge=judge,
        embedder=embedder,
        prices=prices,
    )
    if not harness.prices.has(harness.provider.provider, harness.provider.model):
        raise ConfigError(
            f"no price for {harness.provider.provider}:{harness.provider.model} in models.yaml"
        )

    pred_path = run_dir / PREDICTIONS_NAME
    done: set[str] = set()
    if pred_path.is_file():
        kept = [
            rec for rec in read_records(pred_path) if rec.error is None and rec.judge_error is None
        ]
        n_failed = _count_lines(pred_path) - len(kept)
        if n_failed:
            # A provider error (network drop, timeout, 5xx) on the answer OR on the judge leaves a
            # record that is unscored or unjudged; resuming retries those questions (the answer
            # replays from its cassette when one exists) and replaces the records, never
            # double-counting.
            write_records(pred_path, kept)
            log.info("eval_retrying_failed", run_dir=str(run_dir), n_failed=n_failed)
        done = {rec.financebench_id for rec in kept}
    started_at = datetime.now(UTC)
    if not (run_dir / CONFIG_NAME).is_file():
        _write_config(run_dir, cfg, harness, selected, len(questions), run_cassettes, started_at)

    spent = (
        sum(rec.cost_usd + rec.judge_cost_usd for rec in read_records(pred_path)) if done else 0.0
    )
    log.info(
        "eval_started",
        config=cfg.name,
        run_id=run_id,
        mode=cfg.mode,
        provider=harness.provider.provider,
        model=harness.provider.model,
        judge=harness.judge.model,
        n_questions=len(selected),
        n_dataset=len(questions),
        n_done=len(done),
        cassettes=str(run_cassettes) if run_cassettes else None,
    )

    n_new = 0
    with pred_path.open("a", encoding="utf-8") as fh:
        for index, q in enumerate(selected, start=1):
            if q.id in done:
                continue
            if cfg.max_total_cost_usd is not None and spent >= cfg.max_total_cost_usd:
                log.warning(
                    "eval_budget_exhausted",
                    spent_usd=round(spent, 4),
                    cap_usd=cfg.max_total_cost_usd,
                    n_remaining=len(selected) - index + 1,
                )
                break
            started = time.perf_counter()
            answer: Answer | None = None
            error: str | None = None
            try:
                answer = harness.answer(q)
            except ProviderError as exc:
                error = f"provider: {exc}"
                log.error("eval_question_failed", financebench_id=q.id, error=str(exc))
            record = harness.record(q, answer, error)
            fh.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
            fh.flush()
            done.add(q.id)
            n_new += 1
            spent += record.cost_usd + record.judge_cost_usd
            log.info(
                "eval_question",
                index=index,
                n=len(selected),
                financebench_id=q.id,
                abstained=record.abstained,
                numeric_match=record.numeric_match,
                judge=record.judge.label if record.judge else None,
                failure=record.failure,
                page_recall_10=record.page_recall_10,
                cost_usd=record.cost_usd,
                spent_usd=round(spent, 4),
                seconds=round(time.perf_counter() - started, 2),
                error=record.error,
                judge_error=record.judge_error,
            )

    summary = summarize(pred_path, seed=cfg.seed)
    _update_config(
        run_dir, finished_at=datetime.now(UTC).isoformat(), n_completed=summary.n_completed
    )
    log.info(
        "eval_finished",
        config=cfg.name,
        run_id=run_id,
        n=summary.n,
        n_completed=summary.n_completed,
        n_new=n_new,
        accuracy=summary.metrics.get("accuracy"),
        cost_total_usd=summary.cost_total_usd,
        run_dir=str(run_dir),
    )
    return run_dir


# ---------------------------------------------------------------------------------------------
# fixture corpus (CI smoke rows)
# ---------------------------------------------------------------------------------------------


def build_fixture_index(store: DuckDBStore, embedder: Embedder, pages_path: Path) -> int:
    """Ingest a hand-written fixture corpus (``{doc_name: {..., pages: [text, ...]}}``).

    This is the corpus behind ``tests/fixtures/fb_mini.jsonl`` and the mock configs; it lets
    ``secqa eval --config configs/rag_mock.yaml`` build its own tiny index with no downloads.
    Returns the number of chunks written; the FTS index and manifest counts are rebuilt.

    Raises:
        ConfigError: on a malformed pages file or a read-only store.
    """
    pages_path = Path(pages_path)
    try:
        raw = json.loads(pages_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read fixture pages {pages_path}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{pages_path} must map doc_name -> document")
    n_chunks = 0
    for doc_name, doc in raw.items():
        if not isinstance(doc, dict) or not isinstance(doc.get("pages"), list) or not doc["pages"]:
            raise ConfigError(f"{pages_path}: {doc_name} needs a non-empty 'pages' list")
        texts = [str(text) for text in doc["pages"]]
        pages = [
            Page(doc_name=doc_name, page_num=i, text=text) for i, text in enumerate(texts, start=1)
        ]
        meta = DocumentMeta(
            doc_name=doc_name,
            company=str(doc.get("company") or doc_name),
            ticker=doc.get("ticker"),
            cik=doc.get("cik"),
            form=str(doc.get("form") or "10-K"),
            fiscal_year=doc.get("fiscal_year"),
            period_end=None,
            source_kind="fixture",
            source_url=f"fixture://{pages_path.name}/{doc_name}",
            source_sha256=sha256_hex("\n".join(texts)),
            n_pages=len(pages),
            ingested_at=datetime.now(UTC),
        )
        n_chunks += ingest_document(store, embedder, pages, meta)
    store.rebuild_fts()
    log.info("fixture_index_built", path=str(pages_path), n_documents=len(raw), n_chunks=n_chunks)
    return n_chunks


__all__ = [
    "CASSETTE_MODES",
    "DEFAULT_CASSETTE_DIR",
    "DEFAULT_OUT_DIR",
    "EFFORTS",
    "MAX_K",
    "MODES",
    "RETRIEVAL_ONLY_PROVIDER",
    "STRATEGIES",
    "TOOL_RESULT_CHARS",
    "EvalConfig",
    "build_fixture_index",
    "find_resumable_run",
    "index_sha_of",
    "load_config",
    "make_run_id",
    "run_eval",
]
