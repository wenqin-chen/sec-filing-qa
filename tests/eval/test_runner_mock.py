"""The runner end to end on the six fixture questions with the mock / scripted providers.

Asserts the JSONL schema, that no dataset text is persisted, resume semantics, the summary
keys, the agent path, budget handling, provider-error handling and the config loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from secqa.core.contracts import EvalRecord, FBQuestion, RunSummary
from secqa.core.errors import ConfigError, ProviderError
from secqa.core.ids import sha256_hex
from secqa.embeddings import HashingEmbedder
from secqa.eval.financebench import load_questions_jsonl
from secqa.eval.judge import (
    JUDGE_PROMPT_NAMES,
    JUDGE_VERSION,
    RATIONALE_DIGEST_PREFIX,
    RuleJudge,
)
from secqa.eval.metrics import read_records
from secqa.eval.runner import (
    EvalConfig,
    build_fixture_index,
    find_resumable_run,
    load_config,
    make_run_id,
    run_eval,
)
from secqa.ingest import extract_pdf_pages
from secqa.providers.mock_provider import MockProvider
from secqa.providers.pricing import PriceTable
from secqa.providers.scripted_provider import ScriptedProvider
from secqa.rag import PROMPT_NAMES
from secqa.store import DuckDBStore
from tests.eval.conftest import FB_MINI, FIXTURE_PAGES, TOP_DOC, FixedProvider, fixture_docs

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

# Dataset fields that must never be persisted (SPEC 9). Our prediction may legitimately quote
# the filing sentence, so the evidence *text* is not on this list; its field never is.
DATASET_TEXT = [
    "What were total net sales",  # question text
    "Net sales are stated on the first page",  # justification
    '"evidence"',  # the evidence field itself
    '"justification"',
]


def rag_cfg(**overrides: object) -> EvalConfig:
    base = {
        "name": "rag_mock",
        "mode": "rag",
        "provider": "mock",
        "embedder": "hashing:64",
        "strategy": "hybrid",
        "k": 4,
        "judge": "rule",
        "cassette_mode": "off",
    }
    base.update(overrides)
    return EvalConfig.model_validate(base)


def test_fixture_pdf_pages_carry_the_evidence(fixture_pdfs: dict[str, Path]) -> None:
    """The check_page_indexing invariant on the fixture: evidence text is on its page."""
    questions = load_questions_jsonl(FB_MINI)
    pages = {
        (doc, p.page_num): p.text
        for doc, path in fixture_pdfs.items()
        for p in extract_pdf_pages(path, doc)
    }
    for q in questions:
        for ev in q.evidence:
            assert ev.text in pages[(ev.doc_name, ev.page_num)], (q.id, ev.page_num)


def test_rag_mock_end_to_end(
    store: DuckDBStore,
    questions: list[FBQuestion],
    prices: PriceTable,
    tmp_path: Path,
) -> None:
    run_dir = run_eval(
        rag_cfg(),
        questions,
        store,
        out_dir=tmp_path / "results",
        prices=prices,
        cassette_dir=tmp_path / "cassettes",
    )
    assert run_dir.parent == tmp_path / "results" / "rag_mock"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["config"]["name"] == "rag_mock"
    assert config["provider"] == "mock" and config["model"] == "mock-extractive"
    assert config["n_questions"] == 6 and len(config["question_ids"]) == 6
    assert config["bm25_backend"] in ("duckdb_fts", "python")
    assert config["cassettes"] is None  # cassette_mode off
    assert config["finished_at"] and config["n_completed"] == 6
    assert set(config["prompt_hashes"]) == set(PROMPT_NAMES) | set(JUDGE_PROMPT_NAMES)

    records = read_records(run_dir / "predictions.jsonl")
    assert [r.financebench_id for r in records] == [q.id for q in questions]
    for rec in records:
        assert isinstance(rec, EvalRecord)
        assert rec.mode == "rag" and rec.strategy == "hybrid" and rec.k == 4
        assert rec.embedder == "hashing-64" and rec.provider == "mock"
        assert rec.error is None and rec.terminated_by == "single_shot"
        assert rec.gold_pages and all(p >= 1 for _, p in rec.gold_pages)
        assert 0 < len(rec.retrieved_pages) <= 4
        assert rec.models_yaml_as_of == "2026-09-11"
        assert rec.judge is not None or rec.numeric_match is None  # rule judge decides numerics
        if rec.judge is not None:
            assert rec.judge.judge_model == "rule"
        assert rec.cost_usd == 0.0 and rec.judge_cost_usd == 0.0
    by_id = {r.financebench_id: r for r in records}
    # The extractive mock quotes the top passage verbatim, so its citation verifies.
    net_sales = by_id["fb_mini_001"]
    assert net_sales.page_recall_10 == 1.0 and net_sales.gold_page_mrr > 0
    assert net_sales.citation_verified_rate == 1.0 and net_sales.grounded
    assert net_sales.numeric_match is True and net_sales.failure == "none"
    assert net_sales.overlap_recall_10 == 1.0
    two_numbers = by_id["fb_mini_005"]
    assert two_numbers.numeric_match is None  # gold has two numbers
    assert any(r.numeric_match is False for r in records)  # capex vs operating income

    # No dataset text anywhere in the persisted files.
    persisted = (run_dir / "predictions.jsonl").read_text(encoding="utf-8")
    persisted += (run_dir / "config.json").read_text(encoding="utf-8")
    persisted += (run_dir / "summary.json").read_text(encoding="utf-8")
    for text in DATASET_TEXT:
        assert text not in persisted

    summary = RunSummary.model_validate(
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    )
    assert summary.n == 6 and summary.n_completed == 6
    for key in (
        "accuracy",
        "abstain_rate",
        "hallucination_rate",
        "numeric_match_rate",
        "numeric_coverage",
        "faithfulness",
        "citation_verified_rate",
        "grounded_rate",
        "page_recall_5",
        "page_recall_10",
        "page_recall_20",
        "overlap_recall_10",
        "gold_page_mrr",
    ):
        assert key in summary.metrics
    assert summary.metrics["page_recall_10"] == pytest.approx(1.0)
    assert summary.metrics["faithfulness"] is None  # rule judge never scores faithfulness
    assert summary.provider == "mock" and summary.judge_model == "rule"
    assert summary.judge_version == JUDGE_VERSION and summary.cost_total_usd == 0.0
    assert summary.failures["none"] >= 1


def test_resume_skips_done_ids_and_reuses_run_dir(
    store: DuckDBStore,
    questions: list[FBQuestion],
    prices: PriceTable,
    tmp_path: Path,
) -> None:
    out = tmp_path / "results"
    cfg = rag_cfg()
    first = run_eval(cfg, questions[:3], store, out_dir=out, prices=prices, run_id="fixed_run")
    assert len(read_records(first / "predictions.jsonl")) == 3
    # Corrupt nothing; the resumed run must append only the three new ids.
    second = run_eval(cfg, questions, store, out_dir=out, prices=prices, run_id="fixed_run")
    assert second == first
    records = read_records(second / "predictions.jsonl")
    assert [r.financebench_id for r in records] == [q.id for q in questions]
    # find_resumable_run: an incomplete run of the same config is picked up, a complete one is not
    assert find_resumable_run(out, cfg, 6) is None
    assert find_resumable_run(out, cfg, 7) == first
    assert find_resumable_run(out, rag_cfg(k=3), 7) is None
    # a third run with resume=True and all ids done creates a fresh directory
    third = run_eval(cfg, questions, store, out_dir=out, prices=prices, resume=True)
    assert third != first and len(read_records(third / "predictions.jsonl")) == 6


def test_agent_mock_end_to_end(
    store: DuckDBStore,
    questions: list[FBQuestion],
    prices: PriceTable,
    tmp_path: Path,
) -> None:
    cfg = rag_cfg(name="agent_mock", mode="agent", limit=3)
    run_dir = run_eval(cfg, questions, store, out_dir=tmp_path / "results", prices=prices)
    records = read_records(run_dir / "predictions.jsonl")
    assert len(records) == 3  # limit honoured
    for rec in records:
        assert rec.mode == "agent" and rec.terminated_by == "final_answer"
        assert rec.tool_calls >= 2 and rec.steps >= 2  # search_filings + final_answer
        assert rec.retrieved_pages  # pages the agent saw through search_filings
    assert records[0].numeric_match is True and records[0].citation_verified_rate == 1.0


def test_agent_budget_row_is_classified(
    store: DuckDBStore,
    questions: list[FBQuestion],
    prices: PriceTable,
    tmp_path: Path,
) -> None:
    """A scripted agent that keeps searching hits max_steps -> tools-off final call -> budget."""
    queries = ["net sales", "operating income", "cash", "debt", "capex", "leases", "tax", "eps"]
    scenario: list[dict[str, object]] = [
        {"tool_calls": [{"name": "search_filings", "arguments": {"query": query}}]}
        for query in queries  # 8 distinct searches = max_steps, then the forced final call
    ]
    scenario.append(
        {
            "match": "Stop using tools",
            "text": json.dumps(
                {
                    "answer": "Net sales were $999 million.",
                    "value": 999_000_000,
                    "unit": "USD",
                    "citations": [],
                    "calculation": None,
                    "abstain": False,
                }
            ),
        }
    )
    cfg = rag_cfg(name="agent_scripted", mode="agent", provider="scripted:x", limit=1)
    run_dir = run_eval(
        cfg,
        questions,
        store,
        out_dir=tmp_path / "results",
        prices=prices,
        provider=ScriptedProvider(scenario),
    )
    rec = read_records(run_dir / "predictions.jsonl")[0]
    assert rec.error is None and rec.terminated_by == "max_steps"
    assert rec.steps == 9 and rec.tool_calls == 8
    assert rec.numeric_match is False and rec.failure == "budget"


def test_provider_error_is_recorded_not_raised(
    store: DuckDBStore,
    questions: list[FBQuestion],
    prices: PriceTable,
    tmp_path: Path,
) -> None:
    class Failing(FixedProvider):
        def complete(self, *args: object, **kwargs: object):  # type: ignore[override]
            raise ProviderError("upstream 502", retryable=True, provider="openai")

    cfg = rag_cfg(name="closed_book_test", mode="closed_book", provider="openai:gpt-test", limit=2)
    run_dir = run_eval(
        cfg,
        questions,
        store,
        out_dir=tmp_path / "results",
        prices=prices,
        provider=Failing(provider="openai", model="gpt-test"),
    )
    records = read_records(run_dir / "predictions.jsonl")
    assert len(records) == 2
    assert all(r.error and "upstream 502" in r.error for r in records)
    assert all(r.failure == "tool_error" and r.terminated_by == "error" for r in records)
    summary = RunSummary.model_validate(
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    )
    assert summary.n == 2 and summary.n_completed == 0
    assert summary.metrics["error_rate"] == 1.0
    assert summary.metrics["accuracy"] is None


def test_closed_book_with_priced_provider_records_cost(
    store: DuckDBStore,
    questions: list[FBQuestion],
    prices: PriceTable,
    tmp_path: Path,
) -> None:
    parsed = {
        "answer": "Total net sales were $1,577 million.",
        "value": 1_577_000_000,
        "unit": "USD",
        "citations": [],
        "abstain": False,
    }
    provider = FixedProvider(
        parsed=parsed, text=json.dumps(parsed), provider="openai", model="gpt-test"
    )
    cfg = rag_cfg(name="closed_book_test", mode="closed_book", provider="openai:gpt-test", limit=2)
    run_dir = run_eval(
        cfg, questions, store, out_dir=tmp_path / "results", prices=prices, provider=provider
    )
    records = read_records(run_dir / "predictions.jsonl")
    assert records[0].numeric_match is True and records[1].numeric_match is False
    assert records[0].cost_usd == pytest.approx((500 * 4.0 + 40 * 16.0) / 1e6)
    assert records[0].retrieved_pages == [] and records[0].page_recall_10 == 0.0
    assert records[0].citation_verified_rate is None
    assert records[1].failure == "reasoning_error"  # closed book: no retrieval / citations


def test_total_budget_stops_the_run(
    store: DuckDBStore,
    questions: list[FBQuestion],
    prices: PriceTable,
    tmp_path: Path,
) -> None:
    parsed = {"answer": "x", "value": None, "unit": None, "citations": [], "abstain": True}
    provider = FixedProvider(parsed=parsed, provider="openai", model="gpt-test")
    cost_per_call = (500 * 4.0 + 40 * 16.0) / 1e6
    cfg = rag_cfg(
        name="cb_budget",
        mode="closed_book",
        provider="openai:gpt-test",
        max_total_cost_usd=cost_per_call * 2.5,
    )
    run_dir = run_eval(
        cfg, questions, store, out_dir=tmp_path / "results", prices=prices, provider=provider
    )
    records = read_records(run_dir / "predictions.jsonl")
    assert len(records) == 3  # 3rd call pushes spend past the cap, then the run stops
    summary = RunSummary.model_validate(
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    )
    assert summary.n == 6 and summary.n_completed == 3  # partial row


def test_oracle_mode_uses_gold_pages(
    store: DuckDBStore,
    questions: list[FBQuestion],
    prices: PriceTable,
    tmp_path: Path,
) -> None:
    cfg = rag_cfg(name="oracle_mock", mode="oracle", limit=2)
    run_dir = run_eval(cfg, questions, store, out_dir=tmp_path / "results", prices=prices)
    records = read_records(run_dir / "predictions.jsonl")
    assert all(r.retrieved_pages == r.gold_pages for r in records)
    assert all(r.page_recall_5 == 1.0 for r in records)
    assert records[0].numeric_match is True and records[0].citation_verified_rate == 1.0


def test_unpriced_model_and_mock_judge_are_refused(
    store: DuckDBStore, questions: list[FBQuestion], prices: PriceTable, tmp_path: Path
) -> None:
    provider = FixedProvider(provider="openai", model="gpt-unpriced")
    with pytest.raises(ConfigError, match="no price"):
        run_eval(
            rag_cfg(mode="closed_book", provider="openai:gpt-unpriced"),
            questions,
            store,
            out_dir=tmp_path,
            prices=prices,
            provider=provider,
        )
    with pytest.raises(ConfigError, match="fabricate"):
        run_eval(rag_cfg(judge="mock"), questions, store, out_dir=tmp_path, prices=prices)
    with pytest.raises(ConfigError, match="at least one question"):
        run_eval(rag_cfg(), [], store, out_dir=tmp_path, prices=prices)


def test_llm_judge_path_records_verdicts_and_cost(
    store: DuckDBStore, questions: list[FBQuestion], prices: PriceTable, tmp_path: Path
) -> None:
    judge = FixedProvider(
        parsed={"label": "correct", "rationale": "ok"},
        text='{"label": "correct", "rationale": "ok"}',
        provider="anthropic",
        model="claude-test",
    )
    cfg = rag_cfg(name="rag_judged", judge="anthropic:claude-test", limit=2)
    run_dir = run_eval(
        cfg, questions, store, out_dir=tmp_path / "results", prices=prices, judge=judge
    )
    records = read_records(run_dir / "predictions.jsonl")
    for rec in records:
        assert rec.judge is not None and rec.judge.judge_model == "anthropic:claude-test"
        assert rec.judge_cost_usd > 0
    # The faithfulness judge got the same fixed JSON (no 'claims') -> a judge error. It is
    # recorded in judge_error, NOT in error: the answer, its correctness verdict and its
    # numeric_match are intact, so the record stays completed and scored.
    assert all(rec.error is None for rec in records)
    assert all(rec.judge_error and "judge faithfulness:" in rec.judge_error for rec in records)
    assert all(rec.faith is None for rec in records)
    assert records[0].numeric_match is True and records[0].failure == "none"
    summary = RunSummary.model_validate(
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    )
    assert summary.n == 2 and summary.n_completed == 2
    assert summary.metrics["error_rate"] == 0.0
    assert summary.metrics["judge_error_rate"] == 1.0
    assert summary.metrics["accuracy"] is not None
    assert summary.failures.get("tool_error", 0) == 0
    # judge messages must not include the effort other than 'low'
    assert all(call["effort"] == "low" for call in judge.calls)


def test_correctness_judge_failure_does_not_block_faithfulness(
    store: DuckDBStore, questions: list[FBQuestion], prices: PriceTable, tmp_path: Path
) -> None:
    """The two judge calls are independent: a correctness parse failure still lets the
    faithfulness judge run, and the record keeps its numeric score."""
    faith_only = {"claims": [{"claim": "net sales were $1,577 million", "supported": True}]}
    judge = FixedProvider(
        parsed=faith_only,
        text=json.dumps(faith_only),
        provider="anthropic",
        model="claude-test",
    )
    cfg = rag_cfg(name="rag_judged_corr_fail", judge="anthropic:claude-test", limit=1)
    run_dir = run_eval(
        cfg, questions, store, out_dir=tmp_path / "results", prices=prices, judge=judge
    )
    rec = read_records(run_dir / "predictions.jsonl")[0]
    assert rec.error is None
    assert rec.judge is None and rec.judge_error and "judge correctness:" in rec.judge_error
    assert rec.faith is not None and rec.faith.claims == 1 and rec.faith.score == 1.0
    assert rec.numeric_match is True and rec.failure == "none"
    assert rec.judge_cost_usd > 0  # the faithfulness call was made and paid for
    assert len(judge.calls) == 2  # correctness (failed to parse) + faithfulness
    # The harness hands its store to the judge: every cited chunk is rendered in full from the
    # index (not the <=300-char display snippet), and the gold answer stays hidden.
    faith_prompt = judge.calls[1]["messages"][0].content
    cited = [c for c in rec.citations if c.valid and c.chunk_id]
    assert cited, "the fixture answer must cite a retrieved chunk for this check to mean anything"
    for chunk in store.get_chunks([c.chunk_id for c in cited if c.chunk_id]):
        assert " ".join(chunk.text.split()) in faith_prompt
    assert "Reference answer" not in faith_prompt


def test_llm_judge_rationale_is_persisted_as_a_digest_only(
    store: DuckDBStore, questions: list[FBQuestion], prices: PriceTable, tmp_path: Path
) -> None:
    """The correctness judge sees the gold answer and is asked to name the decisive difference,
    so its rationale restates dataset text; predictions.jsonl must carry only its digest."""
    gold = questions[0]
    rationale = (
        f"The reference answer {gold.answer} to '{gold.question}' matches the prediction; "
        f"the justification says: {gold.justification}"
    )
    judge = FixedProvider(
        parsed={"label": "correct", "rationale": rationale},
        text=json.dumps({"label": "correct", "rationale": rationale}),
        provider="anthropic",
        model="claude-test",
    )
    cfg = rag_cfg(name="rag_judged_leak", judge="anthropic:claude-test", limit=1)
    run_dir = run_eval(
        cfg, questions, store, out_dir=tmp_path / "results", prices=prices, judge=judge
    )
    persisted = (run_dir / "predictions.jsonl").read_text(encoding="utf-8")
    # The gold *answer* is not on this list: the extractive mock quotes the public-domain filing
    # sentence that states the figure, which is our prediction and may be persisted.
    for leak in (rationale, gold.question, gold.justification):
        assert leak not in persisted
    for text in DATASET_TEXT:
        assert text not in persisted
    rec = read_records(run_dir / "predictions.jsonl")[0]
    assert rec.judge is not None and rec.judge.label == "correct"
    assert rec.judge.rationale == f"{RATIONALE_DIGEST_PREFIX}{sha256_hex(rationale)}"
    # The judge itself still saw the full gold context; only the persisted form is redacted.
    assert gold.answer in judge.calls[0]["messages"][0].content


def test_judge_object_override_is_accepted(
    store: DuckDBStore, questions: list[FBQuestion], prices: PriceTable, tmp_path: Path
) -> None:
    cfg = rag_cfg(name="rag_rule_obj", judge="anthropic:claude-sonnet-5", limit=1)
    run_dir = run_eval(
        cfg, questions, store, out_dir=tmp_path / "results", prices=prices, judge=RuleJudge()
    )
    rec = read_records(run_dir / "predictions.jsonl")[0]
    assert rec.judge is not None and rec.judge.judge_model == "rule"


def test_build_fixture_index_matches_pdf_route(embedder: HashingEmbedder) -> None:
    db = DuckDBStore(":memory:", embed_dim=embedder.dim)
    db.init_schema(embedder.name, embedder.dim)
    n_chunks = build_fixture_index(db, embedder, FIXTURE_PAGES)
    assert n_chunks >= 6
    counts = db.counts()
    assert counts["documents"] == 2 and counts["pages"] == 6
    docs = {d.doc_name: d for d in db.list_documents()}
    assert docs[TOP_DOC].ticker == "FIX" and docs[TOP_DOC].source_kind == "fixture"
    page = db.get_pages(TOP_DOC, [1])[0]
    assert page.text == fixture_docs()[TOP_DOC]["pages"][0]
    db.close()


def test_config_yaml_loading_and_validation(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS_DIR / "rag_mock.yaml")
    assert cfg.name == "rag_mock" and cfg.mode == "rag" and cfg.judge == "rule"
    assert cfg.row_kind == "smoke" and cfg.cassette_mode == "off"
    assert cfg.questions_path == Path("tests/fixtures/fb_mini.jsonl")
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        loaded = EvalConfig.from_yaml(path)
        assert loaded.name == path.stem
    retrieval = load_config(CONFIGS_DIR / "retrieval_hybrid.yaml")
    assert retrieval.row_kind == "retrieval" and retrieval.k == 20
    assert "index" in retrieval.pending_reason
    real = load_config(CONFIGS_DIR / "agent_hybrid_claude.yaml")
    assert real.row_kind == "llm" and "ANTHROPIC_API_KEY" in real.pending_reason
    assert real.cassette_mode == "record"

    bad = tmp_path / "bad.yaml"
    bad.write_text("mode: rag\nprovider: mock\nembedder: hashing\nk: 99\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="k"):
        load_config(bad)
    bad.write_text("mode: rag\nprovider: mock\nembedder: hashing\neffort: max\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="effort"):
        load_config(bad)
    bad.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)
    with pytest.raises(ValueError, match="directory name"):
        rag_cfg(name="../escape")


def test_make_run_id_shape() -> None:
    from datetime import UTC, datetime

    assert make_run_id("abcdef0123456789", datetime(2026, 9, 11, 14, 5, tzinfo=UTC)) == (
        "abcdef0_20260911-1405"
    )
    assert make_run_id("unknown").startswith("unknown_")


def test_mock_provider_is_deterministic_across_runs(
    store: DuckDBStore, questions: list[FBQuestion], prices: PriceTable, tmp_path: Path
) -> None:
    a = run_eval(rag_cfg(), questions, store, out_dir=tmp_path / "a", prices=prices)
    b = run_eval(rag_cfg(), questions, store, out_dir=tmp_path / "b", prices=prices)
    strip = ("timestamp", "run_id", "latency_ms", "retrieval_ms", "llm_ms")
    ra = [r.model_dump(exclude=set(strip)) for r in read_records(a / "predictions.jsonl")]
    rb = [r.model_dump(exclude=set(strip)) for r in read_records(b / "predictions.jsonl")]
    assert ra == rb
    assert MockProvider().model == "mock-extractive"
