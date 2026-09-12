"""Fixtures for the eval tests: the fixture corpus (``tests/fixtures/eval_fixture_pages.json``)
rendered to a reportlab PDF, extracted with pypdfium2 and ingested into an in-memory store;
the six ``fb_mini.jsonl`` questions whose evidence lives on those pages; a test price table;
and small synthetic EvalRecords for the metric / report tests. Everything is offline.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from secqa.core.contracts import (
    Citation,
    DocumentMeta,
    EvalRecord,
    FBQuestion,
    JudgeVerdict,
    LLMResponse,
    Message,
    Page,
    StopReason,
    ToolSpec,
    Usage,
)
from secqa.core.ids import sha256_hex
from secqa.embeddings import HashingEmbedder
from secqa.eval.financebench import load_questions_jsonl
from secqa.indexing import ingest_document
from secqa.ingest import extract_pdf_pages
from secqa.providers.base import BaseProvider, Effort
from secqa.providers.pricing import PriceTable
from secqa.store import DuckDBStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE_PAGES = FIXTURES / "eval_fixture_pages.json"
FB_MINI = FIXTURES / "fb_mini.jsonl"
FB_ROWS = FIXTURES / "eval_fb_rows.jsonl"
VERDICTS_DIR = FIXTURES / "eval_judge_verdicts"
DIM = 64
TOP_DOC = "FIXTURE_2023_10K"

PRICES: dict[str, Any] = {
    "as_of": "2026-09-11",
    "models": {
        "openai": {"gpt-test": {"input": 4.0, "cached_input": 0.4, "output": 16.0}},
        "anthropic": {
            "claude-test": {"input": 2.0, "cached_input": 0.2, "output": 10.0},
            "claude-swap": {"input": 1.0, "cached_input": 0.1, "output": 5.0},
        },
    },
}


def fixture_docs() -> dict[str, dict[str, Any]]:
    return json.loads(FIXTURE_PAGES.read_text(encoding="utf-8"))


class FixedProvider(BaseProvider):
    """Return one fixed response for every call and record the calls (judge tests)."""

    def __init__(
        self,
        *,
        text: str = "",
        parsed: dict[str, Any] | None = None,
        stop_reason: StopReason = "end_turn",
        usage: Usage | None = None,
        provider: str = "anthropic",
        model: str = "claude-test",
    ) -> None:
        self.provider = provider
        self.model = model
        self._text = text
        self._parsed = parsed
        self._stop_reason: StopReason = stop_reason
        self._usage = usage or Usage(input_tokens=500, output_tokens=40)
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "messages": messages,
                "system": system,
                "tools": tools,
                "json_schema": json_schema,
                "max_tokens": max_tokens,
                "effort": effort,
            }
        )
        response = LLMResponse(
            text=self._text,
            tool_calls=[],
            usage=self._usage,
            provider=self.provider,
            model=self.model,
            latency_ms=1.0,
            stop_reason=self._stop_reason,
            parsed=self._parsed,
        )
        self._log_response(response)
        return response


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=DIM)


@pytest.fixture(scope="session")
def questions() -> list[FBQuestion]:
    return load_questions_jsonl(FB_MINI)


@pytest.fixture
def prices() -> PriceTable:
    return PriceTable.from_dict(PRICES, source="tests/eval/conftest.py")


@pytest.fixture
def fixture_pdfs(fixture_pdf_factory: Callable[..., Path]) -> dict[str, Path]:
    """One reportlab PDF per fixture document, drawn from ``eval_fixture_pages.json``."""
    return {
        doc_name: fixture_pdf_factory(pages=list(doc["pages"]), name=doc_name)
        for doc_name, doc in fixture_docs().items()
    }


@pytest.fixture
def store(embedder: HashingEmbedder, fixture_pdfs: dict[str, Path]) -> Iterator[DuckDBStore]:
    """In-memory index built from the fixture PDFs through the real ingest path."""
    db = DuckDBStore(":memory:", embed_dim=DIM)
    db.init_schema(embedder.name, embedder.dim)
    docs = fixture_docs()
    for doc_name, path in fixture_pdfs.items():
        pages: list[Page] = extract_pdf_pages(path, doc_name)
        doc = docs[doc_name]
        meta = DocumentMeta(
            doc_name=doc_name,
            company=doc["company"],
            ticker=doc["ticker"],
            cik=doc.get("cik"),
            form=doc["form"],
            fiscal_year=doc["fiscal_year"],
            period_end=None,
            source_kind="fixture",
            source_url=f"fixture://{doc_name}",
            source_sha256=sha256_hex(path.read_bytes()),
            n_pages=len(pages),
            ingested_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
        )
        ingest_document(db, embedder, pages, meta)
    db.rebuild_fts()
    yield db
    db.close()


def make_record(
    financebench_id: str,
    *,
    question_type: str = "metrics-generated",
    mode: str = "rag",
    abstained: bool = False,
    grounded: bool = True,
    value: float | None = 1.0,
    numeric: bool | None = None,
    judge_label: str | None = None,
    retrieved: list[tuple[str, int]] | None = None,
    gold: list[tuple[str, int]] | None = None,
    citations: list[Citation] | None = None,
    terminated_by: str = "single_shot",
    error: str | None = None,
    cost_usd: float = 0.01,
    latency_ms: float = 100.0,
    run_id: str = "abc1234_20260911-1200",
) -> EvalRecord:
    """A synthetic record with sensible defaults (metrics recomputed from retrieved/gold)."""
    from secqa.eval.metrics import gold_page_mrr, page_recall_at_k

    retrieved = retrieved if retrieved is not None else [(TOP_DOC, 1), (TOP_DOC, 2)]
    gold = gold if gold is not None else [(TOP_DOC, 1)]
    judge = (
        JudgeVerdict(
            label=judge_label,  # type: ignore[arg-type]
            rationale="synthetic",
            judge_model="anthropic:claude-test",
            judge_version="v1",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        if judge_label
        else None
    )
    return EvalRecord(
        financebench_id=financebench_id,
        question_type=question_type,
        config_name="synthetic",
        run_id=run_id,
        git_sha="abc1234" + "0" * 33,
        index_sha="idx" + "0" * 61,
        prompt_hashes={"rag_system.md": "0" * 64},
        models_yaml_as_of="2026-09-11",
        provider="openai",
        model="gpt-test",
        mode=mode,  # type: ignore[arg-type]
        embedder="hashing",
        strategy="hybrid",
        k=8,
        answer_text="INSUFFICIENT EVIDENCE" if abstained else "The value was $1 million.",
        value=None if abstained else value,
        unit=None if abstained else "USD",
        abstained=abstained,
        grounded=grounded,
        citations=citations if citations is not None else [],
        retrieved_pages=retrieved,
        gold_pages=gold,
        page_recall_5=page_recall_at_k(retrieved, gold, 5),
        page_recall_10=page_recall_at_k(retrieved, gold, 10),
        page_recall_20=page_recall_at_k(retrieved, gold, 20),
        overlap_recall_10=page_recall_at_k(retrieved, gold, 10),
        gold_page_mrr=gold_page_mrr(retrieved, gold),
        numeric_match=numeric,
        judge=judge,
        faith=None,
        citation_verified_rate=(
            (sum(1 for c in citations if c.verified) / len(citations)) if citations else None
        ),
        failure="none",
        usage=Usage(input_tokens=100, output_tokens=10),
        cost_usd=cost_usd,
        judge_cost_usd=0.001 if judge else 0.0,
        latency_ms=latency_ms,
        retrieval_ms=5.0,
        llm_ms=latency_ms - 5.0,
        steps=1,
        tool_calls=0,
        terminated_by=terminated_by,  # type: ignore[arg-type]
        error=error,
        timestamp=datetime(2026, 9, 11, 12, 30, tzinfo=UTC),
    )


def write_predictions(run_dir: Path, records: list[EvalRecord]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.model_dump(mode="json")) + "\n")
    return path


def verified_citation(verified: bool = True) -> Citation:
    return Citation(
        ref="chunk:" + "a" * 40,
        kind="chunk",
        doc_name=TOP_DOC,
        page_num=1,
        chunk_id="a" * 40,
        quote="Total net sales were $1,577 million in fiscal 2023",
        snippet="Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022.",
        verified=verified,
        valid=True,
    )
