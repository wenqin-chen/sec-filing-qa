"""Tests for secqa.core.contracts: JSON round-trips, immutability, protocols."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any, Literal

import numpy as np
import pytest
from pydantic import ValidationError

from secqa.core.contracts import (
    Answer,
    Chunk,
    Citation,
    CitationRef,
    DocumentMeta,
    Embedder,
    EvalRecord,
    Evidence,
    FactRow,
    FaithVerdict,
    FBQuestion,
    Hit,
    HitView,
    JudgeVerdict,
    LLMProvider,
    LLMResponse,
    Message,
    Page,
    RetrievalFilters,
    RetrievalResult,
    RunSummary,
    SqlResult,
    ToolCall,
    ToolResult,
    ToolSpec,
    TraceStep,
    Usage,
)
from secqa.core.ids import chunk_id


def _chunk(text: str = "Total net sales were $1,577 million in fiscal 2023.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id("FIXTURE_2023_10K", 1, 0, text),
        doc_name="FIXTURE_2023_10K",
        page_num=1,
        chunk_idx=0,
        section="Item 7.",
        text=text,
        n_tokens=12,
    )


def _citation() -> Citation:
    c = _chunk()
    return Citation(
        ref=f"chunk:{c.chunk_id}",
        kind="chunk",
        doc_name=c.doc_name,
        page_num=c.page_num,
        chunk_id=c.chunk_id,
        quote="Total net sales were $1,577 million",
        snippet=c.text[:300],
        verified=True,
        valid=True,
    )


def _answer() -> Answer:
    c = _chunk()
    return Answer(
        request_id="0" * 32,
        question="What were total net sales in fiscal 2023?",
        text="Total net sales were $1,577 million in fiscal 2023.",
        value=1577e6,
        unit="USD",
        abstained=False,
        citations=[_citation()],
        grounded=True,
        calculation=None,
        retrieved=[
            HitView(
                chunk_id=c.chunk_id,
                doc_name=c.doc_name,
                page_num=1,
                section="Item 7.",
                score=0.9,
                snippet=c.text,
            )
        ],
        trace=[
            TraceStep(step=1, kind="retrieval", name="hybrid", latency_ms=3.2),
            TraceStep(
                step=2,
                kind="llm",
                name="mock",
                usage=Usage(input_tokens=100, output_tokens=20),
                latency_ms=1.0,
            ),
        ],
        usage=Usage(input_tokens=100, output_tokens=20),
        cost_usd=0.0,
        latency_ms=4.2,
        retrieval_ms=3.2,
        llm_ms=1.0,
        provider="mock",
        model="mock",
        mode="rag",
        terminated_by="single_shot",
        prompt_hashes={"rag_system.md": "abc"},
    )


def _eval_record() -> EvalRecord:
    return EvalRecord(
        financebench_id="financebench_id_00001",
        question_type="metrics-generated",
        config_name="rag_mock",
        run_id="abc1234_20260911-1200",
        git_sha="abc1234",
        index_sha="deadbeef",
        prompt_hashes={"rag_system.md": "abc"},
        models_yaml_as_of="2026-09-11",
        provider="mock",
        model="mock",
        mode="rag",
        embedder="hashing-384",
        strategy="hybrid",
        k=8,
        answer_text="$1,577 million",
        value=1577e6,
        unit="USD",
        abstained=False,
        grounded=True,
        citations=[_citation()],
        retrieved_pages=[("FIXTURE_2023_10K", 1), ("FIXTURE_2023_10K", 2)],
        gold_pages=[("FIXTURE_2023_10K", 1)],
        page_recall_5=1.0,
        page_recall_10=1.0,
        page_recall_20=1.0,
        overlap_recall_10=1.0,
        gold_page_mrr=1.0,
        numeric_match=True,
        judge=JudgeVerdict(
            label="correct",
            rationale="matches",
            judge_model="rule",
            judge_version="v0",
            usage=Usage(),
        ),
        faith=FaithVerdict(
            claims=1, supported=1, score=1.0, judge_model="rule", judge_version="v0", usage=Usage()
        ),
        citation_verified_rate=1.0,
        failure="none",
        usage=Usage(input_tokens=100, output_tokens=20),
        cost_usd=0.0,
        judge_cost_usd=0.0,
        latency_ms=4.2,
        retrieval_ms=3.2,
        llm_ms=1.0,
        steps=1,
        tool_calls=0,
        terminated_by="single_shot",
        timestamp=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
    )


def test_answer_json_round_trip() -> None:
    answer = _answer()
    payload = answer.model_dump_json()
    restored = Answer.model_validate_json(payload)
    assert restored == answer
    assert json.loads(payload)["citations"][0]["verified"] is True


def test_eval_record_json_round_trip_and_no_dataset_text() -> None:
    record = _eval_record()
    payload = record.model_dump_json()
    restored = EvalRecord.model_validate_json(payload)
    assert restored == record
    assert restored.retrieved_pages[0] == ("FIXTURE_2023_10K", 1)
    # The record carries ids and pages only: no question / gold answer / evidence fields exist.
    fields = set(EvalRecord.model_fields)
    assert not fields & {"question", "answer", "justification", "evidence"}


def test_run_summary_round_trip() -> None:
    summary = RunSummary(
        config_name="rag_mock",
        run_id="abc1234_20260911-1200",
        n=6,
        n_completed=6,
        metrics={"accuracy": 0.5, "faithfulness": None},
        ci95={"accuracy": (0.1, 0.9)},
        by_question_type={"metrics-generated": {"accuracy": 0.5}},
        failures={"retrieval_miss": 1},
        judge_numeric_disagreements=["financebench_id_00002"],
        latency_p50_ms=10.0,
        latency_p95_ms=20.0,
        cost_total_usd=0.0,
        cost_per_q_usd=0.0,
        judge_cost_usd=0.0,
        provider="mock",
        model="mock",
        embedder="hashing-384",
        judge_model="rule",
        judge_version="v0",
        git_sha="abc1234",
        index_sha="deadbeef",
        prompt_hashes={},
        models_yaml_as_of="2026-09-11",
        cassettes=None,
        started_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
        finished_at=datetime(2026, 9, 11, 12, 5, tzinfo=UTC),
    )
    assert RunSummary.model_validate_json(summary.model_dump_json()) == summary
    assert summary.ci95["accuracy"] == (0.1, 0.9)


def test_frozen_models_reject_mutation() -> None:
    answer = _answer()
    with pytest.raises(ValidationError):
        answer.text = "changed"  # type: ignore[misc]
    chunk = _chunk()
    with pytest.raises(ValidationError):
        chunk.page_num = 2  # type: ignore[misc]
    assert hash(chunk) == hash(_chunk())


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Page(doc_name="d", page_num=1, text="t", extra="nope")  # type: ignore[call-arg]


def test_literal_fields_validated() -> None:
    with pytest.raises(ValidationError):
        Message(role="system", content="x")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RetrievalResult(query="q", strategy="fuzzy", k=1, hits=[], latency_ms=1.0)  # type: ignore[arg-type]


def test_document_meta_dates_round_trip() -> None:
    meta = DocumentMeta(
        doc_name="FIXTURE_2023_10K",
        company="Fixture Corp",
        ticker="FIX",
        cik="0000000001",
        form="10-K",
        fiscal_year=2023,
        period_end=date(2023, 12, 31),
        source_kind="fixture",
        source_url="file:///fixture.pdf",
        source_sha256="0" * 64,
        n_pages=3,
        ingested_at=datetime(2026, 9, 11, tzinfo=UTC),
    )
    restored = DocumentMeta.model_validate_json(meta.model_dump_json())
    assert restored.period_end == date(2023, 12, 31)
    assert restored == meta


def test_usage_addition() -> None:
    total = Usage(input_tokens=1, output_tokens=2) + Usage(
        input_tokens=10, output_tokens=20, cache_read_tokens=5, cache_write_tokens=7
    )
    assert total == Usage(
        input_tokens=11, output_tokens=22, cache_read_tokens=5, cache_write_tokens=7
    )
    assert sum([Usage(input_tokens=1), Usage(input_tokens=2)], Usage()) == Usage(input_tokens=3)


def test_fact_row_ref() -> None:
    row = FactRow(
        cik="0000000001",
        ticker="FIX",
        taxonomy="us-gaap",
        tag="Revenues",
        unit="USD",
        fy=2023,
        fp="FY",
        form="10-K",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        val=1577e6,
        accn="0000000001-24-000001",
        filed=date(2024, 2, 1),
        frame="CY2023",
    )
    assert row.ref == "xbrl:Revenues|FY2023|0000000001-24-000001"
    assert FactRow.model_validate_json(row.model_dump_json()) == row


def test_message_and_tool_models() -> None:
    call = ToolCall(id="c1", name="search_filings", arguments={"query": "net sales", "k": 5})
    assistant = Message(role="assistant", content="", tool_calls=[call])
    tool = Message(
        role="tool",
        tool_results=[
            ToolResult(tool_call_id="c1", name="search_filings", content='{"hits": []}'),
            ToolResult(
                tool_call_id="c2", name="calculate", content='{"error": "x"}', is_error=True
            ),
        ],
    )
    assert assistant.tool_results == []
    assert len(tool.tool_results) == 2
    spec = ToolSpec(
        name="calculate",
        description="Evaluate arithmetic",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    )
    assert ToolSpec.model_validate_json(spec.model_dump_json()) == spec
    response = LLMResponse(
        text="",
        tool_calls=[call],
        usage=Usage(input_tokens=5, output_tokens=1),
        provider="mock",
        model="mock",
        latency_ms=0.5,
        stop_reason="tool_use",
    )
    assert response.parsed is None and response.cached is False


def test_misc_models_construct() -> None:
    hit = Hit(chunk=_chunk(), score=0.1, rank=1, source="hybrid", bm25_rank=1, dense_rank=3)
    result = RetrievalResult(
        query="q",
        strategy="hybrid",
        k=1,
        hits=[hit],
        filters=RetrievalFilters(ticker="FIX"),
        latency_ms=1.0,
    )
    assert result.hits[0].chunk.page_num == 1
    sql = SqlResult(columns=["a"], rows=[[1], [None]], row_count=2, truncated=False, sql="SELECT 1")
    assert SqlResult.model_validate_json(sql.model_dump_json()) == sql
    question = FBQuestion(
        id="financebench_id_00001",
        company="Fixture Corp",
        doc_name="FIXTURE_2023_10K",
        question_type="metrics-generated",
        question="q",
        answer="a",
        justification="j",
        evidence=[Evidence(doc_name="FIXTURE_2023_10K", page_num=1, text="e")],
    )
    assert question.evidence[0].page_num == 1
    assert CitationRef(ref="chunk:abc").quote == ""


class _FakeProvider:
    provider = "mock"
    model = "mock"

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        effort: Literal["low", "medium", "high"] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text="ok",
            tool_calls=[],
            usage=Usage(),
            provider=self.provider,
            model=self.model,
            latency_ms=0.0,
            stop_reason="end_turn",
        )


class _FakeEmbedder:
    name = "fake"
    dim = 4

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
        kind: Literal["query", "passage"] = "passage",
    ) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        out[:, 0] = 1.0
        return out


def test_protocols_are_runtime_checkable() -> None:
    assert isinstance(_FakeProvider(), LLMProvider)
    assert isinstance(_FakeEmbedder(), Embedder)
    assert not isinstance(object(), LLMProvider)
    assert _FakeEmbedder().embed(["a", "b"]).shape == (2, 4)
