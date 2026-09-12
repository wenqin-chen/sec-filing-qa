"""Shared contracts: the ONLY definitions of these types; every module imports from here.

The models are frozen pydantic models (``Frozen``) so that values flowing between modules are
immutable and JSON round-trippable. Protocols (``LLMProvider``, ``Embedder``) are
``runtime_checkable`` so tests can assert an implementation satisfies them with ``isinstance``.

Cross-module rules (see CONTRACTS.md):

1. tool results of one agent step go in ONE ``Message(role='tool')``;
2. ``Answer.citations[*].snippet`` is always store-sourced, never model-sourced;
3. providers never send ``temperature`` to Anthropic Opus 5 / Sonnet 5;
4. ``page_num`` is 1-based everywhere;
5. ``EvalRecord`` never contains question/answer/evidence text;
6. every LLM call is appended to ``trace`` with usage;
7. errors live in :mod:`secqa.core.errors`.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Role = Literal["user", "assistant", "tool"]  # system prompt is passed separately
RetrievalStrategy = Literal["bm25", "dense", "hybrid"]
Mode = Literal["closed_book", "oracle", "rag", "agent"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]
Terminated = Literal[
    "final_answer", "single_shot", "budget", "max_steps", "error", "empty_retrieval"
]
FailureClass = Literal[
    "retrieval_miss",
    "reasoning_error",
    "calculation_error",
    "tool_error",
    "budget",
    "unverified_citation",
    "none",
]


class Frozen(BaseModel):
    """Immutable, strict base model: unknown fields are rejected, instances are hashable."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------- documents ----------


class DocumentMeta(Frozen):
    """One ingested filing."""

    doc_name: str  # unique; FinanceBench name ('3M_2022_10K') or '<TICKER>_<FY>_<FORM>' for EDGAR
    company: str
    ticker: str | None = None
    cik: str | None = None  # 10-digit zero-padded
    form: str  # '10-K' | '10-Q' | '10-K/A' | '8-K'
    fiscal_year: int | None = None
    period_end: date | None = None
    source_kind: Literal["financebench_pdf", "edgar_html", "fixture"]
    source_url: str
    source_sha256: str
    n_pages: int
    ingested_at: datetime


class Page(Frozen):
    """Text of one physical page (1-based)."""

    doc_name: str
    page_num: int  # 1-based physical page (pypdfium2 index + 1); FinanceBench evidence_page_num + 1
    text: str


class Chunk(Frozen):
    """A page-bounded chunk of text with a stable id (``core.ids.chunk_id``)."""

    chunk_id: str
    doc_name: str
    page_num: int
    chunk_idx: int  # 0-based within page
    section: str | None = None
    text: str
    n_tokens: int


class Hit(Frozen):
    """One retrieval hit."""

    chunk: Chunk
    score: float  # bm25 score, cosine sim, or RRF score
    rank: int  # 1-based in returned list
    source: RetrievalStrategy
    bm25_rank: int | None = None
    dense_rank: int | None = None


class HitView(Frozen):
    """API / tool-facing projection of a hit."""

    chunk_id: str
    doc_name: str
    page_num: int
    section: str | None
    score: float
    snippet: str  # <= 1200 chars


class RetrievalFilters(Frozen):
    """Optional narrowing of the corpus for one retrieval."""

    ticker: str | None = None
    doc_names: list[str] | None = None
    fiscal_year: int | None = None
    form: str | None = None


class RetrievalResult(Frozen):
    """Result of one ``Retriever.retrieve`` call."""

    query: str
    strategy: RetrievalStrategy
    k: int
    hits: list[Hit]
    filters: RetrievalFilters | None = None
    latency_ms: float


# ---------- XBRL ----------


class FactRow(Frozen):
    """One row of ``xbrl_facts`` (flattened SEC companyfacts)."""

    cik: str
    ticker: str
    taxonomy: str
    tag: str
    unit: str
    fy: int | None
    fp: str | None
    form: str | None
    start_date: date | None
    end_date: date | None
    val: float
    accn: str
    filed: date | None
    frame: str | None
    concept_used: str | None = None  # set by lookup_fact when an alias resolved

    @property
    def ref(self) -> str:
        """Citation ref of this fact: ``xbrl:<tag>|FY<fy>|<accn>``."""
        return f"xbrl:{self.tag}|FY{self.fy}|{self.accn}"


class SqlResult(Frozen):
    """Result of a guarded read-only SQL query."""

    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    sql: str


# ---------- LLM provider ----------


class ToolSpec(Frozen):
    """A tool offered to the model (JSON Schema, additionalProperties=false, required listed)."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(Frozen):
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(Frozen):
    """Result returned to the model; ``content`` is a JSON string <= ``max_result_chars``."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


class Message(Frozen):
    """One conversation message. ``tool_results`` carries ALL results of one step in ONE message."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant only
    tool_results: list[ToolResult] = Field(default_factory=list)  # role == 'tool' only


class Usage(Frozen):
    """Token usage of one or more LLM calls; ``+`` sums component-wise."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, o: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + o.input_tokens,
            output_tokens=self.output_tokens + o.output_tokens,
            cache_read_tokens=self.cache_read_tokens + o.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + o.cache_write_tokens,
        )


class LLMResponse(Frozen):
    """Provider-neutral response of one ``LLMProvider.complete`` call."""

    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    provider: str
    model: str
    latency_ms: float
    stop_reason: StopReason
    parsed: dict[str, Any] | None = None  # json_schema output parsed; None if not requested/failed
    raw_id: str | None = None
    cached: bool = False  # served from a cassette


@runtime_checkable
class LLMProvider(Protocol):
    """Provider-neutral chat completion with tools and JSON-schema output.

    Rules: never raise on refusal (``stop_reason='refusal'``, ``text=''``); raise
    ``ProviderError(retryable=...)`` otherwise; temperature is provider-internal (0 where accepted,
    omitted for Anthropic Opus 5 / Sonnet 5) and recorded in ``provider.params()``.
    """

    provider: str  # 'openai' | 'anthropic' | 'mock' | 'scripted'
    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        effort: Literal["low", "medium", "high"] | None = None,
    ) -> LLMResponse: ...


@runtime_checkable
class Embedder(Protocol):
    """Text -> float32 array of shape (n, dim), L2-normalised."""

    name: str
    dim: int

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
        kind: Literal["query", "passage"] = "passage",
    ) -> np.ndarray: ...


# ---------- answers ----------


class CitationRef(Frozen):
    """Model-supplied citation: a ref plus the span it claims to quote."""

    ref: str  # 'chunk:<chunk_id>' | 'xbrl:<tag>|FY<fy>|<accn>'
    quote: str = ""  # model-supplied span; verified against the chunk text


class Citation(Frozen):
    """A verified (or flagged) citation shown to the caller."""

    ref: str
    kind: Literal["chunk", "xbrl"]
    doc_name: str | None = None
    page_num: int | None = None
    chunk_id: str | None = None
    tag: str | None = None
    fiscal_year: int | None = None
    accn: str | None = None
    value: float | None = None
    quote: str  # model quote (may be '')
    snippet: str  # from the store, never from the model (<= 300 chars)
    verified: bool  # quote is a normalised substring (>=20 chars) of the chunk / fact row returned
    valid: bool  # ref resolved to something this request actually retrieved


class TraceStep(Frozen):
    """One step of an answer trace (LLM call, tool call, retrieval or verification)."""

    step: int
    kind: Literal["llm", "tool", "retrieval", "verify"]
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_preview: str = ""
    latency_ms: float = 0.0
    usage: Usage | None = None
    error: str | None = None


class Answer(Frozen):
    """The full result of answering one question (also the ``/v1/ask`` response body)."""

    request_id: str
    question: str
    text: str
    value: float | None = None  # structured final number when the model gives one
    unit: str | None = None
    abstained: bool
    citations: list[Citation]
    grounded: bool
    calculation: str | None = None  # rendered from actual calculate() calls
    retrieved: list[HitView]
    trace: list[TraceStep]
    usage: Usage
    cost_usd: float
    latency_ms: float
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    provider: str
    model: str
    mode: Mode
    steps: int = 1
    tool_calls: int = 0
    terminated_by: Terminated
    prompt_hashes: dict[str, str] = Field(default_factory=dict)


# ---------- evaluation ----------


class Evidence(Frozen):
    """Gold evidence span; ``page_num`` already 1-based."""

    doc_name: str
    page_num: int
    text: str


class FBQuestion(Frozen):
    """One FinanceBench question (in memory only; never persisted to results files)."""

    id: str
    company: str
    doc_name: str
    question_type: str
    question: str
    answer: str
    justification: str
    doc_link: str | None = None
    doc_period: str | None = None
    evidence: list[Evidence]


class JudgeVerdict(Frozen):
    """Tri-state correctness verdict from the LLM judge."""

    label: Literal["correct", "incorrect", "abstain"]
    rationale: str
    judge_model: str
    judge_version: str
    usage: Usage


class FaithVerdict(Frozen):
    """Faithfulness verdict: atomic claims judged against cited passages only."""

    claims: int
    supported: int
    score: float | None
    judge_model: str
    judge_version: str
    usage: Usage


class EvalRecord(Frozen):
    """One line of ``predictions.jsonl`` — contains NO dataset text."""

    financebench_id: str
    question_type: str
    config_name: str
    run_id: str
    git_sha: str
    index_sha: str
    prompt_hashes: dict[str, str]
    models_yaml_as_of: str
    provider: str
    model: str
    mode: Mode
    embedder: str
    strategy: RetrievalStrategy
    k: int
    answer_text: str
    value: float | None
    unit: str | None
    abstained: bool
    grounded: bool
    citations: list[Citation]
    retrieved_pages: list[tuple[str, int]]  # distinct pages in rank order (top 20)
    gold_pages: list[tuple[str, int]]  # (doc_name, page_num) only — no text
    page_recall_5: float
    page_recall_10: float
    page_recall_20: float
    overlap_recall_10: float
    gold_page_mrr: float
    numeric_match: bool | None
    judge: JudgeVerdict | None
    faith: FaithVerdict | None
    citation_verified_rate: float | None
    failure: FailureClass
    usage: Usage
    cost_usd: float
    judge_cost_usd: float
    latency_ms: float
    retrieval_ms: float
    llm_ms: float
    steps: int
    tool_calls: int
    terminated_by: Terminated
    error: str | None = None
    timestamp: datetime


class RunSummary(Frozen):
    """``summary.json`` of one evaluation run."""

    config_name: str
    run_id: str
    n: int
    n_completed: int
    metrics: dict[str, float | None]  # accuracy, abstain_rate, hallucination_rate, ...
    ci95: dict[str, tuple[float, float]]
    by_question_type: dict[str, dict[str, float | None]]
    failures: dict[str, int]
    judge_numeric_disagreements: list[str]  # financebench_ids
    latency_p50_ms: float
    latency_p95_ms: float
    cost_total_usd: float
    cost_per_q_usd: float
    judge_cost_usd: float
    provider: str
    model: str
    embedder: str
    judge_model: str
    judge_version: str
    git_sha: str
    index_sha: str
    prompt_hashes: dict[str, str]
    models_yaml_as_of: str
    cassettes: str | None
    started_at: datetime
    finished_at: datetime


__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "CitationRef",
    "DocumentMeta",
    "Embedder",
    "EvalRecord",
    "Evidence",
    "FBQuestion",
    "FactRow",
    "FailureClass",
    "FaithVerdict",
    "Frozen",
    "Hit",
    "HitView",
    "JudgeVerdict",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "Mode",
    "Page",
    "RetrievalFilters",
    "RetrievalResult",
    "RetrievalStrategy",
    "Role",
    "RunSummary",
    "SqlResult",
    "StopReason",
    "Terminated",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "TraceStep",
    "Usage",
]
