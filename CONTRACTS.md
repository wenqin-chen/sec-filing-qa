# Shared contracts (every module must agree on these)

```python
# src/secqa/core/contracts.py  — the ONLY definitions of these types; every module imports from here.
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


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------- documents ----------
class DocumentMeta(Frozen):
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
    doc_name: str
    page_num: int  # 1-based physical page (pypdfium2 index + 1); FinanceBench evidence_page_num + 1
    text: str


class Chunk(Frozen):
    chunk_id: str  # core.ids.chunk_id(doc_name, page_num, chunk_idx, text)
    doc_name: str
    page_num: int
    chunk_idx: int  # 0-based within page
    section: str | None = None
    text: str
    n_tokens: int


class Hit(Frozen):
    chunk: Chunk
    score: float  # bm25 score, cosine sim, or RRF score
    rank: int  # 1-based in returned list
    source: RetrievalStrategy
    bm25_rank: int | None = None
    dense_rank: int | None = None


class HitView(Frozen):  # API / tool-facing projection
    chunk_id: str
    doc_name: str
    page_num: int
    section: str | None
    score: float
    snippet: str  # snippet <= 1200 chars


class RetrievalFilters(Frozen):
    ticker: str | None = None
    doc_names: list[str] | None = None
    fiscal_year: int | None = None
    form: str | None = None


class RetrievalResult(Frozen):
    query: str
    strategy: RetrievalStrategy
    k: int
    hits: list[Hit]
    filters: RetrievalFilters | None = None
    latency_ms: float


# ---------- XBRL ----------
class FactRow(Frozen):
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
        return f"xbrl:{self.tag}|FY{self.fy}|{self.accn}"


class SqlResult(Frozen):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    sql: str


# ---------- LLM provider ----------
class ToolSpec(Frozen):
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema, additionalProperties=false, required listed


class ToolCall(Frozen):
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(Frozen):
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False  # content is a JSON string, <= max_result_chars


class Message(Frozen):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant only
    tool_results: list[ToolResult] = Field(
        default_factory=list
    )  # role == 'tool' only; ALL results of one step in ONE message


class Usage(Frozen):
    # input_tokens = UNCACHED prompt tokens; total prompt = input + cache_read + cache_write
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, o: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + o.input_tokens,
            output_tokens=self.output_tokens + o.output_tokens,
            cache_read_tokens=self.cache_read_tokens + o.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + o.cache_write_tokens,
        )


class LLMResponse(Frozen):
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    provider: str
    model: str
    latency_ms: float
    stop_reason: StopReason
    parsed: dict[str, Any] | None = (
        None  # json_schema output parsed (json.loads), None if not requested/failed
    )
    raw_id: str | None = None
    cached: bool = False  # served from a cassette


@runtime_checkable
class LLMProvider(Protocol):
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

    # Rules: never raise on refusal (stop_reason='refusal', text=''); raise ProviderError(retryable: bool) otherwise;
    # temperature is provider-internal (0 where accepted, omitted for Anthropic Opus 5/Sonnet 5), recorded in provider.params().


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
        kind: Literal["query", "passage"] = "passage",
    ) -> np.ndarray: ...

    # float32, shape (n, dim), L2-normalised; blank texts embed to an all-zero row (not unit norm)


# ---------- answers ----------
class CitationRef(Frozen):
    ref: str  # 'chunk:<chunk_id>' | 'xbrl:<tag>|FY<fy>|<accn>'
    quote: str = ""  # model-supplied span; verified against the chunk text


class Citation(Frozen):
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
    verified: (
        bool  # quote is a normalised substring (>=20 chars) of the chunk / fact row was returned
    )
    valid: bool  # ref resolved to something this request actually retrieved


class TraceStep(Frozen):
    step: int
    kind: Literal["llm", "tool", "retrieval", "verify"]
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_preview: str = ""
    latency_ms: float = 0.0
    usage: Usage | None = None
    error: str | None = None


class Answer(Frozen):
    request_id: str
    question: str
    text: str
    value: float | None = None
    unit: str | None = None  # structured final number when the model gives one
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
    prompt_hashes: dict[str, str] = Field(
        default_factory=dict
    )  # {prompt file: sha256} of the prompt(s) THIS request used; rag.prompt_hashes() lists all


# ---------- evaluation ----------
class Evidence(Frozen):
    doc_name: str
    page_num: int
    text: str  # page_num already 1-based


class FBQuestion(Frozen):
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
    label: Literal["correct", "incorrect", "abstain"]
    rationale: str
    judge_model: str
    judge_version: str
    usage: Usage


class FaithVerdict(Frozen):
    claims: int
    supported: int
    score: float | None
    judge_model: str
    judge_version: str
    usage: Usage


FailureClass = Literal[
    "retrieval_miss",
    "reasoning_error",
    "calculation_error",
    "tool_error",
    "budget",
    "unverified_citation",
    "none",
]


class EvalRecord(Frozen):  # one line of predictions.jsonl — contains NO dataset text
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
    error: str | None = None  # the answer failed (ProviderError): record is unscored and not completed
    judge_error: str | None = None  # a judge call failed: answer kept and scored, that verdict is None
    timestamp: datetime


class RunSummary(Frozen):
    config_name: str
    run_id: str
    n: int
    n_completed: int
    n_dataset: int | None = None  # questions available before `limit`; n < n_dataset = subset run
    metrics: dict[
        str, float | None
    ]  # accuracy, abstain_rate, hallucination_rate, numeric_match_rate, numeric_coverage,
    # faithfulness, citation_verified_rate, grounded_rate, page_recall_10, overlap_recall_10, ...
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
```
Cross-module rules: (1) tool results of one agent step go in ONE `Message(role='tool')`; (2) `Answer.citations[*].snippet` is always store-sourced; (3) providers never emit `temperature` to Anthropic Opus 5 / Sonnet 5 and use `thinking={'type':'adaptive'}` + `output_config.effort`; (4) `page_num` is 1-based everywhere; (5) EvalRecord never contains question/answer/evidence text; (6) every LLM call is appended to `trace` with usage; (7) errors: `ProviderError(retryable)`, `SqlRejected`, `CalcRejected`, `IndexMismatch`, `CassetteMiss`, `ConfigError` all live in `secqa.core.errors` (module-specific errors such as `EdgarError` live in their module); (8) `CitationVerifier.verify` maps are keyed by bare `chunk_id` for `chunks` and by `FactRow.ref` (the full `xbrl:<tag>|FY<fy>|<accn>` string) for `facts` — rag and agent must key `seen_chunks` / `seen_facts` that way; (9) the DuckDB view `financials` is owned by `secqa.xbrl.create_financials_view` (CREATE OR REPLACE); the store's schema only creates a baseline `IF NOT EXISTS` so the name always resolves, and whoever writes `xbrl_facts` directly must call `store.set_manifest(n_facts=...)` (or `rebuild_fts()`) so the manifest count is current; (10) `DuckDBStore.readonly_connection()` returns `secqa.store.ReadOnlyConnection` (execute/fetch*/interrupt/close; one SELECT-type statement per call inside a READ ONLY transaction), not a raw `duckdb.DuckDBPyConnection` — DuckDB cannot open one file read-write and read-only in one process. (11) Every ``src/secqa/prompts/*.md`` is listed in `secqa.rag.prompts.PROMPT_NAMES` and pinned in `tests/fixtures/rag_prompt_hashes.json` (the one registry eval records copy); the module that owns a prompt (`secqa.agent` for `agent_system.md`, `secqa.eval.judge` for `judge_correctness.md` / `judge_faithfulness.md`) loads and hashes it itself and records only its own file in `Answer.prompt_hashes`, and the two hashes must agree byte for byte. (12) `ToolRuntime(max_result_chars=4000)` bounds the WHOLE JSON tool result; `get_pages` caps each page at 6000 chars (per-page `truncated` flag) and then shrinks the result structurally to fit, so api/eval that want up to three full pages must construct `ToolRuntime(store, retriever, max_result_chars=20_000)`. (13) `terminated_by='error'` also covers `final_answer` rejected twice (invalid refs) and a provider refusal: both abstain directly, without the tools-off final call, with the reason in the final `verify` TraceStep.error.

## Offline mode

With no API keys and no network, everything below runs from a clean clone with `uv sync --extra dev` (Python 3.11):

1. Tests: `uv run pytest -m "not live and not slow"` (the CI default). Providers resolve to `mock` / `scripted:<yaml>` (deterministic; MockProvider returns an extractive answer quoting the first ≥20-char sentence of the top passage with its `chunk:<id>` ref, so CitationVerifier marks it verified; ScriptedProvider replays turn-indexed YAML scenarios for agent, judge and injection tests). Embedder is `hashing` (sklearn HashingVectorizer, dim 384, no downloads). EDGAR and OpenAI embedding calls are respx-mocked from fixtures. PDFs are generated in-test with reportlab; no dataset rows or third-party PDFs are committed. Live tests (`@pytest.mark.live`) skip when `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` are absent; `@pytest.mark.slow` (bge-small download) is opt-in.

2. Smoke evaluation: `uv run secqa eval --config configs/rag_mock.yaml --limit 6` and `configs/agent_mock.yaml` run the full harness over `tests/fixtures/fb_mini.jsonl` (6 synthetic questions whose evidence lives in the fixture PDF) with provider=mock, embedder=hashing, judge=`rule` (RuleJudge = numeric_match + abstain detection, used only when the configured judge is `rule`), writing `results/rag_mock/<run_id>/{config.json,predictions.jsonl,summary.json}` and asserting the schema; `secqa report` renders the table with every real row as "pending (not run)". Mock rows are labelled `mock` and are excluded from RESULTS.md by design. CI uploads summary.json as an artifact.

3. Real-corpus numbers without keys (network only, still no keys): `secqa data financebench --pdfs` (HF dataset + PDFs), `secqa ingest financebench --embedder local`, then `secqa eval --config configs/retrieval_hybrid.yaml` (mode=rag with provider=`mock:abstain` records retrieval metrics only) yields page_recall@k, overlap recall and MRR for bm25/dense/hybrid — the first honest README rows.

4. Rescoring published runs without keys: download `cassettes/<run_id>.tar.zst` from the GitHub Release and run `uv run secqa rescore --run results/<config>/<run_id>` with `SECQA_CASSETTE_MODE=replay`; ReplayCacheProvider serves every LLM and judge call from the cassette (CassetteMiss fails loudly), and all metrics, CIs and the table regenerate byte-for-byte. `secqa doctor` reports which capabilities are available in the current environment (keys, network, index, torch) so pending cells are explained, not guessed.

5. Container with nothing configured: `docker run -p 8080:8080 ghcr.io/<owner>/sec-filing-qa:latest` builds the bundled 2-doc fixture index at start, serves provider=mock, and `/readyz` returns 200 — the same path the deploy smoke step exercises.
