"""Request and response models of the HTTP API (SPEC section 8).

Every request model forbids unknown fields (a typo such as ``mode_`` is a 422, not a silent
default) and strips surrounding whitespace from strings. Response models reuse the shared
contracts where one exists: :class:`AskResponse` *is* :class:`~secqa.core.contracts.Answer`,
``/v1/search`` returns :class:`~secqa.core.contracts.HitView` rows, ``/v1/filings`` returns
:class:`~secqa.core.contracts.DocumentMeta` and ``/v1/xbrl/query`` returns
:class:`~secqa.core.contracts.SqlResult`. Errors are :class:`ProblemDetail`
(``application/problem+json``, RFC 9457).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from secqa.core.contracts import Answer, HitView, RetrievalFilters, RetrievalStrategy
from secqa.store import IndexManifest

ApiMode = Literal["rag", "agent", "closed_book"]
"""Modes callable over HTTP. ``oracle`` needs gold pages and exists only in the eval harness."""

QUESTION_MAX_CHARS = 2000
K_MIN = 1
K_MAX = 20
DOC_NAMES_MAX = 20
DOC_NAME_MAX_CHARS = 200
SQL_MAX_CHARS = 5000
PROVIDER_SPEC_MAX_CHARS = 100
TICKER_PATTERN = r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$"
FORM_PATTERN = r"^[0-9A-Za-z\-/]{1,12}$"
FISCAL_YEAR_MIN = 1990
FISCAL_YEAR_MAX = 2100


class ApiModel(BaseModel):
    """Base for request bodies: strict about unknown fields, tolerant of stray whitespace."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _FilterFields(ApiModel):
    """The document filters shared by ``/v1/ask`` and ``/v1/search``."""

    ticker: str | None = Field(
        default=None,
        pattern=TICKER_PATTERN,
        description="Restrict to filings of this ticker (case-insensitive).",
        examples=["NVDA"],
    )
    doc_names: list[str] | None = Field(
        default=None,
        max_length=DOC_NAMES_MAX,
        description="Restrict to these document names (for example '3M_2022_10K').",
    )
    fiscal_year: int | None = Field(
        default=None,
        ge=FISCAL_YEAR_MIN,
        le=FISCAL_YEAR_MAX,
        description="Restrict to filings for this fiscal year.",
    )

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, value: str | None) -> str | None:
        return value.upper() if value is not None else None

    @field_validator("doc_names")
    @classmethod
    def _clean_doc_names(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned: list[str] = []
        for name in value:
            name = name.strip()
            if not name:
                raise ValueError("doc_names entries must not be blank")
            if len(name) > DOC_NAME_MAX_CHARS:
                raise ValueError(f"doc_names entries must be <= {DOC_NAME_MAX_CHARS} characters")
            if name not in cleaned:
                cleaned.append(name)
        if not cleaned:
            raise ValueError("doc_names must not be empty when given")
        return cleaned

    def filters(self, *, form: str | None = None) -> RetrievalFilters | None:
        """The :class:`RetrievalFilters` for this request, or ``None`` when unfiltered."""
        if self.ticker is None and self.doc_names is None and self.fiscal_year is None and not form:
            return None
        return RetrievalFilters(
            ticker=self.ticker,
            doc_names=self.doc_names,
            fiscal_year=self.fiscal_year,
            form=form,
        )


class AskRequest(_FilterFields):
    """Body of ``POST /v1/ask``."""

    question: str = Field(
        min_length=1,
        max_length=QUESTION_MAX_CHARS,
        description="The question to answer (1..2000 characters).",
        examples=["What was 3M's total revenue in fiscal 2022?"],
    )
    mode: ApiMode = Field(
        default="rag",
        description=(
            "'rag' = retrieve then one LLM call; 'agent' = tool loop over the local index; "
            "'closed_book' = no evidence at all (baseline)."
        ),
    )
    provider: str | None = Field(
        default=None,
        max_length=PROVIDER_SPEC_MAX_CHARS,
        description=(
            "'mock' | 'mock:abstain' | 'openai:<model>' | 'anthropic:<model>'; the server default "
            "when omitted. Non-default providers need X-API-Key when the server has one."
        ),
        examples=["anthropic:claude-opus-5"],
    )
    k: int = Field(
        default=8,
        ge=K_MIN,
        le=K_MAX,
        description="Passages placed in the prompt (rag mode); agent and closed_book ignore it.",
    )
    max_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Per-request spend cap in USD; clamped to the server cap (SECQA_MAX_COST_USD). "
            "Exceeding it yields 402 with the partial trace."
        ),
    )
    include_trace: bool = Field(
        default=True, description="Include the step-by-step trace in the response."
    )

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("provider")
    @classmethod
    def _provider_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("provider must not be blank when given")
        return value


class AskResponse(Answer):
    """``POST /v1/ask`` response: exactly the shared :class:`~secqa.core.contracts.Answer`.

    Every citation carries ``verified`` (the quote is a normalised substring of the store's
    chunk text) and ``valid`` (the ref resolves to something this request retrieved); unverified
    citations are kept and flagged, never dropped. ``snippet`` always comes from the store.
    """


class SearchRequest(_FilterFields):
    """Body of ``POST /v1/search``."""

    query: str = Field(min_length=1, max_length=QUESTION_MAX_CHARS)
    form: str | None = Field(
        default=None,
        pattern=FORM_PATTERN,
        description="Restrict to this form type ('10-K', '10K' and '10-k' are equivalent).",
    )
    k: int = Field(default=8, ge=K_MIN, le=K_MAX)
    strategy: RetrievalStrategy = Field(
        default="hybrid", description="'bm25', 'dense' or 'hybrid' (reciprocal-rank fusion)."
    )

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    def retrieval_filters(self) -> RetrievalFilters | None:
        """Filters including the search-only ``form`` field."""
        return self.filters(form=self.form)


class SearchResponse(BaseModel):
    """``POST /v1/search`` response."""

    model_config = ConfigDict(frozen=True)

    hits: list[HitView]


class PageResponse(BaseModel):
    """``GET /v1/filings/{doc_name}/pages/{page}`` response (``page_num`` is 1-based)."""

    model_config = ConfigDict(frozen=True)

    doc_name: str
    page_num: int
    text: str


class SqlRequest(ApiModel):
    """Body of ``POST /v1/xbrl/query``.

    The statement passes the same guard as the agent's ``query_xbrl`` tool: one ``SELECT`` /
    ``WITH`` over ``xbrl_facts``, ``financials`` or ``documents``, ``LIMIT 200`` enforced, 5 s
    wall clock. Rows of the wide ``financials`` view carry no accession numbers and are therefore
    not citable; select ``accn`` from ``xbrl_facts`` when provenance matters.
    """

    sql: str = Field(
        min_length=1,
        max_length=SQL_MAX_CHARS,
        examples=["SELECT ticker, fy, val, accn FROM xbrl_facts WHERE tag = 'Revenues' LIMIT 5"],
    )

    @field_validator("sql")
    @classmethod
    def _sql_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sql must not be blank")
        return value


class ProblemDetail(BaseModel):
    """RFC 9457 problem document (``application/problem+json``) returned for every error.

    Some problems add extension members: ``errors`` (422 validation details) and ``answer``
    (402 / 504 partial answer with its trace).
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="URI reference identifying the problem class.")
    title: str
    status: int
    detail: str
    request_id: str


class HealthResponse(BaseModel):
    """``GET /healthz``: process liveness, no dependencies touched."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    """``GET /readyz`` when the index, embedder and default provider are loaded."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ready"] = "ready"
    chunks: int
    documents: int
    facts: int
    embedder: str
    provider: str


class VersionResponse(BaseModel):
    """``GET /version``: everything needed to attribute an answer to code, index and prompts."""

    model_config = ConfigDict(frozen=True)

    version: str
    git_sha: str
    index_manifest: IndexManifest | None = Field(
        default=None, description="Provenance of the loaded index; null while not ready."
    )
    prompt_hashes: dict[str, str] = Field(
        description="sha256 of every prompt file under src/secqa/prompts (all modes)."
    )
    judge_version: str | None = Field(
        default=None,
        description="Version of the evaluation judge prompts, when the eval module is installed.",
    )


def jsonable(model: BaseModel) -> dict[str, Any]:
    """JSON-native dump of a pydantic model (dates and nested models rendered as JSON would)."""
    return model.model_dump(mode="json")


__all__ = [
    "DOC_NAMES_MAX",
    "K_MAX",
    "K_MIN",
    "QUESTION_MAX_CHARS",
    "SQL_MAX_CHARS",
    "ApiMode",
    "ApiModel",
    "AskRequest",
    "AskResponse",
    "HealthResponse",
    "PageResponse",
    "ProblemDetail",
    "ReadyResponse",
    "SearchRequest",
    "SearchResponse",
    "SqlRequest",
    "VersionResponse",
    "jsonable",
]
