"""The HTTP endpoints (SPEC section 8) and the router factory that wires them.

Endpoints are plain synchronous functions: the pipelines block on DuckDB and vendor HTTP, so
FastAPI runs them in its threadpool and the event loop stays free for probes. They are
registered by :func:`build_router` rather than decorated at import time because the slowapi
limit string comes from the app's settings (one limiter per app, tests included).

Outcome mapping for ``POST /v1/ask`` (see :mod:`secqa.api.errors` for the rest):

* the agent loop aborting on its cost cap -> 402, aborting on its wall clock -> 504, both with
  the partial answer (trace included) as the ``answer`` extension member;
* a single-shot answer whose bill exceeds the cap -> 402 with the same payload (the call has
  already been paid for; the client learns exactly what it cost);
* other agent budget stops (tool-call / token limits) are ordinary 200s with
  ``terminated_by='budget'`` in the body.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Header, Path, Query, Request, Response
from slowapi import Limiter

from secqa import __version__
from secqa.agent import COST_ABORT_PREFIX, WALL_CLOCK_ABORT_PREFIX, AgentLoop, ToolRuntime
from secqa.api.deps import (
    API_KEY_HEADER,
    AppState,
    effective_cost_cap,
    get_state,
    is_paid,
    require_api_key,
    resolve_judge_version,
    resolve_provider,
)
from secqa.api.errors import BudgetExceeded, NotFound, Unprocessable, WallClockExceeded
from secqa.api.middleware import SERVER_TIMING_HEADER, merge_server_timing
from secqa.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    PageResponse,
    ProblemDetail,
    ReadyResponse,
    SearchRequest,
    SearchResponse,
    SqlRequest,
    VersionResponse,
)
from secqa.core.contracts import Answer, DocumentMeta, LLMProvider, SqlResult
from secqa.core.logging import bind_context, get_logger
from secqa.rag import RagPipeline, answer_closed_book, prompt_hashes
from secqa.retrieval import to_hit_view
from secqa.store import resolve_git_sha
from secqa.xbrl import run_readonly_sql

log = get_logger(__name__)

AGENT_TOOL_RESULT_CHARS = 20_000
"""``ToolRuntime.max_result_chars`` for the API: room for three full ``get_pages`` pages."""
SQL_TIMEOUT_S = 5.0
COST_ABORT_MARKER = COST_ABORT_PREFIX
WALL_CLOCK_ABORT_MARKER = WALL_CLOCK_ABORT_PREFIX
"""The agent loop's own ``budget`` abort reasons: cost cap -> 402, wall clock -> 504."""
FORM_QUERY_PATTERN = r"^[0-9A-Za-z\-/]{1,12}$"

ApiKeyHeader = Annotated[
    str | None,
    Header(alias=API_KEY_HEADER, description="Demo key; required for agent mode when set."),
]


def _responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries documenting problem+json for the given statuses."""
    titles = {
        400: "SQL rejected by the read-only guard",
        402: "Cost budget exceeded (partial answer attached)",
        403: "Forbidden without X-API-Key",
        404: "Unknown document or page",
        422: "Validation error",
        429: "Rate limited",
        502: "Upstream provider error",
        503: "Service not ready / provider not configured",
        504: "Wall clock or SQL timeout exceeded",
    }
    return {
        status: {
            "model": ProblemDetail,
            "description": f"{titles[status]} (application/problem+json)",
        }
        for status in statuses
    }


# ---- operational endpoints -----------------------------------------------------------------


def healthz() -> HealthResponse:
    """Liveness: the process is up. Touches nothing."""
    return HealthResponse()


def readyz(request: Request) -> ReadyResponse:
    """Readiness: index, embedder and default provider loaded; else 503 with the reason."""
    state = get_state(request)
    handles = state.require_index()
    counts = handles.store.counts()
    provider = state.default_provider
    return ReadyResponse(
        chunks=counts["chunks"],
        documents=counts["documents"],
        facts=counts["facts"],
        embedder=handles.embedder.name,
        provider=f"{provider.provider}:{provider.model}",
    )


def version(request: Request) -> VersionResponse:
    """Code, index and prompt provenance for attributing any answer this server gives."""
    state = get_state(request)
    manifest = state.store.manifest() if state.ready and state.store is not None else None
    return VersionResponse(
        version=__version__,
        git_sha=resolve_git_sha(),
        index_manifest=manifest,
        prompt_hashes=prompt_hashes(),
        judge_version=resolve_judge_version(),
    )


# ---- /v1/ask -------------------------------------------------------------------------------


def ask(
    request: Request,
    body: AskRequest,
    response: Response,
    x_api_key: ApiKeyHeader = None,
) -> AskResponse:
    """Answer one question with verified, page-level citations."""
    state = get_state(request)
    settings = state.settings
    if body.k > settings.max_k:
        raise Unprocessable(f"k={body.k} exceeds this server's maximum of {settings.max_k}")
    if body.mode == "agent":
        require_api_key(state, x_api_key, what="agent mode")
    provider = resolve_provider(state, body.provider, x_api_key)
    cap = effective_cost_cap(state, body.max_cost_usd)
    paid = is_paid(provider)
    if paid and cap <= 0.0:
        raise BudgetExceeded(
            f"max_cost_usd={cap:g} cannot afford a call to {provider.provider}:{provider.model}"
        )
    if paid and state.daily_budget_left() <= 0.0:
        raise BudgetExceeded(
            f"this instance's daily budget of ${settings.daily_budget_usd:.2f} is exhausted"
        )

    rid = str(request.state.request_id)
    bind_context(provider=provider.provider, model=provider.model, mode=body.mode)
    answer = _run(state, body, provider, cap, rid)
    spent = state.charge(answer.cost_usd)
    response.headers[SERVER_TIMING_HEADER] = merge_server_timing(
        f"retrieval;dur={answer.retrieval_ms:.1f}", f"llm;dur={answer.llm_ms:.1f}"
    )
    result = _to_response(answer, include_trace=body.include_trace)
    log.info(
        "ask_completed",
        mode=answer.mode,
        terminated_by=answer.terminated_by,
        abstained=answer.abstained,
        grounded=answer.grounded,
        n_citations=len(answer.citations),
        n_verified=sum(c.verified for c in answer.citations),
        input_tokens=answer.usage.input_tokens,
        output_tokens=answer.usage.output_tokens,
        cost_usd=answer.cost_usd,
        spent_today_usd=round(spent, 6),
        latency_ms=round(answer.latency_ms, 1),
    )

    abort_reason = _abort_reason(answer) if answer.terminated_by == "budget" else ""
    partial = {"answer": result.model_dump(mode="json")}
    if WALL_CLOCK_ABORT_MARKER in abort_reason:
        raise WallClockExceeded(abort_reason, extra=partial)
    if COST_ABORT_MARKER in abort_reason:
        raise BudgetExceeded(abort_reason, extra=partial)
    if answer.cost_usd > cap:
        raise BudgetExceeded(
            f"request cost ${answer.cost_usd:.4f} exceeded the cap of ${cap:.4f}", extra=partial
        )
    return result


def _run(state: AppState, body: AskRequest, provider: LLMProvider, cap: float, rid: str) -> Answer:
    """Dispatch to the pipeline for ``body.mode``; ProviderError propagates (-> 502)."""
    settings = state.settings
    if body.mode == "closed_book":
        return answer_closed_book(body.question, provider, state.prices, request_id=rid)
    handles = state.require_index()
    filters = body.filters()
    if body.mode == "rag":
        pipeline = RagPipeline(handles.retriever, provider, state.verifier, state.prices, k=body.k)
        return pipeline.answer(body.question, filters=filters, request_id=rid)
    runtime = ToolRuntime(
        handles.store,
        handles.retriever,
        max_result_chars=AGENT_TOOL_RESULT_CHARS,
        sql_timeout_s=SQL_TIMEOUT_S,
    )
    loop = AgentLoop(
        provider,
        runtime,
        state.verifier,
        state.prices,
        max_steps=settings.max_agent_steps,
        max_cost_usd=cap,
        wall_clock_s=float(settings.request_timeout_s),
    )
    return loop.run(body.question, filters=filters, request_id=rid)


def _to_response(answer: Answer, *, include_trace: bool) -> AskResponse:
    """Project an ``Answer`` to the response model, dropping the trace when not requested."""
    payload = answer.model_dump()
    if not include_trace:
        payload["trace"] = []
    return AskResponse.model_validate(payload)


def _abort_reason(answer: Answer) -> str:
    """The agent loop's abort reason: the ``error`` of the final ``verify`` trace step."""
    for step in reversed(answer.trace):
        if step.kind == "verify":
            return step.error or ""
    return ""


# ---- /v1/search, /v1/filings, /v1/xbrl/query -------------------------------------------------


def search(request: Request, body: SearchRequest, response: Response) -> SearchResponse:
    """Retrieve chunks without calling a model (bm25 / dense / hybrid).

    ``response`` is unused here but required: slowapi injects its rate-limit headers into it.
    """
    state = get_state(request)
    retriever = state.retriever_for(body.strategy)
    result = retriever.retrieve(body.query, k=body.k, filters=body.retrieval_filters())
    return SearchResponse(hits=[to_hit_view(hit) for hit in result.hits])


def list_filings(
    request: Request,
    response: Response,
    ticker: Annotated[str | None, Query(pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")] = None,
    fiscal_year: Annotated[int | None, Query(ge=1990, le=2100)] = None,
    form: Annotated[str | None, Query(pattern=FORM_QUERY_PATTERN)] = None,
) -> list[DocumentMeta]:
    """Documents in the index, optionally filtered by ticker, fiscal year and form."""
    state = get_state(request)
    handles = state.require_index()
    documents = handles.store.list_documents(ticker=ticker, fiscal_year=fiscal_year)
    if form is not None:
        wanted = _normalise_form(form)
        documents = [doc for doc in documents if _normalise_form(doc.form) == wanted]
    return documents


def get_page(
    request: Request,
    response: Response,
    doc_name: Annotated[str, Path(min_length=1, max_length=200)],
    page: Annotated[int, Path(ge=1, description="1-based physical page number")],
) -> PageResponse:
    """The extracted text of one page (what citations point at)."""
    state = get_state(request)
    handles = state.require_index()
    pages = handles.store.get_pages(doc_name, [page])
    if not pages:
        if not handles.store.list_documents(doc_names=[doc_name]):
            raise NotFound(f"document {doc_name!r} is not in the index")
        raise NotFound(f"document {doc_name!r} has no page {page}")
    found = pages[0]
    return PageResponse(doc_name=found.doc_name, page_num=found.page_num, text=found.text)


def xbrl_query(request: Request, body: SqlRequest, response: Response) -> SqlResult:
    """Run one guarded read-only SELECT over the XBRL tables (400 when the guard refuses)."""
    state = get_state(request)
    handles = state.require_index()
    return run_readonly_sql(handles.store, body.sql, timeout_s=SQL_TIMEOUT_S)


def _normalise_form(form: str) -> str:
    """``'10-k'`` / ``'10K'`` -> ``'10K'`` (same tolerance as the retriever's form filter)."""
    return form.strip().upper().replace("-", "")


# ---- router factory ------------------------------------------------------------------------


def build_router(limiter: Limiter, rate_limit: str) -> APIRouter:
    """Register every endpoint; ``rate_limit`` (``'10/minute'``) guards the ``/v1`` routes."""
    router = APIRouter()
    limited = limiter.limit(rate_limit)

    router.add_api_route("/healthz", healthz, methods=["GET"], tags=["ops"], summary="Liveness")
    router.add_api_route(
        "/readyz",
        readyz,
        methods=["GET"],
        tags=["ops"],
        summary="Readiness",
        responses=_responses(503),
    )
    router.add_api_route(
        "/version", version, methods=["GET"], tags=["ops"], summary="Build and index provenance"
    )
    router.add_api_route(
        "/v1/ask",
        limited(ask),
        methods=["POST"],
        tags=["qa"],
        summary="Answer a question with verified citations",
        response_model=AskResponse,
        responses=_responses(402, 403, 422, 429, 502, 503, 504),
    )
    router.add_api_route(
        "/v1/search",
        limited(search),
        methods=["POST"],
        tags=["qa"],
        summary="Retrieve passages",
        response_model=SearchResponse,
        responses=_responses(422, 429, 503),
    )
    router.add_api_route(
        "/v1/filings",
        limited(list_filings),
        methods=["GET"],
        tags=["filings"],
        summary="List indexed filings",
        response_model=list[DocumentMeta],
        responses=_responses(422, 429, 503),
    )
    router.add_api_route(
        "/v1/filings/{doc_name}/pages/{page}",
        limited(get_page),
        methods=["GET"],
        tags=["filings"],
        summary="Page text",
        response_model=PageResponse,
        responses=_responses(404, 422, 429, 503),
    )
    router.add_api_route(
        "/v1/xbrl/query",
        limited(xbrl_query),
        methods=["POST"],
        tags=["xbrl"],
        summary="Read-only SQL over XBRL facts",
        response_model=SqlResult,
        responses=_responses(400, 422, 429, 503, 504),
    )
    return router


__all__ = [
    "AGENT_TOOL_RESULT_CHARS",
    "SQL_TIMEOUT_S",
    "ask",
    "build_router",
    "get_page",
    "healthz",
    "list_filings",
    "readyz",
    "search",
    "version",
    "xbrl_query",
]
