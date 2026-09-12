"""Single-shot grounded answering: ``rag``, the ``closed_book`` baseline and the ``oracle`` bound.

All three modes share one skeleton (:func:`_single_shot`): build the user prompt from a list of
chunks, make exactly one LLM call with :data:`~secqa.rag.schemas.ANSWER_SCHEMA`, parse the
structured answer, run :class:`~secqa.grounding.CitationVerifier` over the chunks this request
actually supplied, price the usage, and assemble the :class:`~secqa.core.contracts.Answer` with a
full trace. They differ only in where the chunks come from:

* :class:`RagPipeline` -- the :class:`~secqa.retrieval.Retriever` (``mode='rag'``); an empty
  retrieval abstains *without* an LLM call (``terminated_by='empty_retrieval'``).
* :func:`answer_closed_book` -- no chunks at all (``mode='closed_book'``); citations are
  impossible, so the answer never carries any.
* :func:`answer_with_oracle_context` -- the gold evidence pages, wrapped as pseudo-chunks with
  real ``chunk:<id>`` refs (``mode='oracle'``), the retrieval-free upper bound of the harness.

Provider failures (:class:`~secqa.core.errors.ProviderError`) propagate: the API maps them to
502 and the harness records them per question; nothing here retries or swallows them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from secqa.core.contracts import (
    Answer,
    Chunk,
    Hit,
    HitView,
    LLMProvider,
    LLMResponse,
    Message,
    Mode,
    Page,
    RetrievalFilters,
    TraceStep,
    Usage,
)
from secqa.core.ids import chunk_id as make_chunk_id
from secqa.core.ids import request_id as make_request_id
from secqa.core.logging import get_logger
from secqa.grounding import CitationVerifier
from secqa.providers.base import estimate_tokens
from secqa.providers.pricing import PriceTable
from secqa.rag.prompts import (
    CLOSED_BOOK_SYSTEM,
    ORACLE_SYSTEM,
    RAG_SYSTEM,
    build_closed_book_prompt,
    build_user_prompt,
    load_prompt,
    prompt_hash,
)
from secqa.rag.schemas import ABSTAIN_TEXT, ANSWER_SCHEMA, parse_structured_answer
from secqa.retrieval import SNIPPET_CHARS, Retriever, to_hit_view

log = get_logger(__name__)

Effort = Literal["low", "medium", "high"]
EFFORTS: tuple[str, ...] = ("low", "medium", "high")

DEFAULT_K = 8
DEFAULT_MAX_TOKENS = 1024
DEFAULT_EFFORT: Effort = "medium"
ORACLE_PASSAGE_MAX_CHARS = 12_000
"""Displayed length cap per oracle page (a 10-K page is ~3-6k chars; tables can be longer)."""
_PREVIEW_CHARS = 200


@dataclass(frozen=True)
class _Prepared:
    """What a mode hands to the shared single-shot core."""

    mode: Mode
    system_name: str
    user_prompt: str
    chunks: list[Chunk]
    retrieved: list[HitView]
    retrieval_ms: float
    trace: list[TraceStep]


class RagPipeline:
    """Retrieve-then-answer in one LLM call, with verified citations.

    Construct once per process (the retriever validates embedder/store compatibility in its own
    constructor) and call :meth:`answer` per question.

    Args:
        retriever: The retrieval entry point (strategy, filters and timing live there).
        provider: Any :class:`~secqa.core.contracts.LLMProvider`.
        verifier: Citation verifier; shared with the agent and the API so results are identical.
        prices: Price table used to turn usage into ``cost_usd``.
        k: Number of passages placed in the prompt.
        max_tokens: Output token cap for the single call.
        effort: Reasoning effort forwarded to providers that support it.
    """

    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider,
        verifier: CitationVerifier,
        prices: PriceTable,
        k: int = DEFAULT_K,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
    ) -> None:
        _check_positive("k", k)
        _check_positive("max_tokens", max_tokens)
        self.retriever = retriever
        self.provider = provider
        self.verifier = verifier
        self.prices = prices
        self.k = k
        self.max_tokens = max_tokens
        self.effort: Effort = _check_effort(effort)

    def answer(
        self,
        question: str,
        *,
        filters: RetrievalFilters | None = None,
        request_id: str | None = None,
    ) -> Answer:
        """Answer ``question`` from the top-``k`` retrieved passages.

        Passages are numbered ``[1..k]`` in the prompt, each labelled with its ``chunk:<id>`` ref.
        An empty retrieval (blank question, or a filter matching no document) abstains without
        calling the model and is recorded as ``terminated_by='empty_retrieval'``.
        """
        rid = request_id or make_request_id()
        started = time.perf_counter()

        result = self.retriever.retrieve(question, k=self.k, filters=filters)
        hits: list[Hit] = result.hits
        chunks = [hit.chunk for hit in hits]
        retrieval_step = TraceStep(
            step=1,
            kind="retrieval",
            name=result.strategy,
            arguments={
                "k": result.k,
                "filters": filters.model_dump(exclude_none=True) if filters else {},
            },
            result_preview=_hits_preview(hits),
            latency_ms=result.latency_ms,
        )
        prepared = _Prepared(
            mode="rag",
            system_name=RAG_SYSTEM,
            user_prompt=build_user_prompt(question, chunks, filters=filters) if chunks else "",
            chunks=chunks,
            retrieved=[to_hit_view(hit) for hit in hits],
            retrieval_ms=result.latency_ms,
            trace=[retrieval_step],
        )
        if not chunks:
            return _empty_retrieval_answer(
                rid, question, prepared, self.provider, self.verifier, started
            )
        return _single_shot(
            rid,
            question,
            prepared,
            provider=self.provider,
            verifier=self.verifier,
            prices=self.prices,
            max_tokens=self.max_tokens,
            effort=self.effort,
            started=started,
        )


def answer_closed_book(
    question: str,
    provider: LLMProvider,
    prices: PriceTable,
    request_id: str | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str = DEFAULT_EFFORT,
) -> Answer:
    """Answer from the model's parametric memory alone: no passages, no tools, no citations.

    This is the baseline row of the evaluation matrix. Any citation the model invents anyway is
    discarded (there is nothing it could resolve to) and noted in the trace; ``grounded`` follows
    the usual rule, so an answer with numbers is never grounded in this mode.
    """
    if not question.strip():
        raise ValueError("question must not be blank")
    rid = request_id or make_request_id()
    prepared = _Prepared(
        mode="closed_book",
        system_name=CLOSED_BOOK_SYSTEM,
        user_prompt=build_closed_book_prompt(question),
        chunks=[],
        retrieved=[],
        retrieval_ms=0.0,
        trace=[],
    )
    return _single_shot(
        rid,
        question,
        prepared,
        provider=provider,
        verifier=CitationVerifier(),
        prices=prices,
        max_tokens=max_tokens,
        effort=_check_effort(effort),
        started=time.perf_counter(),
    )


def answer_with_oracle_context(
    question: str,
    pages: list[Page],
    provider: LLMProvider,
    verifier: CitationVerifier,
    prices: PriceTable,
    request_id: str | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str = DEFAULT_EFFORT,
) -> Answer:
    """Answer from the given (gold) pages: the retrieval-free upper bound of the harness.

    Each page becomes one pseudo-chunk (``chunk_idx=0``, id from
    :func:`secqa.core.ids.chunk_id`) so citations use the same ``chunk:<id>`` refs and the same
    verifier as the rag mode. With no pages the call abstains without touching the model
    (``terminated_by='empty_retrieval'``).
    """
    if not question.strip():
        raise ValueError("question must not be blank")
    rid = request_id or make_request_id()
    started = time.perf_counter()
    chunks = [page_to_chunk(page) for page in pages]
    prepared = _Prepared(
        mode="oracle",
        system_name=ORACLE_SYSTEM,
        user_prompt=build_user_prompt(question, chunks, max_passage_chars=ORACLE_PASSAGE_MAX_CHARS)
        if chunks
        else "",
        chunks=chunks,
        retrieved=[_oracle_hit_view(index, chunk) for index, chunk in enumerate(chunks, 1)],
        retrieval_ms=0.0,
        trace=[
            TraceStep(
                step=1,
                kind="retrieval",
                name="oracle",
                arguments={"pages": [[page.doc_name, page.page_num] for page in pages]},
                result_preview=f"{len(pages)} gold page(s) supplied",
            )
        ],
    )
    if not chunks:
        return _empty_retrieval_answer(rid, question, prepared, provider, verifier, started)
    return _single_shot(
        rid,
        question,
        prepared,
        provider=provider,
        verifier=verifier,
        prices=prices,
        max_tokens=max_tokens,
        effort=_check_effort(effort),
        started=started,
    )


def page_to_chunk(page: Page) -> Chunk:
    """Wrap a whole page as a single chunk with a content-addressed id (oracle mode)."""
    return Chunk(
        chunk_id=make_chunk_id(page.doc_name, page.page_num, 0, page.text),
        doc_name=page.doc_name,
        page_num=page.page_num,
        chunk_idx=0,
        section=None,
        text=page.text,
        n_tokens=estimate_tokens(page.text),
    )


# ---- shared core -----------------------------------------------------------------------------


def _single_shot(
    rid: str,
    question: str,
    prepared: _Prepared,
    *,
    provider: LLMProvider,
    verifier: CitationVerifier,
    prices: PriceTable,
    max_tokens: int,
    effort: Effort,
    started: float,
) -> Answer:
    """One LLM call -> parse -> verify -> price -> Answer (see the module docstring)."""
    system = load_prompt(prepared.system_name)
    messages = [Message(role="user", content=prepared.user_prompt)]
    trace = list(prepared.trace)
    step = len(trace) + 1

    llm_started = time.perf_counter()
    try:
        response: LLMResponse = provider.complete(
            messages,
            system=system,
            json_schema=ANSWER_SCHEMA,
            max_tokens=max_tokens,
            effort=effort,
        )
    except Exception as exc:
        log.error(
            "rag_llm_failed",
            request_id=rid,
            mode=prepared.mode,
            provider=provider.provider,
            model=provider.model,
            error=str(exc),
        )
        raise
    llm_wall_ms = (time.perf_counter() - llm_started) * 1000.0
    # A cassette hit answers in microseconds; the only real measurement of that call is the one
    # the recording run stored in ``response.latency_ms``. Report it (and below, swap it into the
    # whole-question wall clock) so replays never publish near-zero LLM timings.
    llm_ms = response.latency_ms if response.cached else llm_wall_ms

    parsed = parse_structured_answer(response)
    structured = parsed.answer
    trace.append(
        TraceStep(
            step=step,
            kind="llm",
            name=f"{response.provider}:{response.model}",
            arguments={
                "system": prepared.system_name,
                "max_tokens": max_tokens,
                "effort": effort,
                "n_passages": len(prepared.chunks),
            },
            result_preview=_preview(response.text or structured.text),
            latency_ms=llm_ms,
            usage=response.usage,
            error=_llm_error(response, parsed.parse_error),
        )
    )

    refs = list(structured.citations)
    verify_error: str | None = None
    if prepared.mode == "closed_book" and refs:
        verify_error = f"discarded {len(refs)} citation(s): closed_book has no passages to cite"
        refs = []
    chunks_by_id = {chunk.chunk_id: chunk for chunk in prepared.chunks}
    verify_started = time.perf_counter()
    citations, grounded = verifier.verify(structured.text, refs, chunks_by_id, {}, [])
    trace.append(
        TraceStep(
            step=step + 1,
            kind="verify",
            name="citation_verifier",
            arguments={"n_refs": len(refs), "n_chunks": len(chunks_by_id)},
            result_preview=(
                f"{sum(c.verified for c in citations)}/{len(citations)} verified, "
                f"grounded={grounded}"
            ),
            latency_ms=(time.perf_counter() - verify_started) * 1000.0,
            error=verify_error,
        )
    )

    # Price on the *configured* id (the one the price table and every pre-flight check validate).
    # ``response.model`` is the vendor's echo and may be a dated snapshot ("gpt-5.5-2026-06-01")
    # that is not a key in models.yaml; it is kept on the Answer and trace for provenance only.
    cost = prices.cost_usd(provider.provider, provider.model, response.usage)
    answer = Answer(
        request_id=rid,
        question=question,
        text=structured.text,
        value=structured.value,
        unit=structured.unit,
        abstained=structured.abstained,
        citations=citations,
        grounded=grounded,
        calculation=None,
        retrieved=prepared.retrieved,
        trace=trace,
        usage=response.usage,
        cost_usd=cost,
        latency_ms=(time.perf_counter() - started) * 1000.0 - llm_wall_ms + llm_ms,
        retrieval_ms=prepared.retrieval_ms,
        llm_ms=llm_ms,
        provider=response.provider,
        model=response.model,
        mode=prepared.mode,
        steps=1,
        tool_calls=0,
        terminated_by="single_shot",
        prompt_hashes={prepared.system_name: prompt_hash(prepared.system_name)},
    )
    _log_answer(answer, parse_source=parsed.source)
    return answer


def _empty_retrieval_answer(
    rid: str,
    question: str,
    prepared: _Prepared,
    provider: LLMProvider,
    verifier: CitationVerifier,
    started: float,
) -> Answer:
    """Abstain without an LLM call when there is nothing to answer from."""
    citations, grounded = verifier.verify(ABSTAIN_TEXT, [], {}, {}, [])
    trace = [
        *prepared.trace,
        TraceStep(
            step=len(prepared.trace) + 1,
            kind="verify",
            name="citation_verifier",
            result_preview="no passages: abstained without an LLM call",
        ),
    ]
    answer = Answer(
        request_id=rid,
        question=question,
        text=ABSTAIN_TEXT,
        value=None,
        unit=None,
        abstained=True,
        citations=citations,
        grounded=grounded,
        calculation=None,
        retrieved=prepared.retrieved,
        trace=trace,
        usage=Usage(),
        cost_usd=0.0,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        retrieval_ms=prepared.retrieval_ms,
        llm_ms=0.0,
        provider=provider.provider,
        model=provider.model,
        mode=prepared.mode,
        steps=0,
        tool_calls=0,
        terminated_by="empty_retrieval",
        prompt_hashes={prepared.system_name: prompt_hash(prepared.system_name)},
    )
    _log_answer(answer, parse_source="none")
    return answer


# ---- helpers ---------------------------------------------------------------------------------


def _oracle_hit_view(rank: int, chunk: Chunk) -> HitView:
    """Project an oracle pseudo-chunk into ``Answer.retrieved`` (score = 1/rank, gold order)."""
    return HitView(
        chunk_id=chunk.chunk_id,
        doc_name=chunk.doc_name,
        page_num=chunk.page_num,
        section=chunk.section,
        score=1.0 / rank,
        snippet=chunk.text[:SNIPPET_CHARS],
    )


def _hits_preview(hits: list[Hit]) -> str:
    if not hits:
        return "0 hits"
    pages = ", ".join(f"{h.chunk.doc_name} p.{h.chunk.page_num}" for h in hits[:5])
    more = f" (+{len(hits) - 5} more)" if len(hits) > 5 else ""
    return f"{len(hits)} hits: {pages}{more}"


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _llm_error(response: LLMResponse, parse_error: str | None) -> str | None:
    """Trace-worthy problems with the LLM step: truncation, refusal, unparseable output."""
    problems: list[str] = []
    if response.stop_reason == "max_tokens":
        problems.append("output truncated at max_tokens")
    if parse_error:
        problems.append(parse_error)
    return "; ".join(problems) or None


def _log_answer(answer: Answer, *, parse_source: str) -> None:
    log.info(
        "rag_answer",
        request_id=answer.request_id,
        mode=answer.mode,
        provider=answer.provider,
        model=answer.model,
        terminated_by=answer.terminated_by,
        abstained=answer.abstained,
        grounded=answer.grounded,
        n_retrieved=len(answer.retrieved),
        n_citations=len(answer.citations),
        n_verified=sum(c.verified for c in answer.citations),
        parse_source=parse_source,
        input_tokens=answer.usage.input_tokens,
        output_tokens=answer.usage.output_tokens,
        cost_usd=answer.cost_usd,
        latency_ms=round(answer.latency_ms, 1),
        retrieval_ms=round(answer.retrieval_ms, 1),
        llm_ms=round(answer.llm_ms, 1),
    )


def _check_effort(effort: str) -> Effort:
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {', '.join(EFFORTS)}; got {effort!r}")
    return effort  # type: ignore[return-value]  # validated against EFFORTS above


def _check_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")


__all__ = [
    "DEFAULT_EFFORT",
    "DEFAULT_K",
    "DEFAULT_MAX_TOKENS",
    "ORACLE_PASSAGE_MAX_CHARS",
    "RagPipeline",
    "answer_closed_book",
    "answer_with_oracle_context",
    "page_to_chunk",
]
