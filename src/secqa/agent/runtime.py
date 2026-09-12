"""``ToolRuntime``: executes the agent's tool calls against the local index and keeps the ledger.

Three properties matter for trust, and each is enforced here rather than in the loop:

* **Tools never raise.** :meth:`ToolRuntime.dispatch` always returns a
  :class:`~secqa.core.contracts.ToolResult`; bad arguments, rejected SQL, unknown documents and
  unexpected exceptions all become ``{"error": ...}`` results with ``is_error=True`` so the model
  can correct itself and the loop's failure rules can count them.
* **Citations are bound to what the run saw.** Every chunk returned by ``search_filings`` or
  ``get_pages`` is recorded in :attr:`ToolRuntime.seen_chunks` (keyed by bare ``chunk_id``),
  every fact row from ``lookup_fact`` or a citable ``query_xbrl`` row in
  :attr:`ToolRuntime.seen_facts` (keyed by ``FactRow.ref``), and every ``calculate`` result in
  :attr:`ToolRuntime.calc_results` - exactly the maps :class:`secqa.grounding.CitationVerifier`
  consumes. ``final_answer`` refs that are not in the ledger are rejected as an error result.
* **Results are bounded.** ``ToolResult.content`` is a JSON string of at most
  ``max_result_chars`` characters. Oversized payloads are shrunk *structurally* (shorter
  snippets, fewer rows) and flagged ``"truncated": true``; the JSON is never cut mid-string.

Nothing here touches the network: retrieval, pages, XBRL facts and SQL all come from the one
DuckDB store. ``get_pages`` wraps each page as a pseudo-chunk with the same content-addressed id
the oracle mode uses (``chunk_id(doc_name, page_num, 0, page.text)``), so a page citation is
verified against the exact page text and shares its id across modes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import date
from typing import Any

from pydantic import ValidationError

from secqa.agent.calc import safe_calculate
from secqa.agent.tools import (
    CALCULATE,
    DEFAULT_SEARCH_K,
    FINAL_ANSWER,
    GET_PAGES,
    LOOKUP_COMPANY,
    LOOKUP_FACT,
    MAX_PAGES_PER_CALL,
    MAX_SEARCH_K,
    PAGE_MAX_CHARS,
    QUERY_XBRL,
    SEARCH_FILINGS,
    TOOL_NAMES,
    FinalAnswer,
    decode_final_answer,
)
from secqa.core.contracts import (
    Chunk,
    CitationRef,
    DocumentMeta,
    FactRow,
    Frozen,
    Hit,
    HitView,
    Page,
    RetrievalFilters,
    SqlResult,
    ToolCall,
    ToolResult,
    TraceStep,
)
from secqa.core.errors import CalcRejected, SqlRejected
from secqa.core.ids import chunk_id as make_chunk_id
from secqa.core.logging import get_logger
from secqa.grounding import parse_ref
from secqa.providers.base import estimate_tokens
from secqa.retrieval import Retriever, to_hit_view
from secqa.store import DuckDBStore
from secqa.xbrl import lookup_fact, run_readonly_sql

log = get_logger(__name__)

DEFAULT_MAX_RESULT_CHARS = 4000
DEFAULT_SQL_TIMEOUT_S = 5.0
SEARCH_SNIPPET_CHARS = 1200
_MIN_SNIPPET_CHARS = 80
_MIN_PAGE_CHARS = 200
_PREVIEW_CHARS = 200
_TRUNCATED_MARK = " [truncated]"

# Columns a query_xbrl row must carry to be citable (the ref needs tag, fy, accn; the verifier
# needs val). Everything else is optional and defaulted.
_CITABLE_COLUMNS = frozenset({"tag", "fy", "accn", "val"})

Shrinker = Callable[[dict[str, Any]], bool]
"""Shrink a payload in place one notch; return False when nothing more can be removed."""


class _ToolError(Exception):
    """A tool-level failure the model should see as ``{"error": ...}`` (never propagates)."""


# ---- argument models (extra keys are rejected: the schemas are strict) ---------------------


class _SearchArgs(Frozen):
    query: str = ""
    ticker: str | None = None
    fiscal_year: int | None = None
    form: str | None = None
    k: int | None = None


class _PagesArgs(Frozen):
    doc_name: str
    pages: list[int]


class _CompanyArgs(Frozen):
    name_or_ticker: str


class _FactArgs(Frozen):
    ticker: str
    metric: str
    fiscal_year: int


class _SqlArgs(Frozen):
    sql: str


class _CalcArgs(Frozen):
    expression: str


class ToolRuntime:
    """Execute tool calls for one agent run and remember everything the run saw.

    Args:
        store: The index (documents, pages, chunks, XBRL facts).
        retriever: Retrieval entry point used by ``search_filings``.
        max_result_chars: Hard cap on ``ToolResult.content`` length (JSON characters).
        sql_timeout_s: Wall-clock limit for one ``query_xbrl`` statement.

    One runtime serves one run; call :meth:`reset` to reuse it for the next question.
    """

    def __init__(
        self,
        store: DuckDBStore,
        retriever: Retriever,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
        sql_timeout_s: float = DEFAULT_SQL_TIMEOUT_S,
    ) -> None:
        if isinstance(max_result_chars, bool) or max_result_chars < 200:
            raise ValueError(f"max_result_chars must be an int >= 200, got {max_result_chars!r}")
        if sql_timeout_s <= 0:
            raise ValueError(f"sql_timeout_s must be positive, got {sql_timeout_s!r}")
        self.store = store
        self.retriever = retriever
        self.max_result_chars = max_result_chars
        self.sql_timeout_s = sql_timeout_s
        self.seen_chunks: dict[str, Chunk] = {}
        self.seen_facts: dict[str, FactRow] = {}
        self.calc_results: list[float] = []
        self.calc_log: list[tuple[str, float]] = []
        self.tool_log: list[TraceStep] = []
        self.retrieved: list[HitView] = []
        self.retrieval_ms: float = 0.0
        self.base_filters: RetrievalFilters | None = None
        self._handlers: dict[
            str, Callable[[dict[str, Any]], tuple[dict[str, Any], Shrinker | None]]
        ]
        self._handlers = {
            SEARCH_FILINGS: self._search_filings,
            GET_PAGES: self._get_pages,
            LOOKUP_COMPANY: self._lookup_company,
            LOOKUP_FACT: self._lookup_fact,
            QUERY_XBRL: self._query_xbrl,
            CALCULATE: self._calculate,
            FINAL_ANSWER: self._final_answer,
        }

    # ---- lifecycle ----------------------------------------------------------------------------

    def reset(self, filters: RetrievalFilters | None = None) -> None:
        """Clear the ledger and set the request-level filters every search is constrained by.

        ``filters.doc_names`` always applies; ``ticker`` / ``fiscal_year`` / ``form`` are
        defaults the model may override per call.
        """
        self.seen_chunks = {}
        self.seen_facts = {}
        self.calc_results = []
        self.calc_log = []
        self.tool_log = []
        self.retrieved = []
        self.retrieval_ms = 0.0
        self.base_filters = filters

    # ---- dispatch -----------------------------------------------------------------------------

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Run one tool call and return its result; never raises.

        Successful results carry a JSON object; failures carry ``{"error": "<reason>"}`` with
        ``is_error=True``. Every call is appended to :attr:`tool_log` as a ``TraceStep``.
        """
        started = time.perf_counter()
        handler = self._handlers.get(call.name)
        payload: dict[str, Any]
        shrink: Shrinker | None = None
        is_error = False
        if handler is None:
            payload = {
                "error": f"unknown tool {call.name!r}; available tools: {', '.join(TOOL_NAMES)}"
            }
            is_error = True
        else:
            try:
                payload, shrink = handler(dict(call.arguments or {}))
            except _ToolError as exc:
                payload, is_error = {"error": str(exc)}, True
            except ValidationError as exc:
                payload, is_error = {"error": f"invalid arguments: {_short(exc)}"}, True
            except Exception as exc:  # tools never raise: report and let the loop count it
                log.exception("agent_tool_crashed", tool=call.name)
                payload = {"error": f"{call.name} failed: {type(exc).__name__}: {exc}"}
                is_error = True
        content, truncated_away = self._encode(payload, shrink)
        if truncated_away:
            is_error = True
        latency_ms = (time.perf_counter() - started) * 1000.0
        result = ToolResult(
            tool_call_id=call.id, name=call.name, content=content, is_error=is_error
        )
        self.tool_log.append(
            TraceStep(
                step=len(self.tool_log) + 1,
                kind="tool",
                name=call.name,
                arguments=dict(call.arguments or {}),
                result_preview=_preview(content),
                latency_ms=latency_ms,
                error=str(payload.get("error")) if is_error else None,
            )
        )
        log.info(
            "agent_tool",
            tool=call.name,
            is_error=is_error,
            result_chars=len(content),
            latency_ms=round(latency_ms, 1),
            n_seen_chunks=len(self.seen_chunks),
            n_seen_facts=len(self.seen_facts),
        )
        return result

    def invalid_refs(self, refs: list[CitationRef]) -> list[str]:
        """Refs that do not resolve to a chunk / fact this run saw (malformed ones included)."""
        invalid: list[str] = []
        for ref in refs:
            raw = ref.ref.strip()
            try:
                kind, payload = parse_ref(raw)
            except ValueError:
                invalid.append(raw)
                continue
            known = payload in self.seen_chunks if kind == "chunk" else raw in self.seen_facts
            if not known:
                invalid.append(raw)
        return invalid

    def rendered_calculation(self) -> str | None:
        """``'expr = result; expr = result'`` from the actual ``calculate`` calls, or ``None``."""
        if not self.calc_log:
            return None
        return "; ".join(f"{expression} = {result:g}" for expression, result in self.calc_log)

    # ---- tools --------------------------------------------------------------------------------

    def _search_filings(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], Shrinker]:
        args = _SearchArgs.model_validate(arguments)
        k = DEFAULT_SEARCH_K if args.k is None else args.k
        if k < 1 or k > MAX_SEARCH_K:
            raise _ToolError(f"k must be between 1 and {MAX_SEARCH_K}, got {k}")
        filters = self._merge_filters(args)
        result = self.retriever.retrieve(args.query, k=k, filters=filters)
        self.retrieval_ms += result.latency_ms
        hits: list[dict[str, Any]] = []
        for hit in result.hits:
            self._remember_hit(hit)
            hits.append(
                {
                    "ref": f"chunk:{hit.chunk.chunk_id}",
                    "chunk_id": hit.chunk.chunk_id,
                    "doc_name": hit.chunk.doc_name,
                    "page_num": hit.chunk.page_num,
                    "section": hit.chunk.section,
                    "score": round(hit.score, 6),
                    "snippet": hit.chunk.text[:SEARCH_SNIPPET_CHARS],
                }
            )
        payload: dict[str, Any] = {
            "query": args.query,
            "strategy": result.strategy,
            "filters": filters.model_dump(exclude_none=True) if filters else {},
            "n": len(hits),
            "hits": hits,
        }
        if not hits:
            payload["note"] = (
                "no matching chunks; try different terms, drop a filter, or call lookup_company"
            )
        return payload, _shrink_texts("hits", "snippet", _MIN_SNIPPET_CHARS)

    def _get_pages(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], Shrinker]:
        args = _PagesArgs.model_validate(arguments)
        if not args.pages:
            raise _ToolError("pages must list at least one 1-based page number")
        if len(args.pages) > MAX_PAGES_PER_CALL:
            raise _ToolError(f"at most {MAX_PAGES_PER_CALL} pages per call, got {len(args.pages)}")
        if any(isinstance(page, bool) or page < 1 for page in args.pages):
            raise _ToolError("page numbers are 1-based integers")
        if not self.store.list_documents(doc_names=[args.doc_name]):
            raise _ToolError(
                f"unknown document {args.doc_name!r}; use lookup_company to list documents"
            )
        pages = self.store.get_pages(args.doc_name, args.pages)
        found = {page.page_num for page in pages}
        rendered: list[dict[str, Any]] = []
        for page in pages:
            chunk = page_to_chunk(page)
            self.seen_chunks.setdefault(chunk.chunk_id, chunk)
            text = page.text
            truncated = len(text) > PAGE_MAX_CHARS
            if truncated:
                text = text[:PAGE_MAX_CHARS].rstrip() + _TRUNCATED_MARK
            rendered.append(
                {
                    "page_num": page.page_num,
                    "ref": f"chunk:{chunk.chunk_id}",
                    "chars": len(page.text),
                    "truncated": truncated,
                    "text": text,
                }
            )
        payload: dict[str, Any] = {"doc_name": args.doc_name, "pages": rendered}
        missing = sorted(set(args.pages) - found)
        if missing:
            payload["missing_pages"] = missing
        return payload, _shrink_texts("pages", "text", _MIN_PAGE_CHARS)

    def _lookup_company(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], Shrinker]:
        args = _CompanyArgs.model_validate(arguments)
        needle = " ".join(args.name_or_ticker.split()).casefold()
        if not needle:
            raise _ToolError("name_or_ticker must not be blank")
        documents = self.store.list_documents()
        by_company: dict[tuple[str | None, str], list[DocumentMeta]] = {}
        for doc in documents:
            if not _company_matches(doc, needle):
                continue
            by_company.setdefault((doc.ticker, doc.company), []).append(doc)
        matches = [
            {
                "ticker": ticker,
                "company": company,
                "cik": docs[0].cik,
                "fiscal_years": sorted({d.fiscal_year for d in docs if d.fiscal_year is not None}),
                "documents": [
                    {
                        "doc_name": d.doc_name,
                        "form": d.form,
                        "fiscal_year": d.fiscal_year,
                        "period_end": d.period_end.isoformat() if d.period_end else None,
                        "n_pages": d.n_pages,
                    }
                    for d in docs
                ],
            }
            for (ticker, company), docs in by_company.items()
        ]
        payload: dict[str, Any] = {"query": args.name_or_ticker, "matches": matches}
        if not matches:
            payload["note"] = (
                f"no indexed filings match {args.name_or_ticker!r} "
                f"({len(documents)} documents indexed)"
            )
        return payload, _shrink_list("matches")

    def _lookup_fact(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], Shrinker]:
        args = _FactArgs.model_validate(arguments)
        try:
            rows = lookup_fact(self.store, args.ticker, args.metric, args.fiscal_year)
        except ValueError as exc:
            raise _ToolError(str(exc)) from exc
        facts = [self._remember_fact(row) for row in rows]
        payload: dict[str, Any] = {
            "ticker": args.ticker.strip().upper(),
            "metric": args.metric,
            "fiscal_year": args.fiscal_year,
            "n": len(facts),
            "facts": facts,
        }
        if not facts:
            payload["note"] = (
                "no annual 10-K value under any known tag; try another metric name, "
                "query_xbrl over xbrl_facts, or search_filings for the reported figure"
            )
        return payload, _shrink_list("facts")

    def _query_xbrl(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], Shrinker]:
        args = _SqlArgs.model_validate(arguments)
        try:
            result: SqlResult = run_readonly_sql(self.store, args.sql, timeout_s=self.sql_timeout_s)
        except SqlRejected as exc:  # guard refusals, timeouts and DuckDB execution errors
            raise _ToolError(str(exc)) from exc
        columns = list(result.columns)
        rows = [list(row) for row in result.rows]
        citable = _CITABLE_COLUMNS <= set(columns)
        n_ambiguous = 0
        if citable:
            columns.append("ref")
            n_ambiguous = self._attach_refs(result.columns, rows)
        payload: dict[str, Any] = {
            "sql": result.sql,
            "columns": columns,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "rows": rows,
        }
        if not citable:
            payload["note"] = (
                "rows are not citable: select tag, fy, accn and val from xbrl_facts (or use "
                "lookup_fact) to get refs"
            )
        elif n_ambiguous:
            payload["note"] = (
                f"{n_ambiguous} row(s) have ref null because several values share the same "
                "tag, fy and accn (fy is the filing's fiscal year, so comparative periods "
                "collide); use lookup_fact for the value you need, or filter by end_date"
            )
        return payload, _shrink_list("rows")

    def _attach_refs(self, columns: list[str], rows: list[list[Any]]) -> int:
        """Append a ``ref`` cell to every row; register the unambiguous ones in the ledger.

        A ref is only citable when exactly one value carries it - within this result and
        against anything already in :attr:`seen_facts` - because ``xbrl_facts.fy`` is the
        filing's fiscal-year focus and comparative periods in one 10-K share tag, fy and accn.
        Returns the number of rows left without a ref for that reason.
        """
        facts = [_fact_from_row(columns, row) for row in rows]
        values_by_ref: dict[str, set[float]] = {}
        for fact in facts:
            if fact is not None:
                values_by_ref.setdefault(fact.ref, set()).add(fact.val)
        n_ambiguous = 0
        for row, fact in zip(rows, facts, strict=True):
            if fact is None:
                row.append(None)
                continue
            seen = self.seen_facts.get(fact.ref)
            distinct = values_by_ref[fact.ref] | ({seen.val} if seen is not None else set())
            if len(distinct) != 1:
                n_ambiguous += 1
                row.append(None)
                continue
            row.append(self._remember_fact(fact)["ref"])
        return n_ambiguous

    def _calculate(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], None]:
        args = _CalcArgs.model_validate(arguments)
        try:
            value = safe_calculate(args.expression)
        except CalcRejected as exc:
            raise _ToolError(exc.reason) from exc
        expression = " ".join(args.expression.split())
        self.calc_results.append(value)
        self.calc_log.append((expression, value))
        return {"expression": expression, "result": value}, None

    def _final_answer(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], None]:
        try:
            final = decode_final_answer(arguments)
        except ValueError as exc:
            raise _ToolError(str(exc)) from exc
        invalid = self.invalid_refs(final.citations)
        if invalid:
            raise _ToolError(
                "unknown citation ref(s): "
                + ", ".join(invalid)
                + "; cite only refs returned by tools in this conversation "
                "(chunk:<id> or xbrl:<tag>|FY<fy>|<accn>) or abstain"
            )
        return {"status": "accepted", "abstain": final.abstained}, None

    def final_answer_from(self, arguments: dict[str, Any]) -> FinalAnswer:
        """Decode ``final_answer`` arguments already accepted by :meth:`dispatch`.

        Raises:
            ValueError: if the arguments do not validate (cannot happen after acceptance).
        """
        return decode_final_answer(arguments)

    # ---- internals ----------------------------------------------------------------------------

    def _merge_filters(self, args: _SearchArgs) -> RetrievalFilters | None:
        base = self.base_filters
        ticker = args.ticker or (base.ticker if base else None)
        fiscal_year = (
            args.fiscal_year
            if args.fiscal_year is not None
            else (base.fiscal_year if base else None)
        )
        form = args.form or (base.form if base else None)
        doc_names = base.doc_names if base else None
        if ticker is None and fiscal_year is None and form is None and doc_names is None:
            return None
        return RetrievalFilters(
            ticker=ticker, doc_names=doc_names, fiscal_year=fiscal_year, form=form
        )

    def _remember_hit(self, hit: Hit) -> None:
        if hit.chunk.chunk_id not in self.seen_chunks:
            self.seen_chunks[hit.chunk.chunk_id] = hit.chunk
            self.retrieved.append(to_hit_view(hit))

    def _remember_fact(self, fact: FactRow) -> dict[str, Any]:
        self.seen_facts.setdefault(fact.ref, fact)
        return {
            "ref": fact.ref,
            "concept": fact.concept_used or f"{fact.taxonomy}:{fact.tag}",
            "value": fact.val,
            "unit": fact.unit,
            "fiscal_year": fact.fy,
            "fp": fact.fp,
            "form": fact.form,
            "start_date": fact.start_date.isoformat() if fact.start_date else None,
            "end_date": fact.end_date.isoformat() if fact.end_date else None,
            "accn": fact.accn,
            "filed": fact.filed.isoformat() if fact.filed else None,
        }

    def _encode(self, payload: dict[str, Any], shrink: Shrinker | None) -> tuple[str, bool]:
        """JSON-encode ``payload`` within ``max_result_chars``, shrinking structurally first.

        Returns ``(content, gave_up)``; ``gave_up`` is True when even the smallest form does not
        fit, in which case the content is an error object asking for a narrower request.
        """
        text = _dumps(payload)
        shrunk = False
        while len(text) > self.max_result_chars and shrink is not None and shrink(payload):
            shrunk = True
            payload["truncated"] = True
            text = _dumps(payload)
        if len(text) <= self.max_result_chars:
            if shrunk:
                log.info("agent_tool_result_shrunk", result_chars=len(text))
            return text, False
        log.warning("agent_tool_result_too_large", result_chars=len(text))
        error = {
            "error": (
                f"result exceeds {self.max_result_chars} characters even after truncation; "
                "narrow the request (fewer pages, smaller k, or a more selective query)"
            ),
            "truncated": True,
        }
        return _dumps(error), True


# ---- helpers ---------------------------------------------------------------------------------


def page_to_chunk(page: Page) -> Chunk:
    """Wrap a page as one citable pseudo-chunk (same id as the rag oracle mode's wrapper)."""
    return Chunk(
        chunk_id=make_chunk_id(page.doc_name, page.page_num, 0, page.text),
        doc_name=page.doc_name,
        page_num=page.page_num,
        chunk_idx=0,
        section=None,
        text=page.text,
        n_tokens=estimate_tokens(page.text),
    )


def _company_matches(doc: DocumentMeta, needle: str) -> bool:
    if doc.ticker and doc.ticker.casefold() == needle:
        return True
    if needle in doc.company.casefold():
        return True
    return doc.doc_name.casefold().startswith(needle)


def _fact_from_row(columns: list[str], row: list[Any]) -> FactRow | None:
    """Build a ``FactRow`` from a citable ``query_xbrl`` row; ``None`` if the key cells are null."""
    cells = dict(zip(columns, row, strict=True))
    tag, accn, fy, val = cells.get("tag"), cells.get("accn"), cells.get("fy"), cells.get("val")
    if not tag or not accn or fy is None or val is None:
        return None
    try:
        return FactRow(
            cik=str(cells.get("cik") or ""),
            ticker=str(cells.get("ticker") or ""),
            taxonomy=str(cells.get("taxonomy") or "us-gaap"),
            tag=str(tag),
            unit=str(cells.get("unit") or ""),
            fy=int(fy),
            fp=_optional_str(cells.get("fp")),
            form=_optional_str(cells.get("form")),
            start_date=_optional_date(cells.get("start_date")),
            end_date=_optional_date(cells.get("end_date")),
            val=float(val),
            accn=str(accn),
            filed=_optional_date(cells.get("filed")),
            frame=_optional_str(cells.get("frame")),
        )
    except (TypeError, ValueError, ValidationError):
        return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _shrink_texts(list_key: str, text_key: str, min_chars: int) -> Shrinker:
    """Halve every ``text_key`` in ``payload[list_key]`` down to ``min_chars``, then drop items
    from the end until one is left."""

    def shrink(payload: dict[str, Any]) -> bool:
        items = payload.get(list_key) or []
        # Measure without the marker, otherwise a text already cut to ``min_chars`` looks
        # longer than ``min_chars`` forever and the caller's loop never terminates.
        bodies = [(item, _without_mark(item.get(text_key) or "")) for item in items]
        if any(len(body) > min_chars for _item, body in bodies):
            for item, body in bodies:
                if len(body) > min_chars:
                    cut = max(min_chars, len(body) // 2)  # strictly shorter than len(body)
                    item[text_key] = body[:cut].rstrip() + _TRUNCATED_MARK
                    item["truncated"] = True
            return True
        if len(items) > 1:  # never drop the last item: an empty result would hide the overflow
            items.pop()
            return True
        return False

    return shrink


def _without_mark(text: str) -> str:
    return text[: -len(_TRUNCATED_MARK)] if text.endswith(_TRUNCATED_MARK) else text


def _shrink_list(list_key: str) -> Shrinker:
    """Drop the second half of ``payload[list_key]``; False once one item is left."""

    def shrink(payload: dict[str, Any]) -> bool:
        items = payload.get(list_key) or []
        if len(items) <= 1:  # never drop the last item (see ``_shrink_texts``)
            return False
        del items[len(items) // 2 :]
        return True

    return shrink


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _short(exc: ValidationError) -> str:
    details = exc.errors()
    if not details:
        return "invalid"
    first = details[0]
    location = ".".join(str(part) for part in first["loc"]) or "<root>"
    return f"{location}: {first['msg']}"


__all__ = [
    "DEFAULT_MAX_RESULT_CHARS",
    "DEFAULT_SQL_TIMEOUT_S",
    "SEARCH_SNIPPET_CHARS",
    "ToolRuntime",
    "page_to_chunk",
]
