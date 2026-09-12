"""Tool specifications for the agent (SPEC section 5) and the final-answer contract.

Every tool is a :class:`~secqa.core.contracts.ToolSpec` whose ``input_schema`` follows the
strict-mode subset both vendors accept: ``additionalProperties: false``, every property listed in
``required``, optional values expressed as ``["<type>", "null"]``. Numeric bounds (``k <= 10``,
``pages`` at most 3) are stated in descriptions and enforced by :mod:`secqa.agent.runtime`, not
by schema keywords, because the vendors' strict modes do not agree on which keywords they allow.

The same ``final_answer`` shape serves three purposes: the ``final_answer`` tool's arguments, the
``json_schema`` of the forced last call the loop makes after an abort (``tools=None``), and the
structure a text-only reply must parse into to be accepted as final. :class:`FinalAnswer` is the
validated Python side; :func:`decode_final_answer` and :func:`parse_final_answer_text` turn tool
arguments or free text into it without ever raising anything but ``ValueError``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import Field, ValidationError, field_validator

from secqa.core.contracts import CitationRef, Frozen, ToolSpec

SEARCH_FILINGS = "search_filings"
GET_PAGES = "get_pages"
LOOKUP_COMPANY = "lookup_company"
LOOKUP_FACT = "lookup_fact"
QUERY_XBRL = "query_xbrl"
CALCULATE = "calculate"
FINAL_ANSWER = "final_answer"

MAX_SEARCH_K = 10
DEFAULT_SEARCH_K = 8
MAX_PAGES_PER_CALL = 3
PAGE_MAX_CHARS = 6000

ABSTAIN_TEXT = "INSUFFICIENT EVIDENCE"
"""Canonical abstention text (the same string the rag prompts and the mock provider use)."""

_CITATION_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ref", "quote"],
    "properties": {
        "ref": {
            "type": "string",
            "description": (
                "A ref returned by a tool in this conversation: 'chunk:<id>' from search_filings "
                "or get_pages, or 'xbrl:<tag>|FY<fy>|<accn>' from lookup_fact / query_xbrl."
            ),
        },
        "quote": {
            "type": "string",
            "description": (
                "For chunk refs: a verbatim span of at least 20 characters copied from that "
                "chunk. For xbrl refs: empty string."
            ),
        },
    },
}

FINAL_ANSWER_SCHEMA: dict[str, Any] = {
    "title": "final_answer",
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "value", "unit", "citations", "calculation", "abstain"],
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "One to three sentences answering the question, or exactly "
                "'INSUFFICIENT EVIDENCE' when abstaining."
            ),
        },
        "value": {
            "type": ["number", "null"],
            "description": (
                "The single answer number in base units (dollars not millions; ratios as "
                "fractions, 12% is 0.12), or null when the answer is not one number."
            ),
        },
        "unit": {
            "type": ["string", "null"],
            "description": "Unit of value ('USD', 'shares', 'ratio', 'percent', 'years'), or null.",
        },
        "citations": {
            "type": "array",
            "items": _CITATION_ITEM_SCHEMA,
            "description": "Every ref the answer relies on; every number must be covered.",
        },
        "calculation": {
            "type": ["string", "null"],
            "description": (
                "The arithmetic you performed with the calculate tool, written out, or null."
            ),
        },
        "abstain": {
            "type": "boolean",
            "description": "True when the gathered evidence does not answer the question.",
        },
    },
}

TOOLS: list[ToolSpec] = [
    ToolSpec(
        name=SEARCH_FILINGS,
        description=(
            "Hybrid (BM25 + dense) search over the indexed 10-K / 10-Q pages. Returns up to k "
            "chunks with a 'ref' (chunk:<id>) you can cite, doc_name, page_num and a snippet. "
            "Use short keyword queries such as 'total net sales fiscal 2023'. Filter by ticker "
            "and fiscal_year when you know them; set unknown filters to null."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["query", "ticker", "fiscal_year", "form", "k"],
            "properties": {
                "query": {"type": "string", "description": "Search terms."},
                "ticker": {
                    "type": ["string", "null"],
                    "description": "Restrict to one company's filings by ticker, or null.",
                },
                "fiscal_year": {
                    "type": ["integer", "null"],
                    "description": "Restrict to filings for this fiscal year, or null.",
                },
                "form": {
                    "type": ["string", "null"],
                    "description": "Restrict to a form type such as '10-K' or '10-Q', or null.",
                },
                "k": {
                    "type": ["integer", "null"],
                    "description": f"Number of chunks, 1 to {MAX_SEARCH_K}; null means "
                    f"{DEFAULT_SEARCH_K}.",
                },
            },
        },
    ),
    ToolSpec(
        name=GET_PAGES,
        description=(
            "Read the full text of up to 3 pages of one document (page numbers are 1-based, as "
            "shown by search_filings). Each page comes back with its own citable ref "
            f"(chunk:<id>) and is truncated after {PAGE_MAX_CHARS} characters (flagged). Use it "
            "to read the table or paragraph around a search hit."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["doc_name", "pages"],
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "Exact doc_name from search_filings or lookup_company.",
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": f"1-based page numbers, at most {MAX_PAGES_PER_CALL}.",
                },
            },
        },
    ),
    ToolSpec(
        name=LOOKUP_COMPANY,
        description=(
            "Find which filings are indexed for a company by ticker or (partial) name. Returns "
            "ticker, company, cik and the documents with their form and fiscal year. Call it "
            "first when you are unsure of the ticker or which years are available."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["name_or_ticker"],
            "properties": {
                "name_or_ticker": {
                    "type": "string",
                    "description": "A ticker such as 'MSFT' or part of a company name.",
                }
            },
        },
    ),
    ToolSpec(
        name=LOOKUP_FACT,
        description=(
            "Annual XBRL value of a financial metric from the company's 10-K, with the accession "
            "number, returned as a citable 'xbrl:...' ref. metric is a curated name (revenue, "
            "net_income, total_assets, cfo, capex, long_term_debt, eps_diluted, ...) or a raw "
            "us-gaap concept such as 'Assets'. Prefer this over reading tables for standard "
            "line items; an empty result means the metric is not reported under any known tag."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ticker", "metric", "fiscal_year"],
            "properties": {
                "ticker": {"type": "string", "description": "Company ticker, e.g. 'MSFT'."},
                "metric": {
                    "type": "string",
                    "description": "Curated metric name or a us-gaap / dei concept name.",
                },
                "fiscal_year": {"type": "integer", "description": "Fiscal year, e.g. 2023."},
            },
        },
    ),
    ToolSpec(
        name=QUERY_XBRL,
        description=(
            "Run one read-only SQL SELECT (DuckDB dialect) over the tables xbrl_facts "
            "(cik, ticker, taxonomy, tag, unit, fy, fp, form, start_date, end_date, val, accn, "
            "filed, frame), financials (one row per cik, ticker, fiscal_year with one column "
            "per curated metric, no accession numbers) and documents (doc_name, ticker, cik, "
            "company, form, fiscal_year). At most 200 rows are returned. Rows from financials "
            "are NOT citable; to cite a number select tag, fy, accn and val from xbrl_facts "
            "(such rows get a 'ref' column) or use lookup_fact."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["sql"],
            "properties": {
                "sql": {"type": "string", "description": "A single SELECT statement."},
            },
        },
    ),
    ToolSpec(
        name=CALCULATE,
        description=(
            "Evaluate an arithmetic expression exactly: numbers, + - * / ** %, parentheses and "
            "abs/round/min/max. Write numbers in base units without currency symbols or "
            "thousands separators (1577000000, not $1,577 million). Use it for every "
            "derived number (growth rates, margins, ratios) so the result is recorded."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["expression"],
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "e.g. '(1577000000 - 1408000000) / 1408000000'",
                }
            },
        },
    ),
    ToolSpec(
        name=FINAL_ANSWER,
        description=(
            "Finish with your answer. Cite only refs returned by tools in this conversation; a "
            "citation with an unknown ref is rejected and you get one chance to fix it. Set "
            "abstain to true and answer 'INSUFFICIENT EVIDENCE' when the evidence does not "
            "answer the question."
        ),
        input_schema={key: value for key, value in FINAL_ANSWER_SCHEMA.items() if key != "title"},
    ),
]
"""The six evidence tools plus ``final_answer``, in the order the model sees them."""

TOOL_NAMES: tuple[str, ...] = tuple(tool.name for tool in TOOLS)

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_ABSTAIN_RE = re.compile(r"^\W*insufficient\s+evidence\b", re.IGNORECASE)


class FinalAnswer(Frozen):
    """Validated ``final_answer`` arguments (the Python side of :data:`FINAL_ANSWER_SCHEMA`)."""

    answer: str = ""
    value: float | None = None
    unit: str | None = None
    citations: list[CitationRef] = Field(default_factory=list)
    calculation: str | None = None
    abstain: bool = False

    @field_validator("unit", "calculation", mode="before")
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def abstained(self) -> bool:
        """True when the model set ``abstain`` or wrote the canonical abstention text."""
        return self.abstain or bool(_ABSTAIN_RE.match(self.answer))

    @property
    def text(self) -> str:
        """Answer text to show: the canonical abstention string when abstaining with no text."""
        stripped = self.answer.strip()
        if self.abstained and not stripped:
            return ABSTAIN_TEXT
        return stripped


def abstain_answer(reason: str | None = None) -> FinalAnswer:
    """The canonical abstention (used when the loop has to stop without a model answer)."""
    return FinalAnswer(answer=ABSTAIN_TEXT, abstain=True, calculation=reason)


def decode_final_answer(data: Any) -> FinalAnswer:
    """Validate a decoded mapping as a :class:`FinalAnswer`.

    Tolerates ``null`` citations and a missing answer (both become empty) but rejects unknown
    keys, so a malformed tool call is reported rather than silently trimmed.

    Raises:
        ValueError: with a one-line reason suitable for a tool error message.
    """
    if not isinstance(data, dict):
        raise ValueError(f"final answer must be a JSON object, got {type(data).__name__}")
    unknown = sorted(set(data) - set(FINAL_ANSWER_SCHEMA["properties"]))
    if unknown:
        raise ValueError(f"unknown final_answer field(s): {', '.join(unknown)}")
    known = dict(data)
    if known.get("citations") is None:
        known["citations"] = []
    if known.get("answer") is None:
        known["answer"] = ""
    try:
        return FinalAnswer.model_validate(known)
    except ValidationError as exc:
        raise ValueError(_short(exc)) from exc


def parse_final_answer_text(text: str) -> tuple[FinalAnswer | None, str | None]:
    """Decode a text-only model reply as a final answer: ``(answer, None)`` or ``(None, why)``.

    Accepts a bare JSON object or one wrapped in a Markdown code fence. Anything else, including
    prose, is not accepted as final (SPEC section 5: a text-only reply counts only when the
    ``final_answer`` schema parses).
    """
    candidate = (text or "").strip()
    if not candidate:
        return None, "empty reply"
    fenced = _CODE_FENCE_RE.match(candidate)
    if fenced:
        candidate = fenced.group(1)
    try:
        data = json.loads(candidate)
    except ValueError:
        return None, "reply is not a JSON object matching the final_answer schema"
    try:
        return decode_final_answer(data), None
    except ValueError as exc:
        return None, str(exc)


def _short(exc: ValidationError) -> str:
    details = exc.errors()
    if not details:
        return "invalid final_answer arguments"
    first = details[0]
    location = ".".join(str(part) for part in first["loc"]) or "<root>"
    return f"{location}: {first['msg']}"


__all__ = [
    "ABSTAIN_TEXT",
    "CALCULATE",
    "DEFAULT_SEARCH_K",
    "FINAL_ANSWER",
    "FINAL_ANSWER_SCHEMA",
    "GET_PAGES",
    "LOOKUP_COMPANY",
    "LOOKUP_FACT",
    "MAX_PAGES_PER_CALL",
    "MAX_SEARCH_K",
    "PAGE_MAX_CHARS",
    "QUERY_XBRL",
    "SEARCH_FILINGS",
    "TOOLS",
    "TOOL_NAMES",
    "FinalAnswer",
    "abstain_answer",
    "decode_final_answer",
    "parse_final_answer_text",
]
