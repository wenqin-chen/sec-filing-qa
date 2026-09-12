"""Deterministic citation verification and the ``grounded`` flag (SPEC section 6).

The verifier is the one place where a model's claims are checked against what the request
actually retrieved. It never calls a model and never touches the network, so ``rag``, ``agent``,
``api`` and ``eval`` all get byte-identical results for the same inputs.

Rules (each one is a test in ``tests/grounding/test_verifier.py``):

* A citation ref is ``chunk:<chunk_id>`` or ``xbrl:<tag>|FY<fy>|<accn>``; anything else is
  malformed and kept as an *invalid* citation so the failure is visible, never hidden.
* A ref is **valid** only if it resolves to a chunk / fact row that this request retrieved
  (the ``chunks`` / ``facts`` mappings). Unknown refs are kept with ``valid=False``.
* A chunk citation is **verified** when the model-supplied quote, after
  :func:`secqa.core.textnum.normalize_text`, is at least ``min_quote_chars`` long and a substring
  of the normalised chunk text. Paraphrases are kept with ``verified=False``.
* An XBRL citation is verified exactly when the fact row was returned by a tool in this request
  (the model cannot "quote" a fact row; the accession number is the evidence).
* ``snippet`` always comes from the store (chunk text or a rendering of the fact row), capped at
  300 characters, so nothing shown to a user is model-generated.
* ``grounded`` is true when every numeric token in the answer text appears (within
  :func:`secqa.core.textnum.numbers_equal` tolerance, ratio/percent equivalence included) in a
  verified quote, a verified fact value, or a ``calculate()`` result of this request. Bare
  four-digit years (``fiscal 2023``, ``(2022)``) are dates, not quantities, and are masked
  before extraction; an answer with no numeric tokens is vacuously grounded.
"""

from __future__ import annotations

import re
from typing import Literal

from secqa.core.contracts import Chunk, Citation, CitationRef, FactRow
from secqa.core.logging import get_logger
from secqa.core.textnum import extract_numbers, normalize_text, numbers_equal

log = get_logger(__name__)

RefKind = Literal["chunk", "xbrl"]

CHUNK_PREFIX = "chunk:"
XBRL_PREFIX = "xbrl:"
SNIPPET_MAX_CHARS = 300

_WS_RE = re.compile(r"\s+")

# A bare four-digit year (1900-2099) that is not part of a larger number and is not followed by
# a scale word or unit. ``$2,022``, ``2022 million`` and ``2022.5`` are quantities and stay;
# ``fiscal 2023``, ``(2022)`` and ``from 2021.`` are dates and are masked before extraction.
_YEAR_RE = re.compile(
    r"""
    (?<![$€£¥\d.,])                      # not glued to currency, digits or a thousands separator
    \b(?:19|20)\d{2}\b                   # 1900-2099
    (?!\s*(?:[,.]\d                      # ... not continued as 2,022 or 2022.5
             |%                          # ... not a percentage
             |(?:percent|pct|bps|basis\ points?|thousands?|millions?|billions?|trillions?
               |tn|bn|mm|mn|[kmbtx])\b)) # ... not followed by a unit or scale word
    """,
    re.VERBOSE | re.IGNORECASE,
)
_YEAR_PLACEHOLDER = "YEAR"


def parse_ref(ref: str) -> tuple[RefKind, str]:
    """Split a citation ref into ``(kind, payload)``.

    ``'chunk:<chunk_id>'`` -> ``('chunk', '<chunk_id>')`` and
    ``'xbrl:<tag>|FY<fy>|<accn>'`` -> ``('xbrl', '<tag>|FY<fy>|<accn>')``. The payload is
    returned verbatim (whitespace-stripped) so callers can use it as a lookup key.

    Raises:
        ValueError: if ``ref`` has neither prefix, has an empty payload, or an XBRL payload does
            not have the three ``|``-separated parts.
    """
    if not isinstance(ref, str):
        raise ValueError(f"citation ref must be a string, got {type(ref).__name__}")
    candidate = ref.strip()
    if candidate.startswith(CHUNK_PREFIX):
        payload = candidate[len(CHUNK_PREFIX) :].strip()
        if not payload:
            raise ValueError(f"empty chunk id in citation ref {ref!r}")
        return "chunk", payload
    if candidate.startswith(XBRL_PREFIX):
        payload = candidate[len(XBRL_PREFIX) :].strip()
        parts = payload.split("|")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ValueError(
                f"xbrl citation ref must look like 'xbrl:<tag>|FY<fy>|<accn>', got {ref!r}"
            )
        if not parts[1].strip().upper().startswith("FY"):
            raise ValueError(f"xbrl citation ref fiscal-year part must start with 'FY': {ref!r}")
        return "xbrl", payload
    raise ValueError(f"citation ref must start with 'chunk:' or 'xbrl:', got {ref!r}")


def quote_in_chunk(quote: str, chunk_text: str, min_chars: int = 20) -> bool:
    """True if the normalised ``quote`` is a substring of the normalised ``chunk_text``.

    Normalisation (:func:`secqa.core.textnum.normalize_text`) casefolds, drops punctuation and
    collapses whitespace, so curly quotes, line breaks and a trailing full stop never break an
    otherwise faithful quote. Quotes shorter than ``min_chars`` *after* normalisation are
    rejected: a two-word quote matches almost any page and proves nothing.
    """
    if min_chars < 1:
        raise ValueError(f"min_chars must be positive, got {min_chars}")
    needle = normalize_text(quote)
    if len(needle) < min_chars:
        return False
    haystack = normalize_text(chunk_text)
    return needle in haystack


class CitationVerifier:
    """Verify model citations against retrieved chunks / fact rows and compute ``grounded``.

    Args:
        min_quote_chars: minimum length of a normalised quote for a chunk citation to count as
            verified. Defaults to 20 (SPEC section 6).
    """

    def __init__(self, min_quote_chars: int = 20) -> None:
        if min_quote_chars < 1:
            raise ValueError(f"min_quote_chars must be positive, got {min_quote_chars}")
        self.min_quote_chars = min_quote_chars

    def verify(
        self,
        answer_text: str,
        refs: list[CitationRef],
        chunks: dict[str, Chunk],
        facts: dict[str, FactRow],
        calc_results: list[float],
    ) -> tuple[list[Citation], bool]:
        """Return ``(citations, grounded)`` for one answer.

        Args:
            answer_text: the model's final answer text.
            refs: citation refs (with optional quotes) exactly as the model returned them.
            chunks: chunks retrieved in this request, keyed by ``chunk_id`` (the full
                ``chunk:<id>`` ref is also accepted as a key).
            facts: XBRL fact rows returned by tools in this request, keyed by ``FactRow.ref``
                (``xbrl:<tag>|FY<fy>|<accn>``; the bare payload is also accepted as a key).
            calc_results: results of ``calculate()`` calls made in this request.

        Citations come back in the order given, one per distinct ``(ref, quote)`` pair; nothing
        is dropped. ``grounded`` is defined in the module docstring.
        """
        citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        for ref in refs:
            key = (ref.ref.strip(), ref.quote)
            if key in seen:
                continue
            seen.add(key)
            citations.append(self._verify_one(ref, chunks, facts))

        evidence = self._evidence_numbers(citations, chunks, facts, calc_results)
        answer_numbers = extract_numbers(_mask_years(answer_text))
        unsupported = [
            value
            for value in answer_numbers
            if not any(numbers_equal(value, candidate) for candidate in evidence)
        ]
        grounded = not unsupported

        log.info(
            "citations_verified",
            n_refs=len(refs),
            n_citations=len(citations),
            n_valid=sum(c.valid for c in citations),
            n_verified=sum(c.verified for c in citations),
            n_answer_numbers=len(answer_numbers),
            n_unsupported_numbers=len(unsupported),
            grounded=grounded,
        )
        return citations, grounded

    # ------------------------------------------------------------------ internals

    def _verify_one(
        self, ref: CitationRef, chunks: dict[str, Chunk], facts: dict[str, FactRow]
    ) -> Citation:
        """Resolve and verify a single ref; never raises."""
        raw_ref = ref.ref.strip()
        try:
            kind, payload = parse_ref(raw_ref)
        except ValueError as exc:
            log.warning("citation_ref_malformed", ref=raw_ref, reason=str(exc))
            return Citation(
                ref=raw_ref,
                kind="xbrl" if raw_ref.startswith(XBRL_PREFIX) else "chunk",
                quote=ref.quote,
                snippet="",
                verified=False,
                valid=False,
            )

        if kind == "chunk":
            chunk = chunks.get(payload) or chunks.get(raw_ref)
            if chunk is None:
                log.warning("citation_ref_unknown", ref=raw_ref, kind=kind)
                return Citation(
                    ref=raw_ref,
                    kind="chunk",
                    chunk_id=payload,
                    quote=ref.quote,
                    snippet="",
                    verified=False,
                    valid=False,
                )
            return Citation(
                ref=raw_ref,
                kind="chunk",
                doc_name=chunk.doc_name,
                page_num=chunk.page_num,
                chunk_id=chunk.chunk_id,
                quote=ref.quote,
                snippet=_snippet(chunk.text),
                verified=quote_in_chunk(ref.quote, chunk.text, self.min_quote_chars),
                valid=True,
            )

        fact = facts.get(raw_ref) or facts.get(payload)
        if fact is None:
            log.warning("citation_ref_unknown", ref=raw_ref, kind=kind)
            tag, fy_part, accn = (part.strip() for part in payload.split("|"))
            return Citation(
                ref=raw_ref,
                kind="xbrl",
                tag=tag,
                fiscal_year=_parse_fy(fy_part),
                accn=accn,
                quote=ref.quote,
                snippet="",
                verified=False,
                valid=False,
            )
        return Citation(
            ref=raw_ref,
            kind="xbrl",
            tag=fact.tag,
            fiscal_year=fact.fy,
            accn=fact.accn,
            value=fact.val,
            quote=ref.quote,
            snippet=_snippet(_render_fact(fact)),
            verified=True,
            valid=True,
        )

    @staticmethod
    def _evidence_numbers(
        citations: list[Citation],
        chunks: dict[str, Chunk],
        facts: dict[str, FactRow],
        calc_results: list[float],
    ) -> list[float]:
        """Every number the answer is allowed to contain: verified quotes, facts, calculations."""
        evidence: list[float] = [float(value) for value in calc_results]
        for citation in citations:
            if not citation.verified:
                continue
            if citation.kind == "chunk":
                evidence.extend(extract_numbers(citation.quote))
            elif citation.value is not None:
                evidence.append(citation.value)
        return evidence


def _mask_years(text: str) -> str:
    """Replace bare four-digit years with a non-numeric placeholder (see ``_YEAR_RE``)."""
    if not text:
        return ""
    return _YEAR_RE.sub(_YEAR_PLACEHOLDER, text)


def _snippet(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Whitespace-collapsed prefix of store text, at most ``max_chars`` characters."""
    collapsed = _WS_RE.sub(" ", text or "").strip()
    return collapsed[:max_chars]


def _render_fact(fact: FactRow) -> str:
    """Human-readable, store-sourced rendering of a fact row used as its snippet."""
    period = ""
    if fact.start_date and fact.end_date:
        period = f" {fact.start_date.isoformat()} to {fact.end_date.isoformat()}"
    elif fact.end_date:
        period = f" as of {fact.end_date.isoformat()}"
    fiscal = f" FY{fact.fy}" if fact.fy is not None else ""
    fp = f" {fact.fp}" if fact.fp else ""
    filed = f", filed {fact.filed.isoformat()}" if fact.filed else ""
    form = f" {fact.form}" if fact.form else ""
    return (
        f"{fact.taxonomy}:{fact.tag}{fiscal}{fp}{period} = {fact.val:,} {fact.unit} "
        f"({fact.ticker}{form} accn {fact.accn}{filed})"
    )


def _parse_fy(fy_part: str) -> int | None:
    """``'FY2022'`` -> ``2022``; anything unparsable -> ``None``."""
    digits = fy_part.strip()[2:]
    return int(digits) if digits.isdigit() else None


__all__ = [
    "CHUNK_PREFIX",
    "SNIPPET_MAX_CHARS",
    "XBRL_PREFIX",
    "CitationVerifier",
    "RefKind",
    "parse_ref",
    "quote_in_chunk",
]
