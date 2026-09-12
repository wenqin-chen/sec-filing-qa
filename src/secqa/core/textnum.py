"""Numeric and text normalisation shared by grounding and evaluation metrics.

Design notes (defended in interviews):

* Numbers in filings and model answers come in many surface forms: ``$1.2 billion``,
  ``1,200 million``, ``(1,577)`` for negatives, ``12%``, ``1.5x``, ``3 bps``. All of them are
  reduced to a plain ``float`` in base units (dollars, ratio-as-fraction, multiple) so that
  ``numbers_equal`` can compare an answer against a quote regardless of how it was written.
* Percent and basis points are converted to *fractions* (``12%`` -> ``0.12``). Because gold
  answers sometimes say ``12`` when they mean ``12%``, :func:`numbers_equal` also accepts the
  ratio/percent equivalence (``0.12`` vs ``12``).
* Scale words are only honoured when they are followed by a word boundary, so ``5 months`` is
  ``5`` and not ``5 million``; identifiers such as ``10-K``, ``FY2022`` or ``Q2`` are skipped.
"""

from __future__ import annotations

import math
import re
import unicodedata

_SCALES: dict[str, float] = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mm": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
    "t": 1e12,
}

# Divisors (not multipliers) so that 3 bps is exactly 0.0003 in binary floating point.
_UNIT_DIVISORS: dict[str, float] = {
    "%": 100.0,
    "percent": 100.0,
    "pct": 100.0,
    "bps": 10_000.0,
    "basis points": 10_000.0,
    "basis point": 10_000.0,
    "x": 1.0,  # multiple, e.g. '1.5x'
}

_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])                                   # not glued to a preceding word / decimal
    (?P<open>\(\s*)?                             # optional '(' for accounting negatives
    (?P<sign>[-−–])?\s*                # optional minus (ASCII, U+2212, en dash)
    (?P<cur>US\$|USD\s?|[$€£¥])?\s*  # optional currency marker
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?         # 1,234,567.89
           |\d+(?:\.\d+)?                        # 1234.5
           |\.\d+)                               # .5
    (?![\d,]*\d)                                 # do not stop mid-number
    (?:\s*(?P<scale>trillions?|billions?|millions?|thousands?|tn|bn|mm|mn|[kmbt])
        (?![A-Za-z0-9]))?                        # scale needs a word boundary after it
    (?:\s*(?P<unit>%|percent|pct|bps|basis\ points?|x)
        (?![A-Za-z0-9]))?                        # unit needs a word boundary after it
    (?P<close>\s*\))?                            # optional ')' closing the accounting negative
    (?!-[A-Za-z])                                # skip '10-K', '10-Q'
    (?![A-Za-z])                                 # skip ordinals / identifiers ('5th', '2022A')
    """,
    re.VERBOSE | re.IGNORECASE,
)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _value_of(match: re.Match[str]) -> float:
    """Turn one regex match into a float in base units."""
    raw = match.group("num").replace(",", "")
    value = float(raw)
    scale = match.group("scale")
    if scale:
        key = scale.lower().rstrip("s") if len(scale) > 2 else scale.lower()
        value *= _SCALES[key]
    unit = match.group("unit")
    if unit:
        unit_key = _WS_RE.sub(" ", unit.lower())
        value /= _UNIT_DIVISORS[unit_key]
    negative = bool(match.group("sign")) or (
        bool(match.group("open")) and bool(match.group("close"))
    )
    if negative and value != 0.0:
        value = -value
    return value


def parse_number(text: str) -> float | None:
    """Parse a single numeric literal into a float in base units, or ``None``.

    Examples: ``'$1.2 billion'`` -> ``1.2e9``; ``'(1,577)'`` -> ``-1577.0``; ``'12%'`` -> ``0.12``;
    ``'1.5x'`` -> ``1.5``; ``'3 bps'`` -> ``0.0003``. Text that is not exactly one number
    (including sentences containing several numbers) returns ``None``; use
    :func:`extract_numbers` for free text.
    """
    if text is None:
        return None
    candidate = unicodedata.normalize("NFKC", str(text)).strip()
    if not candidate:
        return None
    match = _NUMBER_RE.fullmatch(candidate)
    if match is None:
        return None
    return _value_of(match)


def extract_numbers(text: str) -> list[float]:
    """Return every numeric token in ``text`` as floats in base units, in order of appearance.

    Identifiers that merely contain digits (``10-K``, ``FY2022``, ``Q3``, ``5th``) are skipped.
    """
    if not text:
        return []
    normalised = unicodedata.normalize("NFKC", text)
    return [_value_of(m) for m in _NUMBER_RE.finditer(normalised)]


def numbers_equal(a: float, b: float, rel_tol: float = 0.01) -> bool:
    """True if ``a`` and ``b`` agree within ``rel_tol`` relative tolerance.

    Also true for the ratio/percent equivalence (``0.12`` vs ``12``), because gold answers and
    filings write percentages both ways. Exact zero only matches zero.
    """
    if a is None or b is None:
        return False
    if math.isnan(a) or math.isnan(b):
        return False
    if a == b:
        return True
    if a == 0.0 or b == 0.0:
        return False
    if math.isclose(a, b, rel_tol=rel_tol):
        return True
    return math.isclose(a * 100.0, b, rel_tol=rel_tol) or math.isclose(
        a, b * 100.0, rel_tol=rel_tol
    )


def normalize_text(s: str) -> str:
    """NFKC-normalise, casefold, replace punctuation with spaces and collapse whitespace.

    Used for quote-in-chunk verification so that curly quotes, ligatures, line breaks and
    stray punctuation never break an otherwise faithful quote.
    """
    if not s:
        return ""
    folded = unicodedata.normalize("NFKC", s).casefold()
    without_punct = _PUNCT_RE.sub(" ", folded)
    return _WS_RE.sub(" ", without_punct).strip()


__all__ = ["extract_numbers", "normalize_text", "numbers_equal", "parse_number"]
