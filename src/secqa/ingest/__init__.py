"""Document ingestion: PDFs and EDGAR HTML -> 1-based pages -> overlapping token chunks.

Public surface (see CONTRACTS.md and the module manifest):

* :func:`extract_pdf_pages` — pypdfium2 text per physical page, ``page_num = index + 1``.
* :func:`extract_html_pages` — EDGAR HTML flattened into pseudo-pages of ~``target_chars``.
* :func:`chunk_pages` — page-bounded, overlapping token windows with section metadata.
* :func:`detect_section` — ``Item 7.`` / ``Item 1A.`` heading detection.
* :func:`doc_meta_from_name` — ``'3M_2022_10K'`` -> ``('3M', 2022, '10-K')``.
"""

from __future__ import annotations

import re

from secqa.ingest.chunk import chunk_pages, count_tokens, detect_section, get_tokenizer
from secqa.ingest.html import extract_html_pages
from secqa.ingest.pdf import extract_pdf_pages, normalize_extracted_text

UNKNOWN_FORM = "unknown"

# FinanceBench: '3M_2022_10K', '3M_2023Q2_10Q', 'JOHNSON_JOHNSON_2022_10K',
# 'FOOTLOCKER_2022_8K_dated-2022-05-20'. EDGAR ingest: '<TICKER>_<FY>_<FORM>' e.g. 'AAPL_2023_10-K'.
_DOC_NAME_RE = re.compile(
    r"^(?P<company>.+?)_(?P<year>(?:19|20)\d{2})(?:Q(?P<quarter>[1-4]))?"
    r"_(?P<form>[0-9A-Za-z\-/]+?)(?:_(?P<suffix>.+))?$"
)
_FORM_BY_KEY = {
    "10K": "10-K",
    "10KA": "10-K/A",
    "10Q": "10-Q",
    "10QA": "10-Q/A",
    "8K": "8-K",
    "8KA": "8-K/A",
    "20F": "20-F",
    "40F": "40-F",
    "S1": "S-1",
    "DEF14A": "DEF 14A",
}


def normalize_form(form: str) -> str:
    """Canonical SEC form label: ``'10K'``, ``'10-k'``, ``'10-K'`` -> ``'10-K'``; unknown as-is."""
    key = re.sub(r"[\s\-/]", "", form).upper()
    return _FORM_BY_KEY.get(key, form.strip().upper())


def doc_meta_from_name(doc_name: str) -> tuple[str, int | None, str]:
    """Split a document name into ``(company, fiscal_year, form)``.

    ``'3M_2022_10K'`` -> ``('3M', 2022, '10-K')``; ``'3M_2023Q2_10Q'`` -> ``('3M', 2023, '10-Q')``.
    Underscores inside the company part are preserved (``'JOHNSON_JOHNSON'``) and a trailing
    suffix such as ``'_dated-2022-05-20'`` is ignored. Names that do not follow the pattern
    return ``(doc_name, None, 'unknown')`` rather than guessing.
    """
    match = _DOC_NAME_RE.match(doc_name.strip())
    if match is None:
        return doc_name, None, UNKNOWN_FORM
    return match.group("company"), int(match.group("year")), normalize_form(match.group("form"))


__all__ = [
    "UNKNOWN_FORM",
    "chunk_pages",
    "count_tokens",
    "detect_section",
    "doc_meta_from_name",
    "extract_html_pages",
    "extract_pdf_pages",
    "get_tokenizer",
    "normalize_extracted_text",
    "normalize_form",
]
