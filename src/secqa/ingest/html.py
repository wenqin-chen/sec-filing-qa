"""EDGAR HTML (incl. inline XBRL) to pseudo-pages.

EDGAR filings have no physical pages, so we manufacture them: the document is walked in reading
order into *blocks* (paragraphs, headings, list items, table rows), tables are flattened row-wise
with ``' | '`` between cells, and blocks are packed into pseudo-pages of roughly ``target_chars``
characters, splitting only at block boundaries. A CSS page break (``page-break-before/after:
always`` or ``break-before/after: page``, which EDGAR renderers emit between printed pages)
always starts a new pseudo-page, so for typical filings pseudo-pages track printed pages.

Dropped on purpose: ``<script>``, ``<style>``, ``<head>``, the hidden inline-XBRL ``<ix:header>``
block, and any element styled ``display:none`` (hidden XBRL context facts would otherwise
pollute retrieval with duplicated numbers).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from bs4.element import Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag

from secqa.core.contracts import Page
from secqa.core.logging import get_logger
from secqa.ingest.pdf import normalize_extracted_text

_log = get_logger(__name__)

_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "caption",
        "center",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "html",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "ul",
    }
)
_SKIP_TAGS = frozenset(
    {"head", "ix:header", "link", "meta", "noscript", "script", "style", "template", "title"}
)
_SKIPPED_STRING_TYPES = (Comment, Declaration, Doctype, ProcessingInstruction)
_PAGE_BREAK_BEFORE_RE = re.compile(r"(?:page-)?break-before\s*:\s*(?:always|page)", re.IGNORECASE)
_PAGE_BREAK_AFTER_RE = re.compile(r"(?:page-)?break-after\s*:\s*(?:always|page)", re.IGNORECASE)
_HIDDEN_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_CELL_SEP = " | "
# Cells that carry only a currency marker attach to the following cell ('$' + '1,577' -> '$1,577');
# cells that carry only a closing marker attach to the preceding one ('(12' + ')' -> '(12)').
_PREFIX_ONLY_CELLS = frozenset({"$", "€", "£", "¥", "US$", "USD", "(", "$("})
_SUFFIX_ONLY_CELLS = frozenset({")", "%", ")%", "%)", "pts", "bps"})

PAGE_BREAK = None  # sentinel in the block stream
Block = str | None


def _style(tag: Tag) -> str:
    style = tag.get("style")
    if isinstance(style, list):
        return " ".join(style)
    return style or ""


def _is_hidden(tag: Tag) -> bool:
    return bool(_HIDDEN_RE.search(_style(tag)))


def _has_page_break(tag: Tag, where: str) -> bool:
    style = _style(tag)
    if not style:
        return False
    pattern = _PAGE_BREAK_BEFORE_RE if where == "before" else _PAGE_BREAK_AFTER_RE
    return bool(pattern.search(style))


def _merge_cells(cells: list[str]) -> list[str]:
    """Glue marker-only cells ('$', ')', '%') onto their neighbours."""
    merged: list[str] = []
    pending_prefix = ""
    for cell in cells:
        if cell in _PREFIX_ONLY_CELLS:
            pending_prefix += cell
            continue
        if cell in _SUFFIX_ONLY_CELLS and merged:
            merged[-1] = merged[-1] + cell
            continue
        merged.append(pending_prefix + cell)
        pending_prefix = ""
    if pending_prefix:
        merged.append(pending_prefix)
    return merged


def flatten_table(table: Tag) -> list[str]:
    """Flatten a ``<table>`` into one string per non-empty row, cells joined by ``' | '``."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        if _is_hidden(tr):
            continue
        cells: list[str] = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            if _is_hidden(cell):
                continue
            text = normalize_extracted_text(cell.get_text(" "))
            if text:
                cells.append(text)
        if cells:
            rows.append(_CELL_SEP.join(_merge_cells(cells)))
    return rows


def _flush(buffer: list[str], out: list[Block]) -> None:
    if buffer:
        text = normalize_extracted_text(" ".join(buffer))
        buffer.clear()
        if text:
            out.append(text)


def _walk(node: Tag, out: list[Block], buffer: list[str]) -> None:
    """Depth-first walk emitting text blocks and page-break sentinels into ``out``."""
    for child in node.children:
        if isinstance(child, NavigableString):
            if isinstance(child, _SKIPPED_STRING_TYPES):
                continue
            buffer.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = (child.name or "").lower()
        if name in _SKIP_TAGS or _is_hidden(child):
            continue
        if _has_page_break(child, "before"):
            _flush(buffer, out)
            out.append(PAGE_BREAK)
        if name == "table":
            _flush(buffer, out)
            out.extend(flatten_table(child))
        elif name == "br":
            buffer.append(" ")
        elif name in _BLOCK_TAGS:
            _flush(buffer, out)
            _walk(child, out, buffer)
            _flush(buffer, out)
        else:  # inline element: keep accumulating into the current block
            _walk(child, out, buffer)
        if _has_page_break(child, "after"):
            _flush(buffer, out)
            out.append(PAGE_BREAK)


def html_to_blocks(html: bytes | str) -> list[Block]:
    """Parse HTML into an ordered stream of text blocks and ``None`` page-break sentinels."""
    with warnings.catch_warnings():
        # Inline-XBRL filings start with an XML prolog; we want lenient HTML parsing regardless.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    out: list[Block] = []
    buffer: list[str] = []
    _walk(soup, out, buffer)
    _flush(buffer, out)
    return out


def _paginate(blocks: Iterable[Block], target_chars: int) -> list[str]:
    """Pack blocks into pseudo-pages of about ``target_chars``; never split a block."""
    pages: list[str] = []
    current: list[str] = []
    length = 0

    def close() -> None:
        nonlocal current, length
        if current:
            pages.append("\n".join(current))
        current, length = [], 0

    for block in blocks:
        if block is PAGE_BREAK:
            close()
            continue
        assert block is not None
        if current and length + 1 + len(block) > target_chars:
            close()
        current.append(block)
        length += len(block) + (1 if length else 0)
    close()
    return pages


def extract_html_pages(html: bytes, doc_name: str, target_chars: int = 3500) -> list[Page]:
    """Convert an EDGAR HTML filing into 1-based pseudo-:class:`Page` records.

    ``target_chars`` bounds a pseudo-page except when a single block (one paragraph or one table
    row) is longer than that; such a block becomes a page of its own.
    """
    if target_chars < 1:
        raise ValueError(f"target_chars must be positive, got {target_chars}")
    blocks = html_to_blocks(html)
    texts = _paginate(blocks, target_chars)
    pages = [
        Page(doc_name=doc_name, page_num=index + 1, text=text) for index, text in enumerate(texts)
    ]
    _log.info(
        "html_extracted",
        doc_name=doc_name,
        n_blocks=sum(1 for b in blocks if b is not PAGE_BREAK),
        n_page_breaks=sum(1 for b in blocks if b is PAGE_BREAK),
        n_pages=len(pages),
        chars=sum(len(p.text) for p in pages),
    )
    return pages


__all__ = ["PAGE_BREAK", "extract_html_pages", "flatten_table", "html_to_blocks"]
