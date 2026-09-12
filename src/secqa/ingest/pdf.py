"""PDF text extraction with pypdfium2.

One :class:`~secqa.core.contracts.Page` per physical page, ``page_num = pdfium index + 1``. This is
the *only* place page numbers are assigned for PDFs, so the FinanceBench mapping
``page_num = evidence_page_num + 1`` (SPEC 4.1) holds by construction.

Text normalisation (:func:`normalize_extracted_text`) is deliberately conservative: NFKC (ligatures
such as ``ﬁ`` become ``fi``), line-break de-hyphenation (``Manage-\\nment`` -> ``Management``),
control characters dropped, whitespace collapsed. Nothing is re-ordered and no characters are
invented, so a model quote can still be verified as a substring of the page text.

Pages with no extractable text (scanned images, blank separators) are kept with ``text=""`` so
that page numbering stays physical; the chunker skips them.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from secqa.core.contracts import Page
from secqa.core.logging import get_logger

_log = get_logger(__name__)

_PDF_MAGIC = b"%PDF"

# Line breaks as pdfium emits them (\r\n on every platform) plus lone \r and form feeds.
_LINE_BREAK_RE = re.compile(r"\r\n|\r|\x0c")
# Control characters other than whitespace (pdfium occasionally leaks \x02 markers).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
# 'Manage-\nment' -> 'Management': a hyphen at end of line followed by a lowercase continuation.
_DEHYPHEN_LOWER_RE = re.compile(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])")
# 'Coca-\nCola' -> 'Coca-Cola': keep the hyphen when the continuation is capitalised or numeric.
_DEHYPHEN_KEEP_RE = re.compile(r"(?<=[A-Za-z0-9])-[ \t]*\n[ \t]*(?=[A-Z0-9])")
_WS_RE = re.compile(r"\s+")


def normalize_extracted_text(text: str, *, keep_newlines: bool = False) -> str:
    """Normalise raw extracted text: NFKC, de-hyphenate line breaks, collapse whitespace.

    With ``keep_newlines=True`` single newlines are preserved (used for HTML block boundaries);
    runs of blank lines still collapse to one newline and spaces/tabs collapse to one space.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _LINE_BREAK_RE.sub("\n", out)
    out = _CONTROL_RE.sub("", out)
    out = _DEHYPHEN_LOWER_RE.sub("", out)
    out = _DEHYPHEN_KEEP_RE.sub("-", out)
    if keep_newlines:
        lines = [_WS_RE.sub(" ", line).strip() for line in out.split("\n")]
        return "\n".join(line for line in lines if line)
    return _WS_RE.sub(" ", out).strip()


def is_pdf_bytes(head: bytes) -> bool:
    """True if ``head`` (the first bytes of a file) carries the ``%PDF`` magic marker."""
    return head.startswith(_PDF_MAGIC)


def extract_pdf_pages(path: Path, doc_name: str) -> list[Page]:
    """Extract every page of ``path`` as a 1-based :class:`Page` record.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError`` if it is not a PDF
    (magic-byte check, so an HTML error page saved as ``.pdf`` fails loudly instead of yielding
    zero pages).
    """
    import pypdfium2 as pdfium  # imported here so the package imports without the native lib

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    with path.open("rb") as fh:
        head = fh.read(len(_PDF_MAGIC))
    if not is_pdf_bytes(head):
        raise ValueError(f"{path} does not start with %PDF (got {head!r}); refusing to parse")

    pages: list[Page] = []
    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception as exc:  # pdfium raises its own error type; surface a plain ValueError
        raise ValueError(f"pypdfium2 could not open {path}: {exc}") from exc
    try:
        n_pages = len(pdf)
        empty = 0
        for index in range(n_pages):
            page = pdf[index]
            try:
                textpage = page.get_textpage()
                try:
                    raw = textpage.get_text_bounded()
                finally:
                    textpage.close()
            finally:
                page.close()
            text = normalize_extracted_text(raw)
            if not text:
                empty += 1
            pages.append(Page(doc_name=doc_name, page_num=index + 1, text=text))
    finally:
        pdf.close()

    _log.info(
        "pdf_extracted",
        doc_name=doc_name,
        path=str(path),
        n_pages=len(pages),
        empty_pages=empty,
        chars=sum(len(p.text) for p in pages),
    )
    return pages


__all__ = ["extract_pdf_pages", "is_pdf_bytes", "normalize_extracted_text"]
