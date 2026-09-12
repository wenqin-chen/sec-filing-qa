"""Page-bounded, overlapping token chunks with section headings and stable ids.

Design (defended in interviews):

* A chunk never crosses a page, so every chunk has exactly one ``(doc_name, page_num)`` and a
  citation to a chunk is a citation to a page (SPEC 6). Pages shorter than ``max_tokens`` become
  a single chunk.
* Windows are cut at token positions but snapped to whitespace so chunks start and end on word
  boundaries; consecutive windows share ``overlap_tokens`` so any span no longer than the
  overlap (a sentence, a table row) is intact in at least one of them.
* Token counts come from ``tiktoken`` (``cl100k_base`` by default). Loading an encoding needs its
  BPE file, which tiktoken fetches from the network on first use; when that is impossible (CI
  without network and without a cache) we fall back to :class:`ApproxTokenizer` (about one token
  per four non-space characters) and log a warning. Chunk *bounds* are then approximate, but the
  pipeline still runs offline as SPEC requires. Pass ``encoding="approx"`` to select it directly.
* ``section`` is a heuristic hint from ``Item N.`` headings, carried forward across chunks and
  pages until the next heading. It is metadata for filtering and display, never evidence.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

from secqa.core.contracts import Chunk, Page
from secqa.core.ids import chunk_id
from secqa.core.logging import get_logger

_log = get_logger(__name__)

APPROX_ENCODING = "approx"
Token = int | str


class Tokenizer(Protocol):
    """Minimal reversible tokenizer used by the chunker."""

    name: str

    def encode(self, text: str) -> list[Token]: ...

    def decode(self, tokens: list[Token]) -> str: ...

    def starts_with_space(self, token: Token) -> bool: ...


class TiktokenTokenizer:
    """``tiktoken`` BPE encoding (exact counts for OpenAI models, close enough for others)."""

    def __init__(self, encoding: str) -> None:
        import tiktoken

        self.name = encoding
        self._enc = tiktoken.get_encoding(encoding)
        self._space_cache: dict[int, bool] = {}

    def encode(self, text: str) -> list[Token]:
        # disallowed_special=() so a filing containing '<|endoftext|>' cannot raise.
        return list(self._enc.encode(text, disallowed_special=()))

    def decode(self, tokens: list[Token]) -> str:
        return self._enc.decode([int(t) for t in tokens])

    def starts_with_space(self, token: Token) -> bool:
        key = int(token)
        cached = self._space_cache.get(key)
        if cached is None:
            piece = self._enc.decode([key])
            cached = bool(piece) and piece[0].isspace()
            self._space_cache[key] = cached
        return cached


class ApproxTokenizer:
    """Offline stand-in: whitespace-prefixed pieces of at most four non-space characters."""

    name = APPROX_ENCODING
    _PIECE_RE = re.compile(r"\s*\S{1,4}|\s+$")

    def encode(self, text: str) -> list[Token]:
        return [m.group(0) for m in self._PIECE_RE.finditer(text)]

    def decode(self, tokens: list[Token]) -> str:
        return "".join(str(t) for t in tokens)

    def starts_with_space(self, token: Token) -> bool:
        piece = str(token)
        return bool(piece) and piece[0].isspace()


@lru_cache(maxsize=8)
def get_tokenizer(encoding: str = "cl100k_base") -> Tokenizer:
    """Return the tokenizer for ``encoding`` (cached), falling back to :class:`ApproxTokenizer`.

    Unknown encoding names raise ``ValueError`` (a configuration error); failures to *load* a
    known encoding (no network, no cache) fall back with a warning so offline runs still work.
    """
    if encoding == APPROX_ENCODING:
        return ApproxTokenizer()
    import tiktoken

    if encoding not in tiktoken.list_encoding_names():
        raise ValueError(
            f"unknown tiktoken encoding {encoding!r}; "
            f"choose one of {sorted(tiktoken.list_encoding_names())} or {APPROX_ENCODING!r}"
        )
    try:
        return TiktokenTokenizer(encoding)
    except Exception as exc:  # network / cache failure loading the BPE file
        _log.warning(
            "tokenizer_fallback",
            encoding=encoding,
            fallback=APPROX_ENCODING,
            error=str(exc),
        )
        return ApproxTokenizer()


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Token count of ``text`` under ``encoding`` (approximate under the offline fallback)."""
    return len(get_tokenizer(encoding).encode(text))


# ---------------------------------------------------------------------------------------------
# section headings
# ---------------------------------------------------------------------------------------------

# 'Item 7.', 'ITEM 1A:', 'Item 9A. Controls', 'Item 7 Management' — but not 'Items 1 and 2',
# 'Item 7.5' (sub-numbering) or 'item 8' glued to a longer identifier.
# 'Item' is matched case-insensitively by hand: a global IGNORECASE flag would also let the
# capital-letter lookahead accept 'item 8 of this'.
_ITEM_RE = re.compile(
    r"\b[Ii][Tt][Ee][Mm]\s+(?P<num>\d{1,2})\s?(?P<letter>[A-Ca-c])?(?!\w)(?!\.\d)"
    r"(?=\s*[.:\-–—]|\s+[A-Z])"
)
# Words that precede a cross-reference rather than a heading: 'see Item 1A.', 'in Item 7.'
_XREF_WORDS = frozenset(
    {
        "in",
        "see",
        "to",
        "of",
        "under",
        "within",
        "and",
        "our",
        "this",
        "per",
        "refer",
        "also",
        "with",
        "from",
        "at",
        "by",
        "or",
    }
)
_PRECEDING_WORD_RE = re.compile(r"([A-Za-z]+)\W*$")


def _is_cross_reference(text: str, start: int) -> bool:
    prefix = text[max(0, start - 16) : start]
    match = _PRECEDING_WORD_RE.search(prefix)
    return match is not None and match.group(1).lower() in _XREF_WORDS


def detect_section(text: str, previous: str | None) -> str | None:
    """Return the 10-K/10-Q section in force at the end of ``text``.

    Scans for ``Item N[A].``-style headings (``'Item 7'``, ``'Item 1A'``), ignoring
    cross-references such as ``'see Item 1A.'``. The *last* heading wins because a chunk that
    contains a heading is mostly about what follows it. Returns ``previous`` when no heading is
    found so callers can carry the section forward across chunks and pages.
    """
    if not text:
        return previous
    section = previous
    for match in _ITEM_RE.finditer(text):
        if _is_cross_reference(text, match.start()):
            continue
        letter = (match.group("letter") or "").upper()
        section = f"Item {int(match.group('num'))}{letter}"
    return section


# ---------------------------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------------------------


def _snap_back(tok: Tokenizer, tokens: list[Token], start: int, end: int, lookback: int) -> int:
    """Move ``end`` back to the nearest token that starts with whitespace (cut *before* it)."""
    floor = max(start + 1, end - lookback)
    for j in range(end, floor, -1):
        if tok.starts_with_space(tokens[j]):
            return j
    return end


def _snap_forward(tok: Tokenizer, tokens: list[Token], start: int, end: int) -> int:
    """Move ``start`` forward to the nearest token that starts with whitespace."""
    for j in range(start, end):
        if tok.starts_with_space(tokens[j]):
            return j
    return start


def _windows(
    text: str, tok: Tokenizer, max_tokens: int, overlap_tokens: int
) -> list[tuple[str, int]]:
    """Split ``text`` into ``(piece, n_tokens)`` windows of at most ``max_tokens`` tokens."""
    tokens = tok.encode(text)
    n = len(tokens)
    if n <= max_tokens:
        return [(text, n)] if text else []

    lookback = max(1, min(64, max_tokens // 4))
    out: list[tuple[str, int]] = []
    start = 0
    while True:
        end = min(start + max_tokens, n)
        if end < n:
            end = _snap_back(tok, tokens, start, end, lookback)
        piece = tok.decode(tokens[start:end]).strip()
        count = len(tok.encode(piece))
        while count > max_tokens and end - start > 1:  # re-tokenisation drift; shrink to fit
            end -= 1
            piece = tok.decode(tokens[start:end]).strip()
            count = len(tok.encode(piece))
        if piece:
            out.append((piece, count))
        if end >= n:
            break
        next_start = max(end - overlap_tokens, start + 1)
        start = _snap_forward(tok, tokens, next_start, end)
    return out


def chunk_pages(
    pages: list[Page],
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    encoding: str = "cl100k_base",
) -> list[Chunk]:
    """Chunk pages into page-bounded, overlapping token windows with stable ids.

    Pages are processed in ``(doc_name, page_num)`` order so the section heading carries
    forward through a document; it resets when the document changes. Empty pages yield no
    chunks. ``chunk_idx`` is 0-based within each page and ``chunk_id`` is
    :func:`secqa.core.ids.chunk_id` of ``(doc_name, page_num, chunk_idx, text)``.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError(
            f"overlap_tokens must satisfy 0 <= overlap < max_tokens, got {overlap_tokens}"
        )
    tok = get_tokenizer(encoding)
    chunks: list[Chunk] = []
    section: str | None = None
    current_doc: str | None = None
    for page in sorted(pages, key=lambda p: (p.doc_name, p.page_num)):
        if page.doc_name != current_doc:
            current_doc, section = page.doc_name, None
        text = page.text.strip()
        if not text:
            continue
        for idx, (piece, n_tokens) in enumerate(_windows(text, tok, max_tokens, overlap_tokens)):
            section = detect_section(piece, section)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(page.doc_name, page.page_num, idx, piece),
                    doc_name=page.doc_name,
                    page_num=page.page_num,
                    chunk_idx=idx,
                    section=section,
                    text=piece,
                    n_tokens=n_tokens,
                )
            )
    _log.info(
        "pages_chunked",
        n_pages=len(pages),
        n_chunks=len(chunks),
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer=tok.name,
    )
    return chunks


__all__ = [
    "APPROX_ENCODING",
    "ApproxTokenizer",
    "TiktokenTokenizer",
    "Tokenizer",
    "chunk_pages",
    "count_tokens",
    "detect_section",
    "get_tokenizer",
]
