"""Prompt files, their hashes, and the single-shot user-prompt layout.

System prompts live as Markdown files in ``src/secqa/prompts/`` so they can be diffed, reviewed
and hashed. :func:`prompt_hashes` returns the SHA-256 of every ``*.md`` file there; the hashes are
recorded in every :class:`~secqa.core.contracts.Answer` and evaluation record so that a result can
never be silently attributed to a different prompt (SPEC section 7: prompts are frozen before the
first real run, and a revision is a new hash).

The user prompt built by :func:`build_user_prompt` has one fixed layout::

    Passages (cite by ref; quote verbatim):

    [1] (ref: chunk:<id>) <doc_name> p.<page> | <section>
    <passage text>

    [2] (ref: chunk:<id>) ...

    Hints: ticker=ACME fiscal_year=2023
    Question: <question on one line>

The header line of each passage carries no sentence terminator and the passage text starts on
the next line: that is the format the deterministic :class:`~secqa.providers.MockProvider`
parses, so CI exercises exactly the prompt real models see.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

from secqa.core.contracts import Chunk, RetrievalFilters
from secqa.core.errors import ConfigError
from secqa.core.ids import sha256_hex

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
"""Directory holding the ``*.md`` system prompts."""

RAG_SYSTEM = "rag_system.md"
CLOSED_BOOK_SYSTEM = "closed_book_system.md"
ORACLE_SYSTEM = "oracle_system.md"
AGENT_SYSTEM = "agent_system.md"
"""The agent loop's system prompt; owned by :mod:`secqa.agent`, hashed here with the rest."""
PROMPT_NAMES: tuple[str, ...] = (RAG_SYSTEM, CLOSED_BOOK_SYSTEM, ORACLE_SYSTEM, AGENT_SYSTEM)
"""Every prompt file under :data:`PROMPTS_DIR`; :func:`prompt_hashes` must cover all of them."""

PASSAGES_HEADER = "Passages (cite by ref; quote verbatim):"
SECTION_MAX_CHARS = 40
_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[\s.!?:;,|-]+$")
# A sentence terminator followed by whitespace would make the header line look like prose to
# the extractive mock's header detection ("Item 7. Management's ..." -> "Item 7 Management's ...").
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)")


@cache
def load_prompt(name: str) -> str:
    """Return the text of ``prompts/<name>`` (cached for the life of the process).

    Raises:
        ConfigError: if the file does not exist or is empty.
    """
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise ConfigError(f"prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ConfigError(f"prompt file is empty: {path}")
    return text


def prompt_hash(name: str) -> str:
    """SHA-256 of one prompt file's bytes (what :func:`prompt_hashes` records)."""
    return sha256_hex(load_prompt(name).encode("utf-8"))


def prompt_hashes() -> dict[str, str]:
    """``{file name: sha256}`` for every ``*.md`` under :data:`PROMPTS_DIR`, sorted by name.

    Reads the files on every call (four small files) so a test that edits a prompt on disk sees
    the new hash without clearing caches.
    """
    files = sorted(PROMPTS_DIR.glob("*.md"))
    if not files:
        raise ConfigError(f"no prompt files found under {PROMPTS_DIR}")
    return {path.name: sha256_hex(path.read_bytes()) for path in files}


def format_passage(index: int, chunk: Chunk, max_chars: int | None = None) -> str:
    """Render one numbered passage: a one-line header, then the chunk text.

    ``max_chars`` truncates the *displayed* text (oracle pages can be long); the verifier still
    checks quotes against the full chunk text, and a quote from the visible part is a substring
    of the whole, so truncation can only lose evidence, never invent it.
    """
    if index < 1:
        raise ValueError(f"passage index is 1-based, got {index}")
    header = f"[{index}] (ref: chunk:{chunk.chunk_id}) {chunk.doc_name} p.{chunk.page_num}"
    section = _clean_section(chunk.section)
    if section:
        header += f" | {section}"
    body = chunk.text.strip()
    if max_chars is not None and max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars].rstrip() + " [truncated]"
    return f"{header}\n{body}"


def build_user_prompt(
    question: str,
    chunks: list[Chunk],
    *,
    filters: RetrievalFilters | None = None,
    max_passage_chars: int | None = None,
) -> str:
    """Build the single-shot user prompt (see the module docstring for the layout)."""
    if not question.strip():
        raise ValueError("question must not be blank")
    parts: list[str] = []
    if chunks:
        parts.append(PASSAGES_HEADER)
        parts.append("")
        for index, chunk in enumerate(chunks, start=1):
            parts.append(format_passage(index, chunk, max_passage_chars))
            parts.append("")
    hints = format_hints(filters)
    if hints:
        parts.append(f"Hints: {hints}")
    parts.append(f"Question: {_one_line(question)}")
    return "\n".join(parts)


def build_closed_book_prompt(question: str) -> str:
    """The closed-book user prompt: just the question, in the same ``Question:`` layout."""
    if not question.strip():
        raise ValueError("question must not be blank")
    return f"Question: {_one_line(question)}"


def format_hints(filters: RetrievalFilters | None) -> str:
    """``'ticker=ACME fiscal_year=2023'`` from the non-empty filter fields ('' if none)."""
    if filters is None:
        return ""
    fields: list[str] = []
    if filters.ticker:
        fields.append(f"ticker={filters.ticker}")
    if filters.fiscal_year is not None:
        fields.append(f"fiscal_year={filters.fiscal_year}")
    if filters.form:
        fields.append(f"form={filters.form}")
    if filters.doc_names:
        fields.append(f"documents={','.join(filters.doc_names)}")
    return " ".join(fields)


def _clean_section(section: str | None) -> str:
    """Collapse whitespace, cap the label, and drop sentence terminators and trailing punctuation.

    The header line must never read as a sentence (see the module docstring), so a label such as
    ``"Item 7. Management's Discussion"`` becomes ``"Item 7 Management's Discussion"``.
    """
    if not section:
        return ""
    flat = _WS_RE.sub(" ", section).strip()
    if len(flat) > SECTION_MAX_CHARS:
        flat = flat[:SECTION_MAX_CHARS].rstrip()
    flat = _SENTENCE_END_RE.sub("", flat)
    return _TRAILING_PUNCT_RE.sub("", flat)


def _one_line(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


__all__ = [
    "AGENT_SYSTEM",
    "CLOSED_BOOK_SYSTEM",
    "ORACLE_SYSTEM",
    "PASSAGES_HEADER",
    "PROMPTS_DIR",
    "PROMPT_NAMES",
    "RAG_SYSTEM",
    "build_closed_book_prompt",
    "build_user_prompt",
    "format_hints",
    "format_passage",
    "load_prompt",
    "prompt_hash",
    "prompt_hashes",
]
