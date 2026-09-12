"""secqa.rag: single-shot grounded answering with verified citations.

Public surface: :class:`RagPipeline` (retrieve -> one LLM call -> verify), the
:func:`answer_closed_book` baseline, the :func:`answer_with_oracle_context` upper bound,
:data:`ANSWER_SCHEMA` (the structured output every mode requests) and :func:`prompt_hashes`
(SHA-256 of every ``prompts/*.md``, recorded on every answer and evaluation record).
"""

from secqa.rag.pipeline import (
    RagPipeline,
    answer_closed_book,
    answer_with_oracle_context,
    page_to_chunk,
)
from secqa.rag.prompts import (
    PROMPT_NAMES,
    PROMPTS_DIR,
    build_closed_book_prompt,
    build_user_prompt,
    load_prompt,
    prompt_hashes,
)
from secqa.rag.schemas import (
    ABSTAIN_TEXT,
    ANSWER_SCHEMA,
    ParsedAnswer,
    StructuredAnswer,
    parse_structured_answer,
)

__all__ = [
    "ABSTAIN_TEXT",
    "ANSWER_SCHEMA",
    "PROMPTS_DIR",
    "PROMPT_NAMES",
    "ParsedAnswer",
    "RagPipeline",
    "StructuredAnswer",
    "answer_closed_book",
    "answer_with_oracle_context",
    "build_closed_book_prompt",
    "build_user_prompt",
    "load_prompt",
    "page_to_chunk",
    "parse_structured_answer",
    "prompt_hashes",
]
