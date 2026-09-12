"""The structured-answer JSON schema and a tolerant parser for what the model returns.

``ANSWER_SCHEMA`` is the single JSON schema every single-shot mode (rag, closed_book, oracle)
requests from the provider (``json_schema=``). It is deliberately flat so that both vendors'
strict structured-output modes accept it: every property is listed in ``required``,
``additionalProperties`` is false, and nullable fields use ``["number", "null"]`` type lists.

:func:`parse_structured_answer` turns an :class:`LLMResponse` into a :class:`StructuredAnswer`.
It prefers ``response.parsed`` (the provider already decoded the JSON), then falls back to decoding
``response.text`` (with or without a Markdown code fence), and finally to treating the raw text as
an uncited free-text answer. The fallback never hides anything: the returned ``parse_error`` says
why the structured path failed, and the caller records it in the trace.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import Field, ValidationError, field_validator

from secqa.core.contracts import CitationRef, Frozen, LLMResponse

ABSTAIN_TEXT = "INSUFFICIENT EVIDENCE"
"""The canonical abstention answer (identical to the mock provider's, on purpose)."""

ANSWER_SCHEMA: dict[str, Any] = {
    "title": "answer",
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "value", "unit", "citations", "abstain"],
    "properties": {
        "answer": {
            "type": "string",
            "description": "Concise answer, or exactly 'INSUFFICIENT EVIDENCE' when abstaining.",
        },
        "value": {
            "type": ["number", "null"],
            "description": "The single answer number in base units (dollars, fraction), or null.",
        },
        "unit": {
            "type": ["string", "null"],
            "description": "Unit of value ('USD', 'shares', 'ratio', 'percent', ...), or null.",
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ref", "quote"],
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Exact ref of a supplied passage: 'chunk:<id>'.",
                    },
                    "quote": {
                        "type": "string",
                        "description": "Verbatim span (>= 20 chars) copied from that passage.",
                    },
                },
            },
        },
        "abstain": {"type": "boolean"},
    },
}

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_ABSTAIN_RE = re.compile(r"^\W*insufficient\s+evidence\b", re.IGNORECASE)


class StructuredAnswer(Frozen):
    """A validated model answer (the Python side of :data:`ANSWER_SCHEMA`)."""

    answer: str = ""
    value: float | None = None
    unit: str | None = None
    citations: list[CitationRef] = Field(default_factory=list)
    abstain: bool = False

    @field_validator("unit", mode="before")
    @classmethod
    def _blank_unit_is_none(cls, value: Any) -> Any:
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


class ParsedAnswer(Frozen):
    """Result of :func:`parse_structured_answer`: the answer plus how it was obtained."""

    answer: StructuredAnswer
    source: str  # 'parsed' | 'text_json' | 'text_fallback' | 'refusal' | 'empty'
    parse_error: str | None = None


def parse_structured_answer(response: LLMResponse) -> ParsedAnswer:
    """Decode a provider response into a :class:`StructuredAnswer`; never raises.

    Order of preference: ``response.parsed`` -> JSON in ``response.text`` (code fences stripped)
    -> raw text as an uncited answer. A refusal (``stop_reason='refusal'``) or an empty response
    becomes an abstention so the harness scores it as such rather than as a wrong answer.
    """
    if response.stop_reason == "refusal":
        return ParsedAnswer(
            answer=StructuredAnswer(answer=ABSTAIN_TEXT, abstain=True),
            source="refusal",
            parse_error="provider reported a refusal",
        )

    parsed_error: str | None = None
    if response.parsed is not None:
        try:
            return ParsedAnswer(answer=_validate(response.parsed), source="parsed")
        except ValidationError as exc:
            parsed_error = f"parsed output failed validation: {_short(exc)}"

    text = response.text.strip()
    if not text:
        return ParsedAnswer(
            answer=StructuredAnswer(answer=ABSTAIN_TEXT, abstain=True),
            source="empty",
            parse_error=parsed_error or "provider returned no text",
        )

    decoded, decode_error = _decode_json(text)
    if decoded is not None:
        try:
            return ParsedAnswer(
                answer=_validate(decoded), source="text_json", parse_error=parsed_error
            )
        except ValidationError as exc:
            decode_error = f"text JSON failed validation: {_short(exc)}"

    fallback = StructuredAnswer(answer=text, abstain=bool(_ABSTAIN_RE.match(text)))
    reason = parsed_error or decode_error or "text is not a JSON object"
    return ParsedAnswer(answer=fallback, source="text_fallback", parse_error=reason)


def _validate(data: dict[str, Any]) -> StructuredAnswer:
    """Validate a decoded mapping, tolerating extra keys and null citations."""
    known = {key: data.get(key) for key in ANSWER_SCHEMA["properties"] if key in data}
    if known.get("citations") is None:
        known["citations"] = []
    if known.get("answer") is None:
        known["answer"] = ""
    return StructuredAnswer.model_validate(known)


def _decode_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Decode ``text`` as a JSON object, stripping a surrounding Markdown code fence."""
    candidate = text
    fenced = _CODE_FENCE_RE.match(text)
    if fenced:
        candidate = fenced.group(1)
    try:
        data = json.loads(candidate)
    except ValueError as exc:
        return None, f"text is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"text decoded to {type(data).__name__}, expected an object"
    return data, None


def _short(exc: ValidationError) -> str:
    """One-line summary of a pydantic validation error for the trace."""
    details = exc.errors()
    if not details:
        return "invalid"
    first = details[0]
    location = ".".join(str(part) for part in first["loc"]) or "<root>"
    return f"{location}: {first['msg']}"


__all__ = [
    "ABSTAIN_TEXT",
    "ANSWER_SCHEMA",
    "ParsedAnswer",
    "StructuredAnswer",
    "parse_structured_answer",
]
