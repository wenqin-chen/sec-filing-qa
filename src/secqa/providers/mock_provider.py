"""Deterministic offline provider used by CI, the smoke evaluation and the public demo.

Behaviour ``extractive`` (default):

* With tools offered and no ``search_filings`` call yet in the conversation, request
  ``search_filings(query=<question>)``.
* Otherwise take the passages the conversation already contains (tool results as JSON hits, or
  numbered passages carrying ``chunk:<id>`` refs in the prompt), pick the first sentence of at
  least 20 characters from the first usable passage, and answer with exactly that sentence, citing
  its ref and quoting it verbatim - so :class:`CitationVerifier` marks the citation verified.
* When ``json_schema`` is requested the same answer is rendered as JSON that follows the schema
  (known answer fields filled, every other property given a type-appropriate default).
* No usable passage -> abstain (``INSUFFICIENT EVIDENCE``).

Behaviour ``abstain`` abstains immediately without calling any tool; the eval harness uses it to
record retrieval-only metrics with no answering model.

Everything is a pure function of the inputs: same messages -> byte-identical response.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from secqa.core.contracts import LLMResponse, Message, ToolCall, ToolSpec
from secqa.core.textnum import extract_numbers
from secqa.providers.base import (
    BaseProvider,
    Effort,
    estimate_usage,
    first_user_text,
)

MockBehaviour = Literal["extractive", "abstain"]

ABSTAIN_TEXT = "INSUFFICIENT EVIDENCE"
MIN_QUOTE_CHARS = 20
SEARCH_TOOL = "search_filings"
FINAL_TOOL = "final_answer"

# A ref as written in prompts and tool results: 'chunk:<hex id>' (sha1 = 40 hex chars, but any
# reasonably long hex token is accepted so fixture ids can be shorter).
_REF_RE = re.compile(r"chunk:([0-9a-fA-F]{6,64})")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_QUESTION_LINE_RE = re.compile(r"^\s*question\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_PASSAGE_TAIL_RE = re.compile(r"\n\s*(?:question|answer|hints?)\s*:", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
_HEADER_MAX_CHARS = 80


class MockProvider(BaseProvider):
    """Deterministic extractive (or abstaining) provider; see the module docstring."""

    provider = "mock"

    def __init__(self, behaviour: MockBehaviour = "extractive"):
        if behaviour not in ("extractive", "abstain"):
            raise ValueError(f"unknown mock behaviour {behaviour!r}")
        self.behaviour: MockBehaviour = behaviour
        self.model = f"mock-{behaviour}"

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        effort: Effort | None = None,
    ) -> LLMResponse:
        """Answer deterministically from the passages present in ``messages``."""
        tool_names = {t.name for t in tools} if tools else set()
        answer = self._decide(messages, tool_names)

        tool_calls: list[ToolCall] = []
        parsed: dict[str, Any] | None = None
        if answer["kind"] == "search":
            tool_calls = [
                ToolCall(id="mock-call-search", name=SEARCH_TOOL, arguments=answer["arguments"])
            ]
            text = ""
        elif FINAL_TOOL in tool_names:
            tool_calls = [
                ToolCall(id="mock-call-final", name=FINAL_TOOL, arguments=answer["fields"])
            ]
            text = ""
        elif json_schema is not None:
            parsed = fill_schema(json_schema, answer["fields"])
            text = json.dumps(parsed, sort_keys=True)
        else:
            text = answer["fields"]["answer"]

        output_for_usage = text or json.dumps([c.arguments for c in tool_calls], sort_keys=True)
        response = LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=estimate_usage(
                messages, system=system, tools=tools, output_text=output_for_usage
            ),
            provider=self.provider,
            model=self.model,
            latency_ms=0.0,
            stop_reason="tool_use" if tool_calls else "end_turn",
            parsed=parsed,
            raw_id=None,
        )
        self._log_response(response)
        return response

    # ---- decision logic ----

    def _decide(self, messages: list[Message], tool_names: set[str]) -> dict[str, Any]:
        if self.behaviour == "abstain":
            return {"kind": "final", "fields": _abstain_fields()}
        if SEARCH_TOOL in tool_names and not _tool_was_called(messages, SEARCH_TOOL):
            return {
                "kind": "search",
                "arguments": {"query": extract_question(first_user_text(messages))},
            }
        passages = extract_passages(messages)
        for ref, passage in passages:
            sentence = first_sentence(passage)
            if sentence is not None:
                numbers = extract_numbers(sentence)
                return {
                    "kind": "final",
                    "fields": {
                        "answer": sentence,
                        "value": numbers[0] if numbers else None,
                        "unit": None,
                        "citations": [{"ref": ref, "quote": sentence}],
                        "calculation": None,
                        "abstain": False,
                    },
                }
        return {"kind": "final", "fields": _abstain_fields()}


def _abstain_fields() -> dict[str, Any]:
    return {
        "answer": ABSTAIN_TEXT,
        "value": None,
        "unit": None,
        "citations": [],
        "calculation": None,
        "abstain": True,
    }


def _tool_was_called(messages: list[Message], name: str) -> bool:
    return any(call.name == name for message in messages for call in message.tool_calls) or any(
        result.name == name for message in messages for result in message.tool_results
    )


def extract_question(user_text: str) -> str:
    """The question inside a prompt: the ``Question:`` line when present, else the whole text."""
    match = _QUESTION_LINE_RE.search(user_text)
    question = match.group(1) if match else user_text
    return " ".join(question.split())


def first_sentence(passage: str, min_chars: int = MIN_QUOTE_CHARS) -> str | None:
    """First sentence of ``passage`` with at least ``min_chars`` characters, or ``None``."""
    flat = " ".join(passage.split())
    if not flat:
        return None
    for sentence in _SENTENCE_SPLIT_RE.split(flat):
        candidate = sentence.strip()
        if len(candidate) >= min_chars:
            return candidate
    return flat if len(flat) >= min_chars else None


def extract_passages(messages: list[Message]) -> list[tuple[str, str]]:
    """Return ``(ref, text)`` passages found in the conversation, in order of appearance.

    Tool results are parsed as JSON first (any list of objects with ``chunk_id``/``ref`` plus
    ``snippet``/``text``); everything else falls back to scanning for ``chunk:<id>`` markers and
    taking the text that follows each marker up to the next one.
    """
    passages: list[tuple[str, str]] = []
    for message in messages:
        if message.role == "tool":
            for result in message.tool_results:
                if result.is_error:
                    continue
                json_hits = _passages_from_json(result.content)
                passages.extend(json_hits if json_hits else _passages_from_text(result.content))
        elif message.role == "user":
            passages.extend(_passages_from_text(message.content))
    return passages


def _passages_from_json(content: str) -> list[tuple[str, str]]:
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return []
    items = _find_hit_list(data)
    out: list[tuple[str, str]] = []
    for item in items:
        ref = item.get("ref")
        if not ref and item.get("chunk_id"):
            ref = f"chunk:{item['chunk_id']}"
        if not isinstance(ref, str):
            continue
        text = item.get("snippet") or item.get("text") or item.get("content") or ""
        if isinstance(text, str) and text.strip():
            out.append((ref, text))
    return out


def _find_hit_list(data: Any) -> list[dict[str, Any]]:
    """Locate the first list of hit-like dicts at the top level or one key down."""
    candidates: list[Any] = [data]
    if isinstance(data, dict):
        candidates.extend(data.values())
    for candidate in candidates:
        if (
            isinstance(candidate, list)
            and candidate
            and all(isinstance(x, dict) and ("chunk_id" in x or "ref" in x) for x in candidate)
        ):
            return candidate
    return []


def _passages_from_text(text: str) -> list[tuple[str, str]]:
    matches = list(_REF_RE.finditer(text))
    out: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        tail = _PASSAGE_TAIL_RE.search(body)
        if tail:
            body = body[: tail.start()]
        # Drop the rest of the header line the ref sat on (e.g. ') FIXTURE_2023_10K p.1') when it
        # is short and not a sentence; passage text normally starts on the next line.
        body = body.lstrip(" \t)]:;,-|")
        if "\n" in body:
            head, rest = body.split("\n", 1)
            if _is_header_remainder(head) and rest.strip():
                body = rest
        if body.strip():
            out.append((f"chunk:{match.group(1)}", body))
    return out


def _is_header_remainder(line: str) -> bool:
    """True for a short ref-line remainder with no sentence terminator ('FIXTURE_2023_10K p.1')."""
    stripped = line.strip()
    return len(stripped) < _HEADER_MAX_CHARS and _SENTENCE_END_RE.search(stripped) is None


# ---- json_schema filling ----


def fill_schema(schema: dict[str, Any], values: dict[str, Any]) -> Any:
    """Produce a value that follows ``schema``, using ``values`` for known property names.

    Unknown properties get a type-appropriate default (``""``, ``0``, ``False``, ``[]``, ``{}``,
    ``None`` when nullable, the first ``enum`` member). Enough for every schema in secqa (answer,
    judge, faithfulness) without special-casing any of them.
    """
    kind = _schema_type(schema)
    if kind == "object":
        properties = schema.get("properties") or {}
        return {
            name: (values[name] if name in values else fill_schema(sub, values))
            for name, sub in properties.items()
        }
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if kind == "null":
        return None
    if kind == "array":
        return []
    if kind == "boolean":
        return False
    if kind in ("number", "integer"):
        return 0
    if kind == "string":
        return ""
    return None


def _schema_type(schema: dict[str, Any]) -> str | None:
    declared = schema.get("type")
    if isinstance(declared, list):
        if "null" in declared:
            return "null"
        return str(declared[0]) if declared else None
    if isinstance(declared, str):
        return declared
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            if any(_schema_type(o) == "null" for o in options if isinstance(o, dict)):
                return "null"
            return _schema_type(options[0]) if isinstance(options[0], dict) else None
    if "properties" in schema:
        return "object"
    return None


__all__ = [
    "ABSTAIN_TEXT",
    "MIN_QUOTE_CHARS",
    "MockBehaviour",
    "MockProvider",
    "extract_passages",
    "extract_question",
    "fill_schema",
    "first_sentence",
]
