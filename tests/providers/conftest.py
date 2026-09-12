"""Shared fixtures for provider tests (all offline: no SDK network calls, no keys)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from secqa.core.contracts import Message, ToolResult, ToolSpec

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
RESPONSES = FIXTURES / "providers_responses"
SCENARIOS = FIXTURES / "providers_scenarios"

CHUNK_A = "0123456789abcdef0123456789abcdef01234567"
CHUNK_B = "89abcdef0123456789abcdef0123456789abcdef"
PASSAGE_A = (
    "Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022. "
    "Growth was broad-based across segments."
)
PASSAGE_B = "Operating income was $245 million and net income was $190 million."

ANSWER_SCHEMA: dict[str, Any] = {
    "title": "answer",
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "value", "unit", "citations", "abstain"],
    "properties": {
        "answer": {"type": "string"},
        "value": {"type": ["number", "null"]},
        "unit": {"type": ["string", "null"]},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ref", "quote"],
                "properties": {"ref": {"type": "string"}, "quote": {"type": "string"}},
            },
        },
        "abstain": {"type": "boolean"},
    },
}


def load_response(name: str) -> dict[str, Any]:
    """Load a recorded-and-scrubbed vendor response fixture by file name."""
    return json.loads((RESPONSES / name).read_text(encoding="utf-8"))


@pytest.fixture
def rag_prompt() -> list[Message]:
    """A single-shot RAG prompt: numbered passages with ``chunk:<id>`` refs, then the question."""
    content = (
        "Answer the question from the passages below.\n\n"
        f"[1] (ref: chunk:{CHUNK_A}) FIXTURE_2023_10K p.1\n{PASSAGE_A}\n\n"
        f"[2] (ref: chunk:{CHUNK_B}) FIXTURE_2023_10K p.2\n{PASSAGE_B}\n\n"
        "Question: What were FIXTURE CORP's total net sales in fiscal 2023?"
    )
    return [Message(role="user", content=content)]


@pytest.fixture
def agent_tools() -> list[ToolSpec]:
    """A minimal subset of the agent tool set (schemas per SPEC section 5)."""
    return [
        ToolSpec(
            name="search_filings",
            description="Hybrid search over indexed filings.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        ),
        ToolSpec(
            name="final_answer",
            description="Finish with a cited answer.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["answer", "citations", "abstain"],
                "properties": {
                    "answer": {"type": "string"},
                    "value": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "citations": {"type": "array"},
                    "calculation": {"type": ["string", "null"]},
                    "abstain": {"type": "boolean"},
                },
            },
        ),
    ]


@pytest.fixture
def search_result_message() -> Message:
    """ONE tool message carrying the JSON result of a ``search_filings`` call."""
    hits = {
        "hits": [
            {
                "chunk_id": CHUNK_A,
                "doc_name": "FIXTURE_2023_10K",
                "page_num": 1,
                "snippet": PASSAGE_A,
            },
            {
                "chunk_id": CHUNK_B,
                "doc_name": "FIXTURE_2023_10K",
                "page_num": 2,
                "snippet": PASSAGE_B,
            },
        ]
    }
    return Message(
        role="tool",
        tool_results=[
            ToolResult(
                tool_call_id="mock-call-search", name="search_filings", content=json.dumps(hits)
            )
        ],
    )
