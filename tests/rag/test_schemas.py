"""ANSWER_SCHEMA shape and the tolerant structured-answer parser."""

from __future__ import annotations

import json
from typing import Any

import pytest

from secqa.core.contracts import CitationRef, LLMResponse, Usage
from secqa.providers.mock_provider import fill_schema
from secqa.rag import ABSTAIN_TEXT, ANSWER_SCHEMA, StructuredAnswer, parse_structured_answer

GOOD: dict[str, Any] = {
    "answer": "Net sales were $1,577 million.",
    "value": 1577e6,
    "unit": "USD",
    "citations": [{"ref": "chunk:" + "a" * 40, "quote": "Total net sales were $1,577 million"}],
    "abstain": False,
}


def response(**overrides: Any) -> LLMResponse:
    base: dict[str, Any] = {
        "text": "",
        "tool_calls": [],
        "usage": Usage(input_tokens=10, output_tokens=5),
        "provider": "test",
        "model": "test",
        "latency_ms": 0.0,
        "stop_reason": "end_turn",
        "parsed": None,
    }
    base.update(overrides)
    return LLMResponse(**base)


# ---- schema -------------------------------------------------------------------------------


def test_schema_is_strict_and_flat() -> None:
    assert ANSWER_SCHEMA["type"] == "object"
    assert ANSWER_SCHEMA["additionalProperties"] is False
    assert (
        set(ANSWER_SCHEMA["required"])
        == set(ANSWER_SCHEMA["properties"])
        == {
            "answer",
            "value",
            "unit",
            "citations",
            "abstain",
        }
    )
    assert ANSWER_SCHEMA["properties"]["value"]["type"] == ["number", "null"]
    assert ANSWER_SCHEMA["properties"]["unit"]["type"] == ["string", "null"]
    items = ANSWER_SCHEMA["properties"]["citations"]["items"]
    assert items["additionalProperties"] is False
    assert set(items["required"]) == set(items["properties"]) == {"ref", "quote"}
    json.dumps(ANSWER_SCHEMA)  # serialisable for both vendors


def test_schema_default_fill_is_a_valid_answer() -> None:
    filled = fill_schema(ANSWER_SCHEMA, {})
    parsed = StructuredAnswer.model_validate(filled)
    assert parsed.citations == [] and parsed.value is None and parsed.abstain is False


# ---- StructuredAnswer ---------------------------------------------------------------------


def test_structured_answer_abstention_rules() -> None:
    explicit = StructuredAnswer(answer="", abstain=True)
    assert explicit.abstained is True and explicit.text == ABSTAIN_TEXT
    textual = StructuredAnswer(answer="Insufficient evidence to answer.", abstain=False)
    assert textual.abstained is True and textual.text == "Insufficient evidence to answer."
    normal = StructuredAnswer(answer="  Yes.  ")
    assert normal.abstained is False and normal.text == "Yes."
    assert StructuredAnswer(unit="  ").unit is None


# ---- parser -------------------------------------------------------------------------------


def test_parse_prefers_parsed_payload() -> None:
    result = parse_structured_answer(response(parsed=GOOD, text="ignored"))
    assert result.source == "parsed" and result.parse_error is None
    assert result.answer.value == 1577e6 and result.answer.unit == "USD"
    assert result.answer.citations == [CitationRef(**GOOD["citations"][0])]
    assert result.answer.abstained is False


def test_parse_falls_back_to_text_json_with_code_fence() -> None:
    text = "```json\n" + json.dumps(GOOD) + "\n```"
    result = parse_structured_answer(response(text=text))
    assert result.source == "text_json" and result.parse_error is None
    assert result.answer.text == GOOD["answer"]


def test_parse_tolerates_extra_keys_and_null_citations() -> None:
    payload = {**GOOD, "citations": None, "confidence": 0.9}
    result = parse_structured_answer(response(parsed=payload))
    assert result.source == "parsed" and result.answer.citations == []


def test_parse_invalid_parsed_then_valid_text() -> None:
    bad = {**GOOD, "value": "a lot"}
    result = parse_structured_answer(response(parsed=bad, text=json.dumps(GOOD)))
    assert result.source == "text_json"
    assert result.parse_error is not None and "value" in result.parse_error


def test_parse_plain_text_fallback() -> None:
    result = parse_structured_answer(response(text="Net sales were $1,577 million."))
    assert result.source == "text_fallback"
    assert result.answer.text == "Net sales were $1,577 million."
    assert result.answer.citations == [] and result.answer.abstained is False
    assert result.parse_error is not None and "JSON" in result.parse_error

    abstain = parse_structured_answer(response(text="INSUFFICIENT EVIDENCE"))
    assert abstain.source == "text_fallback" and abstain.answer.abstained is True


def test_parse_json_array_is_not_an_answer() -> None:
    result = parse_structured_answer(response(text="[1, 2, 3]"))
    assert result.source == "text_fallback"
    assert result.parse_error is not None and "expected an object" in result.parse_error


@pytest.mark.parametrize(
    ("kwargs", "source"),
    [
        ({"stop_reason": "refusal", "text": ""}, "refusal"),
        ({"stop_reason": "refusal", "text": "", "parsed": GOOD}, "refusal"),
        ({"text": "   "}, "empty"),
    ],
)
def test_parse_refusal_and_empty_abstain(kwargs: dict[str, Any], source: str) -> None:
    result = parse_structured_answer(response(**kwargs))
    assert result.source == source
    assert result.answer.abstained is True and result.answer.text == ABSTAIN_TEXT
    assert result.answer.citations == [] and result.parse_error is not None
