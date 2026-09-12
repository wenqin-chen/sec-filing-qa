"""MockProvider: deterministic, extractive, tool-aware, json_schema-honouring."""

from __future__ import annotations

import json

import pytest

from secqa.core.contracts import LLMProvider, Message, ToolCall, ToolSpec
from secqa.providers.mock_provider import (
    ABSTAIN_TEXT,
    MockProvider,
    extract_passages,
    extract_question,
    fill_schema,
    first_sentence,
)
from tests.providers.conftest import ANSWER_SCHEMA, CHUNK_A, PASSAGE_A


def test_satisfies_protocol() -> None:
    provider = MockProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.provider == "mock"
    assert provider.model == "mock-extractive"
    assert provider.params() == {}


def test_rag_answer_is_first_sentence_of_top_passage(rag_prompt: list[Message]) -> None:
    response = MockProvider().complete(rag_prompt, system="You are careful.")
    expected = "Total net sales were $1,577 million in fiscal 2023, an increase of 12% over 2022."
    assert response.text == expected
    assert response.tool_calls == []
    assert response.stop_reason == "end_turn"
    assert response.usage.input_tokens > 0 and response.usage.output_tokens > 0
    assert response.latency_ms == 0.0
    assert response.cached is False


def test_deterministic(rag_prompt: list[Message]) -> None:
    provider = MockProvider()
    first = provider.complete(rag_prompt, json_schema=ANSWER_SCHEMA)
    second = provider.complete(rag_prompt, json_schema=ANSWER_SCHEMA)
    assert first == second
    assert (
        first.model_dump()
        == MockProvider().complete(rag_prompt, json_schema=ANSWER_SCHEMA).model_dump()
    )


def test_json_schema_is_honoured(rag_prompt: list[Message]) -> None:
    response = MockProvider().complete(rag_prompt, json_schema=ANSWER_SCHEMA)
    assert response.parsed is not None
    assert set(response.parsed) == set(ANSWER_SCHEMA["properties"])
    assert response.parsed["abstain"] is False
    assert response.parsed["value"] == 1577e6
    assert response.parsed["citations"] == [
        {"ref": f"chunk:{CHUNK_A}", "quote": response.parsed["answer"]}
    ]
    assert response.parsed["unit"] is None
    assert json.loads(response.text) == response.parsed
    # the quote is a verbatim substring of the passage, so the verifier can confirm it
    assert response.parsed["citations"][0]["quote"] in PASSAGE_A


def test_tool_path_searches_then_answers(
    agent_tools: list[ToolSpec], search_result_message: Message
) -> None:
    provider = MockProvider()
    question = Message(role="user", content="Question: What were total net sales in 2023?")
    step1 = provider.complete([question], tools=agent_tools)
    assert step1.stop_reason == "tool_use"
    assert [c.name for c in step1.tool_calls] == ["search_filings"]
    assert step1.tool_calls[0].arguments == {"query": "What were total net sales in 2023?"}
    assert step1.text == ""

    history = [
        question,
        Message(role="assistant", tool_calls=list(step1.tool_calls)),
        search_result_message,
    ]
    step2 = provider.complete(history, tools=agent_tools)
    assert [c.name for c in step2.tool_calls] == ["final_answer"]
    args = step2.tool_calls[0].arguments
    assert args["abstain"] is False
    assert args["citations"][0]["ref"] == f"chunk:{CHUNK_A}"
    assert args["citations"][0]["quote"] == args["answer"]
    assert args["answer"] in PASSAGE_A
    assert args["value"] == 1577e6


def test_tool_path_abstains_when_search_returned_nothing(agent_tools: list[ToolSpec]) -> None:
    from secqa.core.contracts import ToolResult

    history = [
        Message(role="user", content="Question: anything?"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="search_filings", arguments={"query": "x"})],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(tool_call_id="c1", name="search_filings", content='{"hits": []}')
            ],
        ),
    ]
    response = MockProvider().complete(history, tools=agent_tools)
    assert response.tool_calls[0].name == "final_answer"
    assert response.tool_calls[0].arguments["abstain"] is True
    assert response.tool_calls[0].arguments["answer"] == ABSTAIN_TEXT


def test_abstain_behaviour_never_calls_tools(
    rag_prompt: list[Message], agent_tools: list[ToolSpec]
) -> None:
    provider = MockProvider("abstain")
    assert provider.model == "mock-abstain"
    plain = provider.complete(rag_prompt)
    assert plain.text == ABSTAIN_TEXT and plain.tool_calls == []
    structured = provider.complete(rag_prompt, json_schema=ANSWER_SCHEMA)
    assert structured.parsed is not None and structured.parsed["abstain"] is True
    assert structured.parsed["citations"] == []
    with_tools = provider.complete(rag_prompt, tools=agent_tools)
    assert [c.name for c in with_tools.tool_calls] == ["final_answer"]
    assert with_tools.tool_calls[0].arguments["abstain"] is True


def test_no_passages_means_abstain() -> None:
    response = MockProvider().complete([Message(role="user", content="Question: why?")])
    assert response.text == ABSTAIN_TEXT
    structured = MockProvider().complete(
        [Message(role="user", content="Question: why?")], json_schema=ANSWER_SCHEMA
    )
    assert structured.parsed is not None and structured.parsed["abstain"] is True


def test_invalid_behaviour_rejected() -> None:
    with pytest.raises(ValueError):
        MockProvider("creative")  # type: ignore[arg-type]


def test_helpers() -> None:
    assert extract_question("Hints: ticker=FIX\nQuestion:  What is  x? \n") == "What is x?"
    assert extract_question("plain question") == "plain question"
    assert first_sentence("Short. This sentence is long enough to quote.") == (
        "This sentence is long enough to quote."
    )
    assert first_sentence("tiny") is None
    passages = extract_passages(
        [Message(role="user", content=f"[1] ref chunk:{CHUNK_A}:\n{PASSAGE_A}\nQuestion: q?")]
    )
    assert passages == [(f"chunk:{CHUNK_A}", PASSAGE_A)]


def test_fill_schema_defaults_and_enums() -> None:
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["correct", "incorrect", "abstain"]},
            "rationale": {"type": "string"},
            "score": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "claims": {"type": "array"},
            "nested": {"type": "object", "properties": {"flag": {"type": "boolean"}}},
        },
    }
    assert fill_schema(schema, {"rationale": "because"}) == {
        "label": "correct",
        "rationale": "because",
        "score": None,
        "claims": [],
        "nested": {"flag": False},
    }
