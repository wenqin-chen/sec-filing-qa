"""Anthropic translation on recorded fixtures: request shape both ways, no network."""

from __future__ import annotations

from typing import Any

import pytest

from secqa.core.contracts import LLMProvider, Message, ToolCall, ToolResult, ToolSpec, Usage
from secqa.core.errors import ConfigError, ProviderError
from secqa.providers.anthropic_provider import (
    AnthropicProvider,
    anthropic_to_response,
    build_anthropic_request,
)
from tests.providers.conftest import ANSWER_SCHEMA, load_response

anthropic = pytest.importorskip("anthropic")


# ---------- response -> LLMResponse ----------


def test_tool_use_fixture() -> None:
    response = anthropic_to_response(load_response("anthropic_tool_use.json"), latency_ms=40.0)
    assert response.provider == "anthropic"
    assert response.model == "claude-opus-5"
    assert response.raw_id == "msg_fixture_tool_use"
    assert response.stop_reason == "tool_use"
    assert response.text == "I will look up the filing first."  # thinking block is not text
    assert response.tool_calls == [
        ToolCall(
            id="toolu_fixture_001",
            name="search_filings",
            arguments={
                "query": "FIXTURE CORP total net sales fiscal 2023",
                "ticker": "FIX",
                "k": 5,
            },
        )
    ]
    assert response.usage == Usage(
        input_tokens=200, output_tokens=80, cache_read_tokens=100, cache_write_tokens=900
    )
    assert response.parsed is None


def test_json_fixture_parsed_only_when_expected() -> None:
    raw = load_response("anthropic_text_json.json")
    assert anthropic_to_response(raw, 1.0).parsed is None
    structured = anthropic_to_response(raw, 1.0, expect_json=True)
    assert structured.stop_reason == "end_turn"
    assert structured.parsed is not None and structured.parsed["value"] == 1577000000
    assert structured.parsed["citations"][0]["ref"].startswith("chunk:")
    assert structured.usage == Usage(input_tokens=700, output_tokens=70)


def test_refusal_fixture_is_not_an_error() -> None:
    response = anthropic_to_response(load_response("anthropic_refusal.json"), latency_ms=1.0)
    assert response.stop_reason == "refusal"
    assert response.text == "" and response.tool_calls == []
    assert response.usage.input_tokens == 40


def test_stop_reason_mapping() -> None:
    def with_stop(reason: str) -> dict[str, Any]:
        raw = load_response("anthropic_text_json.json")
        raw["stop_reason"] = reason
        return raw

    assert anthropic_to_response(with_stop("max_tokens"), 0.0).stop_reason == "max_tokens"
    assert anthropic_to_response(with_stop("stop_sequence"), 0.0).stop_reason == "end_turn"
    assert anthropic_to_response(with_stop("pause_turn"), 0.0).stop_reason == "other"
    assert anthropic_to_response(with_stop("model_context_window_exceeded"), 0.0).stop_reason == (
        "other"
    )


def test_non_object_tool_input_raises_retryable() -> None:
    raw = load_response("anthropic_tool_use.json")
    raw["content"][2]["input"] = "not an object"
    with pytest.raises(ProviderError) as info:
        anthropic_to_response(raw, 0.0)
    assert info.value.retryable is True


# ---------- Message/ToolSpec -> request ----------


def test_request_shape_no_temperature_tool_results_in_one_message(
    agent_tools: list[ToolSpec],
) -> None:
    history = [
        Message(role="user", content="Question: net sales?"),
        Message(
            role="assistant",
            content="Searching.",
            tool_calls=[
                ToolCall(id="t1", name="search_filings", arguments={"query": "net sales"}),
                ToolCall(id="t2", name="search_filings", arguments={"query": "revenue"}),
            ],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(tool_call_id="t1", name="search_filings", content='{"hits": []}'),
                ToolResult(
                    tool_call_id="t2",
                    name="search_filings",
                    content='{"error": "x"}',
                    is_error=True,
                ),
            ],
        ),
    ]
    request = build_anthropic_request(
        "claude-opus-5", history, system="SYS", tools=agent_tools, max_tokens=777, effort="high"
    )
    # CONTRACTS rule 3: no sampling parameters, adaptive thinking, effort in output_config
    for forbidden in ("temperature", "top_p", "top_k", "fallbacks"):
        assert forbidden not in request
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "high"}
    assert request["system"] == "SYS"
    assert request["max_tokens"] == 777
    assert request["model"] == "claude-opus-5"
    assert request["tool_choice"] == {"type": "auto"}
    assert [t["name"] for t in request["tools"]] == ["search_filings", "final_answer"]
    assert all(t["strict"] is True for t in request["tools"])
    assert request["tools"][0]["input_schema"] == agent_tools[0].input_schema

    messages = request["messages"]
    assert messages[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "Question: net sales?"}],
    }
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == [
        {"type": "text", "text": "Searching."},
        {"type": "tool_use", "id": "t1", "name": "search_filings", "input": {"query": "net sales"}},
        {"type": "tool_use", "id": "t2", "name": "search_filings", "input": {"query": "revenue"}},
    ]
    # ALL tool results of the step in ONE user message
    assert len(messages) == 3
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": '{"hits": []}',
                "is_error": False,
            },
            {
                "type": "tool_result",
                "tool_use_id": "t2",
                "content": '{"error": "x"}',
                "is_error": True,
            },
        ],
    }


def test_request_defaults_effort_medium_and_json_format() -> None:
    request = build_anthropic_request(
        "claude-sonnet-5", [Message(role="user", content="q")], json_schema=ANSWER_SCHEMA
    )
    assert request["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
    }
    assert "system" not in request and "tools" not in request and "tool_choice" not in request
    assert "temperature" not in request


def test_thinking_blocks_are_replayed_for_own_tool_turns() -> None:
    raw = load_response("anthropic_tool_use.json")
    blocks = raw["content"]
    side_table = {"toolu_fixture_001": blocks}
    history = [
        Message(role="user", content="q"),
        Message(
            role="assistant",
            content="I will look up the filing first.",
            tool_calls=[
                ToolCall(
                    id="toolu_fixture_001",
                    name="search_filings",
                    arguments={
                        "query": "FIXTURE CORP total net sales fiscal 2023",
                        "ticker": "FIX",
                        "k": 5,
                    },
                )
            ],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(tool_call_id="toolu_fixture_001", name="search_filings", content="{}")
            ],
        ),
    ]
    replayed = build_anthropic_request("claude-opus-5", history, assistant_blocks=side_table)
    assert replayed["messages"][1]["content"] == blocks  # thinking + text + tool_use verbatim
    rebuilt = build_anthropic_request("claude-opus-5", history)  # no side table -> rebuilt
    assert rebuilt["messages"][1]["content"][0] == {
        "type": "text",
        "text": "I will look up the filing first.",
    }
    assert all(b["type"] != "thinking" for b in rebuilt["messages"][1]["content"])


# ---------- provider object with a stubbed SDK client ----------


class _StubMessages:
    def __init__(self, raw: dict[str, Any] | Exception):
        self.raw = raw
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if isinstance(self.raw, Exception):
            raise self.raw
        return anthropic.types.Message.model_validate(self.raw)


def _provider(raw: dict[str, Any] | Exception) -> tuple[AnthropicProvider, _StubMessages]:
    provider = AnthropicProvider(api_key="sk-ant-test-not-real", timeout_s=5, max_retries=0)
    stub = _StubMessages(raw)
    provider._client.messages = stub  # type: ignore[assignment]
    return provider, stub


def test_provider_round_trip_and_side_table(agent_tools: list[ToolSpec]) -> None:
    provider, stub = _provider(load_response("anthropic_tool_use.json"))
    assert isinstance(provider, LLMProvider)
    assert provider.model == "claude-opus-5"
    assert provider.params()["temperature"] is None and provider.params()["thinking"] == "adaptive"
    question = Message(role="user", content="q")
    step1 = provider.complete([question], system="S", tools=agent_tools, effort="low")
    sent = stub.requests[0]
    assert "temperature" not in sent and sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "low"}
    assert step1.tool_calls[0].id == "toolu_fixture_001"

    history = [
        question,
        Message(role="assistant", content=step1.text, tool_calls=list(step1.tool_calls)),
        Message(
            role="tool",
            tool_results=[
                ToolResult(tool_call_id="toolu_fixture_001", name="search_filings", content="{}")
            ],
        ),
    ]
    provider.complete(history, system="S", tools=agent_tools)
    replayed_assistant = stub.requests[1]["messages"][1]["content"]
    assert replayed_assistant[0]["type"] == "thinking"  # signed block echoed back verbatim
    assert stub.requests[1]["messages"][2]["content"][0]["type"] == "tool_result"


def test_provider_json_schema_parsed(rag_prompt: list[Message]) -> None:
    provider, stub = _provider(load_response("anthropic_text_json.json"))
    response = provider.complete(rag_prompt, json_schema=ANSWER_SCHEMA)
    assert stub.requests[0]["output_config"]["format"]["type"] == "json_schema"
    assert response.parsed is not None and response.parsed["value"] == 1577000000


def test_refusal_is_returned_not_raised_and_not_retried() -> None:
    provider, stub = _provider(load_response("anthropic_refusal.json"))
    response = provider.complete([Message(role="user", content="q")])
    assert response.stop_reason == "refusal" and response.text == ""
    assert len(stub.requests) == 1
    assert stub.requests[0]["model"] == "claude-opus-5"  # never re-sent to another model


def test_provider_error_translation() -> None:
    httpx2 = pytest.importorskip("httpx2")
    request = httpx2.Request("POST", "https://example.invalid/v1/messages")

    def status(cls: type, code: int) -> Exception:
        return cls("boom", response=httpx2.Response(code, request=request), body=None)

    cases = [
        (status(anthropic.RateLimitError, 429), True),
        (status(anthropic.InternalServerError, 500), True),
        (status(anthropic.APIStatusError, 529), True),  # overloaded
        (anthropic.APITimeoutError(request=request), True),
        (anthropic.APIConnectionError(request=request), True),
        (status(anthropic.BadRequestError, 400), False),
        (status(anthropic.AuthenticationError, 401), False),
    ]
    for exc, retryable in cases:
        provider, _ = _provider(exc)
        with pytest.raises(ProviderError) as info:
            provider.complete([Message(role="user", content="q")])
        assert info.value.retryable is retryable, type(exc).__name__
        assert info.value.provider == "anthropic"


def test_constructor_validation() -> None:
    with pytest.raises(ConfigError, match="model id"):
        AnthropicProvider("", api_key="sk-ant-test")
    assert AnthropicProvider(api_key="sk-ant-test").model == "claude-opus-5"
