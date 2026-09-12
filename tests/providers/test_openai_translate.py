"""OpenAI translation on recorded fixtures: request shape both ways, no network."""

from __future__ import annotations

import json
from typing import Any

import pytest

from secqa.core.contracts import LLMProvider, Message, ToolCall, ToolResult, ToolSpec, Usage
from secqa.core.errors import ConfigError, ProviderError
from secqa.providers.openai_provider import (
    OpenAIProvider,
    build_openai_request,
    is_reasoning_model,
    openai_to_response,
)
from tests.providers.conftest import ANSWER_SCHEMA, load_response

openai = pytest.importorskip("openai")


# ---------- response -> LLMResponse ----------


def test_tool_call_fixture() -> None:
    response = openai_to_response(load_response("openai_tool_call.json"), latency_ms=12.5)
    assert response.provider == "openai"
    assert response.model == "gpt-5.5-2026-06-01"
    assert response.raw_id == "chatcmpl-fixture-tool-call"
    assert response.stop_reason == "tool_use"
    assert response.text == ""
    assert response.tool_calls == [
        ToolCall(
            id="call_fixture_001",
            name="search_filings",
            arguments={
                "query": "FIXTURE CORP total net sales fiscal 2023",
                "ticker": "FIX",
                "k": 5,
            },
        )
    ]
    # prompt_tokens (1200) includes 1000 cached -> input_tokens is the uncached remainder
    assert response.usage == Usage(input_tokens=200, output_tokens=35, cache_read_tokens=1000)
    assert response.latency_ms == 12.5
    assert response.parsed is None and response.cached is False


def test_json_fixture_parsed_only_when_expected() -> None:
    raw = load_response("openai_text_json.json")
    plain = openai_to_response(raw, latency_ms=1.0)
    assert plain.stop_reason == "end_turn" and plain.parsed is None
    assert plain.text.startswith("{")
    structured = openai_to_response(raw, latency_ms=1.0, expect_json=True)
    assert structured.parsed is not None
    assert structured.parsed["value"] == 1577000000
    assert structured.parsed["abstain"] is False
    assert structured.usage == Usage(input_tokens=800, output_tokens=60)


def test_refusal_fixture_is_not_an_error() -> None:
    response = openai_to_response(load_response("openai_refusal.json"), latency_ms=1.0)
    assert response.stop_reason == "refusal"
    assert response.text == ""
    assert response.tool_calls == []


def test_finish_reason_mapping() -> None:
    def with_finish(reason: str) -> dict[str, Any]:
        raw = load_response("openai_text_json.json")
        raw["choices"][0]["finish_reason"] = reason
        return raw

    assert openai_to_response(with_finish("length"), 0.0).stop_reason == "max_tokens"
    assert openai_to_response(with_finish("content_filter"), 0.0).stop_reason == "refusal"
    assert openai_to_response(with_finish("weird"), 0.0).stop_reason == "other"


def test_malformed_tool_arguments_raise_retryable() -> None:
    raw = load_response("openai_tool_call.json")
    raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    with pytest.raises(ProviderError, match="malformed JSON") as info:
        openai_to_response(raw, 0.0)
    assert info.value.retryable is True
    with pytest.raises(ProviderError, match="no choices"):
        openai_to_response({"choices": []}, 0.0)


# ---------- Message/ToolSpec -> request ----------


def test_request_shape_with_tools_and_history(agent_tools: list[ToolSpec]) -> None:
    history = [
        Message(role="user", content="Question: net sales?"),
        Message(
            role="assistant",
            content="Searching.",
            tool_calls=[
                ToolCall(id="c1", name="search_filings", arguments={"query": "net sales"}),
                ToolCall(id="c2", name="search_filings", arguments={"query": "revenue"}),
            ],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(tool_call_id="c1", name="search_filings", content='{"hits": []}'),
                ToolResult(
                    tool_call_id="c2",
                    name="search_filings",
                    content='{"error": "x"}',
                    is_error=True,
                ),
            ],
        ),
    ]
    request = build_openai_request(
        "gpt-5.5", history, system="SYS", tools=agent_tools, max_tokens=321, effort="medium"
    )
    assert request["model"] == "gpt-5.5"
    assert request["max_completion_tokens"] == 321
    assert request["messages"][0] == {"role": "system", "content": "SYS"}
    assert request["messages"][1] == {"role": "user", "content": "Question: net sales?"}
    assistant = request["messages"][2]
    assert assistant["content"] == "Searching."
    assert [c["function"]["name"] for c in assistant["tool_calls"]] == ["search_filings"] * 2
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"query": "net sales"}
    # our ONE tool message fans out to one wire message per tool_call_id, in order
    assert request["messages"][3:] == [
        {"role": "tool", "tool_call_id": "c1", "content": '{"hits": []}'},
        {"role": "tool", "tool_call_id": "c2", "content": '{"error": "x"}'},
    ]
    assert [t["function"]["name"] for t in request["tools"]] == ["search_filings", "final_answer"]
    assert all(
        t["type"] == "function" and t["function"]["strict"] is True for t in request["tools"]
    )
    assert request["tools"][0]["function"]["parameters"] == agent_tools[0].input_schema
    assert request["tool_choice"] == "auto"
    assert request["parallel_tool_calls"] is False
    # gpt-5.x is a reasoning model: no temperature, effort mapped to reasoning_effort
    assert "temperature" not in request
    assert request["reasoning_effort"] == "medium"
    assert "response_format" not in request


def test_request_json_schema_and_temperature_on_non_reasoning_model() -> None:
    request = build_openai_request(
        "gpt-4.1-mini", [Message(role="user", content="q")], json_schema=ANSWER_SCHEMA
    )
    assert request["temperature"] == 0
    assert "reasoning_effort" not in request
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": ANSWER_SCHEMA, "strict": True},
    }
    assert "tools" not in request and "parallel_tool_calls" not in request
    assert request["messages"] == [{"role": "user", "content": "q"}]


def test_reasoning_model_detection() -> None:
    assert is_reasoning_model("gpt-5.5") and is_reasoning_model("o3-mini")
    assert is_reasoning_model("GPT-5.4-mini")
    assert not is_reasoning_model("gpt-4o") and not is_reasoning_model("gpt-4.1")


def test_effort_none_sends_no_reasoning_effort() -> None:
    request = build_openai_request("gpt-5.5", [Message(role="user", content="q")])
    assert "reasoning_effort" not in request and "temperature" not in request


# ---------- provider object with a stubbed SDK client ----------


class _StubCompletions:
    def __init__(self, raw: dict[str, Any] | Exception):
        self.raw = raw
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if isinstance(self.raw, Exception):
            raise self.raw
        return openai.types.chat.ChatCompletion.model_validate(self.raw)


def _provider(
    raw: dict[str, Any] | Exception, model: str = "gpt-5.5"
) -> tuple[OpenAIProvider, _StubCompletions]:
    provider = OpenAIProvider(model, api_key="sk-test-not-real", timeout_s=5, max_retries=0)
    stub = _StubCompletions(raw)
    provider._client.chat.completions = stub  # type: ignore[assignment]
    return provider, stub


def test_provider_round_trip(rag_prompt: list[Message]) -> None:
    provider, stub = _provider(load_response("openai_text_json.json"))
    assert isinstance(provider, LLMProvider)
    assert provider.params()["temperature"] is None  # reasoning model
    response = provider.complete(rag_prompt, system="S", json_schema=ANSWER_SCHEMA, effort="low")
    assert stub.requests[0]["model"] == "gpt-5.5"
    assert stub.requests[0]["reasoning_effort"] == "low"
    assert stub.requests[0]["response_format"]["type"] == "json_schema"
    assert response.parsed is not None and response.parsed["abstain"] is False
    assert response.latency_ms >= 0.0
    assert response.usage.input_tokens == 800


def test_provider_error_translation() -> None:
    httpx2 = pytest.importorskip("httpx2")
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")

    def status(cls: type, code: int) -> Exception:
        return cls("boom", response=httpx2.Response(code, request=request), body=None)

    cases = [
        (status(openai.RateLimitError, 429), True),
        (status(openai.InternalServerError, 503), True),
        (openai.APITimeoutError(request=request), True),
        (openai.APIConnectionError(request=request), True),
        (status(openai.BadRequestError, 400), False),
        (status(openai.AuthenticationError, 401), False),
    ]
    for exc, retryable in cases:
        provider, _ = _provider(exc)
        with pytest.raises(ProviderError) as info:
            provider.complete([Message(role="user", content="q")])
        assert info.value.retryable is retryable, type(exc).__name__
        assert info.value.provider == "openai"
        assert type(exc).__name__ in str(info.value)


def test_constructor_validation() -> None:
    with pytest.raises(ConfigError, match="model id"):
        OpenAIProvider("  ", api_key="sk-test")
    provider = OpenAIProvider("gpt-4.1", api_key="sk-test")
    assert provider.params()["temperature"] == 0
