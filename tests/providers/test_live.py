"""Live vendor smoke tests: ``pytest -m live``; skipped without keys (never run in CI).

The keys are captured at import time because ``tests/conftest.py`` scrubs them from the
environment before every test; they are passed explicitly and never logged.
"""

from __future__ import annotations

import os

import pytest

from secqa.core.contracts import Message, ToolSpec

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()

pytestmark = pytest.mark.live

_ECHO_TOOL = ToolSpec(
    name="echo",
    description="Echo the given word back to the caller.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["word"],
        "properties": {"word": {"type": "string"}},
    },
)
_SCHEMA = {
    "title": "answer",
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "abstain"],
    "properties": {"answer": {"type": "string"}, "abstain": {"type": "boolean"}},
}


@pytest.mark.skipif(not _OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_live_json_and_tool() -> None:
    from secqa.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider("gpt-5.4-mini", api_key=_OPENAI_KEY, timeout_s=60)
    structured = provider.complete(
        [Message(role="user", content="Reply with answer='pong' and abstain=false.")],
        json_schema=_SCHEMA,
        max_tokens=200,
        effort="low",
    )
    assert structured.parsed is not None and structured.parsed["abstain"] is False
    assert structured.usage.input_tokens > 0 and structured.usage.output_tokens > 0

    tooled = provider.complete(
        [Message(role="user", content="Call the echo tool with the word 'ping'.")],
        tools=[_ECHO_TOOL],
        max_tokens=200,
        effort="low",
    )
    assert tooled.stop_reason == "tool_use"
    assert tooled.tool_calls[0].name == "echo"
    assert tooled.tool_calls[0].arguments == {"word": "ping"}


@pytest.mark.skipif(not _ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_live_json_and_tool() -> None:
    from secqa.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider("claude-sonnet-5", api_key=_ANTHROPIC_KEY, timeout_s=120)
    structured = provider.complete(
        [Message(role="user", content="Reply with answer='pong' and abstain=false.")],
        json_schema=_SCHEMA,
        max_tokens=2000,
        effort="low",
    )
    assert structured.parsed is not None and structured.parsed["abstain"] is False
    assert structured.usage.input_tokens > 0 and structured.usage.output_tokens > 0

    tooled = provider.complete(
        [Message(role="user", content="Call the echo tool with the word 'ping'.")],
        tools=[_ECHO_TOOL],
        max_tokens=2000,
        effort="low",
    )
    assert tooled.stop_reason == "tool_use"
    assert tooled.tool_calls[0].name == "echo"
    assert tooled.tool_calls[0].arguments == {"word": "ping"}
