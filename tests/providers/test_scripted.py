"""ScriptedProvider: turn order, regex assertions, exhaustion, YAML validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from secqa.core.contracts import LLMProvider, Message, ToolResult
from secqa.core.errors import ConfigError, ProviderError, ScenarioExhausted
from secqa.providers.scripted_provider import ScriptedProvider
from tests.providers.conftest import CHUNK_A, SCENARIOS


def _turns() -> list[dict]:
    return [
        {"tool_calls": [{"name": "search_filings", "arguments": {"query": "q"}}]},
        {"text": "done", "stop_reason": "end_turn"},
    ]


def test_turns_in_order_from_list() -> None:
    provider = ScriptedProvider(_turns())
    assert isinstance(provider, LLMProvider)
    assert provider.provider == "scripted" and provider.model == "scripted"
    first = provider.complete([Message(role="user", content="hi")])
    assert first.stop_reason == "tool_use"
    assert first.tool_calls[0].name == "search_filings"
    assert first.tool_calls[0].id == "scripted-call-0-0"
    assert first.raw_id == "scripted-0"
    second = provider.complete([Message(role="user", content="hi")])
    assert second.text == "done" and second.stop_reason == "end_turn"
    assert provider.turns_consumed == 2 == provider.turns_total


def test_exhaustion_raises() -> None:
    provider = ScriptedProvider(_turns())
    provider.complete([Message(role="user", content="a")])
    provider.complete([Message(role="user", content="b")])
    with pytest.raises(ScenarioExhausted, match="2 turn"):
        provider.complete([Message(role="user", content="c")])
    provider.reset()
    assert provider.complete([Message(role="user", content="a")]).stop_reason == "tool_use"


def test_yaml_scenario_with_match_regexes() -> None:
    provider = ScriptedProvider(SCENARIOS / "basic.yaml")
    assert provider.model == "scripted-basic"
    question = Message(role="user", content="Question: what were net sales in 2023?")
    step1 = provider.complete([question])
    assert step1.tool_calls[0].id == "call-search"
    tool_msg = Message(
        role="tool",
        tool_results=[
            ToolResult(
                tool_call_id="call-search",
                name="search_filings",
                content=f'{{"hits": [{{"chunk_id": "{CHUNK_A}", "snippet": "x"}}]}}',
            )
        ],
    )
    # the match regex is applied to the LAST user/tool message - here the tool results
    step2 = provider.complete(
        [question, Message(role="assistant", tool_calls=list(step1.tool_calls)), tool_msg]
    )
    assert step2.tool_calls[0].name == "final_answer"
    assert step2.tool_calls[0].arguments["value"] == 1577000000
    step3 = provider.complete([question], json_schema={"type": "object"})
    assert step3.parsed == {"label": "correct", "rationale": "matches the fixture answer"}
    assert step3.usage.input_tokens == 123 and step3.usage.output_tokens == 45


def test_match_failure_is_a_provider_error() -> None:
    provider = ScriptedProvider([{"match": "net sales", "text": "x"}])
    with pytest.raises(ProviderError, match="match 'net sales' failed") as info:
        provider.complete([Message(role="user", content="unrelated")])
    assert info.value.retryable is False


def test_parsed_explicit_beats_text_and_json_only_when_requested() -> None:
    provider = ScriptedProvider(
        [
            {"text": '{"a": 1}', "parsed": {"a": 2}},
            {"text": '{"a": 1}'},
            {"text": "not json"},
        ]
    )
    msgs = [Message(role="user", content="q")]
    assert provider.complete(msgs, json_schema={"type": "object"}).parsed == {"a": 2}
    assert provider.complete(msgs).parsed is None  # no json_schema requested
    assert provider.complete(msgs, json_schema={"type": "object"}).parsed is None


def test_refusal_turn_has_empty_text() -> None:
    provider = ScriptedProvider([{"text": "ignored", "stop_reason": "refusal"}])
    response = provider.complete([Message(role="user", content="q")])
    assert response.stop_reason == "refusal" and response.text == ""


@pytest.mark.parametrize(
    "turns, message",
    [
        ([{"unknown": 1}], "unknown keys"),
        ([{"match": "("}], "bad regex"),
        ([{"tool_calls": [{"arguments": {}}]}], "needs a 'name'"),
        ([{"stop_reason": "banana"}], "stop_reason"),
        (["not a mapping"], "must be a mapping"),
        ([{"parsed": "str"}], "parsed must be a mapping"),
    ],
)
def test_invalid_turns_rejected(turns: list, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        ScriptedProvider(turns)


def test_missing_or_malformed_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        ScriptedProvider(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("just: a mapping without turns\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="list of turns"):
        ScriptedProvider(bad)
