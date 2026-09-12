"""ReplayCacheProvider: record then replay; changed prompt misses; replay miss raises."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from secqa.core.contracts import LLMProvider, LLMResponse, Message, ToolSpec, Usage
from secqa.core.errors import CassetteMiss, ConfigError
from secqa.providers.base import canonical_request, request_key
from secqa.providers.mock_provider import MockProvider
from secqa.providers.replay_provider import ReplayCacheProvider
from secqa.providers.scripted_provider import ScriptedProvider


class CountingProvider:
    """Wraps MockProvider and counts real calls (the thing a cassette must avoid)."""

    provider = "mock"
    model = "mock-extractive"

    def __init__(self) -> None:
        self.inner = MockProvider()
        self.calls = 0

    def params(self) -> dict[str, Any]:
        return {"temperature": 0}

    def complete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return self.inner.complete(messages, **kwargs)


def test_record_then_replay(tmp_path: Path, rag_prompt: list[Message]) -> None:
    inner = CountingProvider()
    recorder = ReplayCacheProvider(inner, cache_dir=tmp_path / "cas", mode="record")
    assert isinstance(recorder, LLMProvider)
    assert recorder.provider == "mock" and recorder.model == "mock-extractive"
    assert recorder.params() == {"temperature": 0}

    first = recorder.complete(rag_prompt, system="sys", max_tokens=512, effort="low")
    assert inner.calls == 1 and first.cached is False
    key = recorder.key_for(rag_prompt, system="sys", max_tokens=512, effort="low")
    entry_path = recorder.path_for(key)
    assert entry_path.is_file()
    entry = json.loads(entry_path.read_text())
    assert entry["key"] == key
    assert entry["request"]["system"] == "sys" and entry["request"]["effort"] == "low"
    assert not list((tmp_path / "cas").glob("*.tmp"))

    # record mode is read-through: a second identical call is served from disk
    again = recorder.complete(rag_prompt, system="sys", max_tokens=512, effort="low")
    assert inner.calls == 1 and again.cached is True
    assert again.model_dump(exclude={"cached"}) == first.model_dump(exclude={"cached"})
    assert recorder.hits == 1 and recorder.misses == 1

    replayer = ReplayCacheProvider(CountingProvider(), cache_dir=tmp_path / "cas", mode="replay")
    replayed = replayer.complete(rag_prompt, system="sys", max_tokens=512, effort="low")
    assert replayed.cached is True and replayed.text == first.text
    assert replayer.inner.calls == 0  # type: ignore[attr-defined]


def test_changed_prompt_misses(tmp_path: Path, rag_prompt: list[Message]) -> None:
    recorder = ReplayCacheProvider(CountingProvider(), cache_dir=tmp_path, mode="record")
    recorder.complete(rag_prompt, system="sys")
    replayer = ReplayCacheProvider(CountingProvider(), cache_dir=tmp_path, mode="replay")
    with pytest.raises(CassetteMiss) as info:
        replayer.complete(rag_prompt, system="sys but different")
    assert info.value.key == replayer.key_for(rag_prompt, system="sys but different")
    # every keyed field participates: tools, json_schema, max_tokens, effort, messages
    tool = ToolSpec(name="t", description="d", input_schema={"type": "object"})
    for kwargs in (
        {"tools": [tool]},
        {"json_schema": {"type": "object"}},
        {"max_tokens": 4},
        {"effort": "high"},
    ):
        with pytest.raises(CassetteMiss):
            replayer.complete(rag_prompt, system="sys", **kwargs)  # type: ignore[arg-type]
    with pytest.raises(CassetteMiss):
        replayer.complete(rag_prompt + [Message(role="user", content="more")], system="sys")
    # the original still hits
    assert replayer.complete(rag_prompt, system="sys").cached is True


def test_replay_miss_never_calls_inner(tmp_path: Path, rag_prompt: list[Message]) -> None:
    inner = CountingProvider()
    replayer = ReplayCacheProvider(inner, cache_dir=tmp_path / "empty", mode="replay")
    with pytest.raises(CassetteMiss):
        replayer.complete(rag_prompt)
    assert inner.calls == 0
    assert not (tmp_path / "empty").exists()  # replay never creates the directory


def test_off_mode_passes_through(tmp_path: Path, rag_prompt: list[Message]) -> None:
    inner = CountingProvider()
    passthrough = ReplayCacheProvider(inner, cache_dir=tmp_path / "off", mode="off")
    passthrough.complete(rag_prompt)
    passthrough.complete(rag_prompt)
    assert inner.calls == 2
    assert not (tmp_path / "off").exists()


def test_key_matches_contract_fields(rag_prompt: list[Message]) -> None:
    request = canonical_request("openai", "gpt-x", rag_prompt, system="s", max_tokens=9)
    assert set(request) == {
        "provider",
        "model",
        "system",
        "messages",
        "tools",
        "json_schema",
        "max_tokens",
        "effort",
    }
    assert request["tools"] is None and request["json_schema"] is None
    key = request_key(request)
    assert len(key) == 64 and key == request_key(dict(reversed(list(request.items()))))


def test_replay_of_scripted_provider_keeps_usage(tmp_path: Path) -> None:
    scripted = ScriptedProvider([{"text": "x", "usage": {"input_tokens": 7, "output_tokens": 3}}])
    recorder = ReplayCacheProvider(scripted, cache_dir=tmp_path, mode="record")
    msgs = [Message(role="user", content="q")]
    recorder.complete(msgs)
    replayed = ReplayCacheProvider(scripted, cache_dir=tmp_path, mode="replay").complete(msgs)
    assert replayed.usage == Usage(input_tokens=7, output_tokens=3)
    assert replayed.provider == "scripted"


def test_bad_mode_and_corrupt_entry(tmp_path: Path, rag_prompt: list[Message]) -> None:
    with pytest.raises(ConfigError, match="cassette mode"):
        ReplayCacheProvider(MockProvider(), cache_dir=tmp_path, mode="sometimes")  # type: ignore[arg-type]
    replayer = ReplayCacheProvider(MockProvider(), cache_dir=tmp_path, mode="replay")
    replayer.path_for(replayer.key_for(rag_prompt)).write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="corrupt cassette"):
        replayer.complete(rag_prompt)
