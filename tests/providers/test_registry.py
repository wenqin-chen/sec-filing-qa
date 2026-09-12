"""get_provider: spec grammar, key checks, cassette wrapping; no SDK network access."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from secqa.core.errors import ConfigError
from secqa.core.settings import Settings
from secqa.providers.mock_provider import MockProvider
from secqa.providers.registry import get_provider, parse_spec
from secqa.providers.replay_provider import ReplayCacheProvider
from secqa.providers.scripted_provider import ScriptedProvider
from tests.providers.conftest import SCENARIOS


def test_parse_spec() -> None:
    assert parse_spec("mock") == ("mock", None)
    assert parse_spec("mock:abstain") == ("mock", "abstain")
    assert parse_spec("OpenAI:gpt-5.5") == ("openai", "gpt-5.5")
    assert parse_spec("scripted:tests/x.yaml") == ("scripted", "tests/x.yaml")
    with pytest.raises(ConfigError):
        parse_spec("")


def test_mock_specs() -> None:
    settings = Settings(_env_file=None)
    default = get_provider("mock", settings)
    assert isinstance(default, MockProvider) and default.behaviour == "extractive"
    abstain = get_provider("mock:abstain", settings)
    assert isinstance(abstain, MockProvider) and abstain.behaviour == "abstain"
    assert isinstance(get_provider("mock:extractive", settings), MockProvider)
    with pytest.raises(ConfigError, match="mock behaviour"):
        get_provider("mock:creative", settings)


def test_default_settings_are_used_when_none_given() -> None:
    provider = get_provider("mock")
    assert isinstance(provider, MockProvider)  # cassette_mode defaults to 'off'


def test_scripted_spec() -> None:
    settings = Settings(_env_file=None)
    provider = get_provider(f"scripted:{SCENARIOS / 'basic.yaml'}", settings)
    assert isinstance(provider, ScriptedProvider)
    with pytest.raises(ConfigError, match="scenario path"):
        get_provider("scripted", settings)
    with pytest.raises(ConfigError, match="not found"):
        get_provider("scripted:/nonexistent/file.yaml", settings)


@pytest.mark.parametrize("spec", ["openai:gpt-5.5", "anthropic:claude-opus-5", "openai"])
def test_vendor_without_key_raises_before_importing_sdk(spec: str) -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(ConfigError, match="_API_KEY"):
        get_provider(spec, settings)


def test_unknown_vendor() -> None:
    with pytest.raises(ConfigError, match="unknown provider vendor"):
        get_provider("cohere:command", Settings(_env_file=None))


def test_openai_with_key_builds_provider(settings_override: Callable[..., Settings]) -> None:
    pytest.importorskip("openai")
    settings = settings_override(openai_api_key="sk-test-not-real", openai_model="gpt-test")
    provider = get_provider("openai", settings)
    assert provider.provider == "openai" and provider.model == "gpt-test"
    explicit = get_provider("openai:gpt-5.4-mini", settings)
    assert explicit.model == "gpt-5.4-mini"


def test_anthropic_with_key_builds_provider(settings_override: Callable[..., Settings]) -> None:
    pytest.importorskip("anthropic")
    settings = settings_override(anthropic_api_key="sk-ant-test-not-real")
    provider = get_provider("anthropic", settings)
    assert provider.provider == "anthropic" and provider.model == "claude-opus-5"
    assert get_provider("anthropic:claude-sonnet-5", settings).model == "claude-sonnet-5"


def test_cassette_mode_wraps_provider(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, cassette_mode="record", cassette_dir=tmp_path / "cas")
    provider = get_provider("mock", settings)
    assert isinstance(provider, ReplayCacheProvider)
    assert provider.mode == "record" and provider.cache_dir == tmp_path / "cas"
    assert isinstance(provider.inner, MockProvider)
    assert provider.provider == "mock" and provider.model == "mock-extractive"
    replay = get_provider("mock", Settings(_env_file=None, cassette_mode="replay"))
    assert isinstance(replay, ReplayCacheProvider) and replay.mode == "replay"


def test_package_import_does_not_import_vendor_sdks() -> None:
    for name in ("openai", "anthropic"):
        sys.modules.pop(name, None)
    import importlib

    import secqa.providers as providers

    importlib.reload(providers)
    get_provider("mock", Settings(_env_file=None))
    assert "openai" not in sys.modules
    assert "anthropic" not in sys.modules
