"""Tests for secqa.core.settings."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from secqa.core.errors import ConfigError
from secqa.core.settings import Settings, get_settings, provider_vendor


def test_defaults_are_offline_safe() -> None:
    s = Settings(_env_file=None)
    assert s.provider == "mock"
    assert s.embedder == "hashing"
    assert s.duckdb_path == Path("data/index.duckdb")
    assert s.openai_api_key is None
    assert s.anthropic_api_key is None
    assert s.sec_user_agent is None
    assert s.cassette_mode == "off"
    assert s.max_k == 20
    assert s.max_agent_steps == 8
    assert s.max_cost_usd == 0.25
    assert s.daily_budget_usd == 5.0
    assert s.log_json is True
    s.validate_provider_keys()  # mock needs no key


def test_openai_provider_without_key_raises() -> None:
    s = Settings(_env_file=None, provider="openai:gpt-5.5")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        s.validate_provider_keys()


def test_anthropic_provider_without_key_raises() -> None:
    s = Settings(_env_file=None, provider="anthropic:claude-opus-5")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        s.validate_provider_keys()


def test_provider_with_key_passes() -> None:
    s = Settings(_env_file=None, provider="openai:gpt-5.5", openai_api_key=SecretStr("sk-test"))
    s.validate_provider_keys()
    assert s.key_for_provider("openai:gpt-5.5") is not None
    assert s.key_for_provider("mock") is None
    assert s.key_for_provider("scripted:tests/fixtures/scenarios/rag.yaml") is None


def test_blank_key_counts_as_missing() -> None:
    s = Settings(_env_file=None, provider="openai:gpt-5.5", openai_api_key=SecretStr("   "))
    with pytest.raises(ConfigError):
        s.validate_provider_keys()


def test_unknown_vendor_rejected() -> None:
    s = Settings(_env_file=None)
    with pytest.raises(ConfigError, match="unknown provider vendor"):
        s.validate_provider_spec("cohere:command")
    with pytest.raises(ConfigError, match="empty"):
        provider_vendor("   ")


def test_secret_is_not_printed() -> None:
    s = Settings(_env_file=None, openai_api_key=SecretStr("sk-secret-value"))
    assert "sk-secret-value" not in repr(s)
    assert "sk-secret-value" not in str(s.openai_api_key)


def test_user_agent_without_email_rejected() -> None:
    with pytest.raises(ValidationError, match="contact email"):
        Settings(_env_file=None, sec_user_agent="Wenqin Chen")


def test_user_agent_with_email_accepted() -> None:
    s = Settings(_env_file=None, sec_user_agent="  Jane Doe jane@example.com ")
    assert s.sec_user_agent == "Jane Doe jane@example.com"
    assert s.require_sec_user_agent() == "Jane Doe jane@example.com"


def test_blank_user_agent_treated_as_unset() -> None:
    s = Settings(_env_file=None, sec_user_agent="   ")
    assert s.sec_user_agent is None
    with pytest.raises(ConfigError, match="SEC_USER_AGENT"):
        s.require_sec_user_agent()


def test_env_override_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECQA_PROVIDER", "anthropic:claude-sonnet-5")
    monkeypatch.setenv("SECQA_MAX_K", "7")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Jane Doe jane@example.com")
    monkeypatch.setenv("SECQA_CASSETTE_MODE", "replay")
    monkeypatch.setenv("SECQA_LOG_JSON", "false")

    from_env = Settings(_env_file=None)
    assert from_env.provider == "anthropic:claude-sonnet-5"
    assert from_env.max_k == 7
    assert from_env.anthropic_api_key is not None
    assert from_env.anthropic_api_key.get_secret_value() == "sk-ant-test"
    assert from_env.sec_user_agent == "Jane Doe jane@example.com"
    assert from_env.cassette_mode == "replay"
    assert from_env.log_json is False
    from_env.validate_provider_keys()

    # explicit constructor arguments beat the environment
    explicit = Settings(_env_file=None, provider="mock", max_k=3)
    assert explicit.provider == "mock"
    assert explicit.max_k == 3


def test_prefixed_alias_for_conventional_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECQA_OPENAI_API_KEY", "sk-prefixed")
    s = Settings(_env_file=None)
    assert s.openai_api_key is not None
    assert s.openai_api_key.get_secret_value() == "sk-prefixed"


def test_dotenv_file_is_read_but_env_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SECQA_PROVIDER=openai:gpt-5.5\nSECQA_MAX_K=4\n", encoding="utf-8")
    monkeypatch.setenv("SECQA_MAX_K", "9")
    s = Settings(_env_file=env_file)
    assert s.provider == "openai:gpt-5.5"
    assert s.max_k == 9


def test_invalid_values_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cassette_mode="sometimes")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_k=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, provider="   ")


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECQA_PROVIDER", "mock:abstain")
    first = get_settings()
    assert first.provider == "mock:abstain"
    assert get_settings() is first
    monkeypatch.setenv("SECQA_PROVIDER", "mock")
    assert get_settings().provider == "mock:abstain"  # still cached
    get_settings.cache_clear()
    assert get_settings().provider == "mock"


def test_settings_override_fixture(settings_override: Callable[..., Settings]) -> None:
    s = settings_override(provider="mock", max_k=5, sec_user_agent="Jane Doe jane@example.com")
    assert s.max_k == 5
    assert s.duckdb_path.name == "index.duckdb"
    assert get_settings().max_k == 5  # env exported by the fixture
