"""Tests for secqa.core.errors."""

from __future__ import annotations

import pytest

from secqa.core.errors import (
    CalcRejected,
    CassetteMiss,
    ConfigError,
    IndexMismatch,
    ProviderError,
    ScenarioExhausted,
    SecqaError,
    SqlRejected,
)


def test_all_errors_share_a_base() -> None:
    for exc in (
        ConfigError("x"),
        ProviderError("x"),
        SqlRejected("x"),
        CalcRejected("x"),
        IndexMismatch("x"),
        CassetteMiss("key"),
        ScenarioExhausted("x"),
    ):
        assert isinstance(exc, SecqaError)
        assert isinstance(exc, Exception)


def test_provider_error_carries_retryable_flag() -> None:
    transient = ProviderError("rate limited", retryable=True, provider="openai")
    permanent = ProviderError("bad key")
    assert transient.retryable is True
    assert permanent.retryable is False
    assert str(transient) == "[openai] rate limited (retryable)"
    assert str(permanent) == "bad key"
    with pytest.raises(ProviderError) as info:
        raise transient
    assert info.value.provider == "openai"


def test_reason_errors_expose_reason() -> None:
    assert SqlRejected("no DDL").reason == "no DDL"
    assert CalcRejected("names not allowed").reason == "names not allowed"
    assert str(SqlRejected("no DDL")) == "no DDL"


def test_cassette_miss_message_and_key() -> None:
    miss = CassetteMiss("abc123")
    assert miss.key == "abc123"
    assert "abc123" in str(miss)
    assert str(CassetteMiss("k", "custom")) == "custom"


def test_package_reexports() -> None:
    import secqa
    from secqa import core

    assert secqa.__version__ == "0.1.0"
    assert core.ConfigError is ConfigError
    assert core.parse_number("12%") == 0.12
    assert callable(core.chunk_id)
