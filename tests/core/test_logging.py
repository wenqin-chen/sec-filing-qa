"""Tests for secqa.core.logging."""

from __future__ import annotations

import json
import logging

import pytest

from secqa.core import logging as seclog


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    yield
    logging.getLogger().handlers[:] = []
    seclog._CONFIGURED["json"] = None
    seclog.clear_context()


def test_json_logging_emits_one_object_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    seclog.configure_logging(json=True)
    assert seclog.is_configured()
    seclog.bind_context(request_id="req-1", provider="mock")
    seclog.get_logger("secqa.test").info("answered", tokens=42)
    seclog.clear_context()
    seclog.get_logger("secqa.test").warning("after clear")

    lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["event"] == "answered"
    assert first["tokens"] == 42
    assert first["request_id"] == "req-1"
    assert first["provider"] == "mock"
    assert first["level"] == "info"
    assert first["logger"] == "secqa.test"
    assert "timestamp" in first
    assert "request_id" not in second
    assert second["level"] == "warning"


def test_stdlib_loggers_go_through_the_same_formatter(capsys: pytest.CaptureFixture[str]) -> None:
    seclog.configure_logging(json=True)
    logging.getLogger("thirdparty").error("boom %s", "x")
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["event"] == "boom x"
    assert record["level"] == "error"


def test_console_logging_is_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    seclog.configure_logging(json=False)
    seclog.get_logger("secqa.test").info("hello", k=1)
    err = capsys.readouterr().err
    assert "hello" in err
    assert "k=1" in err
    with pytest.raises(json.JSONDecodeError):
        json.loads(err.strip())


def test_get_logger_configures_lazily() -> None:
    assert not seclog.is_configured()
    seclog.get_logger("lazy")
    assert seclog.is_configured()
    assert seclog._CONFIGURED["json"] is False


def test_noisy_libraries_are_quietened() -> None:
    seclog.configure_logging(json=True, level="DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("httpx").level == logging.WARNING
