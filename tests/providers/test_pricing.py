"""PriceTable: YAML loading, known usage -> USD, cache tokens priced, failure modes."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from secqa.core.contracts import Usage
from secqa.core.errors import ConfigError
from secqa.providers.pricing import DEFAULT_MODELS_YAML, ModelPrice, PriceTable
from tests.providers.conftest import FIXTURES

TABLE = FIXTURES / "providers_models.yaml"


def test_load_and_as_of() -> None:
    table = PriceTable.load(TABLE)
    assert table.as_of == date(2026, 9, 11)
    assert ("anthropic", "claude-test-large") in table.models
    assert ("openai", "flat-key-model") in table.models
    assert table.has("openai", "gpt-test") and table.has("mock", "anything")
    assert not table.has("openai", "gpt-unpriced")
    assert "as_of=2026-09-11" in repr(table)


def test_known_usage_to_usd() -> None:
    table = PriceTable.load(TABLE)
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
    # 1M input @ $5 + 100k output @ $25/1M = 5 + 2.5
    assert table.cost_usd("anthropic", "claude-test-large", usage) == pytest.approx(7.5)
    # 1M input @ $4 + 100k output @ $16/1M = 4 + 1.6
    assert table.cost_usd("openai", "gpt-test", usage) == pytest.approx(5.6)
    assert table.cost_usd("OpenAI", "gpt-test", usage) == pytest.approx(5.6)  # vendor case


def test_cache_tokens_priced_separately() -> None:
    table = PriceTable.load(TABLE)
    usage = Usage(
        input_tokens=200_000,
        output_tokens=10_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=100_000,
    )
    # small: 0.2M*2 + 1M*0.2 + 0.1M*2.5(cache_write) + 0.01M*10 = 0.4 + 0.2 + 0.25 + 0.1
    assert table.cost_usd("anthropic", "claude-test-small", usage) == pytest.approx(0.95)
    # large has no cache_write rate -> defaults to the input rate (5.0)
    # 0.2M*5 + 1M*0.5 + 0.1M*5 + 0.01M*25 = 1.0 + 0.5 + 0.5 + 0.25
    assert table.cost_usd("anthropic", "claude-test-large", usage) == pytest.approx(2.25)
    # openai cached input at the cached rate
    assert table.cost_usd(
        "openai", "gpt-test", Usage(input_tokens=0, cache_read_tokens=1_000_000)
    ) == pytest.approx(0.4)


def test_embedding_model_only_bills_input() -> None:
    table = PriceTable.load(TABLE)
    price = table.price("openai", "embed-test")
    assert price == ModelPrice(input=0.02, cached_input=0.02, output=0.0, cache_write=0.02)
    assert table.cost_usd("openai", "embed-test", Usage(input_tokens=500_000)) == pytest.approx(
        0.01
    )


def test_free_providers_cost_zero_even_when_unlisted() -> None:
    table = PriceTable.load(TABLE)
    usage = Usage(input_tokens=10**7, output_tokens=10**6)
    assert table.cost_usd("mock", "mock-extractive", usage) == 0.0
    assert table.cost_usd("scripted", "scripted-basic", usage) == 0.0


def test_unknown_model_raises() -> None:
    table = PriceTable.load(TABLE)
    with pytest.raises(ConfigError, match="no price for openai:gpt-unpriced"):
        table.cost_usd("openai", "gpt-unpriced", Usage(input_tokens=1))


def test_zero_usage_costs_zero() -> None:
    assert PriceTable.load(TABLE).cost_usd("openai", "gpt-test", Usage()) == 0.0


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"models": {"openai": {"m": {"input": 1}}}}, "as_of"),
        ({"as_of": "2026-13-01", "models": {"openai": {"m": {"input": 1}}}}, "YYYY-MM-DD"),
        ({"as_of": "2026-09-11"}, "no 'models'"),
        ({"as_of": "2026-09-11", "models": {"openai": {"m": {"output": 1}}}}, "'input' rate"),
        ({"as_of": "2026-09-11", "models": {"openai": {"m": {"input": "cheap"}}}}, "non-numeric"),
        ({"as_of": "2026-09-11", "models": {"openai": "oops"}}, "must map models"),
        ([], "must be a mapping"),
    ],
)
def test_malformed_tables_rejected(raw: object, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        PriceTable.from_dict(raw)


def test_missing_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        PriceTable.load(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("as_of: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        PriceTable.load(bad)


def test_default_path_points_at_eval_models_yaml() -> None:
    assert DEFAULT_MODELS_YAML.parts[-3:] == ("secqa", "eval", "models.yaml")
    if DEFAULT_MODELS_YAML.is_file():  # written by the eval module; not required here
        table = PriceTable.load()
        assert table.has("anthropic", "claude-opus-5") and table.has("openai", "gpt-5.5")


def test_negative_rate_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelPrice(input=-1.0, cached_input=0.0, output=0.0, cache_write=0.0)
