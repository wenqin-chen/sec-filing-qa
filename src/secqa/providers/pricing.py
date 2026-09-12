"""Token -> USD accounting from ``models.yaml``.

The price table is a small YAML file owned by the eval module (``src/secqa/eval/models.yaml``);
this module only defines how it is read. Expected shape (prices are USD per 1M tokens)::

    as_of: 2026-09-11
    models:
      anthropic:
        claude-opus-5: {input: 5.0, cached_input: 0.50, output: 25.0}
      openai:
        gpt-5.5: {input: 5.0, cached_input: 0.50, output: 30.0}
        text-embedding-3-small: {input: 0.02}

``cached_input`` defaults to ``input`` and ``output`` to ``0`` (embedding models), and an optional
``cache_write`` (defaults to ``input``) prices Anthropic's cache-creation tokens. Flat
``"<provider>:<model>"`` keys directly under ``models`` are accepted too. Deterministic providers
(``mock``, ``scripted``) always cost ``0.0``; any other unknown ``(provider, model)`` raises
:class:`ConfigError` so a mispriced run fails loudly instead of reporting ``$0``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from secqa.core.contracts import Frozen, Usage
from secqa.core.errors import ConfigError

DEFAULT_MODELS_YAML = Path(__file__).resolve().parent.parent / "eval" / "models.yaml"

# Providers that never bill tokens (offline / deterministic).
FREE_PROVIDERS = frozenset({"mock", "scripted"})

_PER_MILLION = 1_000_000.0


class ModelPrice(Frozen):
    """USD per 1M tokens for one model."""

    input: float = Field(ge=0.0)
    cached_input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)
    cache_write: float = Field(ge=0.0)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, label: str) -> ModelPrice:
        """Build from a YAML mapping, applying the documented defaults."""
        if not isinstance(data, dict) or "input" not in data:
            raise ConfigError(f"models.yaml entry {label!r} must be a mapping with an 'input' rate")
        try:
            input_rate = float(data["input"])
            cached = float(data.get("cached_input", input_rate))
            output = float(data.get("output", 0.0))
            cache_write = float(data.get("cache_write", input_rate))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"models.yaml entry {label!r} has a non-numeric rate: {exc}") from exc
        return cls(input=input_rate, cached_input=cached, output=output, cache_write=cache_write)


class PriceTable:
    """Price lookup for ``(provider, model)`` pairs with an ``as_of`` date recorded per run."""

    def __init__(self, prices: dict[tuple[str, str], ModelPrice], as_of: date):
        self._prices = dict(prices)
        self.as_of = as_of

    @classmethod
    def load(cls, path: Path = DEFAULT_MODELS_YAML) -> PriceTable:
        """Read ``models.yaml``; raises :class:`ConfigError` when missing or malformed."""
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"price table not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"price table {path} is not valid YAML: {exc}") from exc
        return cls.from_dict(raw, source=str(path))

    @classmethod
    def from_dict(cls, raw: Any, *, source: str = "<dict>") -> PriceTable:
        """Build from an already-parsed mapping (used by tests and by ``load``)."""
        if not isinstance(raw, dict):
            raise ConfigError(f"price table {source} must be a mapping")
        as_of = _parse_as_of(raw.get("as_of"), source)
        models = raw.get("models")
        if not isinstance(models, dict) or not models:
            raise ConfigError(f"price table {source} has no 'models' mapping")
        prices: dict[tuple[str, str], ModelPrice] = {}
        for key, value in models.items():
            if ":" in str(key):
                provider, model = str(key).split(":", 1)
                prices[(provider.strip().lower(), model.strip())] = ModelPrice.from_mapping(
                    value, label=str(key)
                )
                continue
            if not isinstance(value, dict):
                raise ConfigError(f"price table {source}: provider {key!r} must map models")
            for model, rates in value.items():
                label = f"{key}:{model}"
                prices[(str(key).strip().lower(), str(model).strip())] = ModelPrice.from_mapping(
                    rates, label=label
                )
        return cls(prices, as_of)

    def has(self, provider: str, model: str) -> bool:
        """True if ``(provider, model)`` is priced (free providers count as priced)."""
        return provider.lower() in FREE_PROVIDERS or (provider.lower(), model) in self._prices

    def price(self, provider: str, model: str) -> ModelPrice:
        """Rates for a model; raises :class:`ConfigError` if it is not in the table."""
        try:
            return self._prices[(provider.lower(), model)]
        except KeyError:
            raise ConfigError(
                f"no price for {provider}:{model} in models.yaml (as_of {self.as_of.isoformat()})"
            ) from None

    def cost_usd(self, provider: str, model: str, usage: Usage) -> float:
        """USD cost of ``usage``: uncached input + cache reads + cache writes + output.

        ``usage.input_tokens`` is the *uncached* prompt count (see ``providers.base``), so cached
        tokens are never billed twice.
        """
        if provider.lower() in FREE_PROVIDERS:
            return 0.0
        rates = self.price(provider, model)
        cost = (
            usage.input_tokens * rates.input
            + usage.cache_read_tokens * rates.cached_input
            + usage.cache_write_tokens * rates.cache_write
            + usage.output_tokens * rates.output
        ) / _PER_MILLION
        return round(cost, 8)

    @property
    def models(self) -> list[tuple[str, str]]:
        """Every priced ``(provider, model)`` pair, sorted."""
        return sorted(self._prices)

    def __repr__(self) -> str:
        return f"PriceTable(as_of={self.as_of.isoformat()}, models={len(self._prices)})"


def _parse_as_of(value: Any, source: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ConfigError(f"price table {source}: as_of {value!r} is not YYYY-MM-DD") from exc
    raise ConfigError(f"price table {source} must declare 'as_of: YYYY-MM-DD'")


__all__ = ["DEFAULT_MODELS_YAML", "FREE_PROVIDERS", "ModelPrice", "PriceTable"]
