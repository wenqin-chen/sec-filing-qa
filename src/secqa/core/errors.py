"""Exception hierarchy shared by every module.

CONTRACTS.md rule (7): ``ProviderError(retryable)``, ``SqlRejected``, ``CalcRejected``,
``IndexMismatch``, ``CassetteMiss`` and ``ConfigError`` all live here so that callers can catch
them without importing the module that raises them.
"""

from __future__ import annotations


class SecqaError(Exception):
    """Base class for every error raised by secqa."""


class ConfigError(SecqaError):
    """Invalid or missing configuration (for example a provider that needs an absent API key)."""


class ProviderError(SecqaError):
    """An LLM provider call failed.

    ``retryable`` tells the caller whether a retry could succeed (rate limit, transient network
    error, 5xx) as opposed to a permanent failure (bad request, invalid key, unsupported model).
    Refusals are *not* errors: providers return ``stop_reason='refusal'`` instead of raising.
    """

    def __init__(self, message: str, *, retryable: bool = False, provider: str | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.provider = provider

    def __str__(self) -> str:
        base = super().__str__()
        prefix = f"[{self.provider}] " if self.provider else ""
        suffix = " (retryable)" if self.retryable else ""
        return f"{prefix}{base}{suffix}"


class SqlRejected(SecqaError):
    """The read-only SQL guard refused a statement. ``reason`` is safe to show to the model."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class CalcRejected(SecqaError):
    """The calculator refused an expression that is outside the AST whitelist."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class IndexMismatch(SecqaError):
    """An existing DuckDB index was built with a different embedder or embedding dimension."""


class CassetteMiss(SecqaError):
    """Replay mode found no recorded response for this exact request."""

    def __init__(self, key: str, message: str | None = None):
        super().__init__(message or f"no cassette entry for request key {key}")
        self.key = key


class ScenarioExhausted(SecqaError):
    """A scripted provider was asked for more turns than its scenario defines."""


__all__ = [
    "CalcRejected",
    "CassetteMiss",
    "ConfigError",
    "IndexMismatch",
    "ProviderError",
    "ScenarioExhausted",
    "SecqaError",
    "SqlRejected",
]
