"""Runtime settings (pydantic-settings).

Environment variables use the ``SECQA_`` prefix (``SECQA_PROVIDER``, ``SECQA_DUCKDB_PATH`` ...)
except for the three conventional un-prefixed names ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY`` and
``SEC_USER_AGENT``. A ``.env`` file in the working directory is read when present; real environment
variables take precedence over it, and explicit constructor arguments take precedence over both.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from secqa.core.errors import ConfigError

CassetteMode = Literal["off", "record", "replay"]

# Deliberately simple: SEC only asks for "Name email"; we check that an email-shaped token exists.
# Public so the EDGAR client validates a User-Agent with the same rule.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Provider spec prefixes that need no vendor key (offline / deterministic providers).
_KEYLESS_PROVIDER_PREFIXES = ("mock", "scripted")


def provider_vendor(spec: str) -> str:
    """Return the vendor part of a provider spec: ``'openai:gpt-5.5'`` -> ``'openai'``."""
    if not spec or not spec.strip():
        raise ConfigError("provider spec is empty")
    return spec.split(":", 1)[0].strip().lower()


class Settings(BaseSettings):
    """All runtime configuration. Construct via :func:`get_settings` in application code."""

    model_config = SettingsConfigDict(
        env_prefix="SECQA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # storage / index
    duckdb_path: Path = Path("data/index.duckdb")
    index_url: str | None = None

    # models
    provider: str = "mock"
    judge_provider: str = "anthropic:claude-sonnet-5"
    openai_model: str = "gpt-5.5"
    embedder: str = "hashing"

    # credentials (un-prefixed conventional names accepted as well as SECQA_-prefixed ones)
    sec_user_agent: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SEC_USER_AGENT", "SECQA_SEC_USER_AGENT"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "SECQA_OPENAI_API_KEY"),
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "SECQA_ANTHROPIC_API_KEY"),
    )
    api_key: SecretStr | None = None  # SECQA_API_KEY: optional demo key gating agent mode

    # budgets and limits
    max_k: int = Field(default=20, ge=1)
    max_agent_steps: int = Field(default=8, ge=1)
    max_cost_usd: float = Field(default=0.25, ge=0.0)
    request_timeout_s: int = Field(default=90, ge=1)
    daily_budget_usd: float = Field(default=5.0, ge=0.0)
    rate_limit_per_min: int = Field(default=10, ge=1)

    # cassettes / logging
    cassette_mode: CassetteMode = "off"
    cassette_dir: Path = Path("cassettes")
    log_json: bool = True

    @field_validator("sec_user_agent")
    @classmethod
    def _user_agent_has_email(cls, value: str | None) -> str | None:
        """SEC fair-access policy: the declared User-Agent must carry a contact email."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if not EMAIL_RE.search(value):
            raise ValueError(
                "SEC_USER_AGENT must contain a contact email, e.g. 'Jane Doe jane@example.com'"
            )
        return value

    @field_validator("provider", "judge_provider", "embedder")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    # ---- helpers used by providers / api / cli ----

    def key_for_provider(self, spec: str) -> SecretStr | None:
        """Return the vendor key a provider spec needs, or ``None`` for key-less providers."""
        vendor = provider_vendor(spec)
        if vendor.startswith(_KEYLESS_PROVIDER_PREFIXES):
            return None
        if vendor == "openai":
            return self.openai_api_key
        if vendor == "anthropic":
            return self.anthropic_api_key
        raise ConfigError(f"unknown provider vendor {vendor!r} in spec {spec!r}")

    def validate_provider_spec(self, spec: str) -> None:
        """Raise :class:`ConfigError` if ``spec`` needs a vendor key that is not configured."""
        vendor = provider_vendor(spec)
        if vendor.startswith(_KEYLESS_PROVIDER_PREFIXES):
            return
        key = self.key_for_provider(spec)
        if key is None or not key.get_secret_value().strip():
            env_name = f"{vendor.upper()}_API_KEY"
            raise ConfigError(f"provider {spec!r} requires {env_name} but it is not set")

    def validate_provider_keys(self) -> None:
        """Startup check: the configured answering provider must have its key (mock needs none)."""
        self.validate_provider_spec(self.provider)

    def require_sec_user_agent(self) -> str:
        """Return the validated EDGAR User-Agent or raise :class:`ConfigError` if unset."""
        if self.sec_user_agent is None:
            raise ConfigError(
                "SEC_USER_AGENT is required to contact EDGAR; set it to 'Name email@example.com'"
            )
        return self.sec_user_agent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached; call ``get_settings.cache_clear()`` in tests)."""
    return Settings()


__all__ = ["CassetteMode", "Settings", "get_settings", "provider_vendor"]
