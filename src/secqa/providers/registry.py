"""Provider spec -> :class:`LLMProvider` factory.

Specs: ``mock`` | ``mock:abstain`` | ``mock:extractive`` | ``scripted:<path>`` |
``openai[:<model>]`` | ``anthropic[:<model>]``. Vendor keys come from :class:`Settings` (never
read from ``os.environ`` here); a vendor spec without its key raises :class:`ConfigError` before
any SDK is imported. Vendor adapters get ``settings.provider_timeout_s`` (per HTTP attempt) and
``settings.provider_max_retries`` - never ``settings.request_timeout_s``, which is the request
deadline enforced separately through :mod:`secqa.providers.deadline`. When
``settings.cassette_mode`` is not ``off`` the provider is wrapped in a
:class:`ReplayCacheProvider` over ``settings.cassette_dir``.
"""

from __future__ import annotations

from pathlib import Path

from secqa.core.contracts import LLMProvider
from secqa.core.errors import ConfigError
from secqa.core.logging import get_logger
from secqa.core.settings import Settings, get_settings, provider_vendor
from secqa.providers.mock_provider import MockProvider
from secqa.providers.replay_provider import ReplayCacheProvider
from secqa.providers.scripted_provider import ScriptedProvider

log = get_logger("secqa.providers.registry")


def parse_spec(spec: str) -> tuple[str, str | None]:
    """Split ``'vendor:rest'`` into ``(vendor, rest)``; ``rest`` is ``None`` when absent."""
    vendor = provider_vendor(spec)
    rest = spec.split(":", 1)[1].strip() if ":" in spec else None
    return vendor, (rest or None)


def get_provider(spec: str, settings: Settings | None = None) -> LLMProvider:
    """Build the provider named by ``spec`` (see the module docstring for the grammar)."""
    settings = settings if settings is not None else get_settings()
    vendor, rest = parse_spec(spec)
    provider: LLMProvider

    if vendor == "mock":
        behaviour = rest or "extractive"
        if behaviour not in ("extractive", "abstain"):
            raise ConfigError(f"unknown mock behaviour {behaviour!r} in spec {spec!r}")
        provider = MockProvider(behaviour)  # type: ignore[arg-type]  # validated above
    elif vendor == "scripted":
        if not rest:
            raise ConfigError("scripted provider needs a scenario path: 'scripted:<path>'")
        provider = ScriptedProvider(Path(rest))
    elif vendor == "openai":
        settings.validate_provider_spec(spec)
        key = settings.key_for_provider(spec)
        from secqa.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            model=rest or settings.openai_model,
            api_key=key.get_secret_value() if key is not None else None,
            timeout_s=settings.provider_timeout_s,
            max_retries=settings.provider_max_retries,
        )
    elif vendor == "anthropic":
        settings.validate_provider_spec(spec)
        key = settings.key_for_provider(spec)
        from secqa.providers.anthropic_provider import DEFAULT_MODEL, AnthropicProvider

        provider = AnthropicProvider(
            model=rest or DEFAULT_MODEL,
            api_key=key.get_secret_value() if key is not None else None,
            timeout_s=settings.provider_timeout_s,
            max_retries=settings.provider_max_retries,
        )
    else:
        raise ConfigError(
            f"unknown provider vendor {vendor!r} in spec {spec!r}; "
            "expected mock | scripted | openai | anthropic"
        )

    if settings.cassette_mode != "off":
        provider = ReplayCacheProvider(
            provider, cache_dir=settings.cassette_dir, mode=settings.cassette_mode
        )
    log.debug("provider_resolved", spec=spec, provider=provider.provider, model=provider.model)
    return provider


__all__ = ["get_provider", "parse_spec"]
