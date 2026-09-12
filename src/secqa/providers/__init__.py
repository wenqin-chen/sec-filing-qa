"""LLM providers: the only package that imports the ``openai`` / ``anthropic`` SDKs.

Importing this package never imports a vendor SDK; :class:`OpenAIProvider` and
:class:`AnthropicProvider` import theirs lazily at construction. Offline code paths use
:class:`MockProvider`, :class:`ScriptedProvider` and :class:`ReplayCacheProvider`;
:func:`get_provider` resolves a spec string such as ``'anthropic:claude-opus-5'``.
"""

from secqa.providers.anthropic_provider import (
    AnthropicProvider,
    anthropic_to_response,
    build_anthropic_request,
)
from secqa.providers.base import BaseProvider, canonical_request, request_key
from secqa.providers.deadline import CallBudget, call_budget, deadline, remaining_s
from secqa.providers.mock_provider import MockProvider
from secqa.providers.openai_provider import (
    OpenAIProvider,
    build_openai_request,
    openai_to_response,
)
from secqa.providers.pricing import DEFAULT_MODELS_YAML, ModelPrice, PriceTable
from secqa.providers.registry import get_provider, parse_spec
from secqa.providers.replay_provider import ReplayCacheProvider
from secqa.providers.scripted_provider import ScriptedProvider

__all__ = [
    "DEFAULT_MODELS_YAML",
    "AnthropicProvider",
    "BaseProvider",
    "CallBudget",
    "MockProvider",
    "ModelPrice",
    "OpenAIProvider",
    "PriceTable",
    "ReplayCacheProvider",
    "ScriptedProvider",
    "anthropic_to_response",
    "build_anthropic_request",
    "build_openai_request",
    "call_budget",
    "canonical_request",
    "deadline",
    "get_provider",
    "openai_to_response",
    "parse_spec",
    "remaining_s",
    "request_key",
]
