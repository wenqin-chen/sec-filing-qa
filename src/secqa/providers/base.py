"""Shared plumbing for every LLM provider.

Every provider in this package implements :class:`secqa.core.contracts.LLMProvider`. This module
holds the pieces they share so that the vendor adapters stay small and each one can be defended
line by line:

* :class:`BaseProvider` - abstract base with ``provider``/``model`` attributes, ``params()`` (the
  sampling parameters a provider actually sends, recorded per run) and a helper that emits one
  structured log line per call.
* :func:`canonical_request` / :func:`request_key` - the deterministic serialisation used by the
  replay cache; the key is ``sha256`` of ``(provider, model, system, messages, tools, json_schema,
  max_tokens, effort)`` exactly as CONTRACTS.md specifies.
* Small text helpers used by the deterministic providers (``estimate_tokens``, ``first_user_text``,
  ``last_message_text``).

Token semantics (important for cost accounting and budget guards): ``Usage.input_tokens`` counts
the *uncached* prompt tokens. That is Anthropic's native meaning of ``input_tokens``; the OpenAI
translator subtracts ``cached_tokens`` from ``prompt_tokens`` so the two vendors agree. Total
prompt tokens are therefore ``input_tokens + cache_read_tokens + cache_write_tokens``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Literal

from secqa.core.contracts import LLMResponse, Message, ToolSpec, Usage
from secqa.core.ids import sha256_hex
from secqa.core.logging import get_logger

Effort = Literal["low", "medium", "high"]

log = get_logger("secqa.providers")


class BaseProvider(ABC):
    """Abstract base for every provider; subclasses set ``provider`` and ``model``."""

    provider: str
    model: str

    def params(self) -> dict[str, Any]:
        """Sampling parameters this provider sends (``temperature`` etc.), recorded per run.

        The default is empty: deterministic providers have nothing to record.
        """
        return {}

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        effort: Effort | None = None,
    ) -> LLMResponse:
        """Run one chat completion; see :class:`secqa.core.contracts.LLMProvider` for the rules."""

    def _log_response(self, response: LLMResponse) -> None:
        """Emit one structured log line per LLM call (rule 6 of CONTRACTS.md needs usage)."""
        log.info(
            "llm_call",
            provider=response.provider,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_tokens,
            cache_write_tokens=response.usage.cache_write_tokens,
            latency_ms=round(response.latency_ms, 1),
            stop_reason=response.stop_reason,
            tool_calls=len(response.tool_calls),
            cached=response.cached,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(provider={self.provider!r}, model={self.model!r})"


# ---------- canonical request serialisation (replay cache keys) ----------


def canonical_request(
    provider: str,
    model: str,
    messages: list[Message],
    *,
    system: str = "",
    tools: list[ToolSpec] | None = None,
    json_schema: dict[str, Any] | None = None,
    max_tokens: int = 2048,
    effort: Effort | None = None,
) -> dict[str, Any]:
    """Provider-neutral, JSON-serialisable view of one request (what a cassette key hashes)."""
    return {
        "provider": provider,
        "model": model,
        "system": system,
        "messages": [m.model_dump(mode="json") for m in messages],
        "tools": [t.model_dump(mode="json") for t in tools] if tools else None,
        "json_schema": json_schema,
        "max_tokens": max_tokens,
        "effort": effort,
    }


def request_key(request: dict[str, Any]) -> str:
    """``sha256`` of the canonical request with sorted keys and no whitespace."""
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_hex(payload)


# ---------- text helpers for deterministic providers ----------


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token) for offline providers."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_usage(
    messages: list[Message],
    *,
    system: str,
    tools: list[ToolSpec] | None,
    output_text: str,
) -> Usage:
    """Deterministic usage estimate for mock/scripted providers (never billed, always > 0)."""
    prompt_chars = len(system)
    for message in messages:
        prompt_chars += len(message.content)
        for call in message.tool_calls:
            prompt_chars += len(json.dumps(call.arguments, sort_keys=True))
        for result in message.tool_results:
            prompt_chars += len(result.content)
    for tool in tools or []:
        prompt_chars += len(tool.description) + len(json.dumps(tool.input_schema, sort_keys=True))
    return Usage(
        input_tokens=estimate_tokens(" " * prompt_chars),
        output_tokens=estimate_tokens(output_text),
    )


def first_user_text(messages: list[Message]) -> str:
    """Text of the first ``user`` message ('' if none) - the question in every secqa prompt."""
    for message in messages:
        if message.role == "user":
            return message.content
    return ""


def last_message_text(messages: list[Message]) -> str:
    """Text of the last ``user`` or ``tool`` message, tool results joined by newlines.

    The scripted provider matches its optional ``match`` regex against this string.
    """
    for message in reversed(messages):
        if message.role == "user":
            return message.content
        if message.role == "tool":
            return "\n".join(r.content for r in message.tool_results)
    return ""


__all__ = [
    "BaseProvider",
    "Effort",
    "canonical_request",
    "estimate_tokens",
    "estimate_usage",
    "first_user_text",
    "last_message_text",
    "request_key",
]
