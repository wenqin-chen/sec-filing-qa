"""Anthropic adapter (Messages API with adaptive thinking, strict tools and JSON output).

The ``anthropic`` SDK is imported lazily inside ``AnthropicProvider.__init__`` (optional extra
``secqa[anthropic]``). Translation is split into two pure, fixture-testable functions:

* :func:`build_anthropic_request` - ``Message``/``ToolSpec``/``json_schema`` -> ``messages.create``
  kwargs.
* :func:`anthropic_to_response` - a ``Message`` response dict -> :class:`LLMResponse`.

Request shape (CONTRACTS.md rule 3 and SPEC section 5):

* ``thinking={"type": "adaptive"}``; ``output_config={"effort": effort or "medium"}`` plus
  ``"format": {"type": "json_schema", "schema": ...}`` when ``json_schema`` is given.
* Tools carry ``strict: true``; ``tool_choice={"type": "auto"}`` - never forced.
* NO ``temperature`` / ``top_p`` / ``top_k`` (Opus 5 / Sonnet 5 return 400 on them).
* All tool results of one step go back in ONE ``user`` message of ``tool_result`` blocks.
* ``stop_reason='refusal'`` is returned as-is with empty text; there is no fallback model.

Thinking replay: with thinking enabled, an assistant turn that requested tools must be echoed
back with its original content blocks (including the signed thinking block) when the tool results
are sent. Our provider-neutral :class:`Message` cannot carry those blocks, so the provider keeps a
small side table keyed by tool-call id and replays the recorded blocks verbatim when it sees an
assistant message whose tool calls it produced. Assistant messages it did not produce (scripted
history, cassette replays) are rebuilt from text + tool calls.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any

from secqa.core.contracts import (
    LLMResponse,
    Message,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
)
from secqa.core.errors import ConfigError, ProviderError
from secqa.core.logging import get_logger
from secqa.providers.base import BaseProvider, Effort
from secqa.providers.deadline import CallBudget, call_budget, remaining_s

log = get_logger("secqa.providers.anthropic")

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT: Effort = "medium"

_STOP_MAP: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}

# Bound on remembered assistant turns (each is a few KB); an agent run needs <= max_steps.
_SIDE_TABLE_MAX = 256


def build_anthropic_request(
    model: str,
    messages: list[Message],
    *,
    system: str = "",
    tools: list[ToolSpec] | None = None,
    json_schema: dict[str, Any] | None = None,
    max_tokens: int = 2048,
    effort: Effort | None = None,
    assistant_blocks: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Translate one provider-neutral request into ``messages.create`` kwargs.

    ``assistant_blocks`` maps a tool-call id to the raw content blocks of the assistant turn that
    produced it (thinking replay, see the module docstring); ``None`` disables replay.
    """
    wire_messages = [_to_wire_message(m, assistant_blocks or {}) for m in messages]
    output_config: dict[str, Any] = {"effort": effort or DEFAULT_EFFORT}
    if json_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": wire_messages,
        "thinking": {"type": "adaptive"},
        "output_config": output_config,
    }
    if system:
        request["system"] = system
    if tools:
        request["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "strict": True,
            }
            for tool in tools
        ]
        request["tool_choice"] = {"type": "auto"}
    return request


def _to_wire_message(
    message: Message, assistant_blocks: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    if message.role == "user":
        return {"role": "user", "content": [{"type": "text", "text": message.content}]}
    if message.role == "assistant":
        recorded = _recorded_blocks(message, assistant_blocks)
        if recorded is not None:
            return {"role": "assistant", "content": recorded}
        blocks: list[dict[str, Any]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        for call in message.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        return {"role": "assistant", "content": blocks}
    # role == 'tool': ONE user message carrying every result of the step.
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in message.tool_results
        ],
    }


def _recorded_blocks(
    message: Message, assistant_blocks: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]] | None:
    """Recorded content blocks for this assistant turn, if every tool call came from one turn."""
    if not message.tool_calls:
        return None
    first = assistant_blocks.get(message.tool_calls[0].id)
    if first is None:
        return None
    recorded_ids = {b.get("id") for b in first if b.get("type") == "tool_use"}
    if {c.id for c in message.tool_calls} != recorded_ids:
        return None
    return first


def anthropic_to_response(
    raw: dict[str, Any], latency_ms: float, *, expect_json: bool = False
) -> LLMResponse:
    """Translate a Messages API response (as a dict) into :class:`LLMResponse`.

    ``expect_json`` marks that ``output_config.format`` was requested, so the text is parsed into
    ``parsed`` (``None`` when parsing fails).
    """
    stop_raw = str(raw.get("stop_reason") or "")
    stop_reason: StopReason = _STOP_MAP.get(stop_raw, "other")

    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in raw.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            texts.append(block.get("text") or "")
        elif kind == "tool_use":
            arguments = block.get("input")
            if not isinstance(arguments, dict):
                raise ProviderError(
                    f"Anthropic tool_use {block.get('id')!r} input is not an object",
                    retryable=True,
                    provider="anthropic",
                )
            tool_calls.append(
                ToolCall(id=str(block.get("id")), name=str(block.get("name")), arguments=arguments)
            )
    if tool_calls and stop_reason == "end_turn":
        stop_reason = "tool_use"

    text = "" if stop_reason == "refusal" else "".join(texts)
    parsed: dict[str, Any] | None = None
    if expect_json and text:
        try:
            candidate = json.loads(text)
        except ValueError:
            candidate = None
        parsed = candidate if isinstance(candidate, dict) else None

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        usage=_usage_from(raw.get("usage") or {}),
        provider="anthropic",
        model=str(raw.get("model") or ""),
        latency_ms=latency_ms,
        stop_reason=stop_reason,
        parsed=parsed,
        raw_id=raw.get("id"),
    )


def _usage_from(usage: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
    )


class AnthropicProvider(BaseProvider):
    """``LLMProvider`` over the Anthropic Messages API; the model is fixed at construction."""

    provider = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout_s: float = 120,
        max_retries: int = 2,
    ):
        if not model or not model.strip():
            raise ConfigError("AnthropicProvider needs a model id, e.g. 'claude-opus-5'")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ConfigError(
                "the 'anthropic' package is not installed; install secqa[anthropic]"
            ) from exc
        self._sdk = anthropic
        self.model = model.strip()
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._assistant_blocks: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        try:
            self._client = anthropic.Anthropic(
                api_key=api_key, timeout=timeout_s, max_retries=max_retries
            )
        except anthropic.AnthropicError as exc:
            raise ConfigError(f"cannot create Anthropic client: {exc}") from exc

    def params(self) -> dict[str, Any]:
        """Sampling parameters sent for this model (recorded in every run config)."""
        return {
            "temperature": None,
            "top_p": None,
            "thinking": "adaptive",
            "effort_default": DEFAULT_EFFORT,
            "strict_tools": True,
            "tool_choice": "auto",
            "fallbacks": None,
        }

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
        """One ``messages.create`` call translated both ways (refusal is never retried)."""
        request = build_anthropic_request(
            self.model,
            messages,
            system=system,
            tools=tools,
            json_schema=json_schema,
            max_tokens=max_tokens,
            effort=effort,
            assistant_blocks=self._assistant_blocks,
        )
        client = self._client_for(self._call_budget())
        started = time.perf_counter()
        try:
            result = client.messages.create(**request)
        except self._sdk.AnthropicError as exc:
            raise self._translate_error(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw = result.model_dump(mode="json")
        response = anthropic_to_response(raw, latency_ms, expect_json=json_schema is not None)
        self._remember(raw, response)
        self._log_response(response)
        return response

    def _remember(self, raw: dict[str, Any], response: LLMResponse) -> None:
        """Keep the raw content blocks of a tool-using turn for thinking replay."""
        if not response.tool_calls:
            return
        blocks = list(raw.get("content") or [])
        for call in response.tool_calls:
            self._assistant_blocks[call.id] = blocks
        while len(self._assistant_blocks) > _SIDE_TABLE_MAX:
            self._assistant_blocks.popitem(last=False)

    def _call_budget(self) -> CallBudget:
        """Timeout / retries for the next call, shrunk to the active request deadline.

        Raises :class:`ProviderError` when the deadline already passed: starting a call that
        cannot finish would only pay for tokens the caller can no longer use.
        """
        budget = call_budget(self.timeout_s, self.max_retries, remaining_s())
        if budget.timeout_s <= 0:
            raise ProviderError(
                "request deadline exhausted before the Anthropic call was made",
                retryable=True,
                provider=self.provider,
            )
        return budget

    def _client_for(self, budget: CallBudget) -> Any:
        """The SDK client to call: a copy with shorter per-call options only when needed.

        ``with_options`` shares the underlying HTTP connection pool, so the copy is cheap; the
        constructed client is used unchanged when no deadline shortens the budget.
        """
        if budget.timeout_s == self.timeout_s and budget.max_retries == self.max_retries:
            return self._client
        log.debug(
            "provider_call_budget",
            provider=self.provider,
            timeout_s=round(budget.timeout_s, 3),
            max_retries=budget.max_retries,
        )
        return self._client.with_options(timeout=budget.timeout_s, max_retries=budget.max_retries)

    def _translate_error(self, exc: Exception) -> ProviderError:
        sdk = self._sdk
        retryable = isinstance(
            exc,
            (
                sdk.RateLimitError,
                sdk.APITimeoutError,
                sdk.APIConnectionError,
                sdk.InternalServerError,
            ),
        )
        status = getattr(exc, "status_code", None)
        if status is not None and int(status) >= 500:
            retryable = True
        detail = f"Anthropic {type(exc).__name__}"
        if status is not None:
            detail += f" (HTTP {status})"
        return ProviderError(f"{detail}: {exc}", retryable=retryable, provider=self.provider)


__all__ = [
    "DEFAULT_EFFORT",
    "DEFAULT_MODEL",
    "AnthropicProvider",
    "anthropic_to_response",
    "build_anthropic_request",
]
