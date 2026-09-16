"""OpenAI adapter (chat completions with function tools and JSON-schema output).

The ``openai`` SDK is imported lazily inside ``OpenAIProvider.__init__`` so the package is an
optional extra (``secqa[openai]``) and importing ``secqa.providers`` never needs it. Two pure
functions do the translation and are unit-tested on recorded fixtures with no network:

* :func:`build_openai_request` - ``Message``/``ToolSpec``/``json_schema`` -> request kwargs.
* :func:`openai_to_response` - a ``ChatCompletion`` dict -> :class:`LLMResponse`.

Design notes (SPEC section 5):

* Tools are function tools with ``strict: true``; ``parallel_tool_calls`` is disabled while tools
  are offered because OpenAI does not guarantee strict schema adherence for parallel calls.
* ``json_schema`` becomes ``response_format={"type": "json_schema", ..., "strict": true}``.
* ``temperature=0`` is sent only to models that accept it (reasoning models - ``gpt-5*``, ``o*`` -
  reject any value other than the default); what was sent is recorded in :meth:`params`.
* ``effort`` maps to ``reasoning_effort`` on reasoning models and is dropped otherwise; on
  tool-calling turns it is forced to ``'none'`` because chat.completions rejects reasoning with
  function tools (HTTP 400 observed 2026-09-16; Responses-API migration is the recorded follow-up).
* Tool results: our contract carries all results of one step in ONE ``Message(role='tool')``;
  the wire format needs one ``role: tool`` message per ``tool_call_id``, so the adapter fans out.
* ``Usage.input_tokens`` = ``prompt_tokens - cached_tokens`` (see ``providers.base``).
"""

from __future__ import annotations

import json
import time
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

log = get_logger("secqa.providers.openai")

# Models that reject a non-default temperature and accept ``reasoning_effort``.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

_FINISH_TO_STOP: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def is_reasoning_model(model: str) -> bool:
    """True for OpenAI reasoning models (no ``temperature``, ``reasoning_effort`` accepted)."""
    return model.lower().startswith(_REASONING_PREFIXES)


def build_openai_request(
    model: str,
    messages: list[Message],
    *,
    system: str = "",
    tools: list[ToolSpec] | None = None,
    json_schema: dict[str, Any] | None = None,
    max_tokens: int = 2048,
    effort: Effort | None = None,
) -> dict[str, Any]:
    """Translate one provider-neutral request into ``chat.completions.create`` kwargs."""
    wire_messages: list[dict[str, Any]] = []
    if system:
        wire_messages.append({"role": "system", "content": system})
    for message in messages:
        wire_messages.extend(_to_wire_message(message))

    request: dict[str, Any] = {
        "model": model,
        "messages": wire_messages,
        "max_completion_tokens": max_tokens,
    }
    if tools:
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "strict": True,
                },
            }
            for tool in tools
        ]
        request["tool_choice"] = "auto"
        request["parallel_tool_calls"] = False
    if json_schema is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": str(json_schema.get("title") or "response"),
                "schema": json_schema,
                "strict": True,
            },
        }
    if is_reasoning_model(model):
        if tools:
            # Observed 2026-09-16 (gpt-5.4-mini-2026-03-17): chat.completions returns HTTP 400
            # "Function tools with reasoning_effort are not supported ... use /v1/responses or
            # set reasoning_effort to 'none'". Tool-calling turns therefore run without
            # reasoning on OpenAI models until the adapter moves to the Responses API
            # (docs/decisions.md, ADR-009).
            request["reasoning_effort"] = "none"
        elif effort is not None:
            request["reasoning_effort"] = effort
    else:
        request["temperature"] = 0
    return request


def _to_wire_message(message: Message) -> list[dict[str, Any]]:
    if message.role == "user":
        return [{"role": "user", "content": message.content}]
    if message.role == "assistant":
        wire: dict[str, Any] = {"role": "assistant", "content": message.content or None}
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        return [wire]
    # role == 'tool': one wire message per result, in order.
    return [
        {"role": "tool", "tool_call_id": result.tool_call_id, "content": result.content}
        for result in message.tool_results
    ]


def openai_to_response(
    raw: dict[str, Any], latency_ms: float, *, expect_json: bool = False
) -> LLMResponse:
    """Translate a ``ChatCompletion`` (as a dict) into :class:`LLMResponse`.

    ``expect_json`` marks that a ``json_schema`` was requested, so the text is parsed into
    ``parsed`` (``None`` when parsing fails - the caller decides what to do).
    """
    choices = raw.get("choices") or []
    if not choices:
        raise ProviderError("OpenAI response has no choices", retryable=False, provider="openai")
    choice = choices[0]
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")
    stop_reason: StopReason = _FINISH_TO_STOP.get(str(finish), "other")

    refusal = message.get("refusal")
    if refusal:
        stop_reason = "refusal"

    tool_calls: list[ToolCall] = []
    for call in message.get("tool_calls") or []:
        if call.get("type") not in (None, "function"):
            continue
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except ValueError as exc:
            raise ProviderError(
                f"OpenAI tool call {call.get('id')!r} has malformed JSON arguments: {exc}",
                retryable=True,
                provider="openai",
            ) from exc
        if not isinstance(arguments, dict):
            raise ProviderError(
                f"OpenAI tool call {call.get('id')!r} arguments are not an object",
                retryable=True,
                provider="openai",
            )
        tool_calls.append(
            ToolCall(id=str(call.get("id")), name=str(function.get("name")), arguments=arguments)
        )
    if tool_calls and stop_reason == "end_turn":
        stop_reason = "tool_use"

    text = "" if stop_reason == "refusal" else (message.get("content") or "")
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
        provider="openai",
        model=str(raw.get("model") or ""),
        latency_ms=latency_ms,
        stop_reason=stop_reason,
        parsed=parsed,
        raw_id=raw.get("id"),
    )


def _usage_from(usage: dict[str, Any]) -> Usage:
    prompt = int(usage.get("prompt_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    cache_write = int(details.get("cache_write_tokens") or 0)
    return Usage(
        input_tokens=max(prompt - cached - cache_write, 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cache_read_tokens=cached,
        cache_write_tokens=cache_write,
    )


class OpenAIProvider(BaseProvider):
    """``LLMProvider`` over OpenAI chat completions; the model is fixed at construction."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60,
        max_retries: int = 2,
    ):
        if not model or not model.strip():
            raise ConfigError("OpenAIProvider needs a model id, e.g. 'gpt-5.5'")
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ConfigError(
                "the 'openai' package is not installed; install secqa[openai]"
            ) from exc
        self._sdk = openai
        self.model = model.strip()
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        try:
            self._client = openai.OpenAI(
                api_key=api_key, timeout=timeout_s, max_retries=max_retries
            )
        except openai.OpenAIError as exc:
            raise ConfigError(f"cannot create OpenAI client: {exc}") from exc

    def params(self) -> dict[str, Any]:
        """Sampling parameters sent for this model (recorded in every run config)."""
        reasoning = is_reasoning_model(self.model)
        return {
            "temperature": None if reasoning else 0,
            "reasoning_effort": (
                "from effort; 'none' on tool-calling turns (chat.completions rejects "
                "reasoning_effort with function tools)"
                if reasoning
                else None
            ),
            "strict_tools": True,
            "parallel_tool_calls": False,
            "response_format": "json_schema (strict) when json_schema is given",
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
        """One ``chat.completions.create`` call translated both ways."""
        request = build_openai_request(
            self.model,
            messages,
            system=system,
            tools=tools,
            json_schema=json_schema,
            max_tokens=max_tokens,
            effort=effort,
        )
        client = self._client_for(self._call_budget())
        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(**request)
        except self._sdk.OpenAIError as exc:
            raise self._translate_error(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw = completion.model_dump(mode="json")
        response = openai_to_response(raw, latency_ms, expect_json=json_schema is not None)
        self._log_response(response)
        return response

    def _call_budget(self) -> CallBudget:
        """Timeout / retries for the next call, shrunk to the active request deadline.

        Raises :class:`ProviderError` when the deadline already passed: starting a call that
        cannot finish would only pay for tokens the caller can no longer use.
        """
        budget = call_budget(self.timeout_s, self.max_retries, remaining_s())
        if budget.timeout_s <= 0:
            raise ProviderError(
                "request deadline exhausted before the OpenAI call was made",
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
        detail = f"OpenAI {type(exc).__name__}"
        if status is not None:
            detail += f" (HTTP {status})"
        return ProviderError(f"{detail}: {exc}", retryable=retryable, provider=self.provider)


__all__ = ["OpenAIProvider", "build_openai_request", "is_reasoning_model", "openai_to_response"]
