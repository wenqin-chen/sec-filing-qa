"""Turn-indexed scripted provider for agent, judge and prompt-injection tests.

A scenario is a YAML list (or a mapping with ``turns:`` and an optional ``model:``) of turns.
Turn ``i`` answers the ``i``-th ``complete`` call. Each turn may carry::

    - match: "search_filings"        # optional regex asserted against the last user/tool message
      text: "Net sales were ..."     # optional assistant text
      tool_calls:                    # optional
        - name: search_filings
          arguments: {query: "net sales 2023"}
          id: call-1                 # optional; generated when absent
      parsed: {answer: "...", abstain: false}   # optional; used when json_schema is requested
      stop_reason: end_turn          # optional; inferred from tool_calls when absent
      usage: {input_tokens: 100, output_tokens: 20}   # optional; estimated when absent

``match`` is an assertion, not a selector: when it does not match, the provider raises
:class:`ProviderError` with the regex and a preview of the message, so a test fails at the exact
turn where the agent diverged. Asking for more turns than the scenario has raises
:class:`ScenarioExhausted`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import yaml

from secqa.core.contracts import LLMResponse, Message, StopReason, ToolCall, ToolSpec, Usage
from secqa.core.errors import ConfigError, ProviderError, ScenarioExhausted
from secqa.providers.base import BaseProvider, Effort, estimate_usage, last_message_text

_STOP_REASONS: frozenset[str] = frozenset(
    {"end_turn", "tool_use", "max_tokens", "refusal", "other"}
)
_TURN_KEYS: frozenset[str] = frozenset(
    {"match", "text", "tool_calls", "parsed", "stop_reason", "usage"}
)


class ScriptedProvider(BaseProvider):
    """Replay a fixed list of turns; see the module docstring for the YAML format."""

    provider = "scripted"

    def __init__(self, scenario: Path | list[dict[str, Any]]):
        if isinstance(scenario, list):
            turns: list[dict[str, Any]] = scenario
            self.model = "scripted"
            self.source = "<list>"
        else:
            path = Path(scenario)
            if not path.is_file():
                raise ConfigError(f"scripted scenario not found: {path}")
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ConfigError(f"scenario {path} is not valid YAML: {exc}") from exc
            turns, name = _unpack(raw, str(path))
            self.model = name or f"scripted:{path.stem}"
            self.source = str(path)
        self._turns = [_validate_turn(t, i, self.source) for i, t in enumerate(turns)]
        self._cursor = 0

    @property
    def turns_total(self) -> int:
        """Number of scripted turns."""
        return len(self._turns)

    @property
    def turns_consumed(self) -> int:
        """Number of ``complete`` calls answered so far."""
        return self._cursor

    def reset(self) -> None:
        """Rewind to the first turn (reuse one provider across several test cases)."""
        self._cursor = 0

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
        """Return the next scripted turn (asserting its ``match`` regex when present)."""
        if self._cursor >= len(self._turns):
            raise ScenarioExhausted(
                f"scenario {self.source} has {len(self._turns)} turn(s); "
                f"call {self._cursor + 1} has no script"
            )
        index = self._cursor
        turn = self._turns[index]
        self._cursor += 1

        pattern = turn.get("match")
        if pattern is not None:
            target = last_message_text(messages)
            if re.search(pattern, target, flags=re.DOTALL) is None:
                preview = target[:200].replace("\n", " ")
                raise ProviderError(
                    f"scenario {self.source} turn {index}: match {pattern!r} failed "
                    f"against last message {preview!r}",
                    retryable=False,
                    provider=self.provider,
                )

        text: str = turn.get("text") or ""
        tool_calls = [
            ToolCall(
                id=str(call.get("id") or f"scripted-call-{index}-{j}"),
                name=str(call["name"]),
                arguments=dict(call.get("arguments") or {}),
            )
            for j, call in enumerate(turn.get("tool_calls") or [])
        ]
        stop_reason = _stop_reason(turn, tool_calls)
        parsed = _parsed(turn, text, json_schema)
        usage = _usage(turn) or estimate_usage(
            messages,
            system=system,
            tools=tools,
            output_text=text or json.dumps([c.arguments for c in tool_calls], sort_keys=True),
        )
        response = LLMResponse(
            text="" if stop_reason == "refusal" else text,
            tool_calls=tool_calls,
            usage=usage,
            provider=self.provider,
            model=self.model,
            latency_ms=0.0,
            stop_reason=stop_reason,
            parsed=parsed,
            raw_id=f"scripted-{index}",
        )
        self._log_response(response)
        return response


def _unpack(raw: Any, source: str) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(raw, list):
        return raw, None
    if isinstance(raw, dict) and isinstance(raw.get("turns"), list):
        model = raw.get("model")
        return raw["turns"], str(model) if model else None
    raise ConfigError(f"scenario {source} must be a list of turns or a mapping with 'turns'")


def _validate_turn(turn: Any, index: int, source: str) -> dict[str, Any]:
    if not isinstance(turn, dict):
        raise ConfigError(f"scenario {source} turn {index} must be a mapping")
    unknown = set(turn) - _TURN_KEYS
    if unknown:
        raise ConfigError(f"scenario {source} turn {index} has unknown keys {sorted(unknown)}")
    match = turn.get("match")
    if match is not None:
        try:
            re.compile(str(match))
        except re.error as exc:
            raise ConfigError(
                f"scenario {source} turn {index}: bad regex {match!r}: {exc}"
            ) from exc
        turn["match"] = str(match)
    for j, call in enumerate(turn.get("tool_calls") or []):
        if not isinstance(call, dict) or not call.get("name"):
            raise ConfigError(f"scenario {source} turn {index} tool_call {j} needs a 'name'")
        if call.get("arguments") is not None and not isinstance(call["arguments"], dict):
            raise ConfigError(f"scenario {source} turn {index} tool_call {j}: arguments not a map")
    stop = turn.get("stop_reason")
    if stop is not None and stop not in _STOP_REASONS:
        raise ConfigError(f"scenario {source} turn {index}: stop_reason {stop!r} is not valid")
    if turn.get("parsed") is not None and not isinstance(turn["parsed"], dict):
        raise ConfigError(f"scenario {source} turn {index}: parsed must be a mapping")
    return turn


def _stop_reason(turn: dict[str, Any], tool_calls: list[ToolCall]) -> StopReason:
    declared = turn.get("stop_reason")
    if declared is not None:
        return cast(StopReason, declared)  # validated against _STOP_REASONS at load time
    return "tool_use" if tool_calls else "end_turn"


def _parsed(
    turn: dict[str, Any], text: str, json_schema: dict[str, Any] | None
) -> dict[str, Any] | None:
    if json_schema is None:
        return None
    if turn.get("parsed") is not None:
        return dict(turn["parsed"])
    try:
        candidate = json.loads(text) if text else None
    except ValueError:
        return None
    return candidate if isinstance(candidate, dict) else None


def _usage(turn: dict[str, Any]) -> Usage | None:
    raw = turn.get("usage")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("scenario turn usage must be a mapping")
    return Usage(**{k: int(v) for k, v in raw.items()})


__all__ = ["ScriptedProvider"]
